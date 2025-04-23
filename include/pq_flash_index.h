// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <string>

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "neighbor.h"
#include "index.h"
#include "defs.h"

#include "windows_customizations.h"

#define MAX_N_CMPS 16384
#define MAX_N_EDGES 512
#define MAX_PQ_CHUNKS 100

constexpr int kIndexSizeFactor = 2;

enum SearchMode { BEAM_SEARCH = 0, PAGE_SEARCH = 1, PIPE_SEARCH = 2, CORO_SEARCH = 3 };

inline void crash() {
#ifdef ANN_VERIFY
  __builtin_trap();
#endif
}

namespace {
  inline void aggregate_coords(const unsigned *ids, const _u64 n_ids, const _u8 *all_coords, const _u64 ndims,
                               _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  inline void prefetch_chunk_dists(const float *ptr) {
    _mm_prefetch((char *) ptr, _MM_HINT_NTA);
    _mm_prefetch((char *) (ptr + 64), _MM_HINT_NTA);
    _mm_prefetch((char *) (ptr + 128), _MM_HINT_NTA);
    _mm_prefetch((char *) (ptr + 192), _MM_HINT_NTA);
  }

  inline void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts, const _u64 pq_nchunks, const float *pq_dists,
                             float *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);

    prefetch_chunk_dists(pq_dists);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        prefetch_chunk_dists(chunk_dists + 256);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }
}  // namespace

namespace diskann {
  template<typename T>
  struct ThreadData {
    QueryScratch<T> scratch;
    void *ctx;
  };

#define IO_LIMIT __INT_MAX__

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

  template<typename T, typename TagT = uint32_t>
  class PQFlashIndex {
   public:
    // Gopal. Adapting to the new Bing interface. Since the DiskPriorityIO is
    // now a singleton, we have to take it in the DiskANNInterface and
    // pass it around. Since I don't want to pollute this interface with Bing
    // classes, this class takes an AlignedFileReader object that can be
    // created the way we need. Linux will create a simple AlignedFileReader
    // and pass it. Regular Windows code should create a BingFileReader using
    // the DiskPriorityIOInterface class, and for running on XTS, create a
    // BingFileReader using the object passed by the XTS environment. Freeing
    // the
    // reader object is now the client's (DiskANNInterface's) responsibility.
    DISKANN_DLLEXPORT PQFlashIndex(diskann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
                                   SearchMode search_mode, bool tags = false, Parameters *parameters = nullptr);

    DISKANN_DLLEXPORT ~PQFlashIndex();

    // returns region of `node_buf` containing [COORD(T)]
    inline T *offset_to_node_coords(const char *node_buf) {
      return (T *) node_buf;
    }

    // returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
    inline unsigned *offset_to_node_nhood(const char *node_buf) {
      return (unsigned *) (node_buf + data_dim * sizeof(T));
    }

    // obtains region of sector containing node
    inline char *offset_to_node(const char *sector_buf, uint32_t node_id) {
      return offset_to_loc(sector_buf, id2loc(node_id));
    }

    // sector # on disk where node_id is present
    inline uint64_t node_sector_no(uint32_t node_id) {
      return loc_sector_no(id2loc(node_id));
    }

    inline uint64_t u_node_offset(uint32_t node_id) {
      return u_loc_offset(id2loc(node_id));
    }

    // unaligned offset to location
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * max_node_len;
    }

    inline char *offset_to_loc(const char *sector_buf, uint64_t loc) {
      return (char *) sector_buf + (nnodes_per_sector == 0 ? 0 : (loc % nnodes_per_sector) * max_node_len);
    }

    // avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (nnodes_per_sector > 0 ? loc / nnodes_per_sector : loc * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return (sector_no - 1) * nnodes_per_sector + sector_off;
    }

    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(const char *index_prefix, uint32_t num_threads, bool new_index_format = true,
                               bool use_page_search = false);

    DISKANN_DLLEXPORT void load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path,
                                          const _u32 num_threads, const _u32 mem_L);

    DISKANN_DLLEXPORT void load_page_layout(const std::string &index_prefix, const _u64 nnodes_per_sector = 0,
                                            const _u64 num_points = 0);

    DISKANN_DLLEXPORT void load_tags(const std::string &tag_file, size_t offset = 0);

    DISKANN_DLLEXPORT _u64 return_nd();

    // search.
    DISKANN_DLLEXPORT void disk_iterate_to_fixed_point_v(
        const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
        std::vector<Neighbor> &expanded_nodes_info, tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
        QueryStats *stats = nullptr, ThreadData<T> *passthrough_data = nullptr,
        tsl::robin_set<uint32_t> *exclude_nodes = nullptr, bool dyn_search_l = true);

    DISKANN_DLLEXPORT size_t beam_search(const T *query, const _u64 k_search, const _u64 l_search, TagT *res_tags,
                                         float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr,
                                         const _u32 mem_L = 0, tsl::robin_set<uint32_t> *deleted_nodes = nullptr,
                                         bool dyn_search_l = true);

    DISKANN_DLLEXPORT size_t page_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                         TagT *res_tags, float *res_dists, const _u64 beam_width,
                                         QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT size_t pipe_search(const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                         TagT *res_tags, float *res_dists, const _u64 beam_width,
                                         QueryStats *stats = nullptr);

    // best-first search with inter-query parallelism.
    DISKANN_DLLEXPORT size_t coro_search(T **queries, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                         TagT **res_tags, float **res_dists, const _u64 beam_width, int N);

    static constexpr int kMaxCoroPerThread = 8;
    static constexpr int kMaxVectorDim = 512;
    struct alignas(SECTOR_LEN) CoroDataOne {
      // scratch.
      char sectors[SECTOR_LEN * 128];
      T query[kMaxVectorDim];
      _u8 pq_coord_scratch[32768 * 32];
      float pq_dists[32768];
      T data_buf[ROUND_UP(1024 * kMaxVectorDim, 256)];
      float dist_scratch[512];
      _u64 data_buf_idx;
      _u64 sector_idx;

      // search state.
      std::vector<Neighbor> full_retset;
      std::vector<Neighbor> retset;
      tsl::robin_set<_u64> visited;

      std::vector<unsigned> frontier;
      using fnhood_t = std::tuple<unsigned, unsigned, char *>;
      std::vector<fnhood_t> frontier_nhoods;
      std::vector<IORequest> frontier_read_reqs;

      PQFlashIndex<T> *parent;
      unsigned cur_list_size, cmps, k;

      void compute_dists(const unsigned *ids, const _u64 n_ids, float *dists_out) {
        ::aggregate_coords(ids, n_ids, parent->data.data(), parent->n_chunks, pq_coord_scratch);
        ::pq_dist_lookup(pq_coord_scratch, n_ids, parent->n_chunks, pq_dists, dists_out);
      };

      void print() {
        LOG(INFO) << "Full retset size " << full_retset.size() << " retset size: " << retset.size()
                  << " visited size: " << visited.size() << " frontier size: " << frontier.size()
                  << " frontier nhood size: " << frontier_nhoods.size()
                  << " frontier read reqs size: " << frontier_read_reqs.size();
      }

      void reset() {
        data_buf_idx = 0;
        sector_idx = 0;
        visited.clear();  // does not deallocate memory.
        retset.resize(4096);
        retset.clear();
        full_retset.clear();
        cur_list_size = cmps = k = 0;
      }

      void compute_and_add_to_retset(const unsigned *node_ids, const _u64 n_ids) {
        compute_dists(node_ids, n_ids, dist_scratch);
        for (_u64 i = 0; i < n_ids; ++i) {
          auto &item = retset[cur_list_size];
          item.id = node_ids[i];
          item.distance = dist_scratch[i];
          item.flag = true;
          cur_list_size++;
          visited.insert(node_ids[i]);
        }
      };

      void issue_next_io_batch(const _u64 beam_width, void *ctx) {
        if (search_ends()) {
          return;
        }
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        sector_idx = 0;

        _u32 marker = k;
        _u32 num_seen = 0;
        while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
          if (retset[marker].flag) {
            num_seen++;
            frontier.push_back(retset[marker].id);
            retset[marker].flag = false;
          }
          marker++;
        }

        // read nhoods of frontier ids
        std::vector<uint32_t> locked;
        if (!frontier.empty()) {
          for (_u64 i = 0; i < frontier.size(); i++) {
            uint32_t loc = frontier[i];
            uint64_t offset = parent->loc_sector_no(loc) * SECTOR_LEN;
            auto sector_buf = sectors + sector_idx * parent->size_per_io;
            fnhood_t fnhood = std::make_tuple(loc, loc, sector_buf);
            sector_idx++;
            frontier_nhoods.push_back(fnhood);
            // RDMA not supported yet.
            frontier_read_reqs.emplace_back(IORequest(offset, parent->size_per_io, sector_buf, 0, 0));
          }
          parent->reader->send_io(frontier_read_reqs, ctx, false);
        }
      }

      bool io_finished(void *ctx) {
        parent->reader->poll(ctx);
        for (auto &req : frontier_read_reqs) {
          if (!req.finished) {
            return false;
          }
        }
        return true;
      }

      void explore_frontier(uint64_t l_search) {
        auto nk = cur_list_size;

        for (auto &frontier_nhood : frontier_nhoods) {
          auto [id, loc, sector_buf] = frontier_nhood;
          char *node_disk_buf = parent->offset_to_loc(sector_buf, loc);
          unsigned *node_buf = parent->offset_to_node_nhood(node_disk_buf);
          _u64 nnbrs = (_u64) (*node_buf);
          T *node_fp_coords = parent->offset_to_node_coords(node_disk_buf);

          T *node_fp_coords_copy = data_buf + (data_buf_idx * parent->aligned_dim);
          data_buf_idx++;
          memcpy(node_fp_coords_copy, node_fp_coords, parent->data_dim * sizeof(T));
          float cur_expanded_dist =
              parent->dist_cmp->compare(query, node_fp_coords_copy, (unsigned) parent->aligned_dim);

          Neighbor n(id, cur_expanded_dist, true);
          full_retset.push_back(n);

          unsigned *node_nbrs = (node_buf + 1);
          // compute node_nbrs <-> query dist in PQ space
          compute_dists(node_nbrs, nnbrs, dist_scratch);

          // process prefetch-ed nhood
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              cmps++;
              float dist = dist_scratch[m];
              if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
                continue;
              Neighbor nn(id, dist, true);
              // variable search_L for deleted nodes.
              // Return position in sorted list where nn inserted.

              auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

              if (cur_list_size < l_search) {
                ++cur_list_size;
              }

              if (r < nk)
                nk = r;
            }
          }
        }

        if (nk <= k)
          k = nk;  // k is the best position in retset updated in this round.
        else
          ++k;
      }

      bool search_ends() {
        // this->print();
        return k >= cur_list_size;
      }
    };

    struct alignas(SECTOR_LEN) CoroData {
      CoroDataOne data[kMaxCoroPerThread];
      CoroData(PQFlashIndex<T> *parent) {
        for (int i = 0; i < kMaxCoroPerThread; ++i) {
          data[i].parent = parent;
        }
      }
    };

    // gives access to backing thread data buf for easy parallelization
    std::vector<ThreadData<T>> &get_thread_data() {
      return this->thread_data_backing_buf;
    }

    // computes PQ dists between src->[ids] into fp_dists (merge, insert)
    DISKANN_DLLEXPORT void compute_pq_dists(const _u32 src, const _u32 *ids, float *fp_dists, const _u32 count,
                                            uint8_t *aligned_scratch = nullptr);

    // deflates `vec` into PQ ids
    DISKANN_DLLEXPORT std::vector<_u8> deflate_vector(const T *vec);
    std::pair<_u8 *, _u32> get_pq_config() {
      return std::make_pair(this->data.data(), (uint32_t) this->n_chunks);
    }

    DISKANN_DLLEXPORT _u64 get_num_frozen_points() {
      return this->num_frozen_points;
    }

    DISKANN_DLLEXPORT _u64 get_frozen_loc() {
      return this->frozen_location;
    }

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

   protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

   public:
    // data info
    _u64 num_points = 0;
    _u64 init_num_pts = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 aligned_dim = 0;
    _u64 size_per_io = 0;

    std::string _disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;

    std::shared_ptr<AlignedFileReader> &reader;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    std::vector<_u8> data;
    _u64 chunk_size;
    _u64 n_chunks;
    FixedChunkPQTable<T> pq_table;

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;
    std::vector<unsigned> memid2diskid_;

    // distance comparator
    std::shared_ptr<Distance<T>> dist_cmp;
    std::shared_ptr<Distance<float>> dist_cmp_float;

   public:
    // test the estimation efficacy.
    uint32_t beam_width, l_index, range, maxc;
    float alpha;
    // assumed max thread, only the first nthreads are initialized.
    static constexpr int kMaxThreads = 56;
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;
    static constexpr uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;

    // page search
    bool use_page_search_ = true;

    libcuckoo::cuckoohash_map<uint32_t, uint32_t> id2loc_;  // id -> loc (start from 0)
    uint32_t id2loc(uint32_t id) {
      if (likely(no_mapping)) {
        return id;
      } else {
        uint32_t loc = 0;
        if (id2loc_.find(id, loc)) {
          return loc;
        } else {
          LOG(ERROR) << "id " << id << " not found in id2loc";
          crash();
          return kInvalidID;
        }
      }
    }

    uint64_t id2page(uint32_t id) {
      uint32_t loc = id2loc(id);
      if (loc == kInvalidID) {
        return kInvalidID;
      }
      return loc_sector_no(loc);
    }

    static constexpr uint32_t kMaxElemInAPage = 12;
    using PageArr = std::array<uint32_t, kMaxElemInAPage>;
    libcuckoo::cuckoohash_map<uint32_t, PageArr> page_layout;  // page_id (start from loc_sector_no(0)) -> ids

    std::mutex alloc_lock;
    uint32_t loc2id(uint32_t loc) {
      uint32_t page = loc_sector_no(loc);
      uint32_t offset = loc % nnodes_per_sector;
      uint32_t id = kInvalidID;
      page_layout.find_fn(page, [&](PageArr &v) { id = v[offset]; });
      return id;  // kInvalidID if fails.
    }

    void set_loc2id(uint32_t loc, uint32_t id) {
      uint32_t page = loc_sector_no(loc);
      uint32_t offset = loc % nnodes_per_sector;
      page_layout.upsert(page, [&](PageArr &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
            v[i] = kInvalidID;
          }
        }
        v[offset] = id;
      });
    }

    void erase_loc2id(uint32_t loc) {
      uint32_t page = loc_sector_no(loc);
      uint32_t offset = loc % nnodes_per_sector;
      page_layout.upsert(page, [&](PageArr &v) {
        v[offset] = kInvalidID;
        bool empty = true;
        for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
          if (v[i] != kInvalidID) {
            empty = false;
            break;
          }
        }
        if (empty) {
          empty_pages.push(page);
        }
      });
    }

    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids) {
      std::lock_guard<std::mutex> lock(alloc_lock);
      for (uint32_t i = 0; i < new_locs.size(); ++i) {
        set_loc2id(new_locs[i], new_ids[i]);
      }
      for (auto &l : old_locs) {
        erase_loc2id(l);
      }
      // set_loc2id(locs[i], new_nhood[i]);
      // for (auto &l : orig_locs) {
      //   erase_loc2id(l);
      // }
    }

    // returns <loc, need_read>
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read) {
      std::lock_guard<std::mutex> lock(alloc_lock);
      std::vector<uint64_t> ret;
      int cur = 0;
      // reuse.
      uint32_t threshold = (nnodes_per_sector + kIndexSizeFactor - 1) / kIndexSizeFactor;

      uint32_t empty_page = kInvalidID;
      while ((empty_page = empty_pages.pop()) != kInvalidID) {
#ifdef NO_POLLUTE_ORIGINAL
        if (empty_page < loc_sector_no(init_num_pts)) {
          continue;
        }
#endif
        // allocate all the pages.
        page_layout.update_fn(empty_page, [&](PageArr &v) {
          for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
            if (v[i] != kInvalidID) {
              LOG(ERROR) << "Page " << empty_page << " is not empty " << i << " " << v[i];
              crash();
            }
            v[i] = kAllocatedID;
            ret.push_back(sector_to_loc(empty_page, i));
            cur++;
            if (cur == n) {
              return;
            }
          }
        });
        if (cur == n) {
          return ret;
        }
      }

      for (auto &p : hint_pages) {
#ifdef NO_POLLUTE_ORIGINAL
        if (p < loc_sector_no(init_num_pts)) {
          continue;
        }
#endif
        // see the hole number
        page_layout.update_fn(p, [&](PageArr &v) {
          uint32_t cnt = 0;
          for (auto &id : v) {
            if (id == kInvalidID) {
              cnt++;
            }
          }
          if (cnt < threshold) {
            return;
          }

          if (cnt < nnodes_per_sector) {
            page_need_to_read.insert(p);
          }
          // alloc them.
          for (uint32_t i = 0; i < nnodes_per_sector; ++i) {
            if (v[i] == kInvalidID) {
              v[i] = kAllocatedID;
              ret.push_back(sector_to_loc(p, i));
              cur++;
              if (cur == n) {
                return;
              }
            }
          }
        });

        if (cur == n) {
          return ret;
        }
      }

      // Avoid reads as much as possible.
      // to make things simpler, use aligned loc instead of pinning log zones.
      // add holes in cur_loc.
      uint64_t align_loc = cur_loc;
      if (align_loc % nnodes_per_sector != 0) {
        align_loc += nnodes_per_sector - (align_loc % nnodes_per_sector);
      }

      int remaining = n - cur;
      for (int i = 0; i < remaining; i++) {
        set_loc2id(align_loc + i, kAllocatedID);
        ret.push_back(align_loc + i);
      }
      cur_loc = align_loc + remaining;
      return ret;
    }
    std::atomic<_u64> cur_id, cur_loc;

    // merge deletes (NOTE: index read-only during merge.)
    void merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                       const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                       uint32_t nthreads, const uint32_t &n_sampled_nbrs);

   private:
    // Are we dealing with normalized data? This will be true
    // if distance == COSINE and datatype == float. Required
    // because going forward, we will normalize vectors when
    // asked to search with COSINE similarity. Of course, this
    // will be done only for floating point vectors.
    bool data_is_normalized = false;

    // medoid/start info
    uint32_t *medoids = nullptr;     // by default it is just one entry point of graph, we
                                     // can optionally have multiple starting points
    size_t num_medoids;              // by default it is set to 1
    float *centroid_data = nullptr;  // by default, it is empty. If there are multiple
                                     // centroids, we pick the medoid corresponding to the
                                     // closest centroid as the starting point of search

    // nhood_cache
    unsigned *nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T *coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    std::vector<ThreadData<T>> thread_data_backing_buf;
    _u64 max_nthreads;
    bool load_flag = false;
    bool count_visited_nodes = false;

    bool single_index_file = false;  // single index file not supported yet.
    bool no_mapping = false;
    bool sq_poll = true;

    // support for tags and dynamic indexing

    bool enable_tags = false;

    /* diskv2 extra API requirements */
    // ids that don't have disk nhoods, but have in-mem PQ
    tsl::robin_set<_u32> invalid_ids;
    std::mutex invalid_ids_lock;
  };
}  // namespace diskann