// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "aligned_file_reader.h"
#include "logger.h"
#include "pq_flash_index.h"
#include <malloc.h>
#include "liburing.h"
#include "percentile_stats.h"

#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include "parameters.h"
#include "timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "cosine_similarity.h"
#include "tsl/robin_set.h"
#include "linux_aligned_file_reader.h"

namespace diskann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  PQFlashIndex<T, TagT>::PQFlashIndex(diskann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
                                      SearchMode search_mode, bool tags, Parameters *params)
      : reader(fileReader), data_is_normalized(false), single_index_file(false), enable_tags(tags) {
    // initialize the variables based on the search mode.

    // no_mapping: for Starling, an id -> location mapping should be maintained.
    this->no_mapping = (search_mode != SearchMode::PAGE_SEARCH);
    // for pipe and coro, we use SQ polling for faster I/O issue.
    // Other search modes use synchronous I/O, so no need for SQ polling.
    this->sq_poll = (search_mode == SearchMode::PIPE_SEARCH || search_mode == SearchMode::CORO_SEARCH);
#ifndef USE_AIO
    io_uring_flag = this->sq_poll ? IORING_SETUP_SQPOLL : 0;
#endif

    LOG(INFO) << "Using search mode: " << search_mode << ", no_mapping: " << this->no_mapping
              << ", sq_poll: " << this->sq_poll;

    if (m == diskann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        diskann::cout << "Cosine metric chosen for (normalized) float data."
                         "Changing distance to L2 to boost accuracy."
                      << std::endl;
        m = diskann::Metric::L2;
        data_is_normalized = true;

      } else {
        diskann::cerr << "WARNING: Cannot normalize integral data types."
                      << " This may result in erroneous results or poor recall."
                      << " Consider using L2 distance with integral data types." << std::endl;
      }
    }

    this->dist_cmp.reset(diskann::get_distance_function<T>(m));
    this->dist_cmp_float.reset(diskann::get_distance_function<float>(m));

    // this->pq_reader = new LinuxAlignedFileReader();
    if (params != nullptr) {
      this->beam_width = params->Get<uint32_t>("beamwidth");
      this->l_index = params->Get<uint32_t>("L");
      this->range = params->Get<uint32_t>("R");
      this->maxc = params->Get<uint32_t>("C");
      this->alpha = params->Get<float>("alpha");
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
  }

  template<typename T, typename TagT>
  PQFlashIndex<T, TagT>::~PQFlashIndex() {
    assert(!this->thread_data.empty());

    if (centroid_data != nullptr) {
      aligned_free(centroid_data);
    }

    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      delete[] nhood_cache_buf;
      diskann::aligned_free(coord_cache_buf);
    }

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
      // delete reader; //not deleting reader because it is now passed by ref.
    }

    if (medoids != nullptr) {
      delete[] medoids;
    }
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::setup_thread_data(_u64 nthreads) {
    LOG(INFO) << "Setting up thread-specific contexts for nthreads: " << nthreads;

    // omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int) kMaxThreads)
    for (_s64 thread = 0; thread < (_s64) kMaxThreads; thread++) {
#pragma omp critical
      {
        this->reader->register_thread();
        auto ctx = this->reader->get_ctx();
        QueryScratch<T> scratch;
        _u64 coord_alloc_size = ROUND_UP(MAX_N_CMPS * this->aligned_dim, 256);
        diskann::alloc_aligned((void **) &scratch.coord_scratch, coord_alloc_size, 256);
        diskann::alloc_aligned((void **) &scratch.sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, SECTOR_LEN);
        diskann::alloc_aligned((void **) &scratch.aligned_scratch, 256 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_pq_coord_scratch, 32768 * 32 * sizeof(_u8), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch, 25600 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch, 512 * sizeof(float), 256);
        diskann::alloc_aligned((void **) &scratch.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));
        diskann::alloc_aligned((void **) &scratch.aligned_query_float, this->aligned_dim * sizeof(float),
                               8 * sizeof(float));
        diskann::alloc_aligned((void **) &scratch.update_buf, (2 * MAX_N_EDGES + 1) * SECTOR_LEN,
                               SECTOR_LEN);  // 2x for read + write

        scratch.visited = new tsl::robin_set<_u64>(4096);
        scratch.page_visited = new tsl::robin_set<unsigned>(4096);

        memset(scratch.sector_scratch, 0, MAX_N_SECTOR_READS * SECTOR_LEN);
        memset(scratch.aligned_scratch, 0, 256 * sizeof(float));
        memset(scratch.coord_scratch, 0, coord_alloc_size);
        memset(scratch.aligned_query_T, 0, this->aligned_dim * sizeof(T));
        memset(scratch.aligned_query_float, 0, this->aligned_dim * sizeof(float));
        memset(scratch.update_buf, 0, (2 * MAX_N_EDGES + 1) * SECTOR_LEN);

        this->reader->register_buf(scratch.sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, ReaderMRID::kSector);
        this->reader->register_buf(scratch.update_buf, (2 * MAX_N_EDGES + 1) * SECTOR_LEN, ReaderMRID::kUpdate);
        this->reader->register_buf(scratch.aligned_pq_coord_scratch, 32768 * 32 * sizeof(_u8), ReaderMRID::kPQ);

        ThreadData<T> data;
        data.ctx = ctx;
        data.scratch = scratch;
        this->thread_data.push(data);
        this->thread_data_backing_buf.push_back(data);
      }
    }
    load_flag = true;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::destroy_thread_data() {
    // TODO(gh): destruct thread_queue and other readers.
    for (auto &data : this->thread_data_backing_buf) {
      auto &scratch = data.scratch;
      diskann::aligned_free((void *) scratch.coord_scratch);
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);
      diskann::aligned_free((void *) scratch.update_buf);
    }
    this->reader->deregister_all_threads();
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::use_medoids_data_as_centroids() {
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    alloc_aligned(((void **) &centroid_data), num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    // borrow ctx
    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    void *ctx = data.ctx;
    diskann::cout << "Loading centroid data from medoids vector data of " << num_medoids << " medoid(s)" << std::endl;
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      auto medoid = medoids[cur_m];
      // read medoid nhood
      std::vector<IORequest> medoid_read(1);
      medoid_read[0].u_offset = u_loc_offset(medoid);
      medoid_read[0].u_len = max_node_len;
      medoid_read[0].len = size_per_io;
      medoid_read[0].buf = data.scratch.sector_scratch;

      char *medoid_node_buf = nullptr;
      if (likely(no_mapping)) {
        medoid_read[0].offset = loc_sector_no(medoid) * SECTOR_LEN;
        reader->read(medoid_read, ctx);
        // all data about medoid
        medoid_node_buf = offset_to_loc(data.scratch.sector_scratch, medoid);
      } else {
        medoid_read[0].offset = node_sector_no(medoid) * SECTOR_LEN;
        reader->read(medoid_read, ctx);
        // all data about medoid
        medoid_node_buf = offset_to_node(data.scratch.sector_scratch, medoid);
      }
      // add medoid coords to `coord_cache`
      T *medoid_coords = new T[data_dim];
      T *medoid_disk_coords = offset_to_node_coords(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, data_dim * sizeof(T));

      for (uint32_t i = 0; i < data_dim; i++)
        centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
      delete[] medoid_coords;
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path,
                                             const _u32 num_threads, const _u32 mem_L) {
    if (mem_index_path.empty()) {
      diskann::cerr << "mem_index_path is needed" << std::endl;
      exit(1);
    }
    mem_index_ = std::make_unique<diskann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int PQFlashIndex<T, TagT>::load(const char *index_prefix, _u32 num_threads, bool new_index_format,
                                  bool use_page_search) {
    std::string pq_table_bin, pq_compressed_vectors, disk_index_file, centroids_file;

    std::string iprefix = std::string(index_prefix);
    pq_table_bin = iprefix + "_pq_pivots.bin";
    pq_compressed_vectors = iprefix + "_pq_compressed.bin";
    disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    centroids_file = disk_index_file + "_centroids.bin";

    std::ifstream index_metadata(disk_index_file, std::ios::binary);

    size_t tags_offset = 0;
    size_t pq_pivots_offset = 0;
    size_t pq_vectors_offset = 0;
    _u64 disk_nnodes;
    _u64 disk_ndims;
    size_t medoid_id_on_file;
    _u64 file_frozen_id;

    if (new_index_format) {
      _u32 nr, nc;

      READ_U32(index_metadata, nr);
      READ_U32(index_metadata, nc);

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, disk_ndims);

      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      data_dim = disk_ndims;
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
      if (max_degree != this->range) {
        this->range = max_degree;
      }

      LOG(INFO) << "Disk-Index File Meta-data: "
                << "# nodes per sector: " << nnodes_per_sector << ", max node len (bytes): " << max_node_len
                << ", max node degree: " << max_degree << ", npts: " << nr << ", dim: " << nc
                << " disk_nnodes: " << disk_nnodes << " disk_ndims: " << disk_ndims;

      if (nnodes_per_sector > this->kMaxElemInAPage) {
        LOG(ERROR) << "nnodes_per_sector: " << nnodes_per_sector << " is greater than " << this->kMaxElemInAPage
                   << ". Please recompile with a higher value of kMaxElemInAPage.";
        return -1;
      }

      READ_U64(index_metadata, this->num_frozen_points);
      READ_U64(index_metadata, file_frozen_id);
      if (this->num_frozen_points == 1) {
        this->frozen_location = file_frozen_id;
        // if (this->num_frozen_points == 1) {
        diskann::cout << " Detected frozen point in index at location " << this->frozen_location
                      << ". Will not output it at search time." << std::endl;
      }
      READ_U64(index_metadata, tags_offset);
      READ_U64(index_metadata, pq_pivots_offset);
      READ_U64(index_metadata, pq_vectors_offset);

      diskann::cout << "Tags offset: " << tags_offset << " PQ Pivots offset: " << pq_pivots_offset
                    << " PQ Vectors offset: " << pq_vectors_offset << std::endl;
    } else {  // old index file format
      size_t actual_index_size = get_file_size(disk_index_file);
      size_t expected_file_size;
      READ_U64(index_metadata, expected_file_size);
      if (actual_index_size != expected_file_size) {
        diskann::cout << "File size mismatch for " << disk_index_file << " (size: " << actual_index_size << ")"
                      << " with meta-data size: " << expected_file_size << std::endl;
        return -1;
      }

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

      diskann::cout << "Disk-Index File Meta-data: ";
      diskann::cout << "# nodes per sector: " << nnodes_per_sector;
      diskann::cout << ", max node len (bytes): " << max_node_len;
      diskann::cout << ", max node degree: " << max_degree << std::endl;
    }

    size_per_io = SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    LOG(INFO) << "Size per IO: " << size_per_io;

    index_metadata.close();

    if (this->single_index_file) {
      pq_table_bin = disk_index_file;
      pq_compressed_vectors = disk_index_file;
    } else {
      pq_pivots_offset = 0;
      pq_vectors_offset = 0;
    }
    LOG(INFO) << "After single file index check, Tags offset: " << tags_offset
              << " PQ Pivots offset: " << pq_pivots_offset << " PQ Vectors offset: " << pq_vectors_offset;

    size_t npts_u64, nchunks_u64;
    diskann::load_bin<_u8>(pq_compressed_vectors, data, npts_u64, nchunks_u64, pq_vectors_offset);
    this->num_points = this->init_num_pts = npts_u64;
    this->n_chunks = nchunks_u64;

    this->cur_id = this->num_points;

    diskann::cout << "Load compressed vectors from file: " << pq_compressed_vectors << " offset: " << pq_vectors_offset
                  << " num points: " << npts_u64 << " n_chunks: " << nchunks_u64 << std::endl;

    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64, pq_pivots_offset);

    if (disk_nnodes != num_points) {
      diskann::cout << "Mismatch in #points for compressed data file and disk "
                       "index file: "
                    << disk_nnodes << " vs " << num_points << std::endl;
      return -1;
    }

    this->data_dim = pq_table.get_dim();
    this->aligned_dim = ROUND_UP(this->data_dim, 8);

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << num_points
                  << " #dim: " << data_dim << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks << std::endl;

    // read index metadata
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname, true, false);
    // pq_reader->open(pq_compressed_vectors, true, false);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;
    LOG(INFO) << "Setup thread data threads " << num_threads;

    // load page layout and set cur_loc
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      diskann::cout << "Loading tags from " << tag_file << std::endl;
      this->load_tags(tag_file);
    }

    num_medoids = 1;
    medoids = new uint32_t[1];
    medoids[0] = (_u32) (medoid_id_on_file);
    use_medoids_data_as_centroids();
    diskann::cout << "PQFlashIndex loaded successfully." << std::endl;
    return 0;
  }

  template<typename T, typename TagT>
  _u64 PQFlashIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::compute_pq_dists(const _u32 src, const _u32 *ids, float *fp_dists, const _u32 count,
                                               uint8_t *aligned_scratch) {
    const _u8 *src_ptr = this->data.data() + (this->n_chunks * src);
    if (aligned_scratch == nullptr) {
      assert(false);
    }
    // aggregate PQ coords into scratch
    ::aggregate_coords(ids, count, this->data.data(), this->n_chunks, aligned_scratch);
    // compute distances
    this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, fp_dists, count);
  }

  template<typename T, typename TagT>
  std::vector<_u8> PQFlashIndex<T, TagT>::deflate_vector(const T *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    std::vector<float> fp_vec(this->data_dim);
    for (uint32_t i = 0; i < this->data_dim; i++) {
      fp_vec[i] = (float) vec[i];
    }
    this->pq_table.deflate_vec(fp_vec.data(), pq_coords.data());
    return pq_coords;
  }

  template<>
  std::vector<_u8> PQFlashIndex<float>::deflate_vector(const float *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    this->pq_table.deflate_vec(vec, pq_coords.data());
    return pq_coords;
  }

  template<typename T, typename TagT>
  void PQFlashIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      if (!no_mapping) {
#pragma omp parallel for num_threads(max_nthreads)
        for (size_t i = 0; i < this->num_points; ++i) {
          tags.insert_or_assign(i, i);
        }
      }
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      diskann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());
      id2loc_.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
    }
    LOG(INFO) << "Loaded " << tags.size() << " tags";
  }

  // instantiations
  // template class PQFlashIndex<float, int32_t>;
  // template class PQFlashIndex<_s8, int32_t>;
  // template class PQFlashIndex<_u8, int32_t>;
  template class PQFlashIndex<float, uint32_t>;
  template class PQFlashIndex<_s8, uint32_t>;
  template class PQFlashIndex<_u8, uint32_t>;
  // template class PQFlashIndex<float, int64_t>;
  // template class PQFlashIndex<_s8, int64_t>;
  // template class PQFlashIndex<_u8, int64_t>;
  // template class PQFlashIndex<float, uint64_t>;
  // template class PQFlashIndex<_s8, uint64_t>;
  // template class PQFlashIndex<_u8, uint64_t>;
}  // namespace diskann
