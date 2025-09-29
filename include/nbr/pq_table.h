#pragma once

#include "utils.h"
#include <immintrin.h>
#include <sstream>
#include <string_view>

#define NUM_PQ_CENTROIDS 256
#define NUM_PQ_OFFSETS 5

namespace pipeann {
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    //    uint64_t   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    uint64_t   chunk_size;  // chunk_size = chunk size of each dimension chunk
    float *tables = nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    float *centroid = nullptr;
    uint32_t *chunk_offsets = nullptr;
    uint32_t *rearrangement = nullptr;
    float *tables_T = nullptr;  // same as pq_tables, but col-major
    float *all_to_all_dists = nullptr;

   public:
    uint64_t ndims;  // ndims = chunk_size * n_chunks
    uint64_t n_chunks;

    uint64_t all_to_all_dist_size() {
      return sizeof(float) * n_chunks * NUM_PQ_CENTROIDS * NUM_PQ_CENTROIDS;
    }

    FixedChunkPQTable() {
    }

    FixedChunkPQTable &operator=(FixedChunkPQTable &&other) noexcept {
      if (this != &other) {
        this->ndims = other.ndims;
        this->n_chunks = other.n_chunks;
        this->tables = other.tables;
        this->tables_T = other.tables_T;
        this->rearrangement = other.rearrangement;
        this->chunk_offsets = other.chunk_offsets;
        this->centroid = other.centroid;
        this->all_to_all_dists = other.all_to_all_dists;

        other.tables = nullptr;
        other.tables_T = nullptr;
        other.rearrangement = nullptr;
        other.chunk_offsets = nullptr;
        other.centroid = nullptr;
        other.all_to_all_dists = nullptr;
      }
      return *this;
    }

    virtual ~FixedChunkPQTable() {
      destroy_table();
    }

    void destroy_table() {
      if (tables != nullptr) {
        delete[] tables;
        tables = nullptr;
      }
      if (tables_T != nullptr) {
        delete[] tables_T;
        tables_T = nullptr;
      }
      if (rearrangement != nullptr) {
        delete[] rearrangement;
        rearrangement = nullptr;
      }
      if (chunk_offsets != nullptr) {
        delete[] chunk_offsets;
        chunk_offsets = nullptr;
      }
      if (centroid != nullptr) {
        delete[] centroid;
        centroid = nullptr;
      }
      if (all_to_all_dists != nullptr) {
        delete[] all_to_all_dists;
        all_to_all_dists = nullptr;
      }
    }

    uint64_t get_dim() {
      return ndims;
    }

    void load_pq_pivots_new(std::basic_istream<char> &reader, size_t num_chunks, size_t offset) {
      uint64_t nr, nc;
      std::unique_ptr<uint64_t[]> file_offset_data;
      uint64_t *file_offset_data_raw;
      pipeann::load_bin_impl<uint64_t>(reader, file_offset_data_raw, nr, nc, offset);
      file_offset_data.reset(file_offset_data_raw);

      if (nr != NUM_PQ_OFFSETS) {
        LOG(ERROR) << "Pivot offset incorrect, # offsets = " << nr << ", but expecting " << NUM_PQ_OFFSETS;
        crash();
      }

      pipeann::load_bin_impl<float>(reader, tables, nr, nc, file_offset_data[0] + offset);

      if ((nr != NUM_PQ_CENTROIDS)) {
        LOG(ERROR) << "Num centers incorrect, centers = " << nr << " but expecting " << NUM_PQ_CENTROIDS;
        crash();
      }

      this->ndims = nc;
      pipeann::load_bin_impl<float>(reader, centroid, nr, nc, file_offset_data[1] + offset);

      if ((nr != this->ndims) || (nc != 1)) {
        LOG(ERROR) << "Centroid file dim incorrect: row " << nr << ", col " << nc << " expecting " << this->ndims;
        crash();
      }

      pipeann::load_bin_impl<uint32_t>(reader, rearrangement, nr, nc, file_offset_data[2] + offset);
      if ((nr != this->ndims) || (nc != 1)) {
        LOG(ERROR) << "Rearrangement incorrect: row " << nr << ", col " << nc << " expecting " << this->ndims;
        crash();
      }

      pipeann::load_bin_impl<uint32_t>(reader, chunk_offsets, nr, nc, file_offset_data[3] + offset);

      if (nr != (uint64_t) num_chunks + 1 || nc != 1) {
        LOG(ERROR) << "Chunk offsets: nr=" << nr << ", nc=" << nc << ", expecting nr=" << num_chunks + 1 << ", nc=1.";
        crash();
      }

      this->n_chunks = num_chunks;
      LOG(INFO) << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS << ", #dims: " << this->ndims
                << ", #chunks: " << this->n_chunks;
    }

    void save_pq_pivots(const char *pq_pivots_path) {
      std::vector<size_t> offs(NUM_PQ_OFFSETS, 0);
      offs[0] = METADATA_SIZE;
      offs[1] = offs[0] + pipeann::save_bin<float>(pq_pivots_path, tables, NUM_PQ_CENTROIDS, this->ndims, offs[0]);
      offs[2] = offs[1] + pipeann::save_bin<float>(pq_pivots_path, centroid, this->ndims, 1, offs[1]);
      offs[3] = offs[2] + pipeann::save_bin<uint32_t>(pq_pivots_path, rearrangement, this->ndims, 1, offs[2]);
      offs[4] = offs[3] + pipeann::save_bin<uint32_t>(pq_pivots_path, chunk_offsets, this->n_chunks + 1, 1, offs[3]);
      pipeann::save_bin<uint64_t>(pq_pivots_path, offs.data(), offs.size(), 1, 0);
    }

    void post_load_pq_table() {
      // alloc and compute transpose
      pipeann::alloc_aligned((void **) &tables_T, 256 * ndims * sizeof(float), 64);
      // tables_T = new float[256 * ndims];
      for (uint64_t i = 0; i < 256; i++) {
        for (uint64_t j = 0; j < ndims; j++) {
          tables_T[j * 256 + i] = tables[i * ndims + j];
        }
      }

      // added this for easy PQ-PQ squared-distance calculations
      // TODO: Create only for StreamingMerger.
      if (all_to_all_dists != nullptr) {
        delete[] all_to_all_dists;
      }
      all_to_all_dists = new float[256 * 256 * n_chunks];
      std::memset(all_to_all_dists, 0, 256 * 256 * n_chunks * sizeof(float));
      // should perhaps optimize later
      for (uint32_t i = 0; i < 256; i++) {
        for (uint32_t j = 0; j < 256; j++) {
          for (uint32_t c = 0; c < n_chunks; c++) {
            for (uint64_t d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
              float diff = (tables[i * ndims + d] - tables[j * ndims + d]);
              all_to_all_dists[i * 256 * n_chunks + j * n_chunks + c] += diff * diff;
            }
          }
        }
      }
    }

    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks, size_t offset = 0) {
      std::string pq_pivots_path(pq_table_file);
      uint64_t nr, nc;
      get_bin_metadata(pq_table_file, nr, nc, offset);
      std::ifstream reader(pq_table_file, std::ios::binary | std::ios::ate);
      reader.seekg(0);
      load_pq_pivots_new(reader, num_chunks, offset);
      post_load_pq_table();
      LOG(INFO) << "Finished optimizing for PQ-PQ distance compuation";
    }

    void populate_chunk_distances(const T *query_vec, float *dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          uint64_t permuted_dim_in_query = rearrangement[j];
          const float *centers_dim_vec = tables_T + (256 * j);
          for (uint64_t idx = 0; idx < 256; idx++) {
            double diff = centers_dim_vec[idx] - (query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query]);
            chunk_dists[idx] += (float) (diff * diff);
          }
        }
      }
    }

    void populate_chunk_distances_nt(const T *query_vec, float *dist_vec) {
#ifdef USE_AVX512
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (uint64_t chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (uint64_t j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          uint64_t permuted_dim_in_query = rearrangement[j];
          float *centers_dim_vec = tables_T + (256 * j);
          for (uint64_t idx = 0; idx < 256; idx += 16) {
            __m512i center_i = _mm512_stream_load_si512(centers_dim_vec + idx);  // avoid cache thrashing
            __m512 center_f = _mm512_castsi512_ps(center_i);
            __m512 query_f = _mm512_set1_ps(query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query]);
            __m512 diff = _mm512_sub_ps(center_f, query_f);
            __m512 diff_sq = _mm512_mul_ps(diff, diff);
            __m512 chunk_dists_v = _mm512_load_ps(chunk_dists + idx);
            chunk_dists_v = _mm512_add_ps(chunk_dists_v, diff_sq);
            _mm512_store_ps(chunk_dists + idx, chunk_dists_v);  // dist_vec should be in cache.
          }
        }
      }
#else
      // TODO: for non-AVX512 machines, do non-temporal population.
      return populate_chunk_distances(query_vec, dist_vec);
#endif
    }

    // computes PQ distance between comp_src and comp_dsts in efficient manner
    // comp_src: [nchunks]
    // comp_dsts: count * [nchunks]
    // dists: [count]
    // TODO (perf) :: re-order computation to get better locality
    void compute_distances_alltoall(const uint8_t *comp_src, const uint8_t *comp_dsts, float *dists,
                                    const uint32_t count) {
      std::memset(dists, 0, count * sizeof(float));
      for (uint64_t i = 0; i < count; i++) {
        for (uint64_t c = 0; c < n_chunks; c++) {
          dists[i] += all_to_all_dists[(uint64_t) comp_src[c] * 256 * n_chunks +
                                       (uint64_t) comp_dsts[i * n_chunks + c] * n_chunks + c];
        }
      }
    }

    // fp_vec: [ndims]
    // out_pq_vec : [nchunks]
    void deflate_vec(const float *fp_vec, uint8_t *out_pq_vec) {
      // permute the vector according to PQ rearrangement, compute all distances
      // to 256 centroids and choose the closest (for each chunk)
      for (uint32_t c = 0; c < n_chunks; c++) {
        float closest_dist = std::numeric_limits<float>::max();
        for (uint32_t i = 0; i < 256; i++) {
          float cur_dist = 0;
          for (uint64_t d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
            float diff = (tables[i * ndims + d] - ((float) fp_vec[rearrangement[d]] - centroid[rearrangement[d]]));
            cur_dist += diff * diff;
          }
          if (cur_dist < closest_dist) {
            closest_dist = cur_dist;
            out_pq_vec[c] = (uint8_t) i;
          }
        }
      }
    }
  };  // namespace pipeann
}  // namespace pipeann