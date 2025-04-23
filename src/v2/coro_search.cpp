#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "logger.h"
#include "pq_flash_index.h"
#include <malloc.h>
#include <algorithm>
#include <filesystem>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace diskann {
  template<typename T, typename TagT>
  size_t PQFlashIndex<T, TagT>::coro_search(T **queries, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                            TagT **res_tags, float **res_dists, const _u64 beam_width, int N) {
    static __thread CoroData *data;
    if (unlikely(data == nullptr)) {
      data = new CoroData(this);
    }

    if (unlikely(N > kMaxCoroPerThread)) {
      LOG(ERROR) << "N > kMaxCoroPerThread";
      exit(-1);
    }
    if (unlikely(data_is_normalized)) {
      LOG(INFO) << "Unsupported yet";
      exit(-1);
    }
    auto &thread_data = thread_data_backing_buf[omp_get_thread_num()];
    void *ctx = thread_data.ctx;
    // lambda to batch compute query<-> node distances in PQ space

    for (int v = 0; v < N; ++v) {
      auto &coro_data = data->data[v];
      auto &query1 = queries[v];
      memcpy(coro_data.query, query1, this->data_dim * sizeof(T));

      auto &query = coro_data.query;

      // pointers to buffers for data
      T *data_buf = coro_data.data_buf;
      _mm_prefetch((char *) data_buf, _MM_HINT_T1);

      // query <-> PQ chunk centers distances
      float *pq_dists = coro_data.pq_dists;
      pq_table.populate_chunk_distances(query, pq_dists);

      coro_data.reset();

      _u32 best_medoid = medoids[0];

      if (mem_L) {
        std::vector<unsigned> mem_tags(mem_L);
        std::vector<float> mem_dists(mem_L);
        mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
        coro_data.compute_and_add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search));
      } else {
        // Do not use optimized start point.
        coro_data.compute_and_add_to_retset(&best_medoid, 1);
      }
      std::sort(coro_data.retset.begin(), coro_data.retset.begin() + coro_data.cur_list_size);
    }

    // SEARCH!
    for (int i = 0; i < N; ++i) {
      auto &coro_data = data->data[i];
      coro_data.issue_next_io_batch(beam_width, ctx);
    }

    bool all_finished = false;
    while (!all_finished) {
      all_finished = true;
      for (int i = 0; i < N; ++i) {
        auto &coro_data = data->data[i];
        if (!coro_data.search_ends()) {
          all_finished = false;
          if (!coro_data.io_finished(ctx)) {
            continue;
          }
          // LOG(INFO) << "Full retset size: " << coro_data.full_retset.size();
          coro_data.explore_frontier(l_search);
          coro_data.issue_next_io_batch(beam_width, ctx);
        }
      }
    }

    for (int v = 0; v < N; ++v) {
      // re-sort by distance
      auto &full_retset = data->data[v].full_retset;
      std::sort(full_retset.begin(), full_retset.end(),
                [](const Neighbor &left, const Neighbor &right) { return left < right; });

      _u64 t = 0;
      for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
        if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
          continue;  // deduplicate.
        }
        res_tags[v][t] = full_retset[i].id;  // use ID to replace tags
        if (res_dists[v] != nullptr) {
          res_dists[v][t] = full_retset[i].distance;
        }
        t++;
      }
    }

    return 0;
  }

  // instantiations
  //   template class PQFlashIndex<float, int32_t>;
  //   template class PQFlashIndex<_s8, int32_t>;
  //   template class PQFlashIndex<_u8, int32_t>;
  template class PQFlashIndex<float, uint32_t>;
  template class PQFlashIndex<_s8, uint32_t>;
  template class PQFlashIndex<_u8, uint32_t>;
  //   template class PQFlashIndex<float, int64_t>;
  //   template class PQFlashIndex<_s8, int64_t>;
  //   template class PQFlashIndex<_u8, int64_t>;
  //   template class PQFlashIndex<float, uint64_t>;
  //   template class PQFlashIndex<_s8, uint64_t>;
  //   template class PQFlashIndex<_u8, uint64_t>;
}  // namespace diskann
