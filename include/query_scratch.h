#pragma once

#include "utils.h"
#include <bitset>
#include "libcuckoo/cuckoohash_map.hh"
#include "libcuckoo/cuckoohash_util.hh"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#define MAX_N_SECTOR_READS 512

struct IORequest {
  uint64_t offset;    // where to read from (page)
  uint64_t len;       // how much to read
  void *buf;          // where to read into
  bool finished;      // for async IO
  uint64_t u_offset;  // where to read from (unaligned)
  uint64_t u_len;     // how much to read (unaligned)

  IORequest() : offset(0), len(0), buf(nullptr) {
  }

  IORequest(uint64_t offset, uint64_t len, void *buf, uint64_t u_offset, uint64_t u_len)
      : offset(offset), len(len), buf(buf), u_offset(u_offset), u_len(u_len) {
    assert(IS_512_ALIGNED(offset));
    assert(IS_512_ALIGNED(len));
    assert(IS_512_ALIGNED(buf));
    // assert(malloc_usable_size(buf) >= len);
  }
};

namespace diskann {
  template<typename T>
  struct DiskNode {
    uint32_t id = 0;
    T *coords = nullptr;
    uint32_t nnbrs;
    uint32_t *nbrs;

    // id : id of node
    // sector_buf : sector buf containing `id` data
    DiskNode(uint32_t id, T *coords, uint32_t *nhood);
  };

  template<typename T>
  struct QueryScratch {
    T *coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;          // index of next [data_dim] scratch to use

    char *sector_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;             // index of next [SECTOR_LEN] scratch to use

    float *aligned_scratch = nullptr;               // MUST BE AT LEAST [aligned_dim]
    float *aligned_pqtable_dist_scratch = nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch = nullptr;          // MUST BE AT LEAST diskann MAX_DEGREE
    _u8 *aligned_pq_coord_scratch = nullptr;        // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
    T *aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;
    char *update_buf = nullptr;

    tsl::robin_set<_u64> *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;
    IORequest reqs[MAX_N_SECTOR_READS];

    void reset() {
      coord_idx = 0;
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }
  };
};  // namespace diskann