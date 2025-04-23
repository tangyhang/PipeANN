// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include <ctime>
#include <unordered_set>
#include <timer.h>
#include "log.h"

#include "pq_flash_index.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Correct usage: " << argv[0] << " <tag_file> <base_partition_file> <nnodes_per_sector> <dim>"
              << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *tag_file = argv[arg_no++];
  char *partition_file = argv[arg_no++];
  _u64 C = atoll(argv[arg_no++]);

  std::vector<uint32_t> loc2id;
  size_t npts, dim;
  diskann::load_bin(tag_file, loc2id, npts, dim);

  std::string out_partition_file = std::string(partition_file) + ".aligned";

  std::ofstream out_part(out_partition_file.data(), std::ios::binary);
  _u64 partition_nums = (loc2id.size() + C - 1) / C;
  _u64 nd = 0;  // ignore nd.
  out_part.write((char *) &C, sizeof(_u64));
  out_part.write((char *) &partition_nums, sizeof(_u64));
  out_part.write((char *) &nd, sizeof(_u64));

  std::unordered_set<unsigned> visited;
  uint32_t cnt = 0;
  uint32_t tmp_arr[4096];
  for (unsigned i = 0; i < partition_nums; i++) {
    uint32_t s = std::min(C, loc2id.size() - i * C);
    for (uint32_t j = 0; j < s; ++j) {
      tmp_arr[j] = loc2id[i * C + j];
      visited.insert(loc2id[i * C + j]);
      ++cnt;
    }
    for (uint32_t j = s; j < C; ++j) {
      tmp_arr[j] = diskann::PQFlashIndex<uint8_t>::kInvalidID;
    }
    out_part.write((char *) &s, sizeof(unsigned));
    out_part.write((char *) tmp_arr, sizeof(unsigned) * C);
  }
  LOG(INFO) << "Cnt " << cnt << " size " << loc2id.size() << " visited size " << visited.size();
}