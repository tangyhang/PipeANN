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
  if (argc < 4) {
    std::cout << "Correct usage: " << argv[0] << " <tag_file> <head_file> <out_tag_file>" << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *tag_file = argv[arg_no++];
  char *head_file = argv[arg_no++];
  char *out_tag_file = argv[arg_no++];

  std::vector<uint32_t> loc2id;
  size_t npts, dim;
  diskann::load_bin(tag_file, loc2id, npts, dim);

  std::unordered_set<uint32_t> visited;
  for (size_t i = 0; i < loc2id.size(); i++) {
    visited.insert(loc2id[i]);
  }
  visited.erase(INT_MAX);

  auto n_postings = loc2id.size() - visited.size();
  auto head_f = fopen(head_file, "rb");
  std::vector<uint64_t> head_ids(n_postings);
  std::ignore = fread(head_ids.data(), sizeof(uint64_t), n_postings, head_f);
  std::sort(head_ids.begin(), head_ids.end());

  size_t head_pos = 0;
  for (size_t i = 0; i < loc2id.size(); i++) {
    if (loc2id[i] == INT_MAX) {
      loc2id[i] = head_ids[head_pos++];
    }
  }
  if (head_pos != n_postings) {
    std::cout << "Error: " << head_pos << " != " << n_postings << std::endl;
    exit(-1);
  }
  diskann::save_bin(out_tag_file, loc2id.data(), loc2id.size(), 1);
}