// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  uint64_t file_sz;
  FileHandle file_desc;
  void *bad_ctx = nullptr;

#ifdef USE_AIO
  tsl::robin_map<std::thread::id, void *> ctx_map;
  std::mutex ctx_mut;
#endif

 public:
  LinuxAlignedFileReader();
  ~LinuxAlignedFileReader();

  void *get_ctx();

  // register thread-id for a context
  void register_thread();

  // de-register thread-id for a context
  void deregister_thread();

  void deregister_all_threads();

  void register_buf(void *buf, uint64_t buf_size, int mrid);

  // Open & close ops
  // Blocking calls
  void open(const std::string &fname, bool enable_writes, bool enable_create);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false);
  void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false);
  void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx);
  void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx);

  void send_io(IORequest &reqs, void *ctx, bool write);
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write);
  int poll(void *ctx);
  void poll_all(void *ctx);
  void poll_wait(void *ctx);
};

#ifndef USE_AIO
extern int io_uring_flag;
#endif