#pragma once

#define MAX_IO_DEPTH 128

#include <fcntl.h>
#include <unistd.h>

#include "query_scratch.h"

#include <malloc.h>
#include <cstdio>

enum ReaderMRID { kPQ, kSector, kUpdate, kLenReaderMRID };
class AlignedFileReader {
 public:
  // returns the thread-specific context
  // returns (io_context_t)(-1) if thread is not registered
  virtual void *get_ctx() = 0;

  virtual ~AlignedFileReader(){};

  // register thread-id for a context
  virtual void register_thread() = 0;
  // de-register thread-id for a context
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;

  virtual void register_buf(void *buf, uint64_t buf_size, int mrid) = 0;

  // Open & close ops
  // Blocking calls
  virtual void open(const std::string &fname, bool enable_writes, bool enable_create) = 0;
  virtual void close() = 0;

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  virtual void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false) = 0;
  virtual void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false) = 0;
  virtual void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) = 0;
  virtual void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) = 0;

  /*
  virtual int submit_reqs(std::vector<IORequest>& read_reqs, void *ctx) = 0;
  virtual void get_events(void *ctx, int n_ops) = 0;
  */

  virtual void send_io(IORequest &reqs, void *ctx, bool write) = 0;
  virtual void send_io(std::vector<IORequest> &reqs, void *ctx, bool write) = 0;
  // Note that this is not used in update, so no page_ref is adopted (returns n_ios).
  virtual int poll(void *ctx) = 0;
  virtual void poll_all(void *ctx) = 0;
  virtual void poll_wait(void *ctx) = 0;
};