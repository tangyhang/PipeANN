// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <sstream>
#include <mutex>

#include "ann_exception.h"

namespace ANNIndex {
  enum LogLevel { LL_Debug = 0, LL_Info, LL_Status, LL_Warning, LL_Error, LL_Assert, LL_Count };
};

namespace diskann {
  class ANNStreamBuf : public std::basic_streambuf<char> {
   public:
    DISKANN_DLLEXPORT explicit ANNStreamBuf(FILE *fp);
    DISKANN_DLLEXPORT ~ANNStreamBuf();

    DISKANN_DLLEXPORT bool is_open() const {
      return true;  // because stdout and stderr are always open.
    }
    DISKANN_DLLEXPORT void close();
    DISKANN_DLLEXPORT virtual int underflow();
    DISKANN_DLLEXPORT virtual int overflow(int c);
    DISKANN_DLLEXPORT virtual int sync();

   private:
    FILE *_fp;
    char *_buf;
    int _bufIndex;
    std::mutex _mutex;
    ANNIndex::LogLevel _logLevel;

    int flush();
    void logImpl(char *str, int numchars);

    static const int BUFFER_SIZE = 0;

    ANNStreamBuf(const ANNStreamBuf &);
    ANNStreamBuf &operator=(const ANNStreamBuf &);
  };
}  // namespace diskann
