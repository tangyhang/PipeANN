// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "windows_customizations.h"
#include <cosine_similarity.h>
#include <iostream>

namespace diskann {
  //  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
  template<typename T>
  class Distance {
   public:
    virtual float compare(const T *a, const T *b, unsigned length) const = 0;
    virtual ~Distance() {
    }
  };

  class DistanceCosineInt8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
  };

  class DistanceCosineFloat : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t length) const;
  };

  class SlowDistanceCosineUInt8 : public Distance<uint8_t> {
   public:
    virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
  };

  class DistanceL2Int8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
  };

  class DistanceL2UInt8 : public Distance<uint8_t> {
   public:
    virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
  };

  class DistanceL2 : public Distance<float> {
   public:
#ifdef _WINDOWS
    virtual float compare(const float *a, const float *b, uint32_t size) const;
#else
    virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));
#endif
  };

  // Slow implementations of the distance functions to get diskann to
  // work in pre-AVX machines. Performance here is not a concern, so we are
  // using the simplest possible implementation.
  template<typename T>
  class SlowDistanceL2Int : public Distance<T> {
   public:
    // Implementing here because this is a template function
    virtual float compare(const T *a, const T *b, uint32_t length) const {
      uint32_t result = 0;
      for (uint32_t i = 0; i < length; i++) {
        result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) * ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
      }
      return (float) result;
    }
  };

  class SlowDistanceL2Float : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t length) const;
  };

  // AVX implementations. Borrowed from HNSW code.
  class AVXDistanceL2Int8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
  };

  class AVXDistanceL2Float : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t length) const;
  };

}  // namespace diskann
