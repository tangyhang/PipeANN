#pragma once

#include "nbr/abstract_nbr.h"
#include "utils.h"
#include <immintrin.h>
#include <sstream>
#include <string_view>
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index_defs.h"

namespace pipeann {
  template<typename T>
  class DummyNeighbor : public AbstractNeighbor<T> {
   public:
    virtual ~DummyNeighbor() = default;

    static std::string get_name() {
      return "DummyNeighbor";
    }
  };
}  // namespace pipeann