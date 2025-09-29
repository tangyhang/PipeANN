#include "nbr/abstract_nbr.h"
#include "nbr/pq_nbr.h"
#include "omp.h"

#include "aux_utils.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc <= 10) {
    std::cout << "Usage: " << argv[0]
              << " <data_type (float/int8/uint8)>  <data_file.bin>"
                 " <index_prefix_path> <R>  <L>  <PQ_bytes>  <M>  <T>"
                 " <similarity metric (cosine/l2) case sensitive>."
                 " See README for more information on parameters."
              << std::endl;
  } else {
    std::string dist_metric(argv[9]);

    pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
    if (dist_metric != "l2" && m == pipeann::Metric::L2) {
      std::cout << "Metric " << dist_metric << " is not supported. Using L2" << std::endl;
    }

    if (std::string(argv[1]) == std::string("float")) {
      auto nbr_handler = new pipeann::PQNeighbor<float>();
      pipeann::build_disk_index<float>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                       std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    } else if (std::string(argv[1]) == std::string("int8")) {
      auto nbr_handler = new pipeann::PQNeighbor<int8_t>();
      pipeann::build_disk_index<int8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                        std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    } else if (std::string(argv[1]) == std::string("uint8")) {
      auto nbr_handler = new pipeann::PQNeighbor<uint8_t>();
      pipeann::build_disk_index<uint8_t>(argv[2], argv[3], std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[7]),
                                         std::stoi(argv[8]), std::stoi(argv[6]), m, nullptr, nbr_handler);
    } else {
      std::cout << "Error. wrong file type" << std::endl;
    }
  }
}
