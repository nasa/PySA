#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "cdcl.h"


int main(int argc, char* argv[]) {
  // Print the required arguments
  if (argc < 2 || argc > 3){
    std::cerr << "Usage: " << std::filesystem::path(argv[0]).filename().string()
                           << " cnf_file" << std::endl;
  }

  auto t0_ = std::chrono::high_resolution_clock::now();
  std::string cnf_file{argv[1]};
  std::ifstream ifs(cnf_file);
  if(!ifs.good()){
    throw std::runtime_error("Unable to open " + cnf_file);
  }

  FormulaT formula = pysa::algstd::ReadCNF(ifs);
  //std::cout << formula << std::endl;
  CDCL cdcl(std::move(formula));

  auto it_ = std::chrono::high_resolution_clock::now();
  int rc = cdcl.run();
  auto et_ = std::chrono::high_resolution_clock::now();

  if (rc == CDCLSAT) {
    for (uint8_t b: cdcl.prop._state) {
      std::cout << int(b);
    }
    std::cout << std::endl;
  } else if (rc == CDCLUNSAT) {
    std::cout << "unsat" << std::endl;
  } else {
    std::cerr << rc << "\n";
  }
  std::cout << "Pre-processing time "
            << std::chrono::duration_cast<std::chrono::microseconds>(it_ - t0_)
                .count() << " us\n";
  std::cout << "Computation time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(et_ - it_)
                .count() << " us\n";
  std::cout << "Total time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(et_ - t0_)
                .count() << " us\n";
}