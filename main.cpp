/*
Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)

Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
*/

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
    return 1;
  }

  auto t0_ = std::chrono::high_resolution_clock::now();
  std::string cnf_file{argv[1]};
  std::ifstream ifs(cnf_file);
  if(!ifs.good()){
    std::cerr << "Unable to open " << cnf_file << std::endl;
    return 1;
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