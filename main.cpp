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

  std::string cnf_file{argv[1]};
  std::ifstream ifs(cnf_file);
  if(!ifs.good()){
    throw std::runtime_error("Unable to open " + cnf_file);
  }

  FormulaT formula = pysa::algstd::ReadCNF(ifs);
  //std::cout << formula << std::endl;
  CDCL cdcl(std::move(formula));
  int rc = cdcl.run();
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
}