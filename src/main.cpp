

#include <filesystem>
#include <fstream>
#include <iostream>

#include "algstd/sat.hpp"
#include "pysa/sat/walksat.hpp"

int main(int argc, char *argv[]) {
  // Print the required arguments
  if (argc < 3 || argc > 7) {
    std::cerr << "Usage: " << std::filesystem::path(argv[0]).filename().string()
              << " cnf_file max_steps [p = 0.5] [unsat = 0] [seed = 0] "
                 "[cutoff_time = 0]\n";
    std::cerr
        << "Find solutions to a SAT formula in CNF format using WalkSAT.\n\n";
    std::cerr << "   max_steps    Maximum number of steps for each restart.\n";
    std::cerr << "   p            WalkSAT probability of a walk move. (default "
                 "= 0.5)\n";
    std::cerr << "   unsat        Maximum number of unsatisfied clauses to "
                 "target (default = 0)\n";
    std::cerr << "   seed         Random seed (default = 0: seed from entropy)\n";
    std::cerr << "   cutoff_time  The cutoff benchmarking time in seconds "
                 "(default = exit on first solution)";
    std::cerr << std::endl;
    if(argc==2){
      std::cerr << "Both cnf_file and max_steps are required arguments." << std::endl;
    }
    return EXIT_FAILURE;
  }

  std::string cnf_file{argv[1]};
  std::size_t max_steps = 100;
  unsigned long max_unsat = 0;
  float p = 0.5;
  // Default random seed
  std::uint64_t seed = 0;
  double bencht = 0.0;

  // Assign provided values
  switch (argc) {
  case 7:
    bencht = std::stod(argv[6]);
  case 6:
    seed = std::stoull(argv[5]);
  case 5:
    max_unsat = std::stoul(argv[4]);
  case 4:
    p = std::stof(argv[3]);
  case 3:
    max_steps = std::stoull(argv[2]);
  default:;
  }
  // Read formula
  const auto formula = [&cnf_file]() {
    if (auto ifs = std::ifstream(cnf_file); ifs.good())
      return pysa::algstd::ReadCNF(ifs);
    else
      throw std::runtime_error("Cannot open file: '" + cnf_file + "'");
  }();

  pysa::sat::walksat_optimize_bench(formula, max_steps, p, max_unsat, seed, bencht, bencht>0.0);
}
