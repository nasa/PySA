
#include "cdcl.h"

std::vector<std::string> test_cnf{
    "p cnf 1 2\n"
    "-1 0\n"
    "1 0\n",

    "p cnf 3 4\n"
    "1 2 3 0\n"
    "-1 -2 0\n"
    "2 3 0\n"
    "2 -3 0\n",

    "p cnf 8 6\n"
    "1 8 -2 0\n"
    "1 -3 0\n"
    "2 3 4 0\n"
    "-4 -5 0\n"
    "7 -4  -6 0\n"
    "5 6 0\n"
};


int main() {
  for(const std::string& cnf : test_cnf) {
    FormulaT formula = pysa::algstd::ReadCNF(std::stringstream (cnf));
    std::cout << formula << std::endl;
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
    std::cout << "\n---" << std::endl;
  }
  return 0;
}

