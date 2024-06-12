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

#include "libmld/mld.h"
#include <sstream>

void MLDProblem::read_problem(std::istream &input_stream) {
  std::string line;
  std::getline(input_stream, line);
  if (!input_stream) {
    throw MLDException("File I/O failed");
  }
  char prob_type_c;
  std::stringstream ss0(line);
  ss0 >> std::skipws >> prob_type_c >> nvars >> nrows >> y >> w;
  if (!ss0) {
    throw MLDException("Invalid header in first line.");
  }
  switch (prob_type_c) {
  case 'g':
  case 'G':
    prob_type = MLDType::G;
    break;
  case 'h':
  case 'H':
    prob_type = MLDType::H;
    break;
  case 'p':
  case 'P':
    prob_type = MLDType::P;
    break;
  case 'w':
  case 'W':
    prob_type = MLDType::W;
    break;
  default:
    throw MLDException(std::string("Problem type ") + prob_type_c +
                       " not recognized");
  }
  // blcmatrix = std::unique_ptr<BinaryMat>(new BinaryMat(nrows, nvars+1));
  yarr.resize(nrows);
  clauses.resize(nrows);
  if (y > 0) {
    for (size_t i = 0; i < nrows; ++i) {
      yarr[i] = 1;
    }
  }
  size_t i = 0;
  while (std::getline(input_stream, line)) {
    std::stringstream ss(line);
    int64_t js;
    if (i >= nrows) {
      std::ostringstream oss;
      oss << "Line " << i + 2 << " not expected for " << nrows << " variables.";
      throw MLDException(oss.str());
    }
    while (ss >> js) {
      int64_t j;
      if (js < 0) {
        j = -js - 1;
        yarr[i] ^= 1;
      } else if (js == 0) {
        break;
      } else {
        j = js - 1;
      }
      if (j >= nvars) {
        std::ostringstream oss;
        oss << "Unexpected integer " << j << " > " << nvars << " in line "
            << i + 2 << ".";
        throw MLDException(oss.str());
      }
      clauses[i].push_back(j);
    }
    if (!ss.eof()) {
      std::ostringstream oss;
      oss << "Failed to parse line " << i + 2 << ":\n" << line << ".";
      throw MLDException(oss.str());
    }
    i += 1;
  }
}
