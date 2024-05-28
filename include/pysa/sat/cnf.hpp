/*
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

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
#pragma once

#include <iostream>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace pysa::dpll::sat {

    template <typename InputStream, typename ClauseType = std::vector<int32_t>,
            template <typename...> typename Vector = std::vector>
    auto ReadCNF(InputStream &&input_stream) {
        // Initialize formula
        Vector<ClauseType> formula;

        // Initialize current clause
        ClauseType current_cl;

        bool parsed_problem_line = false;
        std::size_t line_num = 0;
        std::string line;

        // For each line in input ...
        while (std::getline(input_stream, line)) {
            // Increment line number
            ++line_num;

            // If line is empty, skip
            if (line.length() == 0) continue;

            // If line is a comment, skip
            if (std::tolower(line[0]) == 'c') continue;

            // If line is problem line, skip
            if (std::regex_match(line, std::regex(R"(^p\s+cnf\s+\d+\s+\d+\s*)"))) {
                if (parsed_problem_line) {
                    std::cerr << "Problem line parsed multiple times" << std::endl;
                    throw std::runtime_error("CNF Parse Error");
                }
                parsed_problem_line = true;

#ifndef NDEBUG
                std::cerr << line << std::endl;
#endif
                continue;
            }

            // Check if problem line appears exactly once
            if (!parsed_problem_line) {
                std::cerr << "CNF file in the wrong format" << std::endl;
                throw std::runtime_error("CNF Parse Error");
            }

            // Check that line is a sequence of numbers
            if (!std::regex_match(line,
                                  std::regex(R"(^\s*-?\d+(\s+-?\d+)*\s+0\s*$)"))) {
                std::cerr << "Failed to parse line " << line_num << ": " << line << ".";
                throw std::runtime_error("CNF Parse Error");
            }

            //  Parse lines
            {
                std::stringstream ss(line);
                typename ClauseType::value_type js;

                // For each token in line ...
                while (ss >> js)

                    // If token is different from zero, append to current clause
                    if (js != 0) current_cl.push_back(js);

                        // Otherwise, append to formula
                    else {
                        formula.push_back(std::move(current_cl));

                        // Remove all elements from current_cl
                        // (in case moving didn't reset it)
                        current_cl.clear();

                        // Ignore everything after zero
                        break;
                    }

                // Only spaces and non-printable characters should remain
                {
                    std::string left_;
                    ss >> left_;
                    if (!ss.eof() ||
                        std::any_of(std::begin(left_), std::end(left_), [](auto &&c) {
                            return !std::isspace(c) && std::isprint(c);
                        })) {
                        std::cerr << "Failed to parse line " << line_num << ": " << line
                                  << ".";
                        throw std::runtime_error("CNF Parse Error");
                    }
                }
            }
        }

        // Return formula
        return formula;
    }
}