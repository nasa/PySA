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
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>
#include <pysa/sat/walksat.hpp>

struct bitvector_hash{
    size_t operator()(const std::vector<uint8_t>& vec) const{
        std::size_t seed = vec.size();
        for(auto x : vec) {
            x = x * 0x3b;
            seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::string bitvector_string(const std::vector<uint8_t>& vec){
    std::string str(vec.size(), '0');
    for (std::size_t i = 0; i < vec.size(); ++i){
        if(vec[i]>0){
            str[i] = '1';
        }
    }
    return str;
}

using namespace std::chrono_literals;
int main(int argc, char* argv[]) {
    // Print the required arguments
    if (argc < 2 || argc > 7) {
        std::cerr << "Usage: " << std::filesystem::path(argv[0]).filename().string()
                  << " cnf_file max_steps [p = 0.5] [unsat = 0] [seed = 0] [cutoff_time = 0]\n";
        std::cerr << "Find solutions to a SAT formula in CNF format using WalkSAT.\n\n";
        std::cerr << "   max_steps    Maximum number of steps for each restart.\n";
        std::cerr << "   p            WalkSAT probability of a walk move. (default = 0.5)\n";
        std::cerr << "   unsat        Maximum number of unsatisfied clauses to target (default = 0)\n";
        std::cerr << "   seed         Random seed (default = seed from entropy)\n";
        std::cerr << "   cutoff_time  The cutoff benchmarking time in seconds (default = exit on first solution)";
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }
    auto it0_ = std::chrono::high_resolution_clock::now();
    // Set filename for cnf formula
    std::string cnf_file{argv[1]};
    // Set default value for maximum number of unsatisfied clauses
    std::size_t max_steps = 100;
    unsigned long max_unsat = 0;
    // Set default value for number of threads (0 = implementation specific)
    float p = 0.5;
    // Default random seed
    std::uint64_t seed = 0;
    // Set default value for verbosity
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
        default: ;
    }
    // Read formula
    const auto formula = [&cnf_file]() {
        if (auto ifs = std::ifstream(cnf_file); ifs.good())
            return pysa::dpll::sat::ReadCNF(ifs);
        else
            throw std::runtime_error("Cannot open file: '" + cnf_file + "'");
    }();

    // Get initial time
    //std::vector<pysa::dpll::BitSet<>> sols;
    size_t nsols = 0;
    size_t nits = 0;
    size_t total_nsteps = 0;
    std::unordered_map<std::vector<uint8_t>, std::pair<size_t, unsigned long>, bitvector_hash> sol_map;
    std::vector<size_t> nstep_hist;
    pysa::sat::WalkSatOptimizer wsopt(formula, seed, p);
    auto it_ = std::chrono::high_resolution_clock::now();
    do {
        //auto [state, nsteps, n_unsat] = pysa::dpll::sat::walksat_optimize(formula, max_steps, seed, p);
        //
        wsopt.restart_state();
        size_t nsteps = 0;
        uint64_t n_unsat = formula.size();
        for(; nsteps < max_steps; ++nsteps){
            n_unsat = wsopt.step();
            if(n_unsat<=max_unsat)
                break;
        }
        auto state = wsopt.state();
//        if(!sol_map.count(state))
//            std::cout << 'U' << n_unsat << " " << std::string(state) << "\n";
        if (n_unsat <= max_unsat) {
            ++nsols;
            nstep_hist.push_back(nsteps);
        }

        if(seed>0) seed += 3;
        sol_map[state].first += 1;
        sol_map[state].second = n_unsat;
        total_nsteps += nsteps;
        nits += 1;
        if(bencht == 0.0)
            break;
    } while (std::chrono::high_resolution_clock::now() - it_ < std::chrono::milliseconds(int(bencht*1000)));
    // Get final time
    auto et_ = std::chrono::high_resolution_clock::now();
    double init_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(it_ - it0_)
            .count()/1000.0;
    double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(et_ - it_)
            .count()/1000.0;
    // Sort solutions lexicographically
    std::vector<std::tuple<std::string, size_t, unsigned >> sols_vec;
    sols_vec.reserve(sol_map.size());
    for(const auto& kv : sol_map) {
        sols_vec.emplace_back(bitvector_string(kv.first), kv.second.first, kv.second.second);
    }
    std::stable_sort(sols_vec.begin(), sols_vec.end(), [](auto& x, auto&y){ return std::get<0>(x) < std::get<0>(y);});
    std::stable_sort(sols_vec.begin(), sols_vec.end(), [](auto& x, auto&y){ return std::get<2>(x) < std::get<2>(y);});
    // Sort distribution of nsteps
    std::sort(nstep_hist.begin(), nstep_hist.end());
    std::cout   << "C Solution count = " << nsols << "\n"
                << "C Unique solutions found = " << sol_map.size() << "\n"
                << "C Initialization time (ms) = " << init_time_ms << "\n"
                << "C Computation time (ms) = " << duration_ms << "\n"
                << "C Total time (ms) = " << init_time_ms + duration_ms << "\n"
                << "C Number of restarts = " << nits << "\n"
                << "C Total inner steps = " << total_nsteps << "\n"
                << "C Avg steps to solution = " << double(total_nsteps) / nsols << "\n"
                << "C Avg time per step (ms) = " << double(duration_ms)/total_nsteps << "\n";

    double qlist[4] {0.5, 0.9, 0.95, 0.99};
    for(double q: qlist)
        if(nits >= 10){ 
            double qn = double(nstep_hist.size()) * q;
            size_t i1 = (size_t)std::floor(qn)-1;
            size_t i2 = (size_t)std::ceil(qn)-1;
            double ip = qn-1.0 - double(i1);
            double qnt = (1.0-ip)*nstep_hist[i1] + ip*nstep_hist[i2];
            std::cout << "C NTS(" <<q*100<<"%) = " << qnt << "\n";
        }
    std::cout << std::endl;
    for(const auto& kv : sols_vec){
        std::cout << "U"<< std::get<2>(kv) << " " << std::get<0>(kv) <<  "  " << std::get<1>(kv)<< "\n";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
