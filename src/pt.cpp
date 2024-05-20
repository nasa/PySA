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

#include <iostream>
#include <fstream>
#include <vector>
#include "boost/program_options.hpp"
#include "boost/timer/timer.hpp"
#if defined(USE_SIMDE)
#include "simde/x86/sse2.h"
#endif
#include "libmld/mld.h"
#include "libmld/simdbitvec.h"
#include "pt.h"

using namespace boost::program_options;


template<typename B>
struct PTRunner{
    BitMatrix<B> G;
    BitMatrix<B> y;
    float beta_min, beta_max;

};

template<typename B>
void run_pt(MLDProblem& mld_problem, float beta_min, float beta_max, size_t num_temperatures,
            size_t max_sweeps,
            uint64_t seed=0){
    int64_t energy_target = mld_problem.Weight();
    BitMatrix<B> G = mld_problem.clauses_as_bitmatrix<B>(false);
    BitVec<B> y = mld_problem.y_as_bitvec<B>();
    size_t n = mld_problem.NVars();
    std::vector<float> beta_arr(num_temperatures);
    std::random_device rd;
    if(seed==0)
        seed = std::uniform_int_distribution<size_t>()(rd);
    pt_rng_t rng(seed);
    // initialize beta array
    double lgbmin = std::log(double(beta_min)), lgbmax = std::log(double(beta_max));
    for(size_t i = 0; i < num_temperatures; ++i){
        double a = double(num_temperatures-1-i)/double(num_temperatures-1);
        double b = 1.0f - a;
        beta_arr[i] = (float)std::exp(a*lgbmin + b*lgbmax);
    }
    // initialize state array
    std::vector<BitVec<B>> x_arr;
    x_arr.reserve(num_temperatures);
    for(size_t i = 0; i < num_temperatures; ++i){
        x_arr.emplace_back(n);
        for(size_t j = 0; j < n; ++j){
            if(std::uniform_int_distribution<uint8_t>()(rng)&1)
                x_arr.back().flip(j);
        }
    }

    ptxor::PTSampler<B> sampler(G, y, beta_arr, std::move(rng), std::move(x_arr));
    boost::timer::cpu_timer ti;
    size_t nsweeps=0;
    for(size_t i = 0; i < max_sweeps; ++i){
        sampler.sweep();
        auto& energies = sampler.current_energies();
        for(size_t j = 0; j < num_temperatures; ++j){
            if(energies[j]<= energy_target){
                std::cout << "Target found after " << i+1 << " sweeps.\n";
                nsweeps=i+1;
                goto _a;
            }
        }
    }
    _a:
    ti.stop();
    auto& energies = sampler.current_energies();
    uint64_t dt = ti.elapsed().user + ti.elapsed().system;
    BitVec<B> sol(n);
    std::cout << "PT i B E Acc\n";
    for(size_t j = 0; j < num_temperatures; ++j){
        std::cout << "PT " << j << ' ' <<  beta_arr[j] << ' ' << energies[j] << ' ' <<  sampler.acceptances()[j] << '\n';
    }
    for(size_t j = 0; j < num_temperatures; ++j){
        if(energies[j]<= energy_target){
            sol = sampler.current_states()[j];
            std::cout << "Decoded vector = " << sol << std::endl;
            std::cout << ti.format(6, "wall %w s | user %u s | system %s s | CPU %t s | (%p%)\n") << std::endl;
            double tperit = (double)dt/1000.0 / double(nsweeps);
            std::cout << "t/iter (us): "<< tperit << '\n';
            std::cout << "TTS (s) " << double(dt)/1e9;
            return;
        }
    }
    std::cout << "Solution not found." << std::endl;
}

int main(int argc, char** argv){
    // read program options
    std::string filenm;
    float beta_min=0.1, beta_max=1.0;
    unsigned num_temperatures=16;
    size_t max_sweeps;
    size_t seed=0;
    positional_options_description posopts;
    options_description desc{"Options"};
    desc.add_options()
            ("help,h", "Help screen")
            ("beta-min", value<float>(&beta_min))
            ("beta-max", value<float>(&beta_max))
            ("num-temps", value<unsigned>(&num_temperatures))
            ("input", value<std::string>(&filenm))
            ("max-sweeps", value<size_t>(&max_sweeps));
    posopts.add("input", 1);
    posopts.add("max-sweeps", 2);
    variables_map vm;
    store(command_line_parser(argc, argv).options(desc).positional(posopts).run(), vm);
    notify(vm);
    if(vm.count("help")){
        std::cout << desc << std::endl;
        return 0;
    }
    if(!vm.count("input") || !vm.count("max-sweeps")){
        std::cout << "[input] and [max-sweeps] are required." << std::endl;
        return 1;
    }
    MLDProblem problem;

    std::ifstream ifs(filenm);
    try{
        problem.read_problem(ifs);
    } catch(MLDException& e){
        std::cerr << e.what() << std::endl;
        return 1;
    }
    if(problem.problem_type() != MLDType::G){
        std::cerr << "MLD problem of type 'G' is required." << std::endl;
        return 1;
    }
    size_t m = problem.NClauses();
    if(m <= 8){
        run_pt<uint8_t>(problem, beta_min, beta_max, num_temperatures, max_sweeps, seed);
    } else if (m<=16){
        run_pt<uint16_t>(problem, beta_min, beta_max, num_temperatures, max_sweeps, seed);
    } else if(m<=32) {
        run_pt<uint32_t>(problem, beta_min, beta_max, num_temperatures, max_sweeps, seed);
    }
#if defined (USE_SIMDE)
    else if(m<=64){
        run_pt<uint64_t>(problem, beta_min, beta_max, num_temperatures, max_sweeps, seed);
    } else {
        run_pt<simde__m128i>(problem, beta_min, beta_max, num_temperatures, max_sweeps, seed);
    }
#else
    else {
        run_pt<uint64_t>(problem, beta_min, beta_max, num_temperatures, max_sweeps, seed);
    }
#endif
}