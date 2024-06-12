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
#include <unordered_set>

#include <pysa/sat/instance.hpp>
#include <pysa/sat/walksat.hpp>

int main(){
    std::size_t max_steps = 100000;
    unsigned long max_unsat = 0;
    float p = 0.5;
    std::uint64_t seed = 1234;

    // Get random SAT problem
    int k = 3;
    int n = 21;
    int m = 60;
    const auto formula = pysa::dpll::sat::GetRandomInstance(k, n, m);
    pysa::sat::WalkSatOptimizer wsopt(formula, seed, p);
    wsopt.restart_state();
    size_t nsteps = 0;
    size_t n_unsat = formula.size();
    for(; nsteps < max_steps; ++nsteps){
        n_unsat = wsopt.step();
        if(n_unsat<=max_unsat)
            break;
    }
    std::cout << "Converged after "<< nsteps << " steps.\n";
    assert(n_unsat==0);


}
