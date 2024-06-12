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

#ifndef PYSA_DPLL_WALKSAT_H
#define PYSA_DPLL_WALKSAT_H

#include <vector>
#include <pysa/sat/instance.hpp>
#include <pysa/sat/cnf.hpp>

namespace pysa::sat {
    using Formula_WS = std::vector<std::vector<int32_t>>;
    using Instance_WS = pysa::dpll::sat::Instance<pysa::dpll::BitSet<>>;
    using rng_WS = std::minstd_rand;
    class WalkSatOptimizer{
    public:
        WalkSatOptimizer(const Formula_WS& sat_formula, uint64_t random_seed=0, double p=0.0):
                formula(sat_formula),
                instance(Instance_WS(formula)),
                _state(instance.n_vars()),
                _unsat_clauses(instance.n_clauses()),
                _unsat_clause_pos(instance.n_clauses()),
                _clause_sat_nlits(instance.n_clauses()),
                p(std::min(std::max(p, 0.0), 1.0)){
            if(random_seed==0)
                rng.seed(std::random_device()());
            size_t k = 1;
            for(auto& clause: instance.clauses){
                k = std::max(k, clause.size());
            }
            _bestvars.resize(k);
            for(size_t v = 0; v < instance.n_vars(); ++v) {
                std::size_t ncl = instance.clauses[v].size();
                var_true_clauses.emplace_back();
                var_false_clauses.emplace_back();
                for (size_t i = 0; i < ncl; ++i) {
                    size_t cl = instance.clauses[v][i];
                    bool sgn = instance.signs[v].test(i);
                    if (sgn) {
                        var_false_clauses[v].push_back(cl);
                    } else {
                        var_true_clauses[v].push_back(cl);
                    }
                }
            }
        }

        int64_t best_var_flip(size_t clause){
            auto& cl_vars = formula[clause];
            auto new_unsats = (int64_t) instance.n_clauses();
            _numbest = 0;
            for (std::int32_t lit: cl_vars) { // find the best variable to flip
                std::size_t var = std::abs(lit) - 1;
                uint8_t sgn = (lit < 0? 1 : 0);
                //auto &clv = instance.clauses[var];
                //auto &cls = instance.signs[var];

                int32_t lit_new_unsats = 0;

                if(sgn){ // Satisfying a false variable will break clauses where it is true
                    for(uint32_t cl: var_true_clauses[var]){
                        if (_clause_sat_nlits[cl] == 1) { // flipping will add an unsat clause
                            lit_new_unsats += 1;
                        }
                    }
                } else {
                    for(uint32_t cl: var_false_clauses[var]){
                        if (_clause_sat_nlits[cl] == 1) {
                            lit_new_unsats += 1;
                        }
                    }
                }
                //assert(lit_new_unsats == _lit_num_break[(2*var+sgn)^1]);
                if (lit_new_unsats < new_unsats) {
                    _numbest = 0;
                    new_unsats = lit_new_unsats;
                }
                if (lit_new_unsats == new_unsats) {
                    _bestvars[_numbest++] = var;
                }
            }

            return new_unsats;
        }
        uint64_t step(){
            if(n_unsat_clauses == 0)
                return 0;
            // randomly select an unsatisfied clause
            //std::uniform_int_distribution<uint32_t> dist(0, n_unsat_clauses-1);
            uint64_t ucl = rng()%n_unsat_clauses;
            uint64_t cl = _unsat_clauses[ucl];
            std::size_t next_var;

            int64_t best_flips_break = best_var_flip(cl);
            if( best_flips_break>0 && p > 0.0 && std::uniform_real_distribution<float>()(rng) < p){
                next_var = select_random_var(cl);
            } else {
                next_var = select_best_var();
            }
            flip_variable(next_var);
            return n_unsat_clauses;
        }
        void restart_state(){
            /// initialize or reset the solver with a random state
            size_t ncl = instance.n_clauses();
            size_t nv = instance.n_vars();
            for(size_t i = 0; i < nv; ++i){
                _state[i] = rng()&1;
            }

            n_unsat_clauses = 0;
            // get satisfied clauses and satisfied literal counts
            for(size_t cl = 0; cl < ncl; ++cl){
                _clause_sat_nlits[cl] = 0;
                auto& cl_lits = formula[cl];
                for(int32_t lit: cl_lits){
                    std::size_t var = std::abs(lit) - 1;
                    uint8_t sgn = (lit < 0 ? 1 : 0);
                    if(_state[var] ^ sgn){
                        _clause_sat_nlits[cl] += 1;
                    }
                }
                if(_clause_sat_nlits[cl] == 0) {
                    _unsat_clause_pos[cl] = n_unsat_clauses;
                    _unsat_clauses[n_unsat_clauses++] = cl;
                }
            }
        }
        std::vector<uint8_t>& state(){
            return _state;
        }
    private:
        int64_t first_true_lit(size_t cl_idx){
            for( int32_t l : formula[cl_idx]){
                uint32_t v2 = abs(l)-1;
                uint8_t s = (l<0);
                if(_state[v2]^s) {
                    return 2 * v2 + s;
                }
            }
            return -1;
        }
        size_t select_random_var(size_t cl_idx){
            // randomly select a variable in the clause to flip
            auto& cl_vars = formula[cl_idx];
            uint32_t ncl = cl_vars.size();
            int32_t next_lit = cl_vars[rng()%ncl];
            size_t next_var = std::abs(next_lit) - 1;
            return next_var;
        }
        size_t select_best_var(){
            if(_numbest>1)
                return _bestvars[rng()%_numbest];
            else
                return _bestvars[0];
        }
        void flip_variable(std::size_t v){
            _state[v] ^= 1;
            bool newval = _state[v];
            int d = (newval ? 1 : -1);

            for(uint32_t cl: var_true_clauses[v]){ // clauses with the true literal
                if(newval && _clause_sat_nlits[cl] == 0) {// variable flipped to true and satisfies a new clause
                    _clause_sat_nlits[cl] = 1;
                    size_t _cl_pos = _unsat_clause_pos[cl];
                    if(_cl_pos < n_unsat_clauses - 1) {
                        std::swap(_unsat_clauses[_cl_pos], _unsat_clauses[n_unsat_clauses - 1]);
                        _unsat_clause_pos[_unsat_clauses[_cl_pos]] = _cl_pos;
                    }
                    n_unsat_clauses--;
                } else if(!newval && _clause_sat_nlits[cl] == 1) {// variable flipped to false and breaks a clause
                    _clause_sat_nlits[cl] = 0;
                    _unsat_clause_pos[cl] = n_unsat_clauses;
                    _unsat_clauses[n_unsat_clauses++] = cl;
                } else {
                    _clause_sat_nlits[cl] += d;
                    assert(_clause_sat_nlits[cl]>0);
                }
            }
            for(uint32_t cl: var_false_clauses[v]){ // clauses with the false literal
                if(newval && _clause_sat_nlits[cl] == 1) {// variable flipped to true and unsats a clause
                    _clause_sat_nlits[cl] = 0;
                    _unsat_clause_pos[cl] = n_unsat_clauses;
                    _unsat_clauses[n_unsat_clauses++] = cl;
                } else if(!newval && _clause_sat_nlits[cl] == 0) {// variable flipped to false and satisfies a new clause
                    _clause_sat_nlits[cl] = 1;
                    size_t _cl_pos = _unsat_clause_pos[cl];
                    if(_cl_pos < n_unsat_clauses - 1) {
                        std::swap(_unsat_clauses[_cl_pos], _unsat_clauses[n_unsat_clauses - 1]);
                        _unsat_clause_pos[_unsat_clauses[_cl_pos]] = _cl_pos;
                    }
                    n_unsat_clauses--;
                } else {
                    _clause_sat_nlits[cl] -= d;
                    assert(_clause_sat_nlits[cl]>0);
                }
            }
        }
        const Formula_WS& formula;
        Instance_WS instance;
        std::vector<std::vector<uint32_t>> var_true_clauses;
        std::vector<std::vector<uint32_t>> var_false_clauses;
        std::vector<uint8_t> _state;
        std::vector<uint32_t> _unsat_clauses;
        std::vector<uint32_t> _unsat_clause_pos;
        uint32_t n_unsat_clauses;
        std::vector<int64_t> _clause_sat_nlits;

        std::vector<uint32_t> _bestvars;
        uint32_t _numbest;

        rng_WS rng;
        float p=1.0;
    };

    std::tuple<std::vector<uint8_t>, uint64_t, uint64_t> walksat_optimize(const Formula_WS& formula, uint64_t max_steps,
                                                                          uint64_t random_seed=0, double p = 1.0){
        /// Performs optimization with walksat and returns the tuple (final_state, num_steps, num_unsat)
        WalkSatOptimizer wsopt(formula, random_seed, p);
        wsopt.restart_state();
        size_t i = 0;
        uint64_t n_unsat = formula.size();
        for(; i < max_steps; ++i){
            n_unsat = wsopt.step();
            if(n_unsat==0)
                break;
        }
        return {wsopt.state(), i, n_unsat};
    }
}

#endif //PYSA_DPLL_WALKSAT_H
