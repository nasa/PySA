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

#ifndef STERN_PT_H
#define STERN_PT_H
#include <vector>
#include <random>
#include <cassert>
#include "libmld/bitvec.h"

typedef std::mt19937_64 pt_rng_t;

namespace ptxor{
    template<typename B>
    int64_t energy(const BitMatrix<B>& G, const BitVec<B>& y, const BitVec<B>& x, BitVec<B>& Gx){
#ifndef NDEBUG
        if(G.cols != x.size()){
            throw std::runtime_error("ptxor::energy dimension mismatch for G.x");
        }
        if(y.size() != G.rows){
            throw std::runtime_error("ptxor::energy dimension mismatch for G and y");
        }
#endif

        int64_t e = 0;
        for(size_t bi = 0; bi < G.row_blocks; ++bi){
            B tmp = BitVecNums<B>::zero_block();
            for(size_t j = 0; j < G.cols; ++j){
                B Gbj = G.get_block(bi, j);
                bool xj = x(j);
                tmp ^= (xj ? Gbj : B{0});
            }
            tmp ^= y.get_block(bi);
            e += BitVecNums<B>::popcount(tmp);
            Gx.get_block(bi) = tmp;
        }
        return e;
    }

    template<typename B>
    int64_t delta_energy(const BitMatrix<B>& G, const BitVec<B>& Gx0, size_t bit){
#ifndef NDEBUG
        if(Gx0.size() != G.rows){
            throw std::runtime_error("ptxor::delta_energy dimension mismatch with G");
        }
#endif

        int64_t de = 0;
        for(size_t bi = 0; bi < G.row_blocks; ++bi){
            B Gblk =  G.get_block(bi, bit);
            B pos_de = Gblk & (~Gx0.get_block(bi));
            B neg_de = Gblk & (Gx0.get_block(bi));
            de += BitVecNums<B>::popcount(pos_de);
            de -= BitVecNums<B>::popcount(neg_de);
        }
        return de;
    }


    template<typename B>
    class PTSampler{
    public:
        PTSampler(const BitMatrix<B>& G, const BitVec<B>& y,
                  std::vector<float>& beta_array, pt_rng_t &&rng, std::vector<BitVec<B>>&& x_arr,
                  int64_t target_energy=0):
                G(G), y(y),
                beta_array(beta_array), x_arr(x_arr), rng(rng), target_energy(target_energy){
            pt_len = beta_array.size();
            delta_beta.resize(pt_len-1);
            //assert(x_arr.size == pt_len);
            num_acceptances.resize(pt_len);
            for(size_t i = 0; i < pt_len - 1; ++i){
                delta_beta[i] = beta_array[i+1] - beta_array[i];
            }

            energies.resize(pt_len);
            Gx_arr.reserve(pt_len);
            for(size_t i = 0; i < pt_len; ++i){
                Gx_arr.push_back(BitVec<B>(y.size()));
                energies[i] = energy(G, y, x_arr[i], Gx_arr[i]);
            }
        }
        bool sweep(){
            // perform replica exchange
            for(size_t i = 0; i < pt_len-1; ++i){
                float dlt = delta_beta[i] * (energies[i+1] - energies[i]);
                if(dlt >= 0.0 || (rand_float(rng) < std::exp(dlt))){
                    std::swap(x_arr[i], x_arr[i+1]);
                    std::swap(Gx_arr[i], Gx_arr[i+1]);
                    std::swap(energies[i], energies[i+1]);
                    num_acceptances[i]++;
                }
            }
            // perform sweeps
            for(size_t i = 0; i < pt_len; ++i){
                auto& x = x_arr[i];
                auto& Gx = Gx_arr[i];
                float beta = beta_array[i];

                size_t n = x.size();
                int64_t sweep_de = 0;
                for(size_t j = 0; j < n; ++j){
                    int64_t de = delta_energy(G, Gx, j);
                    if(de > 0){
                        double p = std::exp(-beta * double(de));
                        double r = std::uniform_real_distribution()(rng);
                        if(r>p){
                            continue;
                        }
                    }
                    //accept move
                    x.flip(j);
                    for(size_t bi = 0; bi < Gx.num_blocks(); ++bi)
                        Gx.get_block(bi) ^= G.get_block(bi, j);
                    sweep_de += de;
                }
                energies[i] += sweep_de;
                if(energies[i]<=target_energy) //terminate early if target found
                    return true;
            }
            return false;
        }
        std::vector<int64_t>& current_energies(){
            return energies;
        }
        std::vector<BitVec<B>>& current_states(){
            return x_arr;
        }
        std::vector<uint64_t>& acceptances(){
            return num_acceptances;
        }
    private:
        const BitMatrix<B>& G;
        const BitVec<B>& y;
        std::vector<float> beta_array;
        std::vector<float> delta_beta;

        std::vector<BitVec<B>> x_arr;
        std::vector<BitVec<B>> Gx_arr;
        std::vector<int64_t> energies;

        std::vector<uint64_t> num_acceptances;
        pt_rng_t rng;
        std::uniform_real_distribution<float> rand_float;
        size_t pt_len;
        int64_t target_energy;

    };

}

#endif //STERN_PT_H
