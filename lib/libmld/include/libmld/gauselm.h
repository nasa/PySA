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

#ifndef LIBMLD_GAUSELM_H
#define LIBMLD_GAUSELM_H
#include <optional>
#include <vector>
#include "libmld/bitvec.h"

template<typename T>
uint64_t gaussian_elimination(BitMatrix<T>& Amat, std::vector<size_t>& columns,
                              std::vector<size_t>& pivot_cols){
    /// Perform binary Gauss-Jordan elimination on the matrix.
    /// Returns the rank of the elimination
    size_t n = Amat.rows;
    size_t m = Amat.cols;

    size_t rowi = 0;
    size_t red_cols = columns.size();

    pivot_cols.reserve(red_cols);
    pivot_cols.clear();

    for(size_t j = 0; j < red_cols; ++j){
        size_t colj = columns[j];
        // find potential pivots
        size_t pivt = n;
        for(size_t i = rowi; i < n; ++i){
            if(Amat(i, colj)){
                pivt=i;
                break;
            }
        }
        if(pivt == n) continue;
        if(pivt != rowi){  // swap rows pivt and rowi
//#pragma acc loop gang
#if defined(__INTEL_COMPILER)
#pragma ivdep
#endif
            for(size_t j2 = 0; j2 < m; ++j2){
                bool tmp = Amat(pivt, j2) ^ Amat(rowi, j2);
                Amat.flip(pivt, j2, tmp);
                Amat.flip(rowi, j2, tmp);
            }
        }
        // Eliminate the rows
        size_t num_blocks = Amat.row_blocks;
        T* Apcj = &Amat.get_block(0, colj);
#if defined(__INTEL_COMPILER)
	__assume(num_blocks%aln==0);
	__assume_aligned(Apcj,aln);
#endif
        // temporarily zero out the pivot element, leaving only the rows to be eliminated
        Amat.set(rowi, colj, false);
        // for each column
        for(size_t j2 = 0; j2 < m; ++j2){
            if(Amat(rowi, j2)){ // at non-zero elements of the pivot row
		T* Apj2 = &Amat.get_block(0, j2);
#if defined(__INTEL_COMPILER)
		__assume_aligned(Apj2,aln);
#endif
                for(size_t bi=0; bi < num_blocks; ++bi){ // eliminate the rows
                    Apj2[bi] ^= Apcj[bi];
                }
            }
        }
        // correctly set the pivot column
        for(size_t bi=0; bi < Amat.row_blocks; ++bi){
            Apcj[bi] = BitVecNums<T>::zero_block();
        }
        Amat.set(rowi, colj, true);

        pivot_cols.push_back(colj);
        ++rowi;
        if(rowi >= n){
            break;
        }
    }
    return rowi;

}

template<typename T>
uint64_t gaussian_elimination(BitMatrix<T>& Amat){
    /// Perform binary Gauss-Jordan elimination on the matrix.
    /// Returns the rank of the elimination
    size_t n = Amat.rows;
    size_t m = Amat.cols;

    size_t rowi = 0;
    size_t red_cols = m;

    for(size_t j = 0; j < red_cols; ++j){
        size_t colj = j;
        // find potential pivots
        size_t pivt = n;
        for(size_t i = rowi; i < n; ++i){
            if(Amat(i, colj)){
                pivt=i;
                break;
            }
        }
        if(pivt == n) continue;
        if(pivt != rowi){  // swap rows pivt and rowi
#if defined(__INTEL_COMPILER)
#pragma ivdep
#endif
            for(size_t j2 = 0; j2 < m; ++j2){
                bool tmp = Amat(pivt, j2) ^ Amat(rowi, j2);
                Amat.flip(pivt, j2, tmp);
                Amat.flip(rowi, j2, tmp);
            }
        }
        // Eliminate the rows
        size_t num_blocks = Amat.row_blocks;
        T* Apcj = &Amat.get_block(0, colj);
#if defined(__INTEL_COMPILER)
        static const size_t aln = BitVecNums<T>::alignment;
        __assume(num_blocks%aln==0);
	__assume_aligned(Apcj,aln);
#endif
        // temporarily zero out the pivot element, leaving only the rows to be eliminated
        Amat.set(rowi, colj, false);
        // for each column
        for(size_t j2 = 0; j2 < m; ++j2){
            if(Amat(rowi, j2)){ // at non-zero elements of the pivot row
                T* Apj2 = &Amat.get_block(0, j2);
#if defined(__INTEL_COMPILER)
                __assume_aligned(Apj2,aln);
#endif
                for(size_t bi=0; bi < num_blocks; ++bi){ // eliminate the rows
                    Apj2[bi] ^= Apcj[bi];
                }
            }
        }
        // correctly set the pivot column
        for(size_t bi=0; bi < Amat.row_blocks; ++bi){
            Apcj[bi] = BitVecNums<T>::zero_block();
        }
        Amat.set(rowi, colj, true);

        ++rowi;
        if(rowi >= n){
            break;
        }
    }
    return rowi;
}
#endif //LIBMLD_GAUSELM_H
