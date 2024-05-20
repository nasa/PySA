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

#ifndef STERN_WF_H
#define STERN_WF_H
#include <cmath>
#include <cstdint>
#include <vector>


inline double log2_binomial(double n, double k){
    //return n*shannon_h(k/n) + 0.5*(std::log2(n) - std::log2(k) - std::log2(n-k));
    return (std::lgamma(n+1) - std::lgamma(k+1) - std::lgamma(n-k+1))/std::log(2.0);
}

inline double log2sumexp(double a, double b){
    // Approximate log2( 2**a + 2**b ) = log2( 2**b(2**(a-b) + 1))
    if(a<b){
        return b + std::log2(1 + std::exp2(a-b));
    } else {
        return a + std::log2(1 + std::exp2(b-a));
    }
}

inline double heavy_stern_iterwf(int n, int k, int p, int l = 0) {
    // Sample: Choose k bits out of n to form the information set.
    // Randomly partition the information set into two sets, I1 and I2
    // This sample is successful if 2p error bits are in the information set, 
    // and each half of the partition contains exactly p error bits
    //
    // Operation:  Diagonalize the parity check matrix with Gaussian elimination ((n-k)^3).
    // For each of I1 and I2, construct every p-linear combination of columns (about 2*((k/2)Cp)*(p * (n-k))
    // For each p-combination v1 from I1, For each p-combination v2 from I2, evaluate the Hamming distance
    // of their sum from y ( (k/2Cp)*(k/2Cp)*(n-k))

    double ge_work_bits = 3 * std::log2(n-k);
    double lc_work_bits = 1.0 + log2_binomial(k/2, p) + std::log2(p) + std::log2(n-k);
    double q_work_bits;
    if (l>0)
        q_work_bits = 2.0*log2_binomial(k/2, p) + std::log2(l);
    else
        q_work_bits = 2.0*log2_binomial(k/2, p) + std::log2(n-k);

    double total_bits = log2sumexp(log2sumexp(ge_work_bits, lc_work_bits), q_work_bits);
    return total_bits;
}

inline double heavy_stern_nll(int n, int k, int w, int p){
    // Sample: Choose k bits out of n to form the information set.
    // Randomly partition the information set into two sets, I1 and I2
    // This sample is successful if 2p error bits are in the information set, 
    // and each half of the partition contains exactly p error bits

    double wf_infoset = log2_binomial(n, k) - log2_binomial(w, 2*p) - log2_binomial(n-w, k-2*p);
    double cond_wf_split = -(log2_binomial(k/2, p) + log2_binomial(k-k/2, p) - log2_binomial(k, 2*p));
    return wf_infoset + cond_wf_split;
}

inline double stern_nll(int n, int k, int w, int p, int l, int m){
    // conditional probability of an ideal stern iteration
    double b0 = heavy_stern_nll(n, k, w, p);
    // Next, we want to calculate the number of ways to distribute (n-k) bits with w-2p error bits
    // into m buckets of l bits such that (X) at least one bucket contains zero [or one] error bits.
    //  not X = No bucket contains zero [or one] error bits
    //        = For all buckets b, (Y) b contains at least [one|two] error bits
    //
    //  not Y = b contains [zero|at most one] error bits
    //        = b contains zero bits [or b contains one bit]

    // Random selection of m collections of l bits from n-k redundancy bits
    // contains
    // One random selection of l bits from the redundancy contains no errors
    double b1 = -log2_binomial(n-k-(w-2*p), l) + log2_binomial(n-k, l); 
    // One random selection of l bits from the redundancy contains at least one error
    double b1inv = -std::log1p(-std::exp2(-b1))/std::log(2.0);
    // m independent random selections of l bits from the redundancy set contain at least one error
    double b2 = m * b1inv;
    // at least one out of m independent random selections of l bits from the redundancy set contains no errors
    double b2inv = -std::log1p(-std::exp2(-b2))/std::log(2.0);

    double btotal = b0 + b2inv;
    return btotal;
}

std::vector<double> combo_dist(long n, long k, long t);

#endif //STERN_WF_H
