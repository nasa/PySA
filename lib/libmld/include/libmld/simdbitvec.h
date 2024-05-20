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

#ifndef STERN_SIMDBITVEC_H
#define STERN_SIMDBITVEC_H

#if defined(USE_SIMDE)
#include "libmld/bitvec.h"
#include "simde/x86/avx.h"
#include "simde/x86/avx2.h"
#include "simde/x86/avx512/popcnt.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

template<>
struct BitVecNums<simde__m128i, typename std::enable_if<true>::type >{
    static const int16_t bits_per_block = 128;
    static const size_t alignment = alignof(simde__m128i);
    static const size_t aligned_num_blocks = alignment / sizeof(simde__m128i);

    inline static const simde__m128i zeros = simde_mm_setzero_si128();
    static inline simde__m128i zero_block(){
        return simde_mm_setzero_si128();
    }
    inline static bool read_bit(simde__m128i* arr, size_t i){
        // treat simd blocks as u64 storage for the purpose of bit access
        auto * arr2 = (uint64_t*)arr;
        return BitVecNums<uint64_t>::read_bit(arr2, i);
    }
    inline static uint64_t popcount(simde__m128i a){
        auto pops =  simde_mm_popcnt_epi64(a);
        uint64_t hw = simde_mm_extract_epi64(pops, 0) + simde_mm_extract_epi64(pops, 1);
        return hw;
    }
    inline static void set_bit(simde__m128i* arr, size_t i, bool val){
        BitVecNums<uint64_t>::set_bit((uint64_t*)arr, i, val);
    }
    inline static void flip_bit(simde__m128i* arr, size_t i, bool val){
        BitVecNums<uint64_t>::flip_bit((uint64_t*)arr, i, val);
    }
    inline static void flip_bit(simde__m128i* arr, size_t i){
        BitVecNums<uint64_t>::flip_bit((uint64_t*)arr, i);
    }
};

template<>
struct BitVecNums<simde__m256i, typename std::enable_if<true>::type >{
    static const int16_t bits_per_block = 256;
    static const size_t alignment = 32;
    static const size_t aligned_num_blocks = alignment / sizeof(simde__m256i);

    inline static const simde__m256i zeros = simde_mm256_setzero_si256();
    static inline simde__m256i zero_block(){
        return simde_mm256_setzero_si256();
    }
    inline static bool read_bit(simde__m256i* arr, size_t i){
        // treat simd blocks as u64 storage for the purpose of bit access
        auto * arr2 = (uint64_t*)arr;
        return BitVecNums<uint64_t>::read_bit(arr2, i);
    }
    inline static uint64_t popcount(simde__m256i a){
	    uint64_t hw = BitVecNums<uint64_t>::popcount(simde_mm256_extract_epi64(a, 0))
		    +  BitVecNums<uint64_t>::popcount(simde_mm256_extract_epi64(a, 1))
		    +  BitVecNums<uint64_t>::popcount(simde_mm256_extract_epi64(a, 2))
		    +  BitVecNums<uint64_t>::popcount(simde_mm256_extract_epi64(a, 3));
//        auto pops =  simde_mm256_popcnt_epi64(a);
//        uint64_t hw = simde_mm256_extract_epi64(pops, 0) + simde_mm256_extract_epi64(pops, 1)
//                + simde_mm256_extract_epi64(pops, 2)+simde_mm256_extract_epi64(pops, 3);
          return hw;
    }
    inline static void set_bit(simde__m256i* arr, size_t i, bool val){
        BitVecNums<uint64_t>::set_bit((uint64_t*)arr, i, val);
    }
    inline static void flip_bit(simde__m256i* arr, size_t i, bool val){
        BitVecNums<uint64_t>::flip_bit((uint64_t*)arr, i, val);
    }
    inline static void flip_bit(simde__m256i* arr, size_t i){
        BitVecNums<uint64_t>::flip_bit((uint64_t*)arr, i);
    }
};

template<>
inline bool nzpopblk<simde__m128i, uint8_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);

    simde__m128i zblocks = simde_mm_cmpeq_epi8(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nzpopblk<simde__m128i, uint16_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);

    simde__m128i zblocks = simde_mm_cmpeq_epi16(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nzpopblk<simde__m128i, uint32_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);

    simde__m128i zblocks = simde_mm_cmpeq_epi32(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nzpopblk<simde__m128i, uint64_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);

    simde__m128i zblocks = simde_mm_cmpeq_epi64(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nz2kpopblk<simde__m128i, uint8_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);
    simde__m128i am1 = simde_mm_sub_epi8(a, simde_mm_set1_epi8(1));
    simde__m128i tst = simde_mm_and_si128(a,am1);
    simde__m128i zblocks = simde_mm_cmpeq_epi8(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nz2kpopblk<simde__m128i, uint16_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);
    simde__m128i am1 = simde_mm_sub_epi16(a, simde_mm_set1_epi16(1));
    simde__m128i tst = simde_mm_and_si128(a,am1);
    simde__m128i zblocks = simde_mm_cmpeq_epi8(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nz2kpopblk<simde__m128i, uint32_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);
    simde__m128i am1 = simde_mm_sub_epi32(a, simde_mm_set1_epi32(1));
    simde__m128i tst = simde_mm_and_si128(a,am1);
    simde__m128i zblocks = simde_mm_cmpeq_epi8(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}


template<>
inline bool nz2kpopblk<simde__m128i, uint64_t>(simde__m128i a){
    simde__m128i zeros = BitVecNums<simde__m128i>::zeros;
    simde__m128i ones = simde_mm_cmpeq_epi64(zeros, zeros);
    simde__m128i am1 = simde_mm_sub_epi64(a, simde_mm_set1_epi64x(1));
    simde__m128i tst = simde_mm_and_si128(a,am1);
    simde__m128i zblocks = simde_mm_cmpeq_epi8(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm_testz_si128(zblocks, ones)==0;
}

template<>
inline bool nzpopblk<simde__m256i, uint8_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);

    simde__m256i zblocks = simde_mm256_cmpeq_epi8(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}

template<>
inline bool nzpopblk<simde__m256i, uint16_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);

    simde__m256i zblocks = simde_mm256_cmpeq_epi16(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}


template<>
inline bool nzpopblk<simde__m256i, uint32_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);

    simde__m256i zblocks = simde_mm256_cmpeq_epi32(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}

template<>
inline bool nzpopblk<simde__m256i, uint64_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);

    simde__m256i zblocks = simde_mm256_cmpeq_epi64(a, zeros); // blocks with zero HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}


template<>
inline bool nz2kpopblk<simde__m256i, uint8_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);
    simde__m256i am1 = simde_mm256_sub_epi8(a, simde_mm256_set1_epi8(1));
    simde__m256i tst = simde_mm256_and_si256(a,am1);
    simde__m256i zblocks = simde_mm256_cmpeq_epi8(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}

template<>
inline bool nz2kpopblk<simde__m256i, uint16_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);
    simde__m256i am1 = simde_mm256_sub_epi16(a, simde_mm256_set1_epi16(1));
    simde__m256i tst = simde_mm256_and_si256(a,am1);
    simde__m256i zblocks = simde_mm256_cmpeq_epi16(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}

template<>
inline bool nz2kpopblk<simde__m256i, uint32_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);
    simde__m256i am1 = simde_mm256_sub_epi32(a, simde_mm256_set1_epi32(1));
    simde__m256i tst = simde_mm256_and_si256(a,am1);
    simde__m256i zblocks = simde_mm256_cmpeq_epi32(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}

template<>
inline bool nz2kpopblk<simde__m256i, uint64_t>(simde__m256i a){
    simde__m256i zeros = BitVecNums<simde__m256i>::zeros;
    simde__m256i ones = simde_mm256_cmpeq_epi64(zeros, zeros);
    simde__m256i am1 = simde_mm256_sub_epi64(a, simde_mm256_set1_epi64x(1));
    simde__m256i tst = simde_mm256_and_si256(a,am1);
    simde__m256i zblocks = simde_mm256_cmpeq_epi64(tst, zeros); // blocks with 0 or 1 HW are all-ones blocks
    return simde_mm256_testz_si256(zblocks, ones)==0;
}
#pragma GCC diagnostic pop
#endif

#endif //STERN_SIMDBITVEC_H
