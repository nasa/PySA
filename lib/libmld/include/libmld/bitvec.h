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

//! BitVector and BitMatrix utilities

//! BitVec and BitMatrix are classes that wrap around bit arrays with convenience functions
//! for individual bit access and manipulation. They are parametrized by a base unsigned integer type
//! as the block size.

#ifndef LIBMLD_Bitarr_H
#define LIBMLD_Bitarr_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <exception>
#include <ostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <memory>
#include "libmld/util.h"
#if defined(__INTEL_COMPILER)
#include <aligned_new>
#endif

template<typename T, typename _enabled=void>
struct BitVecNums;


template<typename T>
struct BitVecNums<T, typename std::enable_if<std::is_integral<T>::value>::type>{
    //! Frequently used static constants for T.
    static const int16_t bits_per_block = std::numeric_limits<T>::digits;
    static const T ones_mask = static_cast<T>(-1);
//    static const size_t alignment = alignof(T);
//#if defined(__AVX2__)
//    static const size_t alignment = 32; // 4*8 byte = 32 byte = 256-bit alignment at the start of each column
//#elif defined(__SSE4_2__)
//    static const size_t alignment = 16;
//#elif defined(__aarch64__)
//    static const size_t alignment = alignof(T);
//#else
    static const size_t alignment = std::max<size_t>(alignof(T),16);
//#endif
    static const size_t aligned_num_blocks = alignment / sizeof(T);

    static inline T zero_block(){
        return T(0);
    }
    inline static bool read_bit(T* arr, size_t i){
        uint64_t b = i / bits_per_block;
        T k = i % bits_per_block;
        return (arr[b]>>k)&1;
    }
    inline static uint64_t popcount(T a){
#if defined(__cplusplus) && (__cplusplus >= 202002L)
        return std::popcount(a);
#else
        return std::bitset<64>(a).count();
#endif
    }
    inline static void set_bit(T* arr, size_t i, bool val){
        uint64_t b = i / bits_per_block;
        T k = i % bits_per_block;
        if(val){
            arr[b] |= (T(1)<<k);
        } else {
            arr[b] &= (ones_mask^(T(1)<<k));
        }
    }
    inline static void flip_bit(T* arr, size_t i, bool val){
        uint64_t b = i / bits_per_block;
        T k = i % bits_per_block;
        arr[b] ^= (T(1)<<k)&(ones_mask*val);
    }
    inline static void flip_bit(T* arr, size_t i){
        uint64_t b = i / bits_per_block;
        T k = i % bits_per_block;
        arr[b] ^= (T(1)<<k);
    }
};

template<typename T, typename U>
inline bool nzpopblk(T a){
    // Treating a as an array of packed U types
    // is there any U element in this array whose popcount is zero?
    static_assert(std::is_integral_v<T>);
    static_assert(sizeof(U)<=sizeof(T));
    static_assert(sizeof(T)%sizeof(U)==0);
    static const unsigned n = sizeof(T) / sizeof(U);
    U arr[n];
    std::copy_n((U*)&a, n, arr);
    unsigned zblocks = 0;
    for(unsigned i = 0; i < n; ++i){
        zblocks += (arr[i]==0);
    }
    return zblocks != 0;
}

template<typename T, typename U>
inline bool nz2kpopblk(T a){
    // Treating a as an array of packed U types
    // is there any U element in this array whose popcount is zero or one?
    static_assert(std::is_integral_v<T>);
    static_assert(sizeof(U)<=sizeof(T));
    static_assert(sizeof(T)%sizeof(U)==0);
    static const unsigned n = sizeof(T) / sizeof(U);
    U arr[n];
    std::copy_n((U*)&a, n, arr);
    unsigned zblocks = 0;
    for(unsigned i = 0; i < n; ++i){
        zblocks += ((arr[i] & (arr[i]-1))==0);
    }
    return zblocks != 0;
}


template<typename T>
class BitVecSlice{
    /// A non-owning view into an array treated as a bit-vector.
    /// Performs basic bound checks in debug config.
public:
    BitVecSlice(T* data, size_t n, size_t block_begin, size_t block_end): data(data), n(n), begin(block_begin), end(block_end) {

    }
    static const int16_t bits_per_block = BitVecNums<T>::bits_per_block;
    bool operator()(uint64_t i){
        /// Retrieve the value of bit i in the view.

#ifndef NDEBUG
        uint64_t bi = i / bits_per_block;
        if(begin + bi >= end || i >= n){
            throw std::range_error("BitVecView index error");
        }
#endif
        return BitVecNums<T>::read_bit(&data[begin], i);
    }
    const T& get_block(uint64_t bi) const{
        /// Retrieve the entire block in the array.
#ifndef NDEBUG
        if(begin + bi >= end){
            throw std::range_error("BitVecView index error");
        }
#endif
        return data[begin + bi];
    }
    friend std::ostream& operator<< (std::ostream& os, const BitVecSlice<T>& slice){
        std::ios::fmtflags os_flags (os.flags());
        os << std::setfill('0')  << std::hex;
        for(size_t bi = slice.begin; bi < slice.end; ++bi){
            os << std::setw(std::numeric_limits<T>::digits/4) << (uint64_t) slice.get_block(bi) << ' ';
        }
        os << '\n';
        os.flags(os_flags);
        return os;
    }
    [[nodiscard]] size_t size() const{
        /// Number of bits in the bit vector.
        return n;
    }
    [[nodiscard]] size_t num_blocks() const{
        /// Number of bits in the bit vector.
        return end - begin;
    }
    [[nodiscard]] size_t hamming_weight() const{
        /// Number of ones in the bit vector.
        /// Sums the number of ones in every block of the array, including the unused bits,
        /// so this is only correct as long as no invalid bits are set in the last block.
        size_t hw = 0;
        for(size_t i = begin; i < end; ++i){
            hw += BitVecNums<T>::popcount(data[i]);
        }
        return hw;
    }

    [[nodiscard]] BitVecSlice<uint8_t> as_u8_slice() const{
        return {(uint8_t*) data, n, sizeof(T)*begin, sizeof(T)*end};
    }

    [[nodiscard]] const T* data_ptr() const{
        return &data[begin];
    }
private:
    T* data;
    size_t n;
    size_t begin;
    size_t end;
};

template<typename T>
class BitVec{
    /// An bit-vector that owns its data array.
    /// The constructor initializes the entire data array to 0.
public:
    BitVec(const BitVec& other) : BitVec(other.n){
        *this = other;
    };
    BitVec& operator=(const BitVec<T>& other){
#ifndef NDEBUG
        if(n != other.n){
            throw std::range_error("BitVec::operator= invalid size");
        }
#endif
        std::copy(other.data, other.data+other.nblocks, data);
        return *this;
    }
    BitVec& operator=(const BitVecSlice<T>& other){
#ifndef NDEBUG
        if(n != other.size()){
            throw std::range_error("BitVec::operator= invalid size");
        }
#endif
        std::copy(other.data_ptr(), other.data_ptr()+other.num_blocks(), data);
        return *this;
    }
    explicit BitVec(uint64_t n): n(n) {
        nblocks = n/bits_per_block + (n%bits_per_block!=0? 1: 0);
        // row_blocks needed so every column is aligned
        size_t alndv = nblocks / aligned_num_blocks;
        size_t alnmod = nblocks % aligned_num_blocks;
        if(alnmod!=0)
            nblocks = (alndv+1)*aligned_num_blocks;
        data = new(std::align_val_t(alignment)) T[nblocks];
        std::fill(data, data+nblocks, BitVecNums<T>::zero_block());
    }
    ~BitVec(){
        operator delete[](data, std::align_val_t(alignment));
    }
    static const int16_t bits_per_block = BitVecNums<T>::bits_per_block;
    static const size_t alignment = BitVecNums<T>::alignment;
    static const size_t aligned_num_blocks = BitVecNums<T>::aligned_num_blocks;
    //std::vector<T> data;

    bool operator()(uint64_t i) const{
#if !defined(__INTEL_COMPILER)
        T* ptr = std::assume_aligned<BitVecNums<T>::alignment>(data);
#else
        T* ptr = data;
#endif
        return BitVecNums<T>::read_bit(ptr, i);
    }
    T& get_block(uint64_t bi){
#ifndef NDEBUG
        if(bi >= nblocks){
            throw std::range_error("BitVec::operator() index error");
        }
#endif
        return data[bi];
    }
    const T& get_block(uint64_t bi) const{
#ifndef NDEBUG
        if(bi >= nblocks){
            throw std::range_error("BitVec::get_block index error");
        }
#endif
        return as_slice().get_block(bi);
    }
    inline void set(uint64_t i, bool val){
        /// Set the value of bit i
        BitVecNums<T>::set_bit(data, i, val);
    }
    void flip(uint64_t i){
        /// Flip the value of the bit i
#ifndef NDEBUG
        if(i >= n){
            throw std::range_error("BitVec::flip() index error");
        }
#endif
        BitVecNums<T>::flip_bit(data, i);
    }
    BitVecSlice<T> as_slice() const{
        /// Wrap the BitVec data as a BitVecSlice.
        ///
        return BitVecSlice<T>(data, n, 0, nblocks);
    }
    [[nodiscard]] BitVecSlice<uint8_t> as_u8_slice() const{
        return {(uint8_t*) data, n, 0, sizeof(T)*nblocks};
    }
    BitVecSlice<T> slice(int64_t begin=-1, int64_t size=-1){
        if(begin < 0){
            begin = 0;
        }
        if (size<0){
            size = nblocks;
        }
        return BitVecSlice(data, begin, begin + size);
    };
    void clear(){
        std::fill(data, data+nblocks, BitVecNums<T>::zero_block());
    }
    uint64_t num_blocks(){
        return nblocks;
    }
    T* data_ptr(){
        return data;
    }
    size_t size() const{
        return n;
    }
    friend std::ostream& operator<<(std::ostream& os, BitVec<T>& bv){
        for(size_t i = 0; i < bv.n; ++i)
            os << (bv(i)? '1': '0');
        return os;
    }
private:
    T* data;
    uint64_t n;
    uint64_t nblocks;
};


template<typename T>
struct BitMatrix{
    BitMatrix(uint64_t rows, uint64_t cols): rows(rows), cols(cols)
    {
        // unaligned row_blocks
        row_blocks = rows / bits_per_block + (rows%bits_per_block !=0 ? 1: 0);
        // row_blocks needed so every column is aligned
        size_t alndv = (sizeof(T) * row_blocks) / alignment;
        size_t alnmod = (sizeof(T) * row_blocks) % alignment;
        if(alnmod!=0)
            row_blocks = (alndv+1)*alignment/sizeof(T);
        data_len = row_blocks*cols;
//#if defined(__INTEL_COMPILER)
//	data = (T*)_mm_malloc(data_len, alignment);
//	std::fill(data, data+data_len,0);
//#else
        data = new(std::align_val_t(alignment)) T[data_len];
        std::fill(data, data+data_len, BitVecNums<T>::zero_block());
//#endif
//#pragma acc enter data copyin(this)
//#pragma acc enter data create(data[0:data_len])
    }
    BitMatrix(const BitMatrix& mat): rows(mat.rows), row_blocks(mat.row_blocks), cols(mat.cols) {
        data_len = mat.data_len;
//#if defined(__INTEL_COMPILER)
//	data = (T*)_mm_malloc(data_len, alignment);
//	std::fill(data, data+data_len,0);
//#else
	data = new(std::align_val_t(alignment)) T[data_len];
        std::fill(data, data+data_len, BitVecNums<T>::zero_block());
//#endif
//#pragma acc enter data copyin(this)
//#pragma acc enter data create(data[0:data_len])
//#pragma acc parallel loop present(data[0:data_len],mat)
        for(uint64_t j = 0; j < cols; ++j){
            for(uint64_t bi = 0; bi < row_blocks; ++bi){
                data[j*row_blocks + bi] = mat.data[j*row_blocks + bi];
            }
        }
    }
    BitMatrix& operator=(const BitMatrix& mat){
        for(uint64_t j = 0; j < cols; ++j){
            for(uint64_t bi = 0; bi < row_blocks; ++bi){
                data[j*row_blocks + bi] = mat.data[j*row_blocks + bi];
            }
        } 
    }
    ~BitMatrix(){
//#pragma acc exit data delete(data)
//#pragma acc exit data delete(this)
//#if defined(__INTEL_COMPILER)
//	_mm_free(data);
//#else
         operator delete[](data, std::align_val_t(alignment));
	 //delete[] data;
//#endif
    }
    static const uint16_t bits_per_block = BitVecNums<T>::bits_per_block;
    static const size_t alignment = BitVecNums<T>::alignment;
    T* data;
    uint64_t rows;
    uint64_t row_blocks;
    uint64_t cols;

    inline bool operator()(uint64_t i, uint64_t j) const{
#ifndef NDEBUG
        if(j >= cols || i >= rows){
            throw std::range_error("BitMatrix::operator() index error");
        }
#endif
        return BitVecNums<T>::read_bit(&data[j*row_blocks], i);
    }
    T& get_block(uint64_t bi, uint64_t j){
#ifndef NDEBUG
        if(j >= cols || bi >= row_blocks){
            std::stringstream ss;
            ss << "BitMatrix::get_block() index error. Index (" << bi <<", " << j<<" ) out of bounds "
                << "( " << row_blocks   << ", " << cols << ").";
            throw std::range_error(ss.str());
        }
#endif
        return data[j*row_blocks + bi];
    }
    BitVecSlice<T> column_slice(uint64_t j){
#ifndef NDEBUG
        if(j >= cols){
            throw std::range_error("BitMatrix::column_slice index error");
        }
#endif
        BitVecSlice<T> slice(data, rows, j*row_blocks, (j+1)*row_blocks);

        return slice;
    }
    BitVecSlice<T> column_slice(uint64_t j) const{
#ifndef NDEBUG
        if(j >= cols){
            throw std::range_error("BitMatrix::column_slice index error");
        }
#endif
        BitVecSlice<T> slice(data, rows, j*row_blocks, (j+1)*row_blocks);

        return slice;
    }
    const T& get_block(uint64_t bi, uint64_t j) const{
        return data[j*row_blocks + bi];
    }
    inline void set(uint64_t i, uint64_t j, bool val){
#ifndef NDEBUG
        if(j >= cols || i >= rows){
            throw std::range_error("BitMatrix::set index error");
        }
#endif
        BitVecNums<T>::set_bit(&data[j*row_blocks], i, val);
    }
    inline void flip(uint64_t i, uint64_t j, bool val){
        BitVecNums<T>::flip_bit(&data[j*row_blocks], i, val);
    }
    inline void flip(uint64_t i, uint64_t j){
        BitVecNums<T>::flip_bit(&data[j*row_blocks], i);
    }

    friend std::ostream& operator<< (std::ostream& os, const BitMatrix<T>& matrix){
        std::ios::fmtflags os_flags (os.flags());
        os << std::setfill('0')  << std::hex;
        for(size_t j= 0; j<matrix.cols; ++j){
            auto col = matrix.column_slice(j).as_u8_slice();
            for(size_t bi = 0; bi < matrix.row_blocks*sizeof(T); ++bi){
                os << std::setw(2) << (uint16_t) col.get_block(bi) << ' ';
            }
            os << '\n';
        }
        os.flags(os_flags);
        return os;
    }
private:
    size_t data_len;
};


template<typename T>
bool axpy(const BitMatrix<T>& A, const BitMatrix<T>& x, const BitVecSlice<T>& y, BitMatrix<T>& result){
    /// Evaluate AX + y mod 2
    if(A.cols != x.rows || A.rows != result.rows || x.cols != result.cols){
        return false;
    }

    for(uint64_t i = 0; i < x.cols; ++i){
        for(uint64_t j = 0; j < A.row_blocks; ++j){
            T block(0);
            for(uint64_t k = 0; k < A.cols; ++k){
                if(x(k, i)){
                    block ^= A.get_block(j, k);
                }
            }
            result.get_block(j, i) = block ^ y.get_block(j);
        }
    }
    return true;
}

template<typename T>
bool axpy(const BitMatrix<T>& A, const BitMatrix<T>& x, const BitVec<T>& y, BitMatrix<T>& result){
    return axpy(A, x, y.as_slice(), result);
}

template<typename T>
bool bitvec_add(const BitVecSlice<T>& a, const BitVecSlice<T>& b, BitVec<T>& result){
    if(a.size() != b.size()){
        return false;
    }
    for(size_t i = 0; i < a.size(); ++i){
        result.get_block(i) = a.get_block(i) ^ b.get_block(i);
    }
    return true;
}

#endif //LIBMLD_Bitarr_H
