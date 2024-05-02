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

#include <cassert>
#include <pysa/archive/archive.hpp>
#include <string>
#include <vector>

namespace pysa::dpll {

template <typename T>
auto popcount(const T &x) {
  if constexpr (std::is_same_v<T, unsigned int>)
    return __builtin_popcount(x);
  else if constexpr (std::is_same_v<T, unsigned long>)
    return __builtin_popcountl(x);
  else if constexpr (std::is_same_v<T, unsigned long long>)
    return __builtin_popcountll(x);
  else
    std::runtime_error("Type not supported for 'popcount'.");
}

template <typename BlockType = std::size_t,
          template <typename...> typename Vector = std::vector>
struct BitSet {
  using block_type = BlockType;
  static constexpr auto block_size = 8 * sizeof(block_type);

  operator std::string() const {
    /*
     * Convert `BitSet` to `std::string`.
     */
    std::string str_;
    for (std::size_t i_ = 0; i_ < _size; ++i_) str_ += test(i_) ? '1' : '0';
    return str_;
  }

  template <typename VectorType,
            typename _T = decltype(VectorType().push_back(true))>
  operator VectorType() const {
    /*
     * Convert `BitSet` to any `VectorType` that supports `push_back`.
     */
    VectorType out_;
    for (std::size_t i_ = 0; i_ < _size; ++i_) out_.push_back(test(i_));
    return out_;
  }

  BitSet(std::size_t n, Vector<block_type> &&buffer)
      : _buffer{std::move(buffer)}, _size{n} {
    /*
     * Initialize from `buffer`.
     */
    _FixUnusedBitsBuffer();
  }

  BitSet(std::size_t n, const Vector<block_type> &buffer)
      : _buffer{buffer}, _size{n} {
    /*
     * Initialize from `buffer`.
     */
    _FixUnusedBitsBuffer();
  }

  BitSet(std::size_t n = 0)
      : _buffer(n / block_size + ((n % block_size) != 0)), _size{n} {
    /*
     * Initialize to the given number of bits.
     */
  }

  BitSet(const std::string &x) : BitSet(std::size(x)) {
    /*
     * Initialize from `std::string`.
     */
    for (std::size_t i_ = 0, size_ = std::size(x); i_ < size_; ++i_)
      switch (x[i_]) {
        case '0':
          break;
        case '1':
          set(i_);
          break;
        default:
          throw std::runtime_error("Tokens must be either '0' or '1'.");
      }
  }

  template <typename VectorType,
            typename _T = decltype(*std::begin(VectorType())),
            std::enable_if_t<std::is_integral_v<_T>, bool> = true>
  BitSet(const VectorType &buffer) : BitSet(std::size(buffer)) {
    /*
     * Initialize from any `VectorType` that supports iterations.
     */
    std::size_t i_ = 0;
    for (const auto &x_ : buffer) {
      if (x_) set(i_);
      ++i_;
    }
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
  BitSet(std::initializer_list<T> buffer) : BitSet(std::size(buffer)) {
    /*
     * Initialize from any `initializer_list`.
     */
    std::size_t i_ = 0;
    for (const auto &x_ : buffer) {
      if (x_) set(i_);
      ++i_;
    }
  }

  auto &buffer() {
    /*
     * Return underlying buffer.
     */
    return _buffer;
  }

  std::size_t size() const {
    /*
     * Return number of bits.
     */
    return _size;
  }

  void apply_or(bool value, std::size_t pos) {
    /*
     * Apply `or` in position `pos` given `value`.
     */
    assert(pos < _size);
    _buffer[pos / block_size] |= block_type{value} << (pos % block_size);
  }

  void apply_xor(bool value, std::size_t pos) {
    /*
     * Apply `xor` in position `pos` given `value`.
     */
    assert(pos < _size);
    _buffer[pos / block_size] ^= block_type{value} << (pos % block_size);
  }

  void apply_and(bool value, std::size_t pos) {
    /*
     * Apply `and` in position `pos` given `value`.
     */
    assert(pos < _size);
    _buffer[pos / block_size] &= ~(block_type{!value} << (pos % block_size));
  }

  void reset(std::size_t pos) { apply_and(false, pos); }

  void set(std::size_t pos) { apply_or(true, pos); }

  void clear(std::size_t pos) { apply_and(false, pos); }

  void flip(std::size_t pos) { apply_xor(true, pos); }

  bool test(std::size_t pos) const {
    assert(pos < _size);
    return (_buffer[pos / block_size] >> (pos % block_size)) & block_type{1};
  }

  std::size_t count() const {
    /*
     * Count the number of bits set.
     */
    std::size_t c_ = 0;
    for (const auto &x_ : _buffer) c_ += popcount(x_);
    return c_;
  }

  void clear() {
    /*
     * Clear `BitSet`.
     */
    _size = 0;
    _buffer.clear();
  }

  void push_back(bool value) {
    /*
     * Push new bit to the back of `BitSet`.
     */
    resize(size() + 1);
    if (value)
      set(size() - 1);
    else
      reset(size() - 1);
  }

  void pop_back() {
    /*
     * Remove last bit from `BitSet`.
     */
    resize(size() - 1);
  }

  void resize(std::size_t n) {
    /*
     * Resize `BitSet` to the given size.
     */
    _buffer.resize(n / block_size + ((n % block_size) != 0));
    _size = n;
    _FixUnusedBitsBuffer();
  }

  std::size_t hash() const {
    /*
     * Return hash of `BitSet`.
     */
    std::size_t seed_ = std::size(_buffer);
    for (const auto &x_ : _buffer)
      seed_ ^= x_ + 0x9e3779b9 + (seed_ << 6) + (seed_ >> 2);

    // Return hash
    return seed_;
  }

  bool operator==(const BitSet &other) const {
    return size() == other.size() && _buffer == other._buffer;
  }

  template <typename Buffer>
  auto dump() const {
    /*
     * Serialize BitSet.
     */
    return pysa::branching::dump<Buffer>(_size) +
           pysa::branching::dump<Buffer>(_buffer);
  }

 private:
  Vector<block_type> _buffer;
  std::size_t _size;

  void _FixUnusedBitsBuffer() {
    /*
     * Set all the unused bits in `_buffer` to zero.
     */
    if (const auto s_ = _size % block_size; s_)
      _buffer.back() &= (block_type{1} << s_) - 1;
  }
};

}  // namespace pysa::dpll

namespace pysa::branching {

template <typename BlockType, template <typename...> typename Vector>
struct Archiver<pysa::dpll::BitSet<BlockType, Vector>> {
  using base_type = pysa::dpll::BitSet<BlockType, Vector>;

  template <typename Buffer>
  static constexpr auto dumps(const base_type &bs) {
    return bs.template dump<Buffer>();
  }

  template <typename Buffer>
  static constexpr auto loads(typename Buffer::const_iterator buffer) {
    auto [h1_, n_] = load<std::size_t, Buffer>(buffer);
    auto [h2_, b_] = load<Vector<BlockType>, Buffer>(h1_);
    return std::pair{h2_, base_type(n_, std::move(b_))};
  }
};

}  // namespace pysa::branching
