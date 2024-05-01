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

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <pysa/archive/archive.hpp>
#include <list>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pysa::branching {

template <typename Container, typename RNG,
          typename T = typename Container::value_type>
auto Random1D(RNG &&rng, std::size_t n) {
  Container v_;
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> uni_;
    std::generate_n(std::back_inserter(v_), n,
                    [&rng, &uni_]() { return uni_(rng); });
  } else {
    std::uniform_int_distribution<T> uni_;
    std::generate_n(std::back_inserter(v_), n,
                    [&rng, &uni_]() { return uni_(rng); });
  }
  return v_;
}

template <typename Container, typename RNG,
          typename T = typename Container::value_type,
          typename std::enable_if_t<std::is_integral_v<T>, bool> = true>
auto RandomSet(RNG &&rng, std::size_t n) {
  Container v_;
  std::uniform_int_distribution<T> uni_;
  for (std::size_t i_ = 0; i_ < n; ++i_) v_.insert(uni_(rng));
  return v_;
}

struct TriviallyCopyable {
  int a;
  float b[10];

  bool operator==(const TriviallyCopyable &other) const {
    return a == other.a && std::transform_reduce(
                               b, b + 10, other.b, true,
                               [](auto &&x, auto &&y) { return x & y; },
                               [](auto &&x, auto &&y) { return x == y; });
  }
};

//-----------------------------------------------------------------------------

template <typename T>
struct NonTriviallyCopyable {
  std::size_t x;
  std::vector<T> a;

  template <typename... Args>
  NonTriviallyCopyable(std::size_t x = 0, Args &&...args)
      : x{x}, a(std::forward<Args>(args)...) {}

  bool operator==(const NonTriviallyCopyable<T> &other) const {
    return x == other.x && a == other.a;
  }
};

template <typename T>
struct Archiver<NonTriviallyCopyable<T>> {
  // Define base type
  using base_type = NonTriviallyCopyable<T>;

  // Overload dumps
  template <typename Buffer>
  static constexpr auto dumps(const NonTriviallyCopyable<T> &obj) {
    return dump<Buffer>(obj.x) + dump<Buffer>(obj.a);
  }

  // Overload loads
  template <typename Buffer>
  static constexpr auto loads(typename Buffer::const_iterator buffer) {
    auto [h1_, x_] =
        load<decltype(std::declval<base_type>().x), Buffer>(buffer);
    auto [h2_, a_] = load<decltype(std::declval<base_type>().a), Buffer>(h1_);
    return std::pair{h2_, base_type{x_, std::move(a_)}};
  }
};

//-----------------------------------------------------------------------------

template <typename T>
struct NonTriviallyCopyableWithInit : NonTriviallyCopyable<T> {
  std::size_t y;

  template <typename... Args>
  NonTriviallyCopyableWithInit(std::size_t y, Args &&...args)
      : NonTriviallyCopyable<T>(std::forward<Args>(args)...), y{y} {}

  bool operator==(const NonTriviallyCopyableWithInit<T> &other) const {
    return y == other.y &&
           *static_cast<const NonTriviallyCopyable<T> *>(this) ==
               *static_cast<const NonTriviallyCopyable<T> *>(&other);
  }
};

template <typename T>
struct Archiver<std::vector<NonTriviallyCopyableWithInit<T>>> {
  // Define base type
  using base_type = NonTriviallyCopyableWithInit<T>;

  // Overload dumps
  template <typename Buffer>
  static constexpr auto dumps(const std::vector<base_type> &obj) {
    // Initialize buffer with size of the vector
    auto buffer_ = dump<Buffer, std::size_t>(std::size(obj));

    // Dump each element
    for (const auto &x_ : obj)
      buffer_ += dump<Buffer, NonTriviallyCopyable<T>>(x_);

    // Return buffer
    return buffer_;
  }

  // Overload loads
  template <typename Buffer>
  static constexpr auto loads(typename Buffer::const_iterator buffer,
                              std::size_t y) {
    // Initialize vector
    std::vector<base_type> obj_;

    // Get size of the vector
    auto [h_, size_] = load<std::size_t, Buffer>(buffer);

    // Load each element and update head
    for (std::size_t i_ = 0; i_ < size_; ++i_) {
      auto [h1_, x_] = load<NonTriviallyCopyable<T>, Buffer>(h_);
      obj_.emplace_back(y, x_.x, std::move(x_.a));
      h_ = h1_;
    }

    // Return head and vector
    return std::pair{h_, std::move(obj_)};
  }
};

//-----------------------------------------------------------------------------

template <typename Object, typename... Args>
[[nodiscard]] bool test_dump(const Object &object, Args &&...args) {
  // Dump
  const auto buffer_ = dump(object);

  // Load
  const auto [head_, obj_] =
      load<Object>(std::begin(buffer_), std::forward<Args>(args)...);

  // Check equality for arrays
  if constexpr (std::is_array_v<Object>) {
    if (std::end(buffer_) != head_ || std::extent_v<Object> != std::size(obj_))
      return false;
    for (std::size_t i_ = 0, end_ = std::size(obj_); i_ < end_; ++i_)
      if (object[i_] != obj_[i_]) return false;
    return true;

    // Check equality for everything else
  } else
    return std::end(buffer_) == head_ && (obj_ == object);
}

void TestArchive() {
  // Initialize random engine
  std::mt19937_64 rng_(std::random_device{}());

  // Check fundamental types
  {
    assert(test_dump(static_cast<bool>(true)));
    assert(test_dump(static_cast<char>(42)));
    assert(test_dump(static_cast<int>(42)));
    assert(test_dump(static_cast<long>(42)));
    assert(test_dump(static_cast<float>(42.42)));
    assert(test_dump(static_cast<double>(42.42)));
    assert(test_dump(static_cast<long double>(42.42)));
    //
    assert(test_dump(std::complex<float>{1.1, 2.2}));
    assert(test_dump(std::complex<double>{1.1, 2.2}));
    assert(test_dump(std::complex<long double>{1.1, 2.2}));
    //
    assert(test_dump(std::pair{1.1, -3}));
    assert(
        test_dump(std::tuple{std::pair{std::complex<float>{1, 2}, 3.3}, -3}));
    //
    assert(test_dump(std::string("123")));
    //
    {
      const float data_[] = {1.1, 2.2, 3.3, 4.4, 5.5};
      assert(test_dump(data_));
    }
    //
    {
      auto *data_ = new int[10];
      for (std::size_t i_ = 0; i_ < 10; ++i_) data_[i_] = rng_();
      const auto buffer_ = dump(data_, 10);
      const auto [head_, obj_] = load<std::vector<int>>(std::begin(buffer_));
      assert(std::end(buffer_) == head_ && std::size(obj_) == 10);
      for (std::size_t i_ = 0; i_ < 10; ++i_) assert(obj_[i_] == data_[i_]);
      delete[] data_;
    }
  }

  // Check trivially/non trivially copyable data
  {
    assert(test_dump(NonTriviallyCopyable<float>(
        12, Random1D<std::vector<float>>(rng_, 1000))));
    assert(test_dump(NonTriviallyCopyable<double>(
        13, Random1D<std::vector<double>>(rng_, 1000))));
    std::vector<NonTriviallyCopyable<float>> v_;
    for (std::size_t i_ = 0; i_ < 10; ++i_)
      v_.emplace_back(14, Random1D<std::vector<float>>(rng_, rng_() % 100));
    assert(test_dump(v_));
  }

  // Check containers
  {
    assert(test_dump(Random1D<std::vector<int>>(rng_, 100)));
    assert(test_dump(Random1D<std::vector<float>>(rng_, 100)));
    assert(test_dump(Random1D<std::vector<double>>(rng_, 100)));
    //
    assert(test_dump(Random1D<std::list<int>>(rng_, 100)));
    assert(test_dump(Random1D<std::list<float>>(rng_, 100)));
    assert(test_dump(Random1D<std::list<double>>(rng_, 100)));
    //
    assert(test_dump(RandomSet<std::set<int>>(rng_, 100)));
    assert(test_dump(RandomSet<std::unordered_set<int>>(rng_, 100)));
    //
    assert(test_dump(std::map<int, float>{{-1, 1.1}, {2, -2.2}}));
    //
    {
      std::vector<std::vector<float>> v_;
      for (std::size_t i_ = 0; i_ < 10; ++i_)
        v_.push_back(Random1D<std::vector<float>>(rng_, rng_() % 100));
      assert(test_dump(v_));
    }
    //
    {
      std::vector<std::list<float>> v_;
      for (std::size_t i_ = 0; i_ < 10; ++i_)
        v_.push_back(Random1D<std::list<float>>(rng_, rng_() % 100));
      assert(test_dump(v_));
    }
    //
    {
      std::list<std::list<float>> v_;
      for (std::size_t i_ = 0; i_ < 10; ++i_)
        v_.push_back(Random1D<std::list<float>>(rng_, rng_() % 100));
      assert(test_dump(v_));
    }
    //
    {
      std::map<int, NonTriviallyCopyable<float>> m_;
      for (std::size_t i_ = 0; i_ < 10; ++i_)
        m_[rng_() % std::numeric_limits<int>::max()] = {
            i_, Random1D<std::vector<float>>(rng_, rng_() % 100)};
      assert(test_dump(m_));
    }
    {
      std::unordered_map<long, NonTriviallyCopyable<double>> m_;
      for (std::size_t i_ = 0; i_ < 10; ++i_)
        m_[rng_() % std::numeric_limits<long>::max()] = {
            i_ * i_, Random1D<std::vector<double>>(rng_, rng_() % 100)};
      assert(test_dump(m_));
    }
    {
      std::vector<NonTriviallyCopyableWithInit<int>> v_;
      for (std::size_t i_ = 0; i_ < 10; ++i_)
        v_.emplace_back(42, rng_(), Random1D<std::vector<int>>(rng_, 1000));
      assert(test_dump(v_, 42));
    }
  }
}

}  // namespace pysa::branching
