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

#include <array>
#include <string>
#include <tuple>
#include <vector>

#define ENABLE_IF(...) std::enable_if_t<__VA_ARGS__, bool> = true

namespace pysa::branching {

//-----------------------------------------------------------------------------

template <typename T>
static constexpr auto is_trivial_v =
    std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

template <typename T>
struct is_pair : std::false_type {};

template <typename X, typename Y>
struct is_pair<std::pair<X, Y>> : std::true_type {};

template <typename T>
static constexpr auto is_pair_v = is_pair<T>::value;

//-----------------------------------------------------------------------------

template <typename T>
struct is_tuple : std::false_type {};

template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};

template <typename T>
static constexpr auto is_tuple_v = is_tuple<T>::value;

//-----------------------------------------------------------------------------

template <typename T>
struct is_array : std::false_type {};

template <typename T, typename Allocator>
struct is_array<std::vector<T, Allocator>> : std::true_type {};

template <typename T, std::size_t N>
struct is_array<std::array<T, N>> : std::true_type {};

template <typename T>
static constexpr auto is_array_v = is_array<T>::value;

//-----------------------------------------------------------------------------

template <typename T>
auto is_constructible_from_iterator_(const T &x)
    -> decltype(T{std::begin(x), std::end(x)}) {}

template <typename T, typename = std::void_t<>>
struct is_constructible_from_iterator : std::false_type {};

template <typename T>
struct is_constructible_from_iterator<
    T,
    std::void_t<decltype(is_constructible_from_iterator_(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
static constexpr auto is_constructible_from_iterator_v =
    is_constructible_from_iterator<T>::value;

//-----------------------------------------------------------------------------

template <typename, typename = std::void_t<>>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
static constexpr auto is_iterable_v = is_iterable<T>::value;

//-----------------------------------------------------------------------------

template <typename, typename = std::void_t<>>
struct has_back_inserter : std::false_type {};

template <typename T>
struct has_back_inserter<
    T, std::void_t<decltype(std::declval<T>().insert(
           std::end(std::declval<T>()), *std::begin(std::declval<T>())))>>
    : std::true_type {};

template <typename T>
static constexpr auto has_back_inserter_v = has_back_inserter<T>::value;

//-------------------------- BUFFER DEFINITION --------------------------------

template <typename BufferType = std::vector<char>>
struct Buffer : BufferType {
  // Constructors
  template <typename... Args>
  Buffer(Args &&...args) : BufferType(std::forward<Args>(args)...) {}

  // Defaults
  Buffer(const Buffer &) = default;
  Buffer(Buffer &&) = default;
  Buffer &operator=(const Buffer &) = default;
  Buffer &operator=(Buffer &&) = default;

  // Append another buffer to this
  template <typename OtherBuffer,
            ENABLE_IF(std::is_same_v<std::decay_t<OtherBuffer>, Buffer>)>
  Buffer &operator+=(OtherBuffer &&buffer) {
    if constexpr (std::is_rvalue_reference_v<OtherBuffer>)
      this->insert(std::end(*this), std::make_move_iterator(std::begin(buffer)),
                   std::make_move_iterator(std::end(buffer)));
    else
      this->insert(std::end(*this), std::begin(buffer), std::end(buffer));
    return *this;
  }

  // Concatenate another buffer with this
  template <typename OtherBuffer,
            ENABLE_IF(std::is_same_v<std::decay_t<OtherBuffer>, Buffer>)>
  Buffer operator+(OtherBuffer &&buffer) const {
    auto out_ = *this;
    out_ += buffer;
    return out_;
  }
};

//---------------------------- PRE-DECLARATION --------------------------------

template <typename Buffer = Buffer<>, typename Object, typename... Args>
auto dump(const Object &obj, Args &&...args);

template <typename Object, typename Buffer = Buffer<>, typename... Args>
auto load(typename Buffer::const_iterator buffer, Args &&...args);

//--------------------------- TRIVIAL OBJECTS ---------------------------------

template <typename Buffer, typename Object, ENABLE_IF(is_trivial_v<Object>)>
auto dump_(const Object &obj) {
  // Cast address of obj to iterator
  const auto ptr_ =
      reinterpret_cast<typename Buffer::const_pointer>(std::addressof(obj));

  // Dump to buffer
  return Buffer(ptr_, ptr_ + sizeof(Object));
}

template <typename Object, typename Buffer,
          ENABLE_IF(!std::is_pointer_v<Object> && !std::is_array_v<Object> &&
                    is_trivial_v<Object>)>
auto load_(typename Buffer::const_iterator buffer) {
  // Recast pointer to Object
  const auto ptr_ = reinterpret_cast<const Object *>(std::addressof(*buffer));

  // Initialize object
  return std::pair{buffer + sizeof(Object), Object{*ptr_}};
}

//---------------------- STATIC ARRAYS OF TRIVIALS  ---------------------------

template <typename Array, typename Buffer,
          typename T_ = typename std::remove_extent_t<Array>,
          auto N_ = std::extent_v<Array>,
          ENABLE_IF(std::is_array_v<Array> &&is_trivial_v<T_>),
          std::size_t... I>
auto load_(typename Buffer::const_iterator buffer, std::index_sequence<I...>) {
  // Recast to the right pointer
  const auto ptr_ = reinterpret_cast<const T_ *>(std::addressof(*buffer));

  // Initialize array
  return std::pair{buffer + N_ * sizeof(T_), std::array<T_, N_>{ptr_[I]...}};
}

template <typename Array, typename Buffer,
          typename T_ = typename std::remove_extent_t<Array>,
          auto N_ = std::extent_v<Array>,
          ENABLE_IF(std::is_array_v<Array> &&is_trivial_v<T_>)>
auto load_(typename Buffer::const_iterator buffer) {
  return load_<Array, Buffer>(buffer, std::make_index_sequence<N_>{});
}

//------------------------------- PAIRS ---------------------------------------

template <typename Buffer, typename Pair,
          ENABLE_IF(!is_trivial_v<Pair> && is_pair_v<Pair>)>
auto dump_(const Pair &pair) {
  // Dump pair
  return dump<Buffer>(pair.first) + dump<Buffer>(pair.second);
}

template <typename Pair, typename Buffer, typename... Args,
          ENABLE_IF(!is_trivial_v<Pair> && is_pair_v<Pair>)>
auto load_(typename Buffer::const_iterator buffer) {
  const auto [h1_, x_] = load<typename Pair::first_type, Buffer>(buffer);
  const auto [h2_, y_] = load<typename Pair::second_type, Buffer>(h1_);
  return std::pair{h2_, Pair{std::move(x_), std::move(y_)}};
}

//------------------------------ TUPLES ---------------------------------------

template <typename Buffer, typename Tuple, std::size_t... I,
          ENABLE_IF(!is_trivial_v<Tuple> && is_tuple_v<Tuple>)>
auto dump_(const Tuple &tuple, std::index_sequence<I...>) {
  return (dump<Buffer>(std::get<I>(tuple)) + ...);
}

template <typename Buffer, typename Tuple,
          ENABLE_IF(!is_trivial_v<Tuple> && is_tuple_v<Tuple>)>
auto dump_(const Tuple &tuple) {
  return dump_<Buffer>(tuple,
                       std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <typename Tuple, typename Buffer, std::size_t I, std::size_t N,
          typename Tuple_, ENABLE_IF(!is_trivial_v<Tuple> && is_tuple_v<Tuple>)>
auto load_(typename Buffer::const_iterator buffer, Tuple_ &&tuple) {
  if constexpr (I < N) {
    // Get type of the Ith element
    using T_ = std::tuple_element_t<I, Tuple>;

    // Load buffer
    auto [buffer_, x_] = load<T_, Buffer>(buffer);

    // Retun buffer + loaded object
    return load_<Tuple, Buffer, I + 1, N>(
        buffer_,
        std::tuple_cat(std::move(tuple), std::tuple<T_>{std::move(x_)}));
  } else

    // End recursion
    return std::pair{buffer, std::move(tuple)};
}

template <typename Tuple, typename Buffer, ENABLE_IF(is_tuple_v<Tuple>)>
auto load_(typename Buffer::const_iterator buffer) {
  return load_<Tuple, Buffer, 0, std::tuple_size_v<Tuple>>(buffer,
                                                           std::tuple{});
}

//-------------------------- ARRAY OF CONTIGUOUS ELEMENTS ---------------------

template <typename Buffer, typename Object,
          typename T_ = typename Object::value_type,
          ENABLE_IF(is_array_v<Object> &&is_trivial_v<T_>)>
auto dump_(const Object &obj) {
  // Cast address of obj underlying data to iterator
  const auto ptr_ =
      reinterpret_cast<typename Buffer::const_pointer>(obj.data());

  // Get size
  const std::size_t size_ = std::size(obj);

  // Dump size and data
  return dump<Buffer>(size_) + Buffer(ptr_, ptr_ + size_ * sizeof(T_));
}

//------------------------------ ITERABLE OBJECTS -----------------------------

template <
    typename Buffer, typename Object,
    typename T_ = std::decay_t<decltype(*std::begin(std::declval<Object>()))>,
    ENABLE_IF(is_iterable_v<Object>),
    ENABLE_IF(!is_array_v<Object> || !is_trivial_v<T_>)>
auto dump_(const Object &obj) {
  // Initialize buffer
  auto buffer_ = dump<Buffer, std::size_t>(std::size(obj));

  // Fill buffer with elements
  for (const auto &x_ : obj) buffer_ += dump<Buffer>(x_);

  // Return buffer
  return buffer_;
}

//----------- OBJECTS THAT CAN BE CREATE FROM ITERATORS OF TRIVIALS -----------

template <
    typename Object, typename Buffer,
    typename T_ = std::decay_t<decltype(*std::begin(std::declval<Object>()))>,
    ENABLE_IF(is_constructible_from_iterator_v<Object>),
    ENABLE_IF(is_trivial_v<T_>)>
auto load_(typename Buffer::const_iterator buffer) {
  // Load size
  const auto [buffer_, size_] = load<std::size_t, Buffer>(buffer);

  // Cast buffer to pointer to data type
  const auto ptr_ = reinterpret_cast<const T_ *>(std::addressof(*buffer_));

  // Load
  return std::pair{buffer_ + size_ * sizeof(T_), Object(ptr_, ptr_ + size_)};
}

//----------------------- OBJECTS THAT HAVE BACK INSERTER ---------------------

template <
    typename Object, typename Buffer,
    typename T_ = std::decay_t<decltype(*std::begin(std::declval<Object>()))>,
    ENABLE_IF(has_back_inserter_v<Object>),
    ENABLE_IF(!is_trivial_v<T_> || !is_constructible_from_iterator_v<Object>)>
auto load_(typename Buffer::const_iterator buffer) {
  // Initialize object
  Object obj_;

  // Get size
  auto [buffer_, size_] = load<std::size_t, Buffer>(buffer);

  // Fill using back inserter
  for (std::size_t i_ = 0; i_ < size_; ++i_) {
    // Load
    auto [tbuffer_, x_] = load<T_, Buffer>(buffer_);
    obj_.insert(std::end(obj_), std::move(x_));
    buffer_ = tbuffer_;
  }

  // Return object
  return std::pair{buffer_, std::move(obj_)};
}

//----------------------------- POINTERS TO TRIVIAL ---------------------------

template <
    typename Buffer, typename Pointer,
    typename T_ = typename std::decay_t<decltype(*std::declval<Pointer>())>,
    ENABLE_IF(std::is_pointer_v<Pointer> &&is_trivial_v<T_>)>
auto dump_(const Pointer &ptr, std::size_t size) {
  // Cast address of obj underlying data to iterator
  const auto ptr_ = reinterpret_cast<typename Buffer::const_pointer>(ptr);

  // Return bew object
  return dump<Buffer>(size) + Buffer(ptr_, ptr_ + size * sizeof(T_));
}

//-----------------------------------------------------------------------------

template <typename Object>
struct Archiver {
  template <typename Buffer, typename... Args>
  static constexpr auto dumps(const Object &obj, Args &&...args) {
    return dump_<Buffer>(obj, std::forward<Args>(args)...);
  }
  template <typename Buffer, typename... Args>
  static constexpr auto loads(typename Buffer::const_iterator buffer,
                              Args &&...args) {
    return load_<Object, Buffer>(buffer, std::forward<Args>(args)...);
  }
};

template <typename Buffer, typename Object, typename... Args>
auto dump(const Object &obj, Args &&...args) {
  return Archiver<Object>::template dumps<Buffer>(obj,
                                                  std::forward<Args>(args)...);
}

template <typename Object, typename Buffer, typename... Args>
auto load(typename Buffer::const_iterator buffer, Args &&...args) {
  return Archiver<Object>::template loads<Buffer>(buffer,
                                                  std::forward<Args>(args)...);
}

}  // namespace pysa::branching

#undef ENABLE_IF
