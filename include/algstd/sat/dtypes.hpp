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

#pragma once

#include <cstdint>

namespace pysa::algstd{
    template<typename T>
    struct lidx {
      //! Indexed Boolean (lidx) struct.
      //! Packs a zero-based index and a boolean bit into storage of type T.
      //! This represents, e.g. a literal in a SAT formula.
      //! The least significant bit contains the sign bit while the remaining bits encode the index.
      T x;

      constexpr explicit lidx(T var) : x(var) {}
      constexpr lidx () : x(~0) {}
      constexpr lidx (const lidx&) = default;
      constexpr lidx (lidx&&) = default;
      constexpr lidx& operator=(const lidx&) = default;
      constexpr lidx& operator=(lidx&&) = default;
      constexpr explicit lidx (T var, bool neg) : x(2 * var + neg) {}

      constexpr lidx<T>  operator~() const { return lidx<T> (x ^ 1); }
      constexpr lidx<T>  operator^(const bool b) const { return lidx<T> (x ^ (T)b); }
      lidx  &operator^=(const bool b) {
          x ^= (T)b;
          return *this;
      }
      constexpr bool sign() const { return x & 1; }
      constexpr T idx() const { return x >> 1; }
      constexpr bool operator==(const lidx<T>  &p) const { return x == p.x || (is_null() && p.is_null()); }
      constexpr bool operator!=(const lidx<T>  &p) const { return !(*this==p); }
      // All-ones (~0) encode a null literal, but we only check the index bits.
      constexpr bool is_null() const {return (x>>1) == (T(~0) >> 1); }
    };

    // A SAT literal is directly represented by a logical index.
    // with the sign bit encoding whether or not the variable is negated.
    template<typename T>
    using SATLiteral = lidx<T>;

    // The canonical Lit type uses uint32 storage for up to 2^31 - 2 variables
    typedef SATLiteral<uint32_t> Lit;

    // A clause index (ClIdx) indexes a clause that a variable participates in,
    // with the sign bit denoting whether or not the variable is negated in that clause.
    typedef lidx<uint64_t> ClIdx;

    // The null literal is interpreted as a sentinel or invalid literal,
    // e.g. in static storage for k-SAT clauses.
    // Equality can be checked through the is_null() method.
    template <typename T>
    const lidx<T> null_literal;

    enum class ClFlag: uint16_t{
      //! Flags describing individidual clauses and operations on clauses
      NONE=0,
      // Unit Clause
      UNIT=1,
      // 2-SAT Clause
      SAT2=2,
      // Clause is satisfied
      SAT=4,
      // Clause was simplified or modified (when ClFlag is a return value)
      CLSIMP=8,
      // Empty/unsatisfied Clause
      UNSAT=16
    };

    inline constexpr ClFlag operator|(ClFlag a, ClFlag b) {
      return a = static_cast<ClFlag>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
    }

    inline constexpr ClFlag operator&(ClFlag a, ClFlag b) {
      return a = static_cast<ClFlag> (static_cast<uint16_t>(a) & static_cast<uint16_t>(b));
    }
}

