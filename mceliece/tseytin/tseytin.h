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

#ifndef MCI_SOLVER_CIRCSAT_H
#define MCI_SOLVER_CIRCSAT_H
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <variant>
#include <vector>

namespace tseytin {

struct Lit {
  uint32_t x;

  constexpr explicit Lit(uint32_t i) : x(i) {}
  constexpr Lit() : x(0) {}
  constexpr explicit Lit(uint32_t var, bool neg) : x(2 * var + neg) {}

  constexpr Lit operator~() const { return Lit(x ^ 1); }
  constexpr Lit operator^(const bool b) const { return Lit(x ^ (uint32_t)b); }
  Lit &operator^=(const bool b) {
    x ^= (uint32_t)b;
    return *this;
  }
  constexpr bool sign() const { return x & 1; }
  constexpr uint32_t var() const { return x >> 1; }
  constexpr bool operator==(const Lit &p) const { return x == p.x; }
  constexpr bool operator!=(const Lit &p) const { return x != p.x; }
};

// output stream implementations
inline std::ostream &operator<<(std::ostream &os, const Lit lit) {
  os << (lit.sign() ? "-" : "") << (lit.var() + 1);
  return os;
}
inline std::ostream &operator<<(std::ostream &co,
                                const std::vector<Lit> &lits) {
  for (uint32_t i = 0; i < lits.size(); i++) {
    co << lits[i];
    if (i != lits.size() - 1)
      co << " ";
  }
  return co;
}

// Struct that contains a mixed SAT/XORSAT problem.
// Compatible with cryptominisat's DimacsParser when Lit_t=CMSat::Lit
template <typename Lit_t> struct XORSatClauses {
  XORSatClauses() = default;
  unsigned int n_vars = 0;
  // Standard SAT clauses
  std::vector<std::vector<Lit_t>> clauses;
  std::vector<std::vector<uint32_t>> xor_rows;
  std::vector<uint8_t> xor_rhs;

  unsigned int nVars() const { return n_vars; }
  void new_var() { n_vars += 1; }
  void new_vars(unsigned int n) { n_vars += n; }
  void add_red_clause(std::vector<Lit_t> &) {}
  void add_xor_clause(const std::vector<uint32_t> &vars, bool rhs) {
    xor_rows.push_back(vars);
    xor_rhs.push_back(rhs ? 1 : 0);
  }
  void add_clause(const std::vector<Lit_t> &lits) { clauses.push_back(lits); }
  std::vector<uint8_t> to_c_array() {
    std::vector<uint8_t> mat_rows;
    size_t n_clauses = xor_rhs.size();
    mat_rows.resize((n_vars + 1) * n_clauses, 0);
    for (int i = 0; i < n_clauses; ++i) {
      for (uint32_t v : xor_rows[i]) {
        mat_rows[i * (n_vars + 1) + v] = 1;
      }
      // final column is the (transformed) ciphertext
      mat_rows[i * (n_vars + 1) + n_vars] = xor_rhs[i];
    }
    return mat_rows;
  }
  std::ostream &write_dimacs(std::ostream &os) const {
    os << "p cnf " << n_vars << " " << clauses.size() + xor_rhs.size()
       << std::endl;
    for (int i = 0; i < xor_rhs.size(); ++i) {
      os << 'x';
      if (xor_rhs[i] == 0) {
        os << '-';
      }
      for (unsigned int b : xor_rows[i]) {
        os << b + 1 << ' ';
      }
      os << "0\n";
    }
    for (const std::vector<Lit_t> &cl : clauses) {
      os << cl << " 0\n";
    }

    return os;
  }
};

// A "wire" in a CircuitSAT problem may be either a literal or a specified
// boolean value
typedef std::variant<bool, Lit> wirevar;

// A NOT gate simply negates either the variable or the literal.
struct NOTGate {
  wirevar operator()(Lit &a) { return {~a}; }
  wirevar operator()(bool &a) { return {!a}; }
};

// Binary visitor classes for use with std::visit on wirevars.
// These visitor classes simplify any gates with at least one boolean input
// If both inputs are literals, a new variable is added to represent the output
// of the gate.
class BinaryVisitor {
public:
  explicit BinaryVisitor(unsigned int &n,
                         std::vector<std::vector<Lit>> &clauses)
      : _n(n), _clauses(clauses) {}
  unsigned int &_n;
  std::vector<std::vector<Lit>> &_clauses;
};

class ANDGate : BinaryVisitor {
public:
  using BinaryVisitor::BinaryVisitor;
  wirevar operator()(Lit &a, Lit &b) {
    unsigned int nv = _n;
    _n += 1;
    Lit c(nv, false);
    const std::vector<Lit> v1({~a, ~b, c});
    const std::vector<Lit> v2({a, ~c});
    const std::vector<Lit> v3({b, ~c});
    _clauses.push_back(v1);
    _clauses.push_back(v2);
    _clauses.push_back(v3);
    return {c};
  }
  wirevar operator()(bool &a, Lit &b) {
    if (a) {
      return {b};
    } else {
      return {false};
    }
  }
  wirevar operator()(Lit &a, bool &b) { return this->operator()(b, a); }
  wirevar operator()(bool &a, bool &b) { return {a && b}; }
};

class ORGate : BinaryVisitor {
public:
  using BinaryVisitor::BinaryVisitor;
  wirevar operator()(Lit &a, Lit &b) {
    unsigned int nv = _n;
    _n += 1;
    Lit c(nv, false);
    std::vector<Lit> v1({a, b, ~c});
    std::vector<Lit> v2({~a, c});
    std::vector<Lit> v3({~b, c});
    _clauses.push_back(v1);
    _clauses.push_back(v2);
    _clauses.push_back(v3);
    return {c};
  }
  wirevar operator()(bool &a, Lit &b) {
    if (a) {
      return {true};
    } else {
      return {b};
    }
  }
  wirevar operator()(Lit &a, bool &b) { return this->operator()(b, a); }
  wirevar operator()(bool &a, bool &b) { return {a || b}; }
};

class XORGate : BinaryVisitor {
public:
  using BinaryVisitor::BinaryVisitor;
  wirevar operator()(Lit &a, Lit &b) {
    unsigned int nv = _n;
    _n += 1;
    Lit c(nv, false);
    std::vector<Lit> v1({~a, ~b, ~c});
    std::vector<Lit> v2({a, b, ~c});
    std::vector<Lit> v3({a, ~b, c});
    std::vector<Lit> v4({~a, b, c});
    _clauses.push_back(v1);
    _clauses.push_back(v2);
    _clauses.push_back(v3);
    _clauses.push_back(v4);
    return {c};
  }
  wirevar operator()(bool &a, Lit &b) {
    if (a) {
      return {~b};
    } else {
      return {b};
    }
  }
  wirevar operator()(Lit &a, bool &b) { return this->operator()(b, a); }
  wirevar operator()(bool &a, bool &b) { return {bool(a xor b)}; }
};

// Construct a SAT problem corresponding to a binary-gate circuit SAT problem
// via the Tseytin reduction.
class TseytinSAT {
public:
  explicit TseytinSAT(unsigned int starting_vars)
      : current_num_vars(starting_vars), extra_clauses() {};
  unsigned int current_num_vars;
  std::vector<std::vector<Lit>> extra_clauses;

  wirevar NOT(wirevar a) {
    NOTGate vis;
    return std::visit(vis, a);
  }
  wirevar AND(wirevar a, wirevar b) {
    ANDGate bvis(current_num_vars, extra_clauses);
    return std::visit(bvis, a, b);
  }
  wirevar OR(wirevar a, wirevar b) {
    ORGate bvis(current_num_vars, extra_clauses);
    return std::visit(bvis, a, b);
  }
  wirevar XOR(wirevar a, wirevar b) {
    XORGate bvis(current_num_vars, extra_clauses);
    return std::visit(bvis, a, b);
  }
  std::tuple<wirevar, wirevar> half_adder(wirevar a, wirevar b) {
    wirevar sum = XOR(a, b);
    wirevar carry = AND(a, b);
    return std::make_tuple(sum, carry);
  }
  // Construct an accumulation circuit whose output sum is acc_out.
  // i.e. exactly acc_out of the input literals have a truth value of 1.
  void accumulate(const std::vector<Lit> &literals, size_t acc_out) {
    size_t n = literals.size();
    // a size_t will never be greater than 2^64
    std::vector<std::vector<wirevar>> acc_circuit;
    acc_circuit.resize(n);
    for (int i = 0; i < n; ++i) {
      // acc_circuit[i] are the bits of an integer whose value is no greater
      // than i+1
      int w = 1 + (int)std::floor(std::log2(i + 1));
      acc_circuit[i].resize(w + 1);
    }
    acc_circuit[0][0] = literals[0];
    for (int i = 1; i < n; ++i) {
      int w = 1 + (int)std::floor(std::log2(i + 1));
      wirevar s, c, c2;
      std::tie(s, c) = half_adder(acc_circuit[i - 1][0], literals[i]);
      acc_circuit[i][0] = s;
      for (int j = 1; j < w; ++j) {
        std::tie(s, c2) = half_adder(acc_circuit[i - 1][j], c);
        acc_circuit[i][j] = s;
        c = c2;
      }
      // assert the final carry is always zero
      if (const Lit *l = std::get_if<Lit>(&c2); l) {
        extra_clauses.push_back({~Lit(*l)});
      } else if (const bool *b = std::get_if<bool>(&c2); b) {
        assert(!(*b));
      }
    }
    int w = 1 + (int)std::floor(std::log2(n));
    for (int i = 0; i < w; ++i) {
      bool bit_i = (acc_out >> i) & 1;
      wirevar wv = acc_circuit[n - 1][i];
      if (const Lit *l = std::get_if<Lit>(&wv); l) {
        if (bit_i) {
          extra_clauses.push_back({Lit(*l)});
        } else {
          extra_clauses.push_back({~Lit(*l)});
        }
      } else if (const bool *b = std::get_if<bool>(&wv); b) {
        assert((*b) == bit_i);
      }
    }
  }
};

// Augment the XORSAT problem with a Hamming weight constraint on the solution
void add_weight_constraint(int t, XORSatClauses<Lit> &solver, int n = -1);

// Reduce all XORSAT clauses into 3-SAT clauses
void into_3sat_only(XORSatClauses<Lit> &solver);

} // namespace tseytin
#endif // MCI_SOLVER_CIRCSAT_H
