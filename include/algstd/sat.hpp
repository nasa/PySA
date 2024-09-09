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

//! Data structures for satisfiability problems

#pragma once
#include <iostream>
#include <vector>
#include <regex>
#include <sstream>
#include <cassert>

#include "algstd/sat/dtypes.hpp"
#include "algstd/sat/clause.hpp"

namespace pysa::algstd{
    template <typename ClauseT>
    struct SATFormula{
      //! k-SAT formula, where the maximum clause size k is specified at compile time
      typedef ClauseT clause_type;
      typedef typename clause_type::literal_type literal_type;
      typedef typename std::vector<ClauseT> clause_container;

      std::vector<ClauseT> _clauses;
      //size_t _num_lits;      
      SATFormula() = default;
      SATFormula(const SATFormula&) = default;
      SATFormula(SATFormula&&) = default;
      SATFormula& operator=(const SATFormula&) = default;
      SATFormula& operator=(SATFormula&&) = default;

      clause_container& clauses(){
        return _clauses;
      }
      const clause_container& clauses() const{
        return _clauses;
      }
      size_t num_clauses() const{
        return _clauses.size();
      }

      void add_clause(clause_type&& clause){
        _clauses.emplace_back(clause);
      }

      clause_type& operator[](size_t i){
        return _clauses[i];
      }
      const clause_type& operator[](size_t i) const{
        return _clauses[i];
      }
      clause_type remove_clause(size_t idx){
        if(idx != _clauses.size() - 1)
          std::swap(_clauses[idx], _clauses.back());
        clause_type cl = std::move(_clauses.back());
        _clauses.pop_back();
        return cl;
      }
    };

    struct SATIndexer{
      //! Map a variable index to the list of clauses it participates in, along with negations.
      std::vector<std::vector<ClIdx>> clauses_by_var;
      size_t _ncl = 0;

      template <typename ClauseT>
      void include_clause(const ClauseT& cl){
        typedef typename ClauseT::literal_type LitT;
        for(const LitT& lit : cl){
          size_t v = lit.idx();
          bool s = lit.sign();
          if(v >= clauses_by_var.size())
            clauses_by_var.resize(v + 1);
          clauses_by_var[v].push_back(ClIdx(_ncl, s));
        }
        _ncl++;
      }

      size_t n_vars() const{
        return clauses_by_var.size();
      }
      size_t n_clauses() const {
        return _ncl;
      }
      template<typename FormulaT>
      explicit SATIndexer(const FormulaT& formula): clauses_by_var{} {
        typedef typename FormulaT::clause_type ClauseT;
        const auto& clauses = formula.clauses();

        for(const ClauseT& cl: clauses){
          include_clause(cl);
        }
      }
    };
}

// output stream implementations
template<typename T>
inline std::ostream &operator<<(std::ostream &os, const pysa::algstd::lidx<T> lit) {
  os << (lit.sign() ? "-" : "") << lit.idx() + 1;
  return os;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os,
                                const std::vector<pysa::algstd::lidx<T>> &lits) {
  for (uint32_t i = 0; i < lits.size(); i++) {
    os << lits[i];
    if (i != lits.size() - 1)
      os << " ";
  }
  return os;
}

template<typename T, unsigned K>
inline std::ostream &operator<<(std::ostream &os,
                                const pysa::algstd::SATClause<T,K> &cl) {
  for (uint32_t i = 0; i < cl.size(); i++) {
    os << cl[i];
    if (i != cl.size() - 1)
      os << " ";
  }
  return os;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os,
                                const pysa::algstd::SATClauseV<T> &cl) {
  for (uint32_t i = 0; i < cl.size(); i++) {
    os << cl[i];
    if (i != cl.size() - 1)
      os << " ";
  }
  return os;
}

// Read in a literal from an istream
template<typename T>
inline std::istream & operator>>(std::istream &is, pysa::algstd::lidx<T>& lit){
  int64_t l;
  if(is >> l){
    bool s = l<0;
    T v = (s ? -l : l) - 1;
    lit = pysa::algstd::lidx<T>(v, s);
  }
  return is;
}

namespace pysa::algstd{

template <typename InputStream, typename ClauseType = SATClauseV<uint32_t>>
SATFormula<ClauseType> ReadCNF(InputStream &&input_stream) {
  typedef typename ClauseType::literal_type literal_type;

  // Initialize formula
  SATFormula<ClauseType> formula;

  // Initialize current clause
  ClauseType current_cl;

  bool parsed_problem_line = false;
  std::size_t line_num = 0;
  std::string line;

  // For each line in input ...
  while (std::getline(input_stream, line)) {
    // Increment line number
    ++line_num;

    // If line is empty, skip
    if (line.length() == 0)
      continue;

    // If line is a comment, skip
    if (std::tolower(line[0]) == 'c')
      continue;

    // If line is problem line, skip
    if (std::regex_match(line, std::regex(R"(^p\s+cnf\s+\d+\s+\d+\s*)"))) {
      if (parsed_problem_line) {
        std::cerr << "Problem line parsed multiple times" << std::endl;
        throw std::runtime_error("CNF Parse Error");
      }
      parsed_problem_line = true;

#ifndef NDEBUG
      std::cerr << line << std::endl;
#endif
      continue;
    }

    // Check if problem line appears exactly once
    if (!parsed_problem_line) {
      std::cerr << "CNF file in the wrong format" << std::endl;
      throw std::runtime_error("CNF Parse Error");
    }

    // Check that line is a sequence of numbers
    if (!std::regex_match(line,
                          std::regex(R"(^\s*-?\d+(\s+-?\d+)*\s+0\s*$)"))) {
      std::cerr << "Failed to parse line " << line_num << ": " << line << ".";
      throw std::runtime_error("CNF Parse Error");
    }

    //  Parse lines
    {
      std::stringstream ss(line);
      literal_type js;

      // For each token in line ...
      while (ss >> js){
        // If token is different from zero, append to current clause
        if (!js.is_null())
          current_cl.add_lit(js);
        // Otherwise, append to formula
        else {
          formula.add_clause(std::move(current_cl));

          // Remove all elements from current_cl
          // (in case moving didn't reset it)
          current_cl = ClauseType();

          // Ignore everything after zero
          break;
        }
      }

      // Only spaces and non-printable characters should remain
      {
        std::string left_;
        ss >> left_;
        if (!ss.eof() ||
            std::any_of(std::begin(left_), std::end(left_), [](auto &&c) {
              return !std::isspace(c) && std::isprint(c);
            })) {
          std::cerr << "Failed to parse line " << line_num << ": " << line
                    << ".";
          throw std::runtime_error("CNF Parse Error");
        }
      }
    }
  }

  // Return formula
  return formula;
}
}