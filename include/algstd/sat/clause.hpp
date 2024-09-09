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

#include "algstd/sat/dtypes.hpp"

namespace pysa::algstd
{
    

    template <typename T, unsigned K>
    struct SATClause{
      //! k-SAT clause using an array of size k for static storage.
      //! A clause of size less than k can be represented by terminating the clause
      //! with null_literal;
      typedef lidx<T> literal_type ;
      typedef std::array<lidx<T>, K> literal_container;

      literal_container _lits;
      uint16_t n=0;
      ClFlag _flags = ClFlag::UNSAT;
      constexpr SATClause() : _lits{} { }
      constexpr SATClause(std::initializer_list<lidx<T>> lits){
        for(lidx<T> l: lits)
          add_lit(l);
      }
      literal_container& literals(){
        return _lits;
      }
      const literal_container& literals() const{
        return _lits;
      }
      literal_type& operator[](uint16_t i){
        return _lits[i];
      }
      const literal_type& operator[](uint16_t i) const{
        return _lits[i];
      }
      ClFlag flags() const{
        return _flags;
      }
      ClFlag& flags_ref(){
        return _flags;
      }

      void add_lit(literal_type lit){
        if(n==K)
          throw std::out_of_range("k-SAT clause exceeded");
        _lits[n++] = lit;
        if(n==1)
          _flags=ClFlag::UNIT;
        else if(n==2)
          _flags=ClFlag::SAT2;
        else
          _flags=ClFlag::NONE;
      }
      bool remove_lit(uint16_t i){
        // Remove the literal in index i and returns true if the removal is successful
        // If the literal to be removed is not the last one in the clause, then
        // it is swapped with the last literal before removal, changing the order
        // of literals.
        if(i < n){
          if(i < n-1)
            std::swap(_lits[i], _lits[n-1]);
          _lits[--n] = null_literal<T>;
          if (n==2)
            _flags = ClFlag::SAT2;
          else if (n==1)
            _flags = ClFlag::UNIT;
          else if (n==0)
            _flags = ClFlag::UNSAT;
          return true;
        } else {
          return false;
        }
      }
      uint16_t size() const{
        return n;
      }
      const lidx<T>* begin() const { return &_lits[0]; }
      const lidx<T>* end() const { return &_lits[0] + n; }
    };

    template <typename T>
    struct SATClauseV{
      //! Dynamically stored SAT clause
      typedef lidx<T> literal_type;
      typedef std::vector<lidx<T>> literal_container;

      literal_container _lits;
      ClFlag _flags=ClFlag::UNSAT;
      SATClauseV() : _lits{} { }
      SATClauseV(std::initializer_list<lidx<T>> lits){
        for(lidx<T> l: lits)
          add_lit(l);
      }
      literal_container& literals(){
        return _lits;
      }
      const literal_container& literals() const{
        return _lits;
      }
      literal_type& operator[](uint16_t i){
        return _lits[i];
      }
      const literal_type& operator[](uint16_t i) const{
        return _lits[i];
      }
      ClFlag flags() const{
        return _flags;
      }
      ClFlag& flags_ref(){
        return _flags;
      }
      void add_lit(literal_type lit){
        _lits.push_back(lit);
        if(size()==1)
          _flags=ClFlag::UNIT;
        else if(size()==2)
          _flags=ClFlag::SAT2;
        else
          _flags=ClFlag::NONE;
      }
      bool remove_lit(uint16_t i){
        // Remove the literal in index i and returns true if the removal is successful
        // If the literal to be removed is not the last one in the clause, then
        // it is swapped with the last literal before removal, changing the order
        // of literals.
        uint16_t n = size();
        if(i < n){
          if(i < n-1)
            std::swap(_lits[i], _lits[n-1]);
          _lits.pop_back();
          n--;
          if (n==2)
            _flags = ClFlag::SAT2;
          else if (n==1)
            _flags = ClFlag::UNIT;
          else if (n==0)
            _flags = ClFlag::UNSAT;
          return true;
        } else {
          return false;
        }
      }
      uint16_t size() const{
        return _lits.size();
      }
      typename literal_container::const_iterator begin() const { return _lits.begin(); }
      typename literal_container::const_iterator end() const { return _lits.end(); }
    };
    
    template <typename ClauseT>
    inline bool is_unit(const ClauseT& clause){
        // True if clause is a (non-empty) unit clause
        return (clause.flags() & ClFlag::UNIT) != ClFlag::NONE;
    }
    template <typename ClauseT>
    inline bool is_empty(const ClauseT& clause){
        // True if clause is a empty, i.e. unsatisfiable
        return (clause.flags() & ClFlag::UNSAT) != ClFlag::NONE;
    }

    template <typename ClauseT>
    inline bool is_binary_disj(const ClauseT& clause){
      // True if clause is a binary disjunction, i.e. contains exactly 2 literals
      return (clause.flags() & ClFlag::SAT2) != ClFlag::NONE;
    }

    template <typename ClauseT>
    void mark_sat(ClauseT& clause){
        // Clause will be flagged SAT,
        // valid until any call to simplify() or propagate_lit()
        clause.flags_ref() =  clause.flags() | ClFlag::SAT;
    }

    template <typename ClauseT>
    bool is_sat(ClauseT& clause){ return (clause.flags() & ClFlag::SAT) != ClFlag::NONE; }
    

    template <typename ClauseT>
    ClFlag simplify(ClauseT& clause){
      // Simplify the clause for duplicate literals and check whether
      // the clause is a tautology (l ^ ~l)
      uint16_t n = clause.size();
      bool simpl = false;
      for(uint16_t i = 0; i < n-1; ++i){ // iterate in reverse order for removal
        for(uint16_t j = 0; j < n-i-1; ++j){
          auto li = clause[n-i-1];
          auto lj = clause[j];
          if(li == lj){
            clause.remove_lit(n-i-1);
            simpl = true;
            break;
          }
          auto nlj = ~ lj;
          if(li == nlj) // Clause is a tautology
            return ClFlag::SAT;
        }
      }
      if(simpl){
        return ClFlag::CLSIMP;
      } else {
        return ClFlag::NONE;
      }
    }
} // namespace pysa::algstd
