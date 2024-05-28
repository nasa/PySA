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

#include <list>
#include <memory>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../bitset/bitset.hpp"
#include "../dpll/dpll.hpp"

namespace pysa::dpll::sat{

    template <typename Instance, typename T, typename = std::void_t<>>
    struct is_pointer_to : std::false_type {};

    template <typename Instance, typename T>
    struct is_pointer_to<
            Instance, T,
            std::void_t<
                    decltype(std::is_same_v<std::decay_t<decltype(*std::declval<T>())>,
                            Instance>)>> : std::true_type {};

    template <typename Instance, typename T>
    static constexpr auto is_pointer_to_v = is_pointer_to<Instance, T>::value;

    template <typename BitSet = BitSet<>,
            template <typename...> typename List = std::list,
            template <typename...> typename Vector = std::vector,
            typename Instance = Instance<BitSet, Vector>>
    struct Branch {
        const std::shared_ptr<const Instance> instance{nullptr};
        const std::size_t max_n_unsat{0};

        // Empty branch
        Branch() {}

        template <
                typename InstancePtr,
                std::enable_if_t<is_pointer_to_v<Instance, InstancePtr>, bool> = true>
        Branch(InstancePtr instance, std::size_t max_n_unsat = 0)
                : instance{instance},
                  max_n_unsat{max_n_unsat},
                  _state(instance->n_vars()),
                  _partial_sat(instance->n_clauses()),
                  _pos{0},
                  _n_unsat{0},
                  _valid{true} {}

        template <typename Clauses,
                std::enable_if_t<!is_pointer_to_v<Instance, Clauses>, bool> = true>
        Branch(const Clauses &clauses, std::size_t max_n_unsat = 0)
                : Branch(std::make_shared<Instance>(clauses), max_n_unsat) {}

        template <typename InstancePtr>
        Branch(InstancePtr instance, std::size_t max_n_unsat, BitSet state,
               BitSet partial_sat, std::size_t pos, std::size_t n_unsat, bool valid)
                : instance{instance},
                  max_n_unsat{max_n_unsat},
                  _state{state},
                  _partial_sat{partial_sat},
                  _pos{pos},
                  _n_unsat{n_unsat},
                  _valid{valid} {}

        const BitSet &state() const { return _state; }

        std::size_t pos() const { return _pos; }

        std::size_t n_unsat() const { return _n_unsat; }

        std::size_t n_sat() const { return _partial_sat.count(); }

        const BitSet &partial_sat() const { return _partial_sat; }

        bool leaf() const {
            /*
             * Check if this node is a leaf.
             */
            // Check that number of sat/unsat match
            assert(_pos < std::size(_state) || !_valid ||
                   instance->n_clauses() - n_sat() == n_unsat());

            // Check if leaf
            return _pos == std::size(_state);
        }

        bool partial() const {
            /*
             * Check if this node is a partial branch.
             */
            return !leaf() || !_valid;
        }

        void next() {
            // Get references
            auto &clauses_ = instance->clauses[_pos];
            auto &signs_ = instance->signs[_pos];

            // For each clause
            for (std::size_t i_ = 0, end_ = std::size(clauses_); i_ < end_; ++i_) {
                const auto cl_ = clauses_[i_];

                // Update the partial satisfiability by checking if this variable has the
                // right sign in clause
                _partial_sat.apply_or(signs_.test(i_) ^ _state.test(_pos), cl_);

                // If this variable is the last variable in clause, also update the
                // number of unsat
                _n_unsat += (instance->last_var[cl_] == _pos) && !_partial_sat.test(cl_);
            }

            // Update position
            if (_n_unsat <= max_n_unsat)
                ++_pos;
            else {
                _valid = false;
                _pos = std::size(_state);
            }
        }

        List<Branch> branch() const {
            // Only branch if this variable is set to zero
            if (_state.test(_pos)) return {};

            // Create a new branch
            auto branch_ = *this;

            // Set this variable to one
            branch_._state.set(_pos);

            // Return
            return {std::move(branch_)};
        }

        template <typename Buffer>
        auto dump() const {
            /*
             * Dump Branch.
             */
            auto buffer_ = pysa::branching::dump<Buffer>(max_n_unsat);
            buffer_ += pysa::branching::dump<Buffer>(_state);
            buffer_ += pysa::branching::dump<Buffer>(_partial_sat);
            buffer_ += pysa::branching::dump<Buffer>(_pos);
            buffer_ += pysa::branching::dump<Buffer>(_n_unsat);
            buffer_ += pysa::branching::dump<Buffer>(_valid);
            return buffer_;
        }

        bool operator==(const Branch &other) const {
            /*
             * Check equality.
             */
            return *instance == *other.instance && max_n_unsat == other.max_n_unsat &&
                   _state == other._state && _partial_sat == other._partial_sat &&
                   _pos == other._pos && _n_unsat == other._n_unsat &&
                   _valid == other._valid;
        }

    private:
        BitSet _state;
        BitSet _partial_sat;
        std::size_t _pos;
        std::size_t _n_unsat;
        bool _valid;
    };

    template <typename StateType, typename SizeType>
    struct Configuration {
        StateType state;
        SizeType n_unsat;

        Configuration() {}
        Configuration(const StateType &state, SizeType n_unsat)
                : state{state}, n_unsat{n_unsat} {}
        Configuration(StateType &&state, SizeType n_unsat)
                : state{std::move(state)}, n_unsat{n_unsat} {}

        auto hash() const {
            const auto seed_ = state.hash();
            return n_unsat + 0x9e3779b9 + (seed_ << 6) + (seed_ >> 2);
        }

        bool operator==(const Configuration &other) const {
            return state == other.state && n_unsat == other.n_unsat;
        }
    };
}


namespace pysa::branching {

    template <typename BitSet, template <typename...> typename List,
            template <typename...> typename Vector, typename Instance>
    struct Archiver<pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>> {
        using base_type = pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>;

        template <typename Buffer>
        static constexpr auto dumps(const base_type &branch) {
            return branch.template dump<Buffer>();
        }

        template <typename Buffer, typename InstancePtr>
        static constexpr auto loads(typename Buffer::const_iterator buffer,
                                    InstancePtr instance) {
            auto [h1_, max_n_unsat_] = load<std::size_t, Buffer>(buffer);
            auto [h2_, state_] = load<BitSet, Buffer>(h1_);
            auto [h3_, partial_sat_] = load<BitSet, Buffer>(h2_);
            auto [h4_, pos_] = load<std::size_t, Buffer>(h3_);
            auto [h5_, n_unsat_] = load<std::size_t, Buffer>(h4_);
            auto [h6_, valid_] = load<bool, Buffer>(h5_);
            return std::pair{
                    h6_, base_type(instance, max_n_unsat_, std::move(state_),
                                   std::move(partial_sat_), pos_, n_unsat_, valid_)};
        }
    };

    template <typename StateType, typename SizeType>
    struct Archiver<pysa::dpll::sat::Configuration<StateType, SizeType>> {
        using base_type = pysa::dpll::sat::Configuration<StateType, SizeType>;

        template <typename Buffer>
        static constexpr auto dumps(const base_type &conf) {
            return dump<Buffer>(conf.state) + dump<Buffer>(conf.n_unsat);
        }

        template <typename Buffer>
        static constexpr auto loads(typename Buffer::const_iterator buffer) {
            using state_type = decltype(std::declval<base_type>().state);
            using n_unsat_type = decltype(std::declval<base_type>().n_unsat);
            auto [h1_, s_] = load<state_type, Buffer>(buffer);
            auto [h2_, n_] = load<n_unsat_type, Buffer>(h1_);
            return std::pair{h2_, base_type{s_, n_}};
        }
    };

    template <typename BitSet, template <typename...> typename List,
            template <typename...> typename Vector, typename Instance>
    struct Archiver<
            std::list<pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>>> {
        using base_type = pysa::dpll::sat::Branch<BitSet, List, Vector, Instance>;

        template <typename Buffer>
        static constexpr auto dumps(const std::list<base_type> &branches) {
            // Dump size
            auto buffer_ = dump<Buffer, std::size_t>(std::size(branches));

            // Dump branches
            for (const auto &br_ : branches) buffer_ += dump<Buffer>(br_);

            // Return buffer
            return buffer_;
        }

        template <typename Buffer, typename InstancePtr>
        static constexpr auto loads(typename Buffer::const_iterator buffer,
                                    InstancePtr instance) {
            // Get size
            auto [buffer_, size_] = load<std::size_t, Buffer>(buffer);

            // Initialize branches
            std::list<base_type> branches_{};

            // Append all branches
            for (std::size_t i_ = 0; i_ < size_; ++i_) {
                // Get branch
                auto [h_, branch_] = load<base_type, Buffer>(buffer_, instance);

                // Append
                branches_.push_back(std::move(branch_));

                // Update head
                buffer_ = h_;
            }

            // Return branches
            return std::pair{buffer_, std::move(branches_)};
        }
    };

}  // namespace pysa::branching