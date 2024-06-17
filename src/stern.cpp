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

#include "sternx/stern.h"
#include "boost/timer/timer.hpp"
#include "libmld/isd.h"
#include "libmld/simdbitvec.h"
#include <iostream>
#include <random>
#ifdef USEMPI
#include "mpi.h"
#endif

// typedef uint64_t stern_uint_t;
#ifdef USEMPI
#define STERN_MPI_INT MPI_UINT32_T
#endif

template <typename stern_uint_t, typename sub_block_t = void,
          bool testhw1 = false>
std::optional<std::vector<uint8_t>> sterncpp(MLDProblem &mld_problem, sternc_opts &opts) {
  int master_rank = 0;
  int rank;
  int64_t complt[3] = {0, -1, -1};
#ifdef USEMPI
  int nproc = 1;
  int mpiflag;
  int64_t compl_buf[3];
  MPI_Request mpireq, mpireq2;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  std::vector<int> rank_nsuccesses(nproc);
  if (rank == master_rank)
    std::cout << "Running sterncpp:MPI ..." << std::endl;
#else
  rank = 0;
#endif
  if (rank == master_rank) {
    if constexpr (std::is_void_v<sub_block_t>)
      std::cout << "Sterncpp / Block bits " << sizeof(stern_uint_t) * 8
                << " / Heavy Stern\n";
    else {
      std::cout << "Sterncpp / Block bits " << sizeof(stern_uint_t) * 8
                << " / Collision Bits " << sizeof(sub_block_t) * 8
                << " / Collision Weight " << (testhw1 ? 1 : 0) << "\n";
    }
    std::cout << "Alignment: " << BitVecNums<stern_uint_t>::alignment
              << " bytes.\n";
  }
  uint64_t n = mld_problem.NVars();
  uint64_t k = mld_problem.CodeDim();
  uint64_t w = mld_problem.Weight();

  // seed the rng
  std::random_device rd;
  stern_rng_t rng(rd());

  BitMatrix<stern_uint_t> hy =
      mld_problem.clauses_as_bitmatrix<stern_uint_t>(true);
  BitVec<stern_uint_t> stern_solution(n);
  SternP1<stern_uint_t> stern_p1(hy, n, k, rng);
#ifdef USEMPI
  if (rank == master_rank) {
    MPI_Irecv(compl_buf, 3, MPI_INT64_T, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD,
              &mpireq);
  } else {
    MPI_Ibcast(compl_buf, 3, MPI_INT64_T, master_rank, MPI_COMM_WORLD, &mpireq);
  }
  uint32_t nsuccess_buf = 0;
#endif
  uint32_t nsuccess = 0;
  uint32_t niters = 0, niters_buf = 0;
  uint64_t dt_sum;
  bool mfound;
  boost::timer::cpu_timer ti;
  for (int32_t i = 0; i < opts.max_iters; ++i) {
    stern_p1.sample_ir_split();
    stern_p1.shuffle_isd();
    if constexpr (std::is_void_v<sub_block_t>) {
      mfound = stern_p1.collision_iteration_heavy(w);
    } else {
      mfound = stern_p1.template collision_iteration<sub_block_t, testhw1>(w);
    }
#ifdef USEMPI
    if (!opts.bench) {
      if (rank == master_rank) {
        MPI_Test(&mpireq, &mpiflag, MPI_STATUS_IGNORE);
        if (mpiflag) {
          if (compl_buf[2] >
              0) { // broadcast the successful completion and terminate
            std::copy(compl_buf, compl_buf + 3, complt);
            MPI_Ibcast(complt, 3, MPI_INT64_T, master_rank, MPI_COMM_WORLD,
                       &mpireq2);
            MPI_Wait(&mpireq2, MPI_STATUS_IGNORE);
            break;
          } else {
            MPI_Irecv(compl_buf, 3, MPI_INT64_T, MPI_ANY_SOURCE, 1,
                      MPI_COMM_WORLD, &mpireq);
          }
        }
      } else { // check if master rank has broadcasted completion
        MPI_Test(&mpireq, &mpiflag, MPI_STATUS_IGNORE);
        if (mpiflag) {
          std::copy(compl_buf, compl_buf + 3, complt);
          break;
        }
      }
    }
#endif
    if (mfound) {
      if (nsuccess == 0)
        stern_solution = stern_p1.get_solution_vec();
      complt[0] = rank;
      complt[1] = i;
      complt[2] = 1;
      nsuccess++;
      niters = i + 1;
#ifdef USEMPI
      if (rank == master_rank && !opts.bench) {
        MPI_Ibcast(complt, 3, MPI_INT64_T, master_rank, MPI_COMM_WORLD,
                   &mpireq2);
        MPI_Wait(&mpireq2, MPI_STATUS_IGNORE);
      } else {
        MPI_Send(complt, 3, MPI_INT64_T, master_rank, 1, MPI_COMM_WORLD);
      }
#endif
      if (!opts.bench)
        break;
    }
    niters = i + 1;
  }
  ti.stop();
  uint64_t dt = ti.elapsed().user + ti.elapsed().system;
#ifdef USEMPI
  if (opts.bench) {
    // will wait until all ranks finish
    MPI_Allreduce(&nsuccess, &nsuccess_buf, 1, MPI_UINT32_T, MPI_SUM,
                  MPI_COMM_WORLD);
    niters = opts.max_iters * nproc;
    if (nsuccess_buf > 0) { // transmit the solution to the master rank from the
                            // first rank to find it
      MPI_Allgather(&nsuccess, 1, MPI_UINT32_T, rank_nsuccesses.data(), 1,
                    MPI_UINT32_T, MPI_COMM_WORLD);
      for (int i = 0; i < nproc; ++i) {
        if (rank_nsuccesses[i] > 0) {
          if (i != master_rank) { // ping pong
            if (rank == master_rank)
              MPI_Recv(stern_solution.data_ptr(),
                       (int)stern_solution.num_blocks(), STERN_MPI_INT, i, 2,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank == i)
              MPI_Send(stern_solution.data_ptr(),
                       (int)stern_solution.num_blocks(), STERN_MPI_INT,
                       master_rank, 2, MPI_COMM_WORLD);
            if (rank == master_rank)
              MPI_Recv(complt, 3, MPI_INT64_T, i, 3, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
            if (rank == i)
              MPI_Send(complt, 3, MPI_INT64_T, master_rank, 3, MPI_COMM_WORLD);
          }
        }
      }
    }
    nsuccess = nsuccess_buf;
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    // synchronize one more time to break if two ranks found a solution
    // simultaneously before the master rank could broadcast
    MPI_Bcast(complt, 3, MPI_INT64_T, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(stern_solution.data_ptr(), (int)stern_solution.num_blocks(),
              STERN_MPI_INT, complt[0], MPI_COMM_WORLD);
    MPI_Reduce(&niters, &niters_buf, 1, MPI_UINT32_T, MPI_SUM, master_rank,
               MPI_COMM_WORLD);
    if (rank == master_rank && complt[2] > 0) {
      std::cout << "Solution found in rank " << complt[0] << " Iter "
                << complt[1] << std::endl;
    }
  }
  MPI_Reduce(&dt, &dt_sum, 1, MPI_UINT64_T, MPI_SUM, master_rank,
             MPI_COMM_WORLD);
#else
  dt_sum = dt;
  niters_buf = niters;
#endif
  if (rank == master_rank) {
    std::cout << ti.format(
                     6,
                     "wall %w s | user %u s | system %s s | CPU %t s | (%p%)\n")
              << std::endl;
    if (opts.bench) {
      double tperit = (double)dt_sum / 1000.0 / niters;
      double success_prob = (double)nsuccess / niters;
      double tts = (tperit / 1e6) * std::log(0.01) / std::log1p(-success_prob);
      std::cout << "t/iter (us): " << tperit << '\n';
      std::cout << "Success Count: " << nsuccess << " / " << niters << '\n';
      std::cout << "Success Prob: " << success_prob << '\n';
      std::cout << "TTS(99%) (s): " << tts << '\n';
      std::cout << "lg TTS(99%): " << std::log2(tts) << std::endl;
    } else {
      std::cout << "Iterations: " << niters_buf << std::endl;
    }

    if (complt[2] > 0) {
      // don't trust if processor is not little-endian
      std::cout << "Error vector = " << stern_solution.as_u8_slice();
    }
  }
  if (complt[2] > 0) {
    return stern_solution.as_slice().as_vec();
  } else {
    return {};
  }
}

template <bool testhw1>
std::optional<std::vector<uint8_t>> sterncpp_switch(MLDProblem &mld_problem, sternc_opts &opts,
                     size_t block_size) {
  // sorry
  switch (block_size) {
  case 8: // uint8_t
    switch (opts.l) {
    case 8:
      return sterncpp<uint8_t, uint8_t, testhw1>(mld_problem, opts);
    default:
      return sterncpp<uint8_t, void>(mld_problem, opts);
    }
  case 16: // uint16_t
    switch (opts.l) {
    case 16:
      return sterncpp<uint16_t, uint16_t, testhw1>(mld_problem, opts);
    case 8:
      return sterncpp<uint16_t, uint8_t, testhw1>(mld_problem, opts);
    default:
      return sterncpp<uint16_t, void>(mld_problem, opts);
    }
  case 32: // uint32_t
    switch (opts.l) {
    case 32:
      return sterncpp<uint32_t, uint32_t, testhw1>(mld_problem, opts);
    case 16:
      return sterncpp<uint32_t, uint16_t, testhw1>(mld_problem, opts);
    case 8:
      return sterncpp<uint32_t, uint8_t, testhw1>(mld_problem, opts);
    default:
      return sterncpp<uint32_t, void>(mld_problem, opts);
    }
    break;
#if defined(USE_SIMDE)
  case 128: // simd 128
    switch (opts.l) {
    case 64:
      return sterncpp<simde__m128i, uint64_t, testhw1>(mld_problem, opts);
    case 32:
      return sterncpp<simde__m128i, uint32_t, testhw1>(mld_problem, opts);
    case 16:
      return sterncpp<simde__m128i, uint16_t, testhw1>(mld_problem, opts);
    case 8:
      return sterncpp<simde__m128i, uint8_t, testhw1>(mld_problem, opts);
    default:
      return sterncpp<simde__m128i, void>(mld_problem, opts);
    }
#endif
  case 64: // uint64_t
  default:
    switch (opts.l) {
    case 64:
      return sterncpp<uint64_t, uint64_t, testhw1>(mld_problem, opts);
    case 32:
      return sterncpp<uint64_t, uint32_t, testhw1>(mld_problem, opts);
    case 16:
      return sterncpp<uint64_t, uint16_t, testhw1>(mld_problem, opts);
    case 8:
      return sterncpp<uint64_t, uint8_t, testhw1>(mld_problem, opts);
    default:
      return sterncpp<uint64_t, void>(mld_problem, opts);
    }
  }
}

std::optional<std::vector<uint8_t>> sterncpp_main(MLDProblem &mld_problem, sternc_opts &opts) {
  if (opts.test_hw1) {
    return sterncpp_switch<true>(mld_problem, opts, opts.block_size);
  } else {
    return sterncpp_switch<false>(mld_problem, opts, opts.block_size);
  }
}

std::optional<sternc_opts> sterncpp_adjust_opts(const sternc_opts &opts) {
  sternc_opts new_opts(opts);
  int mpi_rank = 0;
#ifdef USEMPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  if (opts.l <= 0) {
    new_opts.l = 0;
  } else if (opts.l <= 8) {
    new_opts.l = 8;
  } else if (opts.l <= 16) {
    new_opts.l = 16;
  } else if (opts.l <= 32) {
    new_opts.l = 32;
  } else {
    if (mpi_rank == 0)
      std::cout << "Option -l " << opts.l << " not supported.\n";
    return {};
  }
  if (opts.m <= 0) {
    new_opts.m = 0;
  } else if (opts.m <= 1) {
    new_opts.m = 1;
  } else if (opts.m <= 2) {
    new_opts.m = 2;
  } else if (opts.m <= 4) {
    new_opts.m = 4;
  } else if (opts.m <= 8) {
    new_opts.m = 8;
  } else if (opts.m <= 16) {
    new_opts.m = 16;
  } else if (opts.m <= 32) {
    new_opts.m = 32;
  } else {
    if (mpi_rank == 0)
      std::cout << "Option -m " << opts.m << " not supported.\n";
    return {};
  }
  size_t block_size;
  if (opts.m > 0 &&
      opts.l > 0) { // determine block size from passed -l and -m options
    block_size = (opts.l) * opts.m;
#if defined(USE_SIMDE)
    if (block_size > 128) {
      if (mpi_rank == 0)
        std::cout << "-l and -m combination not supported.\n";
#else
    if (block_size > 64) {
      if (mpi_rank == 0)
        std::cout << "-l and -m combination not supported.\n";
#endif
      return {};
    }
    if (mpi_rank == 0)
      std::cout << "Set block size " << block_size << "\n";
    if (opts.block_size > 0) {
      if (mpi_rank == 0)
        std::cout << "Note: -l and -m will override --block-size\n";
    }
  } else if (opts.block_size > 0) { //
    switch (opts.block_size) {
    case 8:
    case 16:
    case 32:
    case 64:
#if defined(USE_SIMDE)
    case 128:
#endif
      //            case 256:
      block_size = opts.block_size;
      break;
    default:
      if (mpi_rank == 0)
        std::cout << "Option --block-size " << opts.block_size
                  << " not valid.\n";
      return {};
    }
  } else { // use number of clauses (rows) to determine block size
    if (opts.nclauses <= 8) {
      block_size = 8;
    } else if (opts.nclauses <= 16) {
      block_size = 16;
    } else if (opts.nclauses <= 32) {
      block_size = 32;
    }
#if defined(USE_SIMDE)
    else if (opts.nclauses <= 64) {
      block_size = 64;
    } else {
      block_size = 128;
    }
#else
    else {
      block_size = 64;
    }
#endif
    new_opts.block_size = block_size;
  }
  return new_opts;
}
