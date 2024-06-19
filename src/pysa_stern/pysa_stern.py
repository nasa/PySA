import argparse
import sys
from pysa_stern.mld import MLDProblem
from pysa_stern.bindings import SternOpts, sterncpp_adjust_opts
from pysa_stern.sterncpp import pysa_sternx
import numpy as np
try:
    # Make sure bindings were compiled with MPI
    from pysa_stern.bindings import __pysa_mpi__
    # Check that mpi4py is installed
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except:
    comm = None
    rank = 0


def PySA_Stern_main():
    adjusted_opts = None
    
    parser = argparse.ArgumentParser("PySA-Stern")
    parser.add_argument("input", help="MLD format input file")
    # Stern algorithm parameters
    parser.add_argument(
        "--col-size", "-l",
        type=int,
        default=0,
        help="Size of collision set to check. "
        'By default, check the entire column sum ("Heavy" Stern).',
    )
    parser.add_argument(
        "--col-sets", "-m",
        type=int,
        default=1,
        help="Number of collision sets to check.",
    )
    # Additional parameters
    parser.add_argument(
        "--block-size", 
        type=int, 
        default=0,
        help="Block size of parity check bit columns."
    )
    # Maximum resources and benchmarking
    parser.add_argument(
        "--max-iters",
        type=int, default=1000,
        help="Maximum number of iterations."
    )
    parser.add_argument(
        "--bench", 
        action='store_true', 
        help="Continue until max_iters iterations and count how often the solution "
    )
    # Experimental ...
    parser.add_argument(
        "--test-hw1",
        action='store_true',
    )

    if(comm is not None and rank == 0):
        print("--- PySA-Stern (MPI) ---")
    elif comm is None:
        print("--- PySA-Stern---")
    args = parser.parse_args()
    # Gather and check options
    stern_opts = SternOpts() 
    stern_opts.l = args.col_size if args.col_size is not None else -1
    stern_opts.m = args.col_sets 
    stern_opts.block_size = args.block_size
    stern_opts.bench = 1 if args.bench else 0
    stern_opts.test_hw1 = 1 if args.test_hw1 else 0
    stern_opts.max_iters = args.max_iters
    try:
        adjusted_opts = sterncpp_adjust_opts(stern_opts)
    except RuntimeError as e:
        print(f"Problem with passed options.", file=sys.stderr)
        print(e, file=sys.stderr)
        adjusted_opts = None
        if comm is not None:
            MPI.Finalize()
        exit(1)
    # Read in the MLD problem
    problem = MLDProblem()
    problem_ok = np.asarray([1])
    if rank == 0:
        try:
            with open(args.input, 'r') as f:
                problem.read_problem(f)
        except Exception as e:
            problem_ok[0] = 0
            print(f"Problem reading MLD file ({args.input}).", file=sys.stderr)
            print(e, file=sys.stderr)
    if comm is not None:
        comm.Bcast(problem_ok, root=0)
    if not problem_ok[0]:
        if comm is not None:
            MPI.Finalize()
        exit(1)
    if comm is not None:
        comm.bcast(problem, root=0)
    
    # Call the Stern algorithm solver 
    sol_arr = pysa_sternx(problem, adjusted_opts)

    if rank==0:
        if sol_arr is not None:
            print(f"[PySA-Stern] Solution found: ")
            for i in range(len(sol_arr)):
                print(f"{sol_arr[i]} ", end="")
                if (i+1)%32 == 0:
                    print("\n", end="")
                elif (i+1)%8 == 0:
                    print(" ", end="")
            print()
        else:
            print(f"[PySA-Stern] Solution not found.")
    if comm is not None:
        MPI.Finalize()

if __name__ == "__main__":
    PySA_Stern_main()