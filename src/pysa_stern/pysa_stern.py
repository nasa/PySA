import argparse

from pysa_stern.mld import MLDProblem
from pysa_stern.bindings import SternOpts, sterncpp_adjust_opts
from pysa_stern.sterncpp import pysa_sternx

def PySA_Stern_main():
    parser = argparse.ArgumentParser("PySA-Stern")
    parser.add_argument("input", help="MLD format input file")
    # Stern algorithm parameters
    parser.add_argument(
        "--col-size", "-l",
        type=int,
        default=None,
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
        default=None,
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

    args = parser.parse_args()
    stern_opts = SternOpts()
    problem = MLDProblem()
    try:
        with open(args.input) as f:
            problem.read_problem(f)
    except RuntimeError as e:
        print(e)
        exit(1)
    stern_opts.l = args.l if args.col_size is not None else -1
    stern_opts.m = args.col_sets 
    stern_opts.bench = 1 if args.bench else 0
    stern_opts.test_hw1 = 1 if args.test_hw1 else 0
    stern_opts.max_iters = args.max_iters
    adjusted_opts = sterncpp_adjust_opts(stern_opts)
    sol_arr = pysa_sternx(problem, adjusted_opts)
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

if __name__ == "__main__":
    PySA_Stern_main()