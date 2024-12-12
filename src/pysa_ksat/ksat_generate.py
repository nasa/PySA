import argparse
from pathlib import Path

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pysa_ksat.util.scipy_survival import ecdf

from pysa_ksat import KSatAdvancedException
from pysa_ksat.random_ksat import KSatGenerator
from pysa_ksat.util.sat import write_cnf, SATFormula
from pysa_ksat.util.scipy_survival import EmpiricalDistributionFunction



def plot_chf(sf: EmpiricalDistributionFunction, ax, **kwargs):
    delta = np.ptp(sf.quantiles) * 0.05
    q = sf.quantiles
    q = [q[0] - delta] + list(q) + [q[-1] + delta]
    return ax.step(q, -np.log(sf.evaluate(q)), **kwargs)

def plot_chf_ci(sf: EmpiricalDistributionFunction, confidence_level, ax, **kwargs):
    cil, cih = sf.confidence_interval(confidence_level, method='log-log')
    delta = np.ptp(sf.quantiles) * 0.05
    q = sf.quantiles
    q = [q[0] - delta] + list(q) + [q[-1] + delta]
    return ax.fill_between(q, -np.log(cih.evaluate(q)), -np.log(cil.evaluate(q)), **kwargs)


def ksat_generate_main():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument("n", type=int)
    parser.add_argument("num_inst", type=int)
    parser.add_argument("--nclauses", type=int, default=None)
    parser.add_argument("--ctov", type=float, default=None)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gen-only", action="store_true", help="Generate instances only; do not use advanced instance filters")

    #Advanced Options
    try:
        import pysa_ksat.opt
        parser.add_argument("--filter-sat", default='cdcl', choices=['off', 'cdcl', 'walksat'])
        parser.add_argument("--max-tries-per-inst", type=int, default=200)
        parser.add_argument("--ws-reps", type=int, default=100)
        parser.add_argument("--ws-sweeps-per-rep", type=int, default=100000)
        parser.add_argument("--categories", type=int, default=3)
        parser.add_argument("--cat-shuffle", action='store_true')
        parser.add_argument("--enumerate-solutions", action='store_true')
        parser.add_argument("--eval-backbone", action='store_true')
        KSAT_ADVANCED = True
    except KSatAdvancedException:
        KSAT_ADVANCED = False

    args = parser.parse_args()
# Argument checks
    if args.nclauses is None and args.ctov is None:
        print(f"Either --nclauses or --ctov must be specified.")
        exit(1)
    elif args.nclauses is not None:
        m = args.nclauses
        alpha = m / args.n
    else:
        print(f"alpha = {args.ctov}")
        alpha = args.ctov
        m = int(np.round(alpha * args.n))
        print(f"m = {m}")

    rng = np.random.default_rng(args.seed)
    n = args.n
    k = args.k
    num_insts = args.num_inst
    prename = f"satgen_n{n}_m{m}_k{k}_{num_insts}_{args.seed}"
    rgen = KSatGenerator(k, n, m)

    if args.out_dir is None:
        out_dir = Path('.')
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_dir():
            if out_dir.exists():
                print(f"Non-directory {out_dir} exits, exiting.")
                exit(1)
            else:
                out_dir.mkdir()

    if KSAT_ADVANCED and not args.gen_only:
        from pysa_ksat.opt.bench import SatGen, bench_walksat
        if args.categories < 2:
            print("--categories must be at least 2")
            exit(1)

        sg_name = out_dir/(prename+"_sg.pkl")
        if sg_name.is_file():
            print(f"Loading from {sg_name}")
            with open(sg_name, 'rb') as f:
                gen: SatGen = pickle.load(f)
        else:
            gen = SatGen(args.n, m, k=args.k)
        # Generate instances, or load if available
            print("Generating instances ...")
            gen.add_instances(rgen, num_insts, max_tries=args.max_tries_per_inst*num_insts,
                              filter_sat=(None if args.filter_sat == 'off' else args.filter_sat), rng=rng)
        # Evaluate the backbone
            if args.eval_backbone:
                print("Evaluating backbones ...")
                gen.evaluate_backbones()
        # Enumerate all solutions with DPLL
            if args.enumerate_solutions:
                print("Enumerating solutions ...")
                gen.enumerate_solutions()
            with open(sg_name, 'wb') as f:
                pickle.dump(gen, f)
    # Bench walksat on instances
        ws_name = out_dir / (prename + "_ws.pkl")
        if ws_name.is_file():
            print(f"Loading from {ws_name}")
            with open(ws_name, 'rb') as f:
                bench_results = pickle.load(f)
        else:
            print("Running walksat")
            bench_results = bench_walksat(gen._instances, max_steps=args.ws_sweeps_per_rep * n, reps=args.ws_reps, rng=rng)
            with open(ws_name, 'wb') as f:
                pickle.dump(bench_results, f)

    # TTS(99)
        tts99 =np.asarray([np.quantile(res["runtime_iterations"], 0.99) for res in bench_results])
        sort_tts = np.argsort(tts99)
        #sol_clusters = [solution_cluster(x, threshold=2) for x in gen._solutions]
        #nclusters = [len(x) for x in sol_clusters]

        # Survival function analysis
        ecdfs = [ecdf(res["runtime_iterations"]) for res in bench_results]
        num_ecdfs = min(len(ecdfs), 50)
        lines_per_plot = 5
        ncols = 2
        nrows = max(num_ecdfs // (lines_per_plot * ncols), 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 16), layout='tight')
        for axi, ax in enumerate(axes.flatten()):
            for idxi in range(lines_per_plot):
                ii = len(ecdfs) - num_ecdfs + lines_per_plot * axi +  idxi
                if ii > len(ecdfs):
                    break
                i = sort_tts[ii]
                plot_chf(ecdfs[i].sf, ax, ls='--', color=f'C{i}')
                plot_chf_ci(ecdfs[i].sf, 0.90, ax, alpha=0.2, color=f'C{i}')

            ax.set_xlim(1, 1.5*np.max(tts99))
            ax.set_xscale('log')
            ax.set_xlabel('WS Iterations')
            ax.set_ylim(0.01, 10)
            ax.set_yscale('log')
            ax.set_ylabel('CHF')
        plt.savefig(out_dir / "sample_chf.pdf")

    # Categorization
        cat_directories = [out_dir/"instances"/f"C{i:02}" for i in range(args.categories)]
        for dir in cat_directories:
            dir.mkdir(parents=True, exist_ok=True)
        ncats = args.categories

        cat_quantiles = np.linspace(0, 1, ncats+1)[1:]
        tts_quantiles = np.quantile(tts99, cat_quantiles)
        print(f"TTS Quantiles: ")
        for q, t in zip(cat_quantiles, tts_quantiles):
            print(f"  {q*100:7.3f}% {t:4.3e}")
        cat_idxs = []
        tts_bounds = np.concatenate([[0.0], tts_quantiles])
        for i, (t0, tf) in enumerate(zip(tts_bounds[:-1], tts_bounds[1:])):
            idxi = np.nonzero(np.logical_and(np.greater(tts99, t0), np.less_equal(tts99, tf)))[0]
            if args.cat_shuffle:
                rng.shuffle(idxi)
            for ji, j in enumerate(idxi):
                inst = gen._instances[j]
                cnf_file = cat_directories[i]/f"{ji:03}.cnf"
                with open(cnf_file, 'w') as f:
                    write_cnf(SATFormula(inst), f, comments=[gen.instance_sha(j)])
        log10tts99 = np.log10(tts99)
        if args.enumerate_solutions:
            nsols = np.log10([len(x) for x in gen._solutions])
            fig, ax = plt.subplots()
            ax.scatter(nsols, log10tts99)
            ax.set_xlabel("Number of Solutions")
            ax.set_ylabel("$\\log_{10}$ TTS99")
            fig.savefig(out_dir/"tts_nsols.pdf")
        if args.eval_backbone:
            fig, ax = plt.subplots()
            ax.scatter(gen._backbones, log10tts99)
            ax.set_xlabel("Backbone Size")
            ax.set_ylabel("$\\log_{10}$ TTS99")
            fig.savefig(out_dir / "tts_bb.pdf")
    else:
        new_instances = list(rgen.generate_n(num_insts))
        for  i, inst in enumerate(new_instances):
            cnf_file = out_dir/ f"{i:03}.cnf"
            with open(cnf_file, 'w') as f:
                write_cnf(SATFormula(inst), f)

if __name__ == '__main__':
    ksat_generate_main()