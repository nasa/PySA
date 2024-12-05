import argparse
from pathlib import Path

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gamma, skew, lognorm, kurtosistest, normaltest, expon, weibull_min, ecdf

from pysa_ksat.satgen import SatGen
from pysa_ksat.random_ksat import KSatGenerator
from pysa_ksat.util.sat import write_cnf, SATFormula
from pysa_walksat.bindings import walksat_optimize
from scipy.stats._survival import EmpiricalDistributionFunction



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


def bench_walksat(instances, max_steps=100000, p=0.5, reps=100, rng:np.random.Generator = None):
    bench_results = []
    rng = np.random.default_rng(rng)
    for i, inst in enumerate(instances):
        result_dict = {
            "solver": "pysa:walksat",
            "solver_parameters": {"max_steps": max_steps, "p": p},
            "hardware": "CPU:Apple M2:1",
            "set": "ANONYMOUS_BATCH",
            "instance_idx": i,
            "cutoff_type": "iterations",
            "cutoff": max_steps,
            "runs_attempted": reps,
            "runs_solved": 0,
            "runtime_seconds": [],
            "runtime_iterations": [],
            "hardware_time_seconds": [],
            "n_unsat_clauses": [],
            "configurations": []
        }

        for i in range(reps):
            wsres = walksat_optimize(inst, max_steps, p, 0, int(rng.integers(0,2**31-1)))
            if wsres.num_unsat == 0:
                result_dict["runs_solved"] += 1
            result_dict["runtime_seconds"].append((wsres.preproc_time_us + wsres.computation_time_us)*1e-6)
            result_dict["runtime_iterations"].append(wsres.iterations)
            result_dict["hardware_time_seconds"].append(wsres.computation_time_us*1e-6)
            result_dict["n_unsat_clauses"].append(wsres.num_unsat)
            result_dict["configurations"].append(''.join([str(b) for b in np.asarray(wsres.result_state) ]))

        # inline analysis
        runtime_iters = np.asarray(result_dict["runtime_iterations"])
        params = gamma.fit(runtime_iters,  floc=np.min(runtime_iters)-1.0, method='mle')
        params_lognorm = lognorm.fit(runtime_iters, floc=0, method='mle')
        params_exp = expon.fit(runtime_iters, floc=np.min(runtime_iters)-0.5, method='mle')
        params_weib = weibull_min.fit(runtime_iters,floc=np.min(runtime_iters)-0.5,  method='mle')
        result_dict["statistics"] = {
            'mean': np.mean(runtime_iters),
            'std': np.std(runtime_iters),
            'skew': skew(runtime_iters),
            'mean_log': np.mean(np.log(runtime_iters)),
            'std_log': np.std(np.log(runtime_iters)),
            'skew_log': skew(np.log(runtime_iters)),
            'kurtosis_test': kurtosistest(np.log(runtime_iters)),
            'normal_test': normaltest(np.log(runtime_iters)),
        }
        result_dict["mle"] = {
            "gamma": list(params),
            "lognormal": list(params_lognorm),
            "expon": list(params_exp),
            "weibull": list(params_weib)
        }
        result_dict["ecdf"] = ecdf(runtime_iters)
        bench_results.append(result_dict)
    return bench_results


def ksat_generate_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)

    parser.add_argument("num_inst", type=int)
    parser.add_argument("--nclauses", type=int, default=None)
    parser.add_argument("--ctov", type=float, default=None)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--max-tries-per-inst", type=int, default=200)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ws-reps", type=int, default=100)
    parser.add_argument("--categories", type=int, default=3)
    parser.add_argument("--cat-shuffle", action='store_true')
    parser.add_argument("--enumerate-solutions", action='store_true')
    parser.add_argument("--eval-backbone", action='store_true')
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
    if args.categories < 2:
        print("--categories must be at least 2")
        exit(1)

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

    rng = np.random.default_rng(args.seed)
    n = args.n
    k = args.k
    num_insts = args.num_inst
    prename = f"satgen_n{n}_m{m}_k{k}_{num_insts}_{args.seed}"
    rgen = KSatGenerator(k, n, m)

    sg_name = out_dir/(prename+"_sg.pkl")
    if sg_name.is_file():
        print(f"Loading from {sg_name}")
        with open(sg_name, 'rb') as f:
            gen: SatGen = pickle.load(f)
    else:
        gen = SatGen(args.n, m, k=args.k)
    # Generate instances, or load if available
        print("Generating instances ...")
        gen.add_instances(rgen, num_insts, max_tries=args.max_tries_per_inst*num_insts, filter_sat=True, rng=rng)
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
        bench_results = bench_walksat(gen._instances, reps=args.ws_reps, rng=rng)
        with open(ws_name, 'wb') as f:
            pickle.dump(bench_results, f)

# TTS(99)
    tts99 =np.asarray([np.quantile(res["runtime_iterations"], 0.99) for res in bench_results])
    sort_tts = np.argsort(tts99)
    #sol_clusters = [solution_cluster(x, threshold=2) for x in gen._solutions]
    #nclusters = [len(x) for x in sol_clusters]

    # Survival function analysis
    ecdfs = [ecdf(res["runtime_iterations"]) for res in bench_results]
    lines_per_plot = 5
    ncols = 2
    nrows = max(len(ecdfs) // (lines_per_plot * ncols), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 16), layout='tight')
    for axi, ax in enumerate(axes.flatten()):
        for idxi in range(lines_per_plot):
            ii = lines_per_plot * axi +  idxi
            if ii > len(ecdfs):
                break
            i = sort_tts[ii]
            plot_chf(ecdfs[i].sf, ax, ls='--', color=f'C{i}')
            plot_chf_ci(ecdfs[i].sf, 0.90, ax, alpha=0.2, color=f'C{i}')

        ax.set_xlim(1, 1.5*np.max(tts99))
        #ax.set_xscale('log')
        ax.set_ylim(0.01, 10)
        #ax.set_yscale('log')
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
### Process

    # skew_arr = [res["statistics"]["skew"] for res in bench_results]
    # skewlog_arr = [res["statistics"]["skew_log"] for res in bench_results]
    # normalp =  [res["statistics"]["normal_test"].pvalue for res in bench_results]
    # lognormsigma =  [res["mle"]["lognormal"][0] for res in bench_results]
    # lognormmu = [res["mle"]["lognormal"][2] for res in bench_results]
    # gamma_k = [res["mle"]["gamma"][0] for res in bench_results]
    # sort_mu = np.argsort(lognormmu)
    # sort_k = np.argsort(gamma_k)
    # sort_w = np.argsort( [res["mle"]["weibull"][1] for res in bench_results])

    #
    # _x=np.power(10.0, np.linspace(0,5,500))
    # ###
    # fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 16))
    # for idx in [0, 2, 20, -3, -1]:
    #     i = sort_mu[idx]
    #     _iters = bench_results[i]['runtime_iterations']
    #     _params = bench_results[i]['mle']
    #     #ax.ecdf(bench_results[i]['runtime_iterations'], ls='--', color=f'C{i}')
    #     ecdfs[i].cdf.plot(ax, ls='--', color=f'C{i}')
    #     plot_ecdf_ci(ecdfs[i].cdf, ax, alpha=0.2, color=f'C{i}')
    #     ax.plot(_x, lognorm.cdf(_x, *_params['lognormal']), ls='solid', color=f'C{i}',
    #              label=f'$\\sigma={_params['lognormal'][0]:2.1f}$, $\\bar{{\\tau}}={_params['lognormal'][2]:3.1f}$')
    # for idx in [0, 2, 20, -3, -1]:
    #     i = sort_k[idx]
    #     _params = bench_results[i]['mle']
    #     ax2.ecdf(bench_results[i]['runtime_iterations'], ls='--', color=f'C{i}')
    #     ax2.plot(_x, gamma.cdf(_x, *_params['gamma']), ls='solid', color=f'C{i}',
    #             label=f'k={_params['gamma'][0]:2.1f}, $\\bar{{\\tau}}={_params['gamma'][2]:3.1f}$')
    # for idx in [0, 2, 20, -3, -1]:
    #     i = sort_w[idx]
    #     _params = bench_results[i]['mle']
    #     ax3.ecdf(bench_results[i]['runtime_iterations'], ls='--', color=f'C{i}')
    #     ax3.plot(_x, weibull_min.cdf(_x, *_params['weibull']), ls='solid', color=f'C{i}',
    #              label=f'c={_params['weibull'][0]:2.1f}, $\\bar{{\\tau}}={_params['weibull'][2]:3.1f}$')
    #     #ax.plot(_x, gamma_distribution.pdf(_x, *gms[i]), ls='solid', color=f'C{i}')
    # #plot_quantile([benchmark_raw_data['01']['0.5'][i]['runtime_iterations'] for i in range(2)], fig=fig, ax=ax, xlog=True)
    # ax.set_xscale('log')
    # ax.set_yscale('logit')
    # ax.set_ylim(0.5/args.ws_reps, 1 - 0.5/args.ws_reps)
    # ax.legend()
    # ax2.set_xscale('log')
    # ax2.set_yscale('logit')
    # ax2.set_ylim(0.5 / args.ws_reps, 1 - 0.5 / args.ws_reps)
    # ax2.legend()
    # ax3.set_xscale('log')
    # ax3.set_yscale('logit')
    # ax3.set_ylim(0.5 / args.ws_reps, 1 - 0.5 / args.ws_reps)
    # ax3.legend()
    # plt.savefig('lognorm.pdf')

if __name__ == '__main__':
    ksat_generate_main()