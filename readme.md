# K-SAT Generator Module for PySA

Generates k-sat instances and characterizes them by hardness with Walksat

## Basic Generator

Generates random k-SAT instances as CNF files. 
This generator does not check for satisfiability or hardness.

Usage:
```
python -m pysa_ksat.ksat_generate  [-h] [--nclauses NCLAUSES] [--ctov CTOV] [--k K] 
    [--max-tries-per-inst MAX_TRIES_PER_INST] [--seed SEED] [--out-dir OUT_DIR] 
    n num_inst
```


Example: Generate 50 instances of a 40 variable k=4 SAT problem,
with 10 clauses per variable. 
```
python -m pysa_ksat.ksat_generate 40 50 --k 4 --seed 1234  --ctov 10.0 --out-dir test_k4_n40_a4.5 
```

## Advanced Generator

Requires the use of PySA SAT solvers. The advanced executable is installed via `pip install '.[advanced]'`

Usage:
```
python -m pysa_ksat.ksat_generate  [-h] [--nclauses NCLAUSES] [--ctov CTOV] [--k K] 
    [--max-tries-per-inst MAX_TRIES_PER_INST] [--seed SEED] [--out-dir OUT_DIR] 
    [--gen-only]
    [--ws-reps WS_REPS] [--categories CATEGORIES]
    [--cat-shuffle] [--enumerate-solutions] [--eval-backbone]
    n num_inst
```

Example: Generate 50 instances of a 40 variable k=4 SAT problem,
with 10 clauses per variable. Sort and categorize by hardness
with walksat in 3 equal parts.
```
python -m pysa_ksat.ksat_generate 40 50 --k 4 --seed 1234 --ws-reps 1000 --ctov 10.0 --categories 3 --out-dir test_k4_n40_a4.5 --cat-shuffle
```

For small instances, additional instance hardness characteristics can be gathered with the 
`--enumerate-solutions` and  `--eval-backbone` options. The behavior of the basic generator can be enabled
by passing the `--gen-only` flag.
