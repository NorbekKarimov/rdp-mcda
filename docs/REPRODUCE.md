# Reproducing the results

All commands are run from the repository root with the virtual environment active. Paths use `/`; on Windows `\` works as well. Close other applications and keep the machine on mains power with sleep mode disabled (a run interrupted by sleep is not a measurement and is excluded, see Section 4.4 of the paper).

Directions for all commands: `--dirs + - - + - + - -`.

## 1. Ablation and scaling (Tables 5 and 6, Figures 4–6)

```
python src/e1_passes.py --data data/orol_v2_30m/data.npy --dirs + - - + - + - - --variants B1 A1 A2 P P5 --threads 1 2 4 6 8 12 --repeats 10 --pause 5 --out results/final/e1_30m
python src/e1_passes.py --data data/orol_v2_100m/data.npy --dirs + - - + - + - - --variants B0 B1 A1 A2 P --threads 1 12 --repeats 10 --pause 5 --out results/final/e1_100m
```

## 2. Memory (Table 7, Figure 7)

```
python src/e1_passes.py --data data/orol_v2_1000m/data.npy --dirs + - - + - + - - --variants B0 P --threads 1 12 --repeats 3 --block 4096 --out results/e3/mem3_1000m
python src/e1_passes.py --data data/orol_v2_100m/data.npy --dirs + - - + - + - - --variants B0 P --threads 1 12 --repeats 3 --out results/e3/mem3_100m
python src/e1_passes.py --data data/orol_v2_30m/data.npy --dirs + - - + - + - - --variants P --threads 1 12 --repeats 3 --out results/e3/mem3_30m
```

## 3. Stack larger than main memory (Section 5.4, Table 8, Figure 8)

About 15 GB of free disk space is needed.

```
python src/make_replicated.py --src data/orol_v2_30m/data.npy --k 4 --dst data/orol_v2_30m_x4/data.npy
python src/e1_passes.py --data data/orol_v2_30m_x4/data.npy --dirs + - - + - + - - --variants P --threads 1 12 --repeats 3 --pause 5 --no_check --out results/e3/x4_30m
python src/verify_replicated.py --src data/orol_v2_30m/data.npy --k 4 --dirs + - - + - + - - --summary results/e3/x4_30m/summary.json
python src/e1_passes.py --data data/orol_v2_30m_x4/data.npy --dirs + - - + - + - - --variants A2 P P5 --threads 12 --threads_all --repeats 5 --pause 10 --no_check --out results/e3/x4_write
```

## 4. Light pass versus P5 with cached data (Section 5.3)

```
python src/e1_passes.py --data data/orol_v2_30m/data.npy --dirs + - - + - + - - --variants P P5 --threads 1 12 --repeats 5 --pause 5 --out results/e3/cache_p5
```

## 5. Determinism (Table 9, Figure 9)

```
python src/e2_determinism.py --data data/orol_v2_30m/data.npy --dirs + - - + - + - - --threads 4 6 12 --threads_det 1 4 8 12 --runs 100 --out results/e2/full_30m
```

## 6. Figures

```
python figures/make_figures.py
```

The figure script uses the medians reported in the paper; each value is taken from the `summary.json` file named in the figure caption.

## Where each table and figure comes from

| Paper item | Result folder |
|---|---|
| Table 5, Figure 4 | `results/final/e1_30m`, `results/final/e1_100m` |
| Table 6, Figures 5–6 | `results/final/e1_30m` |
| Table 7, Figure 7 | `results/e3/mem3_*`, `results/e3/x4_30m`, `results/final/e1_100m` (B0) |
| Section 5.4 (out-of-core times) | `results/e3/x4_30m` |
| Table 8, Figure 8 | `results/e3/x4_write` |
| Section 5.3 (P versus P5) | `results/final/e1_30m`, `results/e3/cache_p5` |
| Table 9, Figure 9 | `results/e2/full_30m` |
