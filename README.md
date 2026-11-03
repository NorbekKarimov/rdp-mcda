# RDP-MCDA

Code and measurement data for the manuscript

> **RDP-MCDA: Pass-Minimal and Deterministic Out-of-Core Computation of Entropy-Weighted Spatial MCDA for Large Rasters**
> [Authors]. Submitted to *ISPRS International Journal of Geo-Information*.

RDP-MCDA evaluates the objective-weighted MCDA chain (percentile normalization, entropy weights, SAW, TOPSIS and VIKOR) on large rasters with four full passes and one light pass over the data, a fixed merge order that makes the result independent of the number of threads, and additional memory that does not grow with the raster size.

## Contents

| Path | Content |
|---|---|
| `src/e1_passes.py` | Ablation, scaling, memory and out-of-core experiments (variants B0, B1, A1, A2, P, P5) |
| `src/e2_determinism.py` | Determinism experiment (P versus order-dependent reduction A3, float32 and float64) |
| `src/make_replicated.py` | Builds the stack larger than main memory by concatenating a real stack *k* times |
| `src/verify_replicated.py` | Checks the weights of the concatenated stack against the analytical prediction |
| `figures/make_figures.py` | Draws Figures 1–9 of the manuscript |
| `results/` | `runs.csv` and `summary.json` of every measurement series reported in the paper |
| `data/` | Description of the input stacks (the stacks themselves are archived on Zenodo) |
| `gee/` | Google Earth Engine script used to prepare the criteria layers |
| `docs/REPRODUCE.md` | Exact commands for every experiment and where each table and figure comes from |

## Requirements

Python 3.11 with the packages in `requirements.txt`. All results in the paper were obtained on Windows (10.0.22000), AMD Ryzen 5 5600H (6 cores, 12 threads), 7.4 GB of usable main memory and an internal SSD.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```
python src/e1_passes.py --data data/orol_v2_100m/data.npy --dirs + - - + - + - - --variants P --threads 1 12 --repeats 3 --out results/try_100m
```

Criteria order and directions are fixed for all stacks: Precipitation (+), LST (−), WindSpeed (−), NDWI (+), SI2 (−), NDVI (+), Elevation (−), Slope (−).

## Script versions

Every `summary.json` records the SHA-256 prefix of the script that produced it. The final series in the paper were produced by `e1_passes.py` with hash prefix `eecdc88f15019d44`, which is the version in this repository.

## Data

The input stacks (`orol_v2_1000m`, `orol_v2_100m`, `orol_v2_30m`; NumPy `.npy` arrays of shape (8, K), float32, with `meta.json`) are archived on Zenodo: [Zenodo DOI]. See `data/README.md`.

## Licence

Code: MIT licence (`LICENSE`). Measurement results in `results/`: CC BY 4.0. Input data: see `data/README.md` for the licences of the source products.

## Citation

See `CITATION.cff`. Please cite the article once published, and the software archive: [Zenodo DOI].
