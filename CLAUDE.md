# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IEEE conference paper: **Bayesian Change-Point Detection in Nigerian Rainfall Patterns**. Applies hierarchical Bayesian change-point models (PyMC5) to detect climate regime shifts across Nigeria's three climate zones (Sahel, Guinea Savanna, Coastal/Rainforest) using Open-Meteo reanalysis data (1950–2023).

## Environment

- **Conda env**: `bap3` — PyMC 5.8.0, ArviZ 0.16.1, Bambi 0.13.0, Pandas, NumPy, Xarray, Matplotlib
- **Secondary env**: `homl3` — scikit-learn, TensorFlow, Seaborn, Shapely (for extra plotting/geo if needed)
- **Platform**: Windows 11, PowerShell

## Running Scripts

All scripts are numbered for sequential execution:
```
conda activate bap3
python notebooks/01_data_acquisition.py
python notebooks/02_data_processing.py
python notebooks/03_eda.py
python notebooks/04_single_changepoint.py
python notebooks/05_hierarchical_model.py
python notebooks/06_model_comparison.py
python notebooks/07_paper_figures.py
```

MCMC scripts (04, 05, 06) take 5–15 minutes each. All use `random_seed=42`.

**Important**: PyTensor's C compiler (mingw g++) segfaults on this machine. All MCMC scripts must include `os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,cxx="` before importing PyMC. This disables C compilation and uses the slower Python backend.

## Compiling the Paper

```
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Architecture

- `src/utils.py` — shared zone definitions (9 cities, 3 zones), Open-Meteo API fetcher with chunking, IEEE matplotlib style
- `notebooks/` — numbered Python scripts (not Jupyter) run sequentially; each reads from `data/` and writes outputs to `data/` or `figures/`
- `data/raw/` — cached API JSON responses; `data/traces/` — saved MCMC NetCDF traces
- `figures/` — PDF/PNG output consumed by `paper/main.tex`
- `paper/` — IEEEtran conference LaTeX source

## Data Source

Open-Meteo Historical Weather API (no API key): `archive-api.open-meteo.com/v1/archive`. Fetched in 10-year chunks to avoid timeouts. Nine stations across three Nigerian climate zones.

## Running MCMC on Google Colab (the working approach)

Local MCMC is unusable because PyTensor's C compiler segfaults (see above). All MCMC was run on Google Colab via `google-colab-cli` from WSL. Here is what works and what doesn't.

### WSL command pattern
```powershell
wsl --exec bash -c "export HOME=/home/batesthommie; export PATH=/home/batesthommie/.local/bin:/usr/local/bin:/usr/bin:/bin; colab <command>"
```

### What WORKS: `colab run` with a single combined script
```bash
colab run --keep --timeout 3600 -s <session_name> colab_mcmc_combined.py
```
- **Combine ALL steps** (single CP, hierarchical, model comparison, figures) into ONE script. Idle gaps between separate `colab exec` calls let Colab's free tier kill the VM.
- Use `--keep` to prevent session cleanup after the run.
- Use `--timeout 3600` (1 hour) for safety margin.
- Use `-s <name>` to name your session for easy reference.
- Embed CSV data directly in the script — no file upload needed.
- Save checkpoints (NetCDF traces) after each model so partial progress survives.
- **2000 draws + 1000 tune is enough** (R-hat < 1.01 for most params). Don't use 4000 draws — it doubles runtime for marginal gain.
- Total runtime with 2000 draws: ~27 minutes on Colab CPU.
- GPU does NOT help NUTS sampling (sequential per chain, overhead > benefit).

### What FAILS or is unreliable
- **`colab exec -f script.py`**: WebSocket drops after 10–20 min of MCMC. The kernel keeps running but CLI loses connection and reports failure. Unusable for anything over ~10 min.
- **Multiple separate scripts via `colab exec`**: Idle time between scripts lets the free tier kill the VM. Session dies with 404/401 errors.
- **`colab install`**: Uses `uv` which only installs metadata, not actual packages. Use `pip install` via `colab exec` instead.
- **Upgrading numpy on Colab**: Breaks pandas C extensions. Accept Colab's pre-installed numpy.
- **30-second timeouts for check scripts**: Kernel is busy sampling, won't schedule your check. Use 1200s+ timeout.

### Step-by-step recipe (next time)
1. `colab new` → creates a new session (note the session ID)
2. Upload a PyMC install script via `colab exec -f install_pymc.py --timeout 300`:
   ```python
   import subprocess, sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pymc>=5.0", "arviz>=0.16"])
   ```
3. Restart kernel: `colab restart-kernel -s <session>`
4. Run combined script: `colab run --keep --timeout 3600 -s <session> colab_mcmc_combined.py`
5. Download results: `colab download -s <session> /content/mcmc_output.zip`
6. Extract: unzip to `data/traces/` and `figures/`
7. Stop session: `colab stop -s <session>`

### Colab PyMC version differences
Colab installs PyMC 5.28+ / ArviZ 0.22+ / pandas 2.x (or PyMC 6.x / ArviZ 1.2+ / pandas 3.x). Key API differences from local bap3 (PyMC 5.8.0):
- PyMC 6: `trace.groups()` → `trace.children`; call `pm.compute_log_likelihood(trace)` separately
- ArviZ 1.2: HDI uses `hdi_prob=` kwarg
- pandas 3.0: `az.summary()` returns string-typed columns — must use `pd.to_numeric()` before numeric comparisons

### Key file
`colab_mcmc_combined.py` — the script that produced all results. Self-contained with embedded data, all 4 pipeline steps, checkpoints, and zip packaging.

## MCMC Results Summary (completed 2026-06-19)

All traces in `data/traces/`, all figures in `figures/`.

| Zone | Change-Point | 94% HDI | Shift (mm/yr) |
|------|-------------|---------|---------------|
| Sahel (single) | 1958 | [1957, 1960] | −215 |
| Guinea Savanna (single) | 2016 | [2014, 2017] | −366 |
| Coastal (single) | 2015 | [1996, 2022] | −236 |
| Group (hierarchical) | 1989 | [1974, 2003] | — |

LOO model comparison: two_cp marginally preferred for Sahel/Guinea Savanna; one_cp preferred for Coastal; null decisively rejected everywhere.

Convergence: mostly good (R-hat < 1.01, ESS > 400). Minor issues: τ_sigma R-hat ≈ 1.02, Coastal single CP has broad posterior. These are genuine data features, not sampling failures.

## Word Document

`Bayesian_Changepoint_Nigerian_Rainfall.docx` — full paper with all 7 figures embedded. Generated by `create_word_paper.py`. Author placeholders: `[AUTHOR 1 NAME]`, `[AUTHOR 2 NAME]`, `[AUTHOR 3 NAME]`.

## Kruschke-Style Model Diagram

`notebooks/08_model_diagram.py` generates `figures/fig_model_diagram.pdf/png` using the `daft-pgm` package (NOT the Daft data framework). The correct `daft` is the PGM plate diagram library: `pip install daft`. If the wrong daft gets installed, uninstall and recreate `site-packages/daft/__init__.py` with `from ._core import PGM, Node, Edge, Plate`.
