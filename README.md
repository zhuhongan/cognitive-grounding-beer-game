# Cognitive grounding in the Beer Distribution Game

Code, data and the trained inference network for

> Hongan Zhu and Babak Heydari. *Cognitive grounding improves LLM simulation of collective human behavior.*

Seven large language models (LLMs) play the Beer Distribution Game, a four-role supply-chain experiment, under three prompt conditions. Their order trajectories are compared with a human-calibrated target generated from Sterman's stock-management model. An invertible neural network then infers each session's latent decision parameters.

This repository is a copy of the Code Ocean capsule ([doi:10.24433/CO.2254887.v1](https://doi.org/10.24433/CO.2254887.v1)), reorganized into folders. Every code and data file is byte-identical to the capsule.

## Contents

```
simulation/                    LLM gameplay (calls models through the OpenRouter API)
  README.md                    how to run the games, parameters, what each agent sees
  full_state/                  primary protocol: one API call per decision, history in the prompt
  persistent_session/          one conversation per role (Supplementary Section 9)
  extract_simulated_data.py    converts raw game output into the files the notebooks read
analysis/
  run                          Code Ocean entry point (executes beer_role_test.ipynb)
  beer_role_test.ipynb         trajectory analysis: relative log-MSE, bullwhip, temporal diagnostics, tests
  beer_role_infer.ipynb        parameter inference with BayesFlow; aggregate alignment, PCA and distances
data/
  beer data/                   LLM trajectories, primary protocol (main text; Supplementary Sections 1, 3-5, 7, 8)
  beer data session/           LLM trajectories, persistent conversation (Supplementary Section 9)
  beer data 2/                 LLM trajectories, supplementary information protocol (Supplementary Section 10)
models/model_role_cf.keras     trained inference network used by beer_role_infer.ipynb
environment/                   Code Ocean Dockerfile and pinned requirement files
```

## Data

Each data folder contains one subfolder per model and condition, named `<model>-<condition>`. Each subfolder holds 11 sessions (`sim_0_extracted.json` … `sim_10_extracted.json`). Each file has the per-round orders, inventory, backlog and shipments for the Retailer, Wholesaler, Distributor and Factory (35 rounds), plus each role's costs, revenue and profit.

| Model tag | Model |
|---|---|
| `kimi` | Kimi K2 0905 |
| `llama3` | Llama-3.1 8B |
| `llama70b` | Llama-3.3 70B |
| `qwen3` | Qwen-3.6-35B-A3B |
| `mistral` | Mistral Small 4 |
| `oss20b` | GPT-OSS 20B |
| `oss120b` | GPT-OSS 120B |

| Condition | Meaning |
|---|---|
| `-baseline` | baseline prompt (game setup, role, objective, role-local state) |
| `-prompt` | heuristic-aware prompt (adds a qualitative description of Sterman's heuristic) |
| `-alt` | cognitive-load prompt (describes the local decision environment) |
| `oss*-high-*` (`beer data/`) | the same conditions at high reasoning effort (Supplementary Section 5) |
| `oss20b-t02-*`, `oss20b-t06-*` (`beer data/`) | GPT-OSS 20B at temperature 0.2 and 0.6 (Supplementary Section 3) |

All other runs use temperature 1.0 and low reasoning effort where the endpoint supports it.

The three protocols differ only in what each agent is told:

- **Primary (`beer data/`).** A fresh API request for every decision, containing the role's current state, the shipment arriving in two rounds and the complete round-by-round record sheet.
- **Persistent conversation (`beer data session/`).** Each role keeps one conversation for the whole game, so its earlier state messages, orders and rationales stay in context instead of a record sheet.
- **Supplementary protocol (`beer data 2/`).** Adds total outstanding orders and the pipeline quantities arriving in two and four rounds, but gives only a rolling four-round history instead of the complete record sheet.

## Reproducing the results

**On Code Ocean.** Open the capsule and click *Reproducible Run*. This runs `analysis/run`, which executes `beer_role_test.ipynb` on `beer data/` and writes its figures (PDF) and tables (CSV) to `/results`.

**Locally.** The notebooks use Code Ocean paths, set in their first code cell: `DATA_ROOT = Path("/code/beer data")`, `RESULTS_ROOT = Path("/results")` and, in `beer_role_infer.ipynb`, `MODEL_PATH = Path("/code/model_role_cf.keras")`. To run them from this repository, start Jupyter in `analysis/` and change these lines to:

```python
DATA_ROOT = Path("../data/beer data")
RESULTS_ROOT = Path("../results")
MODEL_PATH = Path("../models/model_role_cf.keras")   # beer_role_infer.ipynb only
```

The two notebooks use different environments:

- **Trajectory analysis** (`beer_role_test.ipynb`): the Code Ocean environment, Python 3.12 with the versions in `environment/requirements-analysis.txt`. It runs in about a minute on a laptop.

  ```bash
  python3.12 -m venv .venv-analysis && . .venv-analysis/bin/activate
  pip install -r environment/requirements-analysis.txt
  cd analysis && jupyter nbconvert --to html --execute beer_role_test.ipynb
  ```

- **Parameter inference** (`beer_role_infer.ipynb`): Python 3.10 with BayesFlow 2.0.7 and Keras 3.10.0 on the JAX 0.6.0 backend (`environment/requirements-inference.txt`). The notebook selects the JAX backend itself. Posterior sampling for all sessions needs a high-memory CPU machine.

**Supplementary Sections 9 and 10.** Set `DATA_ROOT` to `data/beer data session` (Section 9) or `data/beer data 2` (Section 10) and run the notebook. The temperature and reasoning-effort cells of `beer_role_test.ipynb` then report missing folders, because those runs exist only for the primary protocol; all other cells run.

**From raw game output.** The simulation scripts write `<tag>-<condition>/sim_N.json`. Convert a folder into the format the notebooks read with

```bash
python3 simulation/extract_simulated_data.py --input-dir <tag>-<condition> --output-dir "data/beer data/<model>-<condition>"
```

and name the output folder with the model tags above. Applied to the raw runs, this reproduces every file in `data/beer data/` byte for byte.

**Running new games.** See `simulation/README.md`. The scripts need an OpenRouter API key in `OPENROUTER_API_KEY`. Models are sampled at temperature 1 without a seed, so re-running reproduces the distribution of trajectories, not individual games.

## Citation

If you use this code or data, please cite the paper and the capsule:

> Zhu, H. & Heydari, B. Cognitive grounding improves LLM simulation of collective human behavior. Code Ocean (2026). https://doi.org/10.24433/CO.2254887.v1

## License

MIT; see `LICENSE`.
