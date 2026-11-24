# Do Pruning and Quantization Compose?

**Order and Interaction in Post-Training Compression of Language Models**

Antoun Yaacoub, Martin Cousseau and Stanislas Laborde. ESANN 2027 (submitted).

Post-training pruning and quantization are sometimes stacked to compress language models. This repository holds the code,
the result lines and the analysis of a controlled study of how their errors combine. One Hessian-compensated sweep
(the OBS family: SparseGPT for pruning, GPTQ for quantization) implements both methods and three ways of combining
them at equal theoretical compression:
- a **joint** pass;
- **prune-then-quantize** (P→Q);
- **quantize-then-prune** (Q→P).

The study covers seven models from 135M to 3B parameters (SmolLM2-135M, -360M and -1.7B, Qwen2.5-0.5B, -1.5B and -3B,
OLMo-2-1B), 25, 33 and 50% unstructured sparsity, and 3 and 4 bits.

Every number of the paper is generated from the result lines by `scripts/make_tables.py`
(`paper/generated/macros.tex`). The figures below come from the same file.

## Key results

- **Mask-first orders are sub-additive.** With compensation, the joint pass and P→Q have a smaller layer error than
  the sum of the errors of pruning alone and quantization alone, in all 84 cases (7 models × 6 settings × 2 orders).
- **The order matters.** By WikiText-2 perplexity:
  - Q→P is the worst order in 35 of 42 settings, up to 1.77× the best;
  - it is above the better mask-first order in 41 of 42;
  - over three calibration seeds, it is the worst in 57 of 72 runs.
- **Accuracy agrees.** In the 19 settings where every order has benchmarks, P→Q is more accurate than Q→P on
  ARC-Easy, PIQA and HellaSwag in 17, significantly in 13 (paired bootstrap over items). It is never
  significantly less accurate.
- **A "weaker first" heuristic does not carry over.** It recommends quantizing first in 14 settings; Q→P is the
  best order in none of them.
- **Mechanism.** With one Hessian shared by all orders, the gap between the layer errors of Q→P and P→Q splits exactly
  into three parts:
  - the quantization grid changes the error by at most 0.5%;
  - the order of the compensated updates costs +4.5 to +18.1%, in all 42 cases;
  - the mask can go either way.

  Once the weights are on the grid, compensating a pruning error requires rounding again.

## Repository layout

| Path | Content |
|---|---|
| `src_v2/pipeline.py` | compression pipeline: the OBS sweep for pruning, quantization and the three orders; calibration, perplexity, KL, lm-eval with per-item logs, run metadata |
| `scripts/run_grid.py` | resume-safe runner: one JSON line per compressed model state |
| `scripts/make_queues.py`, `colab/queues/` | the experiment definitions (families E1 to E9) and the queues that were run |
| `scripts/run_e2.py`, `scripts/run_e9.py`, `review_code/` | layer-level controls: one shared Hessian (E2) and no compensation (E9), with their ablation arms |
| `scripts/make_tables.py`, `scripts/analysis/` | analysis: every table, figure and paper number from the result lines |
| `scripts/make_supplement.py` | the supplementary tables (`paper/supplement/`) |
| `results/v2/` | result lines per family and model, and per-item benchmark logs (`items/`) |
| `reports/analysis/`, `reports/figures/` | analysis tables (CSV/JSON, `summary.md`) and figures |
| `paper/generated/` | the paper's numbers (`macros.tex`, documented in `MACROS.md`) and figures |
| `release/` | loader for the compressed models (`load_esann2027.py`) and the Hugging Face model cards |
| `tests/` | checks of the pipeline and the analysis |
| `src/`, `notebooks/`, `analysis/`, `results/*.json*` | the first release of the code and results (October 2026) |

## Reproduce the analysis (CPU)

```bash
pip install -r requirements-analysis.txt
python scripts/make_tables.py        # tables, figures and paper numbers from results/v2 (about a minute)
python scripts/make_supplement.py    # supplementary tables (the PDF needs pdflatex)
python -m pytest tests/test_analysis.py tests/test_analysis_extended.py tests/test_supplement.py
```

## Re-run the compression (GPU)

The runs used one GPU type (NVIDIA L4), 32-bit weights and Hessians and deterministic kernels. `requirements-v2.txt`
and `colab/locked_fingerprint.json` pin the libraries and the image.

```bash
pip install -r requirements-v2.txt
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-135M \
    --queue colab/queues/E1__SmolLM2-135M.json --locked colab/locked_fingerprint.json
# add --smoke for a quick code-path test (4 calibration windows, 2 evaluation windows, 2 items per task)
```

Each result line records the state, its measurements, the GPU, the libraries and a hash of the compression code.
20 states re-run on another GPU reproduce exactly, with a bit-identical WikiText-2 log-likelihood.

## Data, models and runs

- Results and supplementary tables: [huggingface.co/datasets/martincousseau/esann2027-v2-results](https://huggingface.co/datasets/martincousseau/esann2027-v2-results)
- Compressed model states (exact codes, scales and masks): [huggingface.co/martincousseau/esann2027-compose-v2](https://huggingface.co/martincousseau/esann2027-compose-v2), loaded with `release/load_esann2027.py`
- One Weights & Biases run per state: [wandb.ai/martin-css/esann2027-compose-v2](https://wandb.ai/martin-css/esann2027-compose-v2)

## Citation

```bibtex
@inproceedings{yaacoub2027compose,
  title     = {Do Pruning and Quantization Compose? Order and Interaction in Post-Training Compression of Language Models},
  author    = {Yaacoub, Antoun and Cousseau, Martin and Laborde, Stanislas},
  booktitle = {European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year      = {2027},
  note      = {Submitted}
}
```

This study follows: S. Laborde, M. Cousseau, A. Yaacoub and L. Prevost, *Semantic retention and extreme compression in
LLMs: Can we have both?*, IJCNN 2025 ([arXiv:2505.07289](https://arxiv.org/abs/2505.07289)).

## License

MIT, see [LICENSE](LICENSE).

## Author

**Martin Cousseau**: [Portfolio](https://martincousseau.com) · [Research](https://martincousseau.com/research) ·
[Google Scholar](https://scholar.google.com/citations?user=KzMJqUcAAAAJ) · [ORCID](https://orcid.org/0009-0000-5228-3565) ·
[LinkedIn](https://www.linkedin.com/in/cousseaumartin/) · [GitHub](https://github.com/martin-cousseau) ·
[Hugging Face](https://huggingface.co/martincousseau)
