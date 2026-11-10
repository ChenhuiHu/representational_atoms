# Representational Atoms

Code for **Representational Atoms in Neural Networks**. The repository provides a unified pipeline for extracting neural representations, training sparse autoencoders (SAEs), evaluating fidelity and stability, measuring monosemanticity, and analyzing semantic alignment and cross-layer composition.

```text
representation extraction
        ↓
SAE training
        ↓
fidelity / stability evaluation
        ↓
monosemanticity analysis
        ↓
HowNet semantic alignment and cross-layer composition
```

## Quick start: end-to-end smoke test

The commands below run a small HowNet/Gemma-2-2B experiment through the complete pipeline. They are intended to verify the environment, model loading, data interfaces, checkpointing, evaluation, and semantic-analysis code. The reduced data size, SAE width, and training duration are **not intended to reproduce the reported experimental results**.

The smoke test uses 512 HowNet entities and three Gemma-2-2B layers (`0`, `12`, and `25`). It requires access to `google/gemma-2-2b` and one CUDA-capable GPU.

From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
mkdir -p data/smoke
```

Create a deterministic 512-entity subset:

```bash
python - <<'PY'
import json
import random
from pathlib import Path

src = "data/hownet_all_entities.json"
dst = "data/smoke/hownet_512.json"

with open(src, "r", encoding="utf-8") as f:
    values = json.load(f)

values = list(dict.fromkeys(
    x.strip() for x in values if isinstance(x, str) and x.strip()
))
random.Random(42).shuffle(values)
values = values[:512]

Path(dst).parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(values, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(values)} entities to {dst}")
PY
```

### 1. Extract representations

```bash
python scripts/extract_representations.py \
  --config configs/extraction/hownet_gemma2_2b.yaml \
  --set dataset.path=data/smoke/hownet_512.json \
  --set extraction.layers='[0,12,25]' \
  --set extraction.shuffle=false \
  --set extraction.forward_batch_size=32 \
  --set output.root=data/smoke/representations
```

### 2. Train small BiJumpReLU SAEs

```bash
python scripts/train_sae.py \
  --config configs/training/hownet_gemma2_2b.yaml \
  --sae-config configs/sae/bijumprelu.yaml \
  --set experiment.layers='[0,12,25]' \
  --set data.path_template='data/smoke/representations/hownet/google__gemma-2-2b/post_feedforward_layernorm_output/layer_{layer}' \
  --set data.max_samples=512 \
  --set sae.expansion_factor=1 \
  --set training.num_epochs=3 \
  --set training.batch_size=128 \
  --set training.logical_batch_size=128 \
  --set training.eval_every=1 \
  --set training.log_every=1 \
  --set training.eval_batch_size=128 \
  --set selection.max_l0_for_save=10000 \
  --set selection.first_save_l0=10000 \
  --set selection.first_save_r2=-10000 \
  --set output.dir_template='saved_models/smoke/hownet/gemma2_2b/{architecture}/layer_{layer}'
```

### 3. Evaluate fidelity and stability

```bash
python scripts/evaluate_fidelity_stability.py \
  --config configs/evaluation/hownet_gemma2_2b.yaml \
  --sae-config configs/sae/bijumprelu.yaml \
  --set experiment.layers='[0,12,25]' \
  --set representations.path_template='data/smoke/representations/hownet/google__gemma-2-2b/post_feedforward_layernorm_output/layer_{layer}' \
  --set representations.max_samples=512 \
  --set sae.run_dir_template='saved_models/smoke/hownet/gemma2_2b/{architecture}/layer_{layer}' \
  --set evaluation.stats_batch_size=512 \
  --set evaluation.inference_batch_size=128 \
  --set evaluation.stability.max_mu_samples=20000 \
  --set evaluation.stability.pair_batch_size=5000 \
  --set evaluation.stability.full_metric_chunk_size=512 \
  --set output.root='results/smoke/fidelity_stability/HowNet/gemma2_2b/{architecture}'
```

### 4. Mine top-activating entities

```bash
python scripts/prepare_monosemanticity.py \
  --config configs/monosemanticity/hownet_gemma2_2b.yaml \
  --stage candidates \
  --set experiment.layers='[0,12,25]' \
  --set sparse_activations.path_template='results/smoke/fidelity_stability/HowNet/gemma2_2b/bijumprelu/layer_{layer}/sae_activations.pt' \
  --set source.path_template='data/smoke/representations/hownet/google__gemma-2-2b/post_feedforward_layernorm_output/layer_{layer}' \
  --set candidate_mining.top_k_examples=100 \
  --set candidate_mining.feature_pool_size=500 \
  --set output.root='results/smoke/monosemanticity/HowNet/gemma2_2b/bijumprelu'
```

### 5. Run semantic alignment and composition

Prepare the shared sparse set representation:

```bash
python scripts/evaluate_semantic_alignment.py \
  --config configs/semantic_alignment/hownet_gemma2_2b.yaml \
  --stage prepare \
  --set experiment.layers='[0,12,25]' \
  --set features.path_template='results/smoke/monosemanticity/HowNet/gemma2_2b/bijumprelu/layer_{layer}/candidates/feature_examples.jsonl' \
  --set features.top_k_examples=100 \
  --set output.root='results/smoke/semantic_alignment/HowNet/gemma2_2b'
```

Run the best-single-atom search:

```bash
python scripts/evaluate_semantic_alignment.py \
  --config configs/semantic_alignment/hownet_gemma2_2b.yaml \
  --stage single \
  --set experiment.layers='[0,12,25]' \
  --set features.path_template='results/smoke/monosemanticity/HowNet/gemma2_2b/bijumprelu/layer_{layer}/candidates/feature_examples.jsonl' \
  --set features.top_k_examples=100 \
  --set output.root='results/smoke/semantic_alignment/HowNet/gemma2_2b'
```

Run a reduced cross-layer composition search:

```bash
python scripts/evaluate_semantic_alignment.py \
  --config configs/semantic_alignment/hownet_gemma2_2b.yaml \
  --stage composition \
  --set experiment.layers='[0,12,25]' \
  --set features.path_template='results/smoke/monosemanticity/HowNet/gemma2_2b/bijumprelu/layer_{layer}/candidates/feature_examples.jsonl' \
  --set features.top_k_examples=100 \
  --set search.max_candidates_per_sememe=200 \
  --set search.max_features_to_merge=10 \
  --set search.sememe_batch_size=16 \
  --set output.root='results/smoke/semantic_alignment/HowNet/gemma2_2b'
```

Run a reduced hyperparameter grid:

```bash
python scripts/evaluate_semantic_alignment.py \
  --config configs/semantic_alignment/hownet_gemma2_2b.yaml \
  --stage grid \
  --set experiment.layers='[0,12,25]' \
  --set features.path_template='results/smoke/monosemanticity/HowNet/gemma2_2b/bijumprelu/layer_{layer}/candidates/feature_examples.jsonl' \
  --set features.top_k_examples=100 \
  --set search.max_candidates_per_sememe=200 \
  --set search.sememe_batch_size=16 \
  --set grid_search.min_precision_values='[0.4,0.5]' \
  --set grid_search.max_features_to_merge_values='[5,10]' \
  --set grid_search.extended.enabled=false \
  --set output.root='results/smoke/semantic_alignment/HowNet/gemma2_2b'
```

A successful run verifies the complete software path from entity input to cross-layer semantic composition. Numerical values from this smoke test should not be used for scientific comparison.

---

## Installation

The repository is tested with Python 3.10. The experimental environment used for validation includes PyTorch 2.11 with CUDA 12.8 support.

Install the dependencies appropriate for the target CUDA platform, then make the package importable from the repository root:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

For editable installation, use:

```bash
pip install -e .
```

## Repository structure

```text
configs/
  extraction/                  representation-extraction settings
  training/                    representation source and optimization settings
  sae/                         SAE architecture settings
  evaluation/                  fidelity/stability settings and summaries
  monosemanticity/             candidate mining and review-set settings
  semantic_alignment/          HowNet semantic alignment and composition

data/
  hownet_all_entities.json
  wikidata_filtered_entities.json

scripts/
  extract_representations.py
  train_sae.py
  evaluate_fidelity_stability.py
  summarize_fidelity_stability.py
  prepare_monosemanticity.py
  summarize_monosemanticity.py
  evaluate_semantic_alignment.py
  analyze_semantic_alignment.py

src/representational_atoms/
  config.py
  representation_io.py
  extraction/
  sae/
  evaluation/
  monosemanticity/
  semantic_alignment/
```

Configuration files define the experiment; shared implementation lives under `src/representational_atoms/`; command-line entry points live under `scripts/`. Most scripts support repeated `--set key=value` arguments for temporary overrides without modifying the YAML files.

## Data and representation extraction

The repository includes the compact entity lists used by the HowNet and Wikidata experiments:

```
data/hownet_all_entities.json
data/wikidata_filtered_entities.json
```

Wikipedia and LAION-COCO are downloaded separately. 

All representation extraction uses:

```
python scripts/extract_representations.py \
  --config <config.yaml>
```

The expected input paths are defined in the corresponding files under `configs/extraction/`.

### Wikipedia

The language-model experiments use the English `20231101.en` release of Wikipedia. The dataset should be stored as Parquet files under:

```
data/wikipedia/20231101.en/
```

so that the extraction config can read:

```
data/wikipedia/20231101.en/*.parquet
```

The dataset is available from the Hugging Face `wikimedia/wikipedia` collection using the `20231101.en` configuration.

Representations can then be extracted with:

```
python scripts/extract_representations.py \
  --config configs/extraction/wikipedia.yaml
```

The extractor processes Wikipedia as a continuous token stream, separates documents with the configured separator, and partitions the stream into fixed-length token chunks before recording the selected internal representations.

Per-layer representations are stored with shape:

```
[num_chunks, tokens_per_chunk, hidden_dim]
```

SAE training and evaluation flatten the first two dimensions into a token population of shape:

```
[num_tokens, hidden_dim]
```

### Wikidata and HowNet

The entity populations used by these experiments are provided directly in the repository:

```
data/wikidata_filtered_entities.json
data/hownet_all_entities.json
```

Wikidata representations can be extracted with:

```
python scripts/extract_representations.py \
  --config configs/extraction/wikidata.yaml
```

Model-specific HowNet configs are provided, for example:

```
python scripts/extract_representations.py \
  --config configs/extraction/hownet_gemma2_2b.yaml
```

For each entity, the extractor records the representation of the final non-padding token at the configured model layer and representation site.

Entity representations are stored as per-layer shards:

```
{
    "shard_id": int,
    "sample_ids": list[str],
    "activations": Tensor[num_samples, hidden_dim],
}
```

The `sample_ids` preserve the mapping between each representation and its source entity for downstream monosemanticity and semantic-alignment analyses.

### LAION-COCO

The diffusion-model experiments use the [LAION-COCO Aesthetic](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic) dataset.

Download the dataset to the repository root as:

```
laion-coco-aesthetic/
```

The extraction configs for both Stable Diffusion v1.5 and FLUX.2-klein-9B use:

```
path: ./laion-coco-aesthetic
```

The dataset provides the captions used as generation prompts. Images are generated by the corresponding diffusion model during representation extraction, so the original LAION image collection is not required.

Stable Diffusion v1.5 representations are extracted with:

```
python scripts/extract_representations.py \
  --config configs/extraction/sd15.yaml
```

FLUX.2-klein-9B representations are extracted with:

```
python scripts/extract_representations.py \
  --config configs/extraction/flux2_klein.yaml
```

SD1.5 records representations from the configured U-Net block at selected diffusion timesteps. FLUX.2-klein-9B records image-token representations from selected transformer blocks and inference steps.

### Diffusion representation format

SD1.5 and FLUX.2 representations are stored as WebDataset tar archives because each generated image contributes multiple spatial representations and may be evaluated at multiple diffusion or inference stages.

Each spatial representation is assigned a stable global identifier:

```
patch_idx = h * grid_width + w
global_id = sample_id * patches_per_image + patch_idx
```

These identifiers are preserved through SAE training, fidelity/stability evaluation, and monosemanticity analysis, allowing each activation to be mapped back to its generated image and spatial position.

### Representation outputs

Extracted representations are written to the output location specified by each extraction config. LLM entity experiments use a layer-oriented layout such as:

```
data/representations/
└── <dataset>/
    └── <model>/
        └── <representation_site>/
            ├── config.json
            ├── layer_0/
            │   └── shard_000000.pt
            └── layer_.../
```

Wikipedia follows the same layer-oriented organization for continuous-token representations. Diffusion extraction stores WebDataset archives together with the metadata required to identify samples, spatial positions, layers, and diffusion or inference stages.

The generated representations are consumed directly by the SAE training configs under `configs/training/`.

## SAE training

Training combines a data/training config with an architecture config:

```bash
python scripts/train_sae.py \
  --config configs/training/wikipedia.yaml \
  --sae-config configs/sae/bijumprelu.yaml
```

The repository implements five SAE architectures:

- ReLU
- TopK
- AbsTopK
- JumpReLU
- BiJumpReLU

The shared trainer implements the representation-derived initialization, architecture-specific activation rules, Adam optimization, gradient accumulation, decoder-gradient projection, decoder-row normalization, R2/L0 monitoring, and checkpoint selection used throughout the experiments.

### Representation preprocessing

Normalization is controlled by config:

```yaml
preprocessing:
  normalization: rms
```

Supported modes are:

```text
rms          per-representation RMS normalization
global_std   one global standard-deviation scaling factor
none         no additional scaling
```

Newly trained SAEs in the provided experiment configs use RMS normalization. External pretrained baselines such as GemmaScope use the preprocessing specified by their own evaluation configs.

## Fidelity and stability

Run the shared evaluator with a dataset/model config and an SAE config:

```bash
python scripts/evaluate_fidelity_stability.py \
  --config configs/evaluation/wikidata_gemma2_2b.yaml \
  --sae-config configs/sae/bijumprelu.yaml
```

The evaluator reports:

- flattened reconstruction R2;
- per-sample L0 (`K`);
- active feature IDs and feature utilization;
- sparse SAE activations in CSR form;
- atomic-inner-product coherence statistics;
- retention and stability scores.

The fidelity metric is

```text
R2 = 1 - MSE / Var(X)
```

For stability, normalized decoder directions are evaluated under the representation-induced metric. The implementation searches for the largest quantile `alpha` satisfying

```text
mu_alpha < 1 / (K_alpha - 1)
mu_alpha < 1 / (2 K_alpha - 1)
```

The first condition measures retention/coherence; the second measures stability/distinguishability.

### Metric construction for undercomplete active sets

With `metric_policy: auto`, the evaluator chooses the metric source according to the number of active decoder directions:

```text
n_active >= d_model  -> metric from active decoder directions
n_active <  d_model  -> metric from all normalized decoder directions
```

Only active directions are scored in the second case. Cholesky factorization is preferred, with a Moore-Penrose pseudoinverse available for singular cases.

Each `stability.json` records the selected metric source, solver, active-feature count, model dimension, and the resulting coherence statistics.

### Evaluation outputs

Layer-based experiments produce:

```text
<output-root>/
├── layer_0/
│   ├── metrics.json
│   ├── sparsity.pt
│   ├── features_id.pt
│   ├── sae_activations.pt
│   └── stability.json
├── layer_.../
├── all_points_metrics.json
├── all_points_stability.json
├── all_layers_metrics.json
└── all_layers_stability.json
```

Diffusion experiments use condition names such as `cond_t500` and `layer15_step3` in place of layer-only names.

## Monosemanticity evaluation

The monosemanticity pipeline has three stages.

1. **Candidate mining** reads the sparse SAE activations and collects the strongest positive, negative, and absolute-magnitude activating examples for each feature.
2. **Review-set construction** deterministically samples candidate FRUs and their activating instances. Wikipedia uses contextualized tokens, Wikidata uses entity names, and LAION-COCO uses generated images with the activated spatial cell highlighted.
3. **Score aggregation** records the size `M` of the largest coherent semantic group among `N` reviewed examples and defines the feature score as `M/N`.

Prepare candidate pools and review sets with, for example:

```bash
python scripts/prepare_monosemanticity.py \
  --config configs/monosemanticity/wikipedia_gemma2_2b.yaml
```

Each evaluated condition can produce:

```text
candidates/
  feature_examples.jsonl
  feature_statistics.pt
review/
  evaluation_prompts.jsonl
  sampled_features.jsonl
  quick_review.md
  annotations_template.jsonl
  visuals/                    # image models only
```

After completing the review annotations, aggregate scores with:

```bash
python scripts/summarize_monosemanticity.py \
  --input results/monosemanticity/Wikipedia/gemma2-2b/bijumprelu
```

For HowNet, candidate mining supplies the signed entity sets used by the semantic-alignment pipeline; review-set generation is disabled in the corresponding configs.

## HowNet semantic alignment and cross-layer composition

Semantic alignment is implemented once and configured separately for:

```text
Gemma-2-2B
Gemma-2-9B
Llama-3.1-8B
```

The pipeline consumes the signed high-activation entity sets produced by HowNet candidate mining:

```text
sae_activations.pt
      ↓
prepare_monosemanticity.py --stage candidates
      ↓
layer_k/candidates/feature_examples.jsonl
      ↓
evaluate_semantic_alignment.py
```

Features with both positive and negative responses contribute separate signed candidates. Candidate atom sets and HowNet sememe sets with fewer than two unique entities are excluded.

Run the complete semantic-alignment workflow with:

```bash
python scripts/evaluate_semantic_alignment.py \
  --config configs/semantic_alignment/hownet_gemma2_2b.yaml \
  --stage all
```

The stages can also be run independently:

```text
prepare       build reusable sparse feature/sememe set matrices
single        find the best individual signed atom for each sememe
composition   perform greedy cross-layer atom composition
grid          evaluate the configured search hyperparameter grid
```

### Single-atom alignment

A common entity vocabulary is built over the signed atom sets and HowNet sememe sets. Sparse binary matrix multiplication yields all atom-sememe intersection sizes, from which precision, recall, and F1 are computed. The highest-F1 candidate across layers is retained for each sememe.

### Cross-layer composition

For each sememe, candidates are filtered by minimum precision and non-zero overlap. The greedy search repeatedly adds the atom whose entity-set union gives the largest positive F1 improvement, subject to the configured candidate and composition limits.

The default manuscript configuration uses:

```yaml
search:
  min_precision: 0.5
  max_candidates_per_sememe: 4000
  max_features_to_merge: 800
  require_positive_gain: true
```

The result records the selected signed atoms, source layers, stepwise gains, final precision/recall/F1, recovered entity sets, and cross-layer width.

Post-search layer-width statistics can be computed with:

```bash
python scripts/analyze_semantic_alignment.py \
  --results results/semantic_alignment/hownet/gemma2_2b/composition/composition_results.json
```

## Experiment coverage

The provided configs cover the principal model/data combinations used in the repository:

- **Wikipedia:** Gemma-2-2B, Gemma-2-9B, Qwen3.5-9B, and GemmaScope baselines;
- **Wikidata:** Gemma-2-2B, Gemma-2-9B, Olmo-Hybrid-7B, and GemmaScope baselines;
- **HowNet:** full-depth BiJumpReLU runs for Gemma-2-2B, Gemma-2-9B, and Llama-3.1-8B;
- **LAION-COCO:** Stable Diffusion v1.5 and FLUX.2-klein-9B.

For HowNet full-depth training, for example:

```bash
python scripts/train_sae.py \
  --config configs/training/hownet_gemma2_2b.yaml \
  --sae-config configs/sae/bijumprelu.yaml
```

The corresponding full-depth fidelity/stability evaluation is:

```bash
python scripts/evaluate_fidelity_stability.py \
  --config configs/evaluation/hownet_gemma2_2b.yaml \
  --sae-config configs/sae/bijumprelu.yaml
```

## Aggregating fidelity/stability results

Aggregate evaluation points with a two-sided 95% Student-t confidence interval using:

```bash
python scripts/summarize_fidelity_stability.py \
  --config configs/evaluation/summary_wikipedia.yaml

python scripts/summarize_fidelity_stability.py \
  --config configs/evaluation/summary_wikidata.yaml

python scripts/summarize_fidelity_stability.py \
  --config configs/evaluation/summary_laion.yaml
```

## Reproducibility notes

- Use the YAML files under `configs/` as the authoritative experiment definitions.
- Use `--set key=value` for temporary overrides such as smoke-test sizes, selected layers, output roots, or batch sizes.
- Keep preprocessing consistent between SAE training and evaluation.
- Large extracted representations, checkpoints, and runtime results should remain outside Git or in ignored output directories.
- Small final result tables intended for paper reproduction may be stored separately from runtime artifacts.
