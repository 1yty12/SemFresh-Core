# Semantic Freshness-Aware Edge Video Analytics — Core Components and Data

Core algorithm components and supporting data for *Semantic Freshness-Aware Task Partitioning and Resource Allocation for Edge Video Analytics*.

This repository contains standalone core algorithm components, a subset of research inputs, and single-run outputs. **It is not a complete reproduction package for all paper experiments.** No training environment or complete evaluation pipeline is provided here.

## Directory guide

| Directory / file | Contents |
|---|---|
| `SemFresh-Core/` | Standalone TensorFlow implementations of PDS, Hard-STE, mathematical resource guidance, and an actor-loss composition interface, with a synthetic demonstration. |
| `experiment_results/algorithm_comparison_seed41/` | Six MAT results, raw mean metrics, metadata and filename/hash mappings. |
| `processed/accmpeg_importance_cache/` | 15 NPZ caches of frame-level tile importance and available frame-size metadata for the default WD inputs. |
| `workload/` | Original `ECW_newapp.pkl`, a numeric normalized NPY trace, CSV and processing description. |
| `provenance/` | Video/cache mapping, input-selection records, preprocessing/model source snapshots, and historical copy manifests. |

## Core algorithm components: SemFresh-Core

[SemFresh-Core](SemFresh-Core/README.md) provides the core mechanisms used in semantic-freshness-aware task scheduling:

- **Post-decision state (PDS):** differentiable state construction, discrete state construction with aggregate-based hidden-window reconstruction, and ADoI/energy cost estimation.
- **Straight-through estimation (Hard-STE):** hard forward scheduling decisions with sigmoid surrogate gradients for optimization.
- **Mathematical resource guidance:** CPU-frequency and transmit-power targets for the selected workloads.
- **Actor-loss composition:** an interface combining a caller-provided value function, the PDS cost, and the resource-guidance loss.

| File | Purpose |
|---|---|
| `SemFresh-Core/semantic_core/core.py` | PDS construction, decision gates, resource targets, and cost estimation. |
| `SemFresh-Core/semantic_core/config.py` | Explicit caller-supplied configuration and parameter validation. |
| `SemFresh-Core/semantic_core/losses.py` | Composition of the critic, PDS-cost, and resource-guidance loss terms. |
| `SemFresh-Core/semantic_core/__init__.py` | Public imports: `CoreConfig`, `PDSModel`, and `core_actor_loss`. |
| `SemFresh-Core/demo.py` | Synthetic checks of tensor shapes, finite outputs, hard decisions, and gradient flow. |
| `SemFresh-Core/requirements.txt` | NumPy and TensorFlow dependencies. |
| `SemFresh-Core/README.md`, `README_zh.md`, `VALIDATION.md` | API documentation, Chinese overview, and recorded validation results. |
| `SemFresh-Core/MANIFEST.json` | Original package checksum manifest; it may mention files not retained in this snapshot. |

From the repository root (this `data` directory), run:

```sh
cd SemFresh-Core
python -m pip install -r requirements.txt
python demo.py
```

The demonstration uses two artificial users and two artificial tasks. It does not load the sibling `processed/`, `workload/`, or `experiment_results/` directories, and it does not reproduce the paper's numerical results. Connecting these assets to the core requires a compatible simulator, input adapter, training loop, and evaluation protocol, which are not included. The mathematical guidance provides clipped resource targets rather than a guarantee of global optimality.

## AI City dataset

The source dataset is **AICity22_Track1_MTMC_Tracking**, the 2022 AI City Challenge Track 1 multi-target multi-camera vehicle-tracking dataset.

- Official 2022 data/access page: https://www.aicitychallenge.org/2022-data-and-evaluation/
- Official challenge homepage: https://www.aicitychallenge.org/

Obtain the dataset through the official access process and comply with its license. 

The 15 selected source cameras have resolutions 1920x1080 (6 cameras), 2560x1920 (5), and 1600x1200 (4). AccMPEG inputs were resized to 1280x720. See `provenance/video_to_cache.csv` for exact camera/cache associations.

## Processed cache format

Each NPZ contains `importance_maps` and metadata such as `frame_sizes`, `fps`, `video_paths`, `tile_size`, and `source_resolution`; some additionally contain `frame_data_sizes` and `frame_data_size_source`. Read the exact per-file array schema in `provenance/input_selection.json`; do not assume optional fields exist in every file. Cached `source_resolution` describes the resized predictor input, not necessarily the raw video resolution.

## Workload

`ECW_newapp.pkl` is the original workload asset. The prepared trace selects sample 0, feature 0 (720 entries) and applies the normalization recorded in `workload/processing.json`. For ordinary reading, prefer the numeric NPY or CSV; pickle files require trusted inputs.

## Scope and limitations

Only seed 41/default-N15 results are included. These files do not establish the paper's five-seed 95% intervals, independent frozen-policy test protocol, other budgets/scales, or all ablation/sensitivity results. 
