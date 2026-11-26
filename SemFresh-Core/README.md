# PDS / STE / Mathematical Resource Guidance Core

A standalone core-method excerpt from a research implementation of semantic-freshness-aware scheduling. **This is not an experimental reproduction repository.**

## Included
- Differentiable post-decision state (PDS) construction and immediate ADoI/energy cost.
- Exact discrete PDS path, with aggregate-based hidden-window reconstruction.
- Hard forward decisions with sigmoid straight-through gradients (Hard-STE).
- CPU/power resource targets used for mathematical guidance.
- A small actor-loss composition function accepting a caller-owned value function.

## Excluded
No videos, importance caches, measured traces, trained weights, experiment outputs, simulator, task-generation pipeline, TD3 networks, replay buffer, optimizer/training loop, evaluation script, plotting code or paper configurations are distributed. Supplying these missing components is necessary to run actual experiments. Publishing the mechanism does not prevent independent reimplementation.

## Run the synthetic check
From this directory:

```sh
python -m pip install -r requirements.txt
python demo.py
```

The example uses two artificial users and two artificial tasks. It tests tensor shapes and gradients, not paper performance. Dependency ranges are compatibility bounds, not a promise that every release was tested.

## API
```python
from semantic_core import CoreConfig, PDSModel, core_actor_loss
# All physical parameters must be supplied by the caller.
model = PDSModel(config, e_max=per_user_slot_budgets)
post = model.build_pds(state, action)
target = model.get_math_optimal_action(state, action[:, :, 2:])
cost = model.estimate_cost(state, action, lagrange)
```

`state`: float32 `[B, N, 1 + 3*M + P + 2]`:
1. Channel gain multiplied by 1e10.
2. M triples: task size/d_ub, task age, semantic importance.
3. P optional prediction features passed through unchanged (no predictor supplied).
4. log1p(total queue load/d_ub), log1p(total queue semantic-age penalty).

`action`: float32 `[B, N, 2+M]`, in [-1, 1]. First two entries map affinely to CPU frequency and transmit power. Each remaining decision below -0.33 selects local service; above 0.33 selects offloading; otherwise defers. `lagrange` and `e_max` have shape `[N]`; budgets are energy **per slot**, consistent with d_t. Optional parent IDs, priorities and importances have shape `[B,N,M]`; trace indices have shape `[B]`.

Use consistent physical units: bits, seconds, Hz, watts, joules, cycles/bit. `gamma` is the CPU dynamic-power coefficient. `lambda_adoi` controls age decay. `F_edge_max` is edge CPU capacity; workload fractions lie in [0,1].

The differentiable path is a smooth surrogate and is not asserted equal to the discrete path. Guidance targets are detached in the loss; gradients still flow through the actor action. `value_fn` must return differentiable values from the supplied PDS. See SOURCE_PROVENANCE.md for inherited numerical/model limitations, including the guidance/rate interference-order discrepancy.
