# goals/specs/

One YAML per goal, validated by `rsi_bench.goals.spec.GoalSpec`. Copy
`../GOAL_TEMPLATE.yaml`, rename to `<id>.yaml`, and fill it in as you validate goals
from the proposal. Use the `status` field (`proposed` → `validated` → `selected`, or
`rejected`) to track your pass.

A spec becomes a live `Goal` by pairing it with an oracle `ModelEval`:

```python
from rsi_bench.goals import GoalSpec, goal_from_spec
spec = GoalSpec.from_yaml("src/rsi_bench/goals/specs/g1_less_refusal.yaml")
goal = goal_from_spec(spec, my_refusal_oracle)   # my_refusal_oracle: ModelEval
```

The oracle (`probe_set` + `scorer`) is the per-goal code/data we build separately; the
secret held-out probe payloads must stay out of git (see the repo `.gitignore`).
