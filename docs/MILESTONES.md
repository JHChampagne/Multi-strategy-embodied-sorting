# Milestones

| Stage | Deliverable | Exit condition |
| --- | --- | --- |
| M1 | Unified simulation prototype and public repository | Acceptance report passes; real model evaluations, trained MPC and demo dataset exist |
| M2 | ACT and SmolVLA trained for sorting | Balanced demonstrations; held-out sorting and instruction counterfactual evaluations |
| M3 | RL / model-based skill study | Three training seeds, fixed environment-interaction budgets, Reach/Lift comparisons, validated skill handovers |
| M4 | Engineering evidence and application portfolio | Perturbations, recovery, ablations, measured limits and interview-ready technical narrative |

M1 requires 20 sorting scenes each: expert >=90%, vision >=80%, hybrid >=70%.
World-model Reach must achieve >=80% on 20 independent targets. ACT/VLA each
require 20 real compatibility-task evaluations, with no M1 success threshold.
SAC Reach and Lift each require >=4096 training interactions, five evaluations,
changed parameters and checkpoint action parity. The dataset gate requires ten
successful exported demonstrations and readback.

Training seeds occupy 0–9999; sorting evaluation seeds are 10000–10019 and skill
evaluation seeds 50000–50019. The fixed test cases are a small regression set,
not a claim of broad statistical generalization. Later stages add independent
validation/test splits before tuning learning policies.

M1 does not claim trained multi-instruction behavior, solved RL grasping,
contact-rich world-model planning, sim-to-real transfer, deployment to humans or
quadrupeds, or production readiness. Resume statements should distinguish the
author's integration and new sorting/hybrid implementation from upstream
algorithms and public pretrained models. Add numerical claims only from the
published measured report.
