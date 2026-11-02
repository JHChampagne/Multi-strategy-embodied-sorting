# M1 measured results

Measured: 2026-11-02T18:18:03.479758+08:00

All task execution and learning jobs used physical GPU 0 (NVIDIA L40).
Physics uses CPU. These are 20 fixed regression scenes, not a generalization claim.

| Method | Task | Successful episodes | Status |
| --- | --- | --- | --- |
| expert | sorting | 20/20 | pass |
| vision | sorting | 20/20 | pass |
| hybrid | sorting | 20/20 | pass |
| act | pickplace_compat | 20/20 | pass |
| vla | pickplace_compat | 18/20 | pass |
| learned dynamics + CEM-MPC | Reach, home/rest starts | 20/20 | pass |
| SAC | reach, 4096 training steps | 0/5 | training/update/checkpoint smoke only |
| SAC | lift, 4096 training steps | 0/5 | training/update/checkpoint smoke only |

ACT and SmolVLA use upstream pretrained weights on their original single-block task.
Their scores do not establish multi-material sorting or general language understanding.
SAC's short training budget verifies the pipeline and is not presented as solving the skills.

## Hybrid execution evidence

Across 20 scenes, MPC executed 910 control actions; IK executed 10142. Recorded controller changes: 122. Expert fallbacks: 0.

MPC controls only open-gripper approach. Geometry/IK controls contact and transport.

## Demonstrations and reproduction

Exported and read back 10 LeRobot demonstrations (1280 frames), with task labels and front/wrist RGB.

Original locked Part A baseline: 20/20, using its original success predicate.
Original locked Part C baseline: 20/20, using its original success predicate.

## Diagnostic findings

- A front tray obstructed some jaw approaches; the final third tray is beside the arm, with verified reachable transport/lowering waypoints.
- Reach freezes the jaw. CEM must also freeze that action dimension before input normalization; otherwise zero-variance training inputs become extreme out-of-distribution planner inputs.
- Nominal 15 Hz control uses 33 steps at 0.002 s, so actual action duration is 0.066 s; dataset video metadata retains the upstream nominal 15 fps.

The selected world model was fitted to 8,000 real transitions. Additional diagnostic transitions not used by this checkpoint: 3,000. Total training interactions stayed within the 16,000 budget.

## Pretrained sorting transfer

| Policy | Complete scenes | Evaluation budget |
| --- | --- | --- |
| act | 0/5 | up to 900 actions per scene, no fine-tuning |
| vla | 0/5 | up to 900 actions per scene, no fine-tuning |

These limited transfer probes motivate M2 multi-material demonstrations and fine-tuning.

Full per-episode measurements, seeds, gates and versions: [results/m1.json](results/m1.json).

Overall M1 acceptance: **PASS**.
