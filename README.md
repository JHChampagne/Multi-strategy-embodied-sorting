# Multi-strategy embodied sorting

**SO-ARM100 七物料连续视觉分拣工作站**：三个容器、七个 24 mm 方块，
颜色数量随机分配为 3/2/2，位置和朝向随机。统一任务执行器管理整场任务，基础策略可插拔，
局部优化器预留 Residual SAC 和世界模型控制增强接口。

当前七物料能力、逐回合记录和限制见 [独立验收报告](docs/RESULTS_BASIC.md)。
[七物料演示](docs/media/sorting_basic.mp4) · [架构](docs/ARCHITECTURE.md) · [阶段里程碑](docs/MILESTONES.md)

## 当前能力

| 模块 | 七物料阶段范围 |
| --- | --- |
| Task Manager | 多实例跟踪、目标调度、三落点预留、视觉验证、重试、自动重置 |
| 传统视觉＋IK | 非特权 RGB-D 感知，连续多件抓放，独立分拣 |
| ACT / SmolVLA | 统一基础策略适配、episode/目标缓存重置；公开单物块权重不声明七物料能力 |
| 局部增强 | 默认 NoOp，阶段白名单、动作信任域和轨迹记录；本阶段未训练 Residual SAC |
| 世界模型 | 保留 M1 Reach 独立实验；抓取后果模型与局部 MPC 留待后续数据和训练 |

运行时不用仿真物体位姿、物体 ID、分割、接触或真值成功反馈。视觉完成判定与独立真值
评测分别记录，不用审计结果补做控制。源区、容器内物料都由 RGB-D 检测；视觉 ID 与仿真
实例 ID 分开。成功、失败和超时均自动进入下一回合，复用模型和渲染资源。

## 安装

Python 3.12、NVIDIA 驱动和单张可用 GPU。开发机已配置 `.venv`，可直接使用
`.venv/bin/workcell`。全新环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
pip install --no-deps -e .
workcell doctor
```

只运行几何基线可以 `pip install -e '.[test]'`；学习模块额外需要 `.[learning]`。
完整已测版本锁定在 `requirements.lock.txt`，上游代码/公开模型版本见 `upstream.lock.json`。
默认所有渲染、推理和训练串行使用物理 GPU 0，CLI 持有 GPU 文件锁。

## 七物料运行与验收

```bash
workcell calibrate
# 连续 20 场，固定独立验收种子 20000–20019，同一进程自动重置。
workcell evaluate --base-policy geometric --episodes 20 --seed 20000 --output outputs/basic --video
python scripts/acceptance_basic.py
```

`bash scripts/validate_basic.sh` 顺序运行依赖/代码检查、CPU 测试、标定、真实渲染测试和
20 场验收。门槛是至少 18/20 整场成功、133/140 单件正确，所有失败均计入分母。
输出包括 `report.json`、`manifest.json`、逐步 `traces/*.jsonl` 和首场视频。
开发布局用 0–9999，验收布局用 20000–20019；不是跨扰动或真机泛化结果。

```bash
# 开发试跑，使用单独输出目录。
workcell evaluate --base-policy geometric --episodes 3 --seed 0 --output outputs/dev
# 公开权重尚未适配七物料：此命令会明确报告 not ready。
workcell evaluate --base-policy act --output outputs/act-basic
# 未来经七物料训练的本地模型，须提供 task_capabilities.json、权重及处理器。
workcell evaluate --base-policy smolvla --checkpoint checkpoints/sorting-vla --output outputs/vla-basic
```

七物料容器扩宽到 120 mm，并配置三个分离落点，给活动夹指留净空；光照、摩擦、形状固定。
方块采样失败明确报告初始化失败，不减少数量。几何策略使用适合方块宽度的预抓开度，
检查夹指闭合扫掠、用编码器检测抓空，最多两次额外重试；无专家兜底。

## 保留的 M1 任务与历史

`v0.1.0-m1` 保留上阶段历史，三物料和单物块通过显式任务运行。
M1 使用原尺寸托盘和原相机配置，需要单独标定：

```bash
workcell calibrate --task sorting
workcell evaluate --task sorting --policy vision --output outputs/m1/vision
workcell evaluate --task sorting --policy expert --output outputs/m1/expert
workcell evaluate --task sorting --policy hybrid --checkpoint outputs/world_model/final.pt --output outputs/m1/hybrid
workcell evaluate --task pickplace_compat --base-policy act --output outputs/m1/act
workcell evaluate --task pickplace_compat --base-policy smolvla --output outputs/m1/vla
workcell train-rl --task reach --output outputs/m1/rl_reach
workcell train-rl --task lift --output outputs/m1/rl_lift
workcell record
```

历史 M1 结果：专家/视觉/hybrid 三物料各 20/20；8,000 交互训练的世界模型 Reach 20/20；
公开 ACT/SmolVLA 单物块 20/20、18/20，直接三物料迁移均 0/5；SAC Reach/Lift
各 4,096 步仅验证训练/恢复，任务成功均 0/5；LeRobot 示范 10 条读回通过。
这些数字**不代表七物料结果**。完整证据见 [M1 历史报告](docs/RESULTS.md)。

数据、权重、训练产物及完整轨迹留在忽略的 `outputs/`、`data/`、`checkpoints/`。
选定的小型报告和演示进入仓库。项目基于 Ahmed Sohail Butt 的 WorkCell A–D，保留
Apache-2.0 署名与许可证；来源及修改见 `NOTICE`。初期仿真项目，不涉及真机部署。

CI 模板位于 `configs/github-actions-ci.yml`，当前未启用远程 workflow；本地验证命令可直接运行。
