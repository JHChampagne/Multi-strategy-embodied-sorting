"""LeRobot checkpoint adapters, including the checkpoint's pre/post processors."""

import numpy as np

MODELS = {
    "act": ("ahmedsohail2003/act-so101-pickplace-dr", "fb128aa2f37a31324d2651bf3b8745717771a157"),
    "vla": ("ahmedsohail2003/smolvla-so101-pickplace-v2", "f43f9f3a1c81b050f9747919c53019fcae2caec0"),
}
VLM = ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467")


class LeRobotPolicy:
    supported_tasks = frozenset({"pickplace_compat", "sorting"})

    def __init__(self, kind="act", checkpoint=None, device="cuda"):
        import torch
        from lerobot.policies import make_pre_post_processors
        from huggingface_hub import snapshot_download

        self.kind = kind
        self.device = device
        torch.set_num_threads(4)
        if kind == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as PolicyClass
        elif kind == "vla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyClass
        else:
            raise ValueError(kind)
        repo, revision = MODELS[kind]
        if checkpoint is None:
            checkpoint = snapshot_download(
                repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]
            )
        self.checkpoint = str(checkpoint)
        config = PolicyClass.config_class.from_pretrained(self.checkpoint)
        config.device = device
        overrides = {"device_processor": {"device": device}}
        if kind == "act":
            config.pretrained_backbone_weights = None  # complete backbone is in the checkpoint
        else:
            base = snapshot_download(VLM[0], revision=VLM[1], allow_patterns=["*.json", "*.model", "*.txt"])
            config.vlm_model_name = base
            config.load_vlm_weights = False  # build architecture, then strictly restore the complete policy
            overrides["tokenizer_processor"] = {"tokenizer_name": base}
        self.policy = (
            PolicyClass.from_pretrained(self.checkpoint, config=config, strict=True).to(device).eval()
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=self.checkpoint, preprocessor_overrides=overrides
        )

    def reset(self, context):
        import torch

        torch.manual_seed(context.seed)
        self.context = context
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def act(self, observation):
        import torch
        from ..types import ActionCommand

        batch = {
            "observation.state": torch.from_numpy(observation["state"]).unsqueeze(0).to(self.device),
            "task": [self.context.task.instruction],
        }
        for camera, image in observation["images"].items():
            batch[f"observation.images.{camera}"] = (
                torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255
            )
        with torch.inference_mode():
            batch = self.preprocessor(batch)
            action = self.postprocessor(self.policy.select_action(batch))
        values = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return ActionCommand(values, controller=self.kind, phase="learned_pickplace")
