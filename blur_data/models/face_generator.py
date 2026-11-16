"""The diffusion model of FM-Anonymize-Gen (diffusers, fp16).

:class:`FaceGenerator` edits one head crop: SD 1.5 inpainting with two
ControlNets (template-head depth, OpenPose) and IP-Adapter-FaceID
plus-v2, started from the crop as given (SDEdit at ``strength``).

The identity comes from a *synthetic reference face*: a second, plain
SD 1.5 inpainting pipeline paints a portrait from the seed alone (full
mask on grey, no image content), its ArcFace embedding and aligned crop
condition FaceID. One seed per track gives one face per track; the
seed -> identity cache lives in memory only. The plain pipeline is
separate because the FaceID adapter changes the attention layers of
the guided pipeline's UNet. The NSFW safety checker of SD 1.5 is not
loaded: it blacks out false positives on face crops, and the output is
only ever pasted inside a head mask.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from blur_data.anonymize.blur import U8
from blur_data.generate.config import GenConfig, split_ref
from blur_data.generate.pose_guide import Guide
from blur_data.utils.masks import BoolMask

#: synthetic reference faces and identities kept in memory (by seed)
CACHE = 256
#: reference seeds tried per identity until SCRFD finds a face in it
REFERENCE_TRIES = 4


def _scheduler(pipe: Any, cfg: GenConfig) -> None:
    from diffusers import DPMSolverMultistepScheduler

    if cfg.scheduler == "dpmpp":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    elif cfg.scheduler != "default":
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


def load_image_encoder(cfg: GenConfig) -> Any:
    """The CLIP image encoder of the FaceID plus-v2 adapter, at the
    pinned revision (diffusers' ``load_ip_adapter`` would load it from
    "main", which fails offline)."""
    import torch
    from transformers import CLIPVisionModelWithProjection

    repo, rev = split_ref(cfg.image_encoder)
    return CLIPVisionModelWithProjection.from_pretrained(
        repo,
        subfolder="models/image_encoder",
        revision=rev,
        dtype=torch.float16,
    )


def _place(pipe: Any, device: str, offload: bool) -> Any:
    pipe.set_progress_bar_config(disable=True)
    if offload:  # the GPU is shared with SAM 3D Body
        pipe.enable_model_cpu_offload(device=device)
        return pipe
    return pipe.to(device)


class FaceGenerator:
    """A seeded ``(crop, mask, seed, guide) -> crop`` head generator."""

    def __init__(
        self,
        cfg: GenConfig,
        arcface: Path,
        device: str = "cuda",
        offload: bool = True,
    ) -> None:
        import torch
        from diffusers import (
            AutoPipelineForInpainting,
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
        )

        from blur_data.models.faceid import FaceEmbedder

        fp16: dict[str, Any] = {
            "torch_dtype": torch.float16,
            "variant": "fp16",
        }
        no_checker = {"safety_checker": None, "requires_safety_checker": False}
        # plain inpainting pipeline: paints the synthetic reference faces
        plain = AutoPipelineForInpainting.from_pretrained(
            cfg.model, revision=cfg.revision, **fp16, **no_checker
        )
        _scheduler(plain, cfg)
        self.plain = _place(plain, device, offload)
        # guided pipeline: depth + OpenPose ControlNets, FaceID
        nets = [
            ControlNetModel.from_pretrained(repo, revision=rev, **fp16)
            for repo, rev in map(
                split_ref, (cfg.controlnet_depth, cfg.controlnet_pose)
            )
        ]
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            cfg.model,
            revision=cfg.revision,
            controlnet=nets,
            image_encoder=load_image_encoder(cfg),
            **fp16,
            **no_checker,
        )
        _scheduler(pipe, cfg)
        repo, rev = split_ref(cfg.faceid_adapter)
        pipe.load_ip_adapter(
            repo,
            subfolder=None,
            weight_name=cfg.faceid_weight,
            image_encoder_folder=None,  # the pinned encoder above
            revision=rev,
        )
        pipe.set_ip_adapter_scale(cfg.identity_scale)
        self.pipe = _place(pipe, device, offload)
        self.embedder = FaceEmbedder(arcface)
        self.cfg = cfg
        self.refs: OrderedDict[int, Image.Image] = OrderedDict()
        self.ids: OrderedDict[int, tuple[Any, Any]] = OrderedDict()

    def reference(self, seed: int) -> Image.Image:
        """The synthetic reference face of ``seed`` (text to image: full
        mask on grey, strength 1.0), cached."""
        if seed in self.refs:
            self.refs.move_to_end(seed)
            return self.refs[seed]
        import torch

        res = self.cfg.resolution
        grey = np.full((res, res, 3), 127, np.uint8)
        out: Image.Image = self.plain(
            prompt=self.cfg.reference_prompt,
            negative_prompt=self.cfg.negative_prompt or None,
            image=Image.fromarray(grey),
            mask_image=Image.fromarray(np.full((res, res), 255, np.uint8)),
            height=res,
            width=res,
            strength=1.0,
            num_inference_steps=self.cfg.reference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="pil",
        ).images[0]
        self.refs[seed] = out
        if len(self.refs) > CACHE:
            self.refs.popitem(last=False)
        return out

    def identity(self, seed: int) -> tuple[Any, Any]:
        """FaceID inputs (ArcFace and CLIP embeddings) of the synthetic
        face of ``seed``, cached; the first of a few derived seed
        variants whose reference shows a detectable face."""
        if seed in self.ids:
            return self.ids[seed]
        import torch

        for k in range(REFERENCE_TRIES):
            ref = self.reference((seed + 7919 * k) & (2**63 - 1))
            emb, crop, found = self.embedder(np.asarray(ref.convert("RGB")))
            if found:
                break
        dev, dt = torch.device("cuda"), torch.float16
        pos = torch.from_numpy(emb)[None, None, None]  # 1 x 1 x 1 x 512
        id_embeds = torch.cat([torch.zeros_like(pos), pos]).to(dev, dt)
        with torch.no_grad():  # outside the pipeline call: no graph kept
            clip = (
                self.pipe.prepare_ip_adapter_image_embeds(
                    [Image.fromarray(crop)], None, dev, 1, True
                )[0]
                .to(dt)
                .detach()
            )
        self.ids[seed] = (id_embeds, clip)
        if len(self.ids) > CACHE:
            self.ids.popitem(last=False)
        return id_embeds, clip

    def __call__(
        self, image: U8, mask: BoolMask, seed: int, guide: Guide | None = None
    ) -> U8:
        """Edit ``mask`` of the square ``image`` (SDEdit at
        ``strength``); without a ``guide`` the ControlNets are off
        (scale 0) and only the identity steers the head."""
        import torch

        cfg, res = self.cfg, self.cfg.resolution
        if image.shape[:2] != (res, res):
            raise ValueError(f"expected {res}x{res}, got {image.shape}")
        on = 1.0
        if guide is None:
            blank = np.zeros((res, res, 3), np.uint8)
            guide, on = Guide(blank, blank), 0.0
        id_embeds, clip = self.identity(seed)
        proj = self.pipe.unet.encoder_hid_proj.image_projection_layers[0]
        proj.clip_embeds, proj.shortcut = clip, True  # plus-v2
        out = self.pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt or None,
            image=Image.fromarray(image),
            mask_image=Image.fromarray(mask.astype(np.uint8) * 255),
            control_image=[
                Image.fromarray(guide.depth),
                Image.fromarray(guide.pose),
            ],
            controlnet_conditioning_scale=[
                on * cfg.depth_scale,
                on * cfg.pose_scale,
            ],
            ip_adapter_image_embeds=[id_embeds],
            height=res,
            width=res,
            strength=cfg.strength,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images[0]
        arr: U8 = np.clip(np.round(out * 255), 0, 255).astype(np.uint8)
        return arr
