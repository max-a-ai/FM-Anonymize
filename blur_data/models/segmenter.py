"""SAM3 text-prompted instance segmentation (image model).

Wraps ``sam3.model.sam3_image_processor.Sam3Processor``: one backbone
pass per image window, then one cheap grounding pass per prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from blur_data.models.detection import (
    Box,
    Detection,
    Window,
    dedupe,
    place,
    square_tiles,
    touches_inner_edge,
)


class Sam3Segmenter:
    """Detect every instance of a text prompt, with masks.

    ``thresholds`` maps prompt -> minimum score; SAM3's own threshold is
    set to the lowest of them and each prompt is filtered afterwards.
    """

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=device,
        )
        self.device = device
        self.processor = Sam3Processor(
            model, device=device, confidence_threshold=0.05
        )

    def _run(
        self, image: Image.Image, thresholds: dict[str, float], source: str
    ) -> list[Detection]:
        """All prompts on one window; boxes/masks in window coords."""
        dets: list[Detection] = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state: dict[str, Any] = self.processor.set_image(image)
            for prompt, thr in thresholds.items():
                self.processor.reset_all_prompts(state)
                out = self.processor.set_text_prompt(prompt, state)
                scores = out["scores"].float().cpu().numpy()
                boxes = out["boxes"].float().cpu().numpy()
                masks = out["masks"][:, 0].cpu().numpy()
                for s, b, m in zip(scores, boxes, masks, strict=True):
                    if s >= thr:
                        box: Box = (
                            float(b[0]),
                            float(b[1]),
                            float(b[2]),
                            float(b[3]),
                        )
                        dets.append(
                            Detection(prompt, float(s), box, m, source)
                        )
        return dets

    def detect(
        self,
        image: Image.Image,
        thresholds: dict[str, float],
        tiles: bool = True,
        iou_thr: float = 0.5,
    ) -> dict[str, list[Detection]]:
        """Full frame plus square tiles, merged per prompt."""
        frame = image.size
        found = self._run(image, thresholds, "full")
        windows = square_tiles(*frame) if tiles else []
        for i, win in enumerate(windows):
            for d in self._run(image.crop(win), thresholds, f"tile{i}"):
                box = _shift(d.box, win)
                if touches_inner_edge(box, win, frame):
                    continue
                found.append(
                    Detection(
                        d.prompt,
                        d.score,
                        box,
                        place(d.mask, win, frame),
                        d.source,
                    )
                )
        return {
            p: dedupe([d for d in found if d.prompt == p], iou_thr)
            for p in thresholds
        }

    def detect_in_box(
        self,
        image: Image.Image,
        box: Box,
        thresholds: dict[str, float],
        pad: float = 0.1,
    ) -> list[Detection]:
        """Run the prompts on a padded crop around ``box`` (zoom-in)."""
        win = crop_window(box, image.size, pad)
        out: list[Detection] = []
        for d in self._run(image.crop(win), thresholds, "crop"):
            out.append(
                Detection(
                    d.prompt,
                    d.score,
                    _shift(d.box, win),
                    place(d.mask, win, image.size),
                    "crop",
                )
            )
        return out


def crop_window(box: Box, frame: tuple[int, int], pad: float) -> Window:
    """Integer window around ``box`` grown by ``pad`` x its size."""
    w, h = frame
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = int(max(0, np.floor(box[0] - pad * bw)))
    y0 = int(max(0, np.floor(box[1] - pad * bh)))
    x1 = int(min(w, np.ceil(box[2] + pad * bw)))
    y1 = int(min(h, np.ceil(box[3] + pad * bh)))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _shift(box: Sequence[float], win: Window) -> Box:
    return (
        float(box[0]) + win[0],
        float(box[1]) + win[1],
        float(box[2]) + win[0],
        float(box[3]) + win[1],
    )
