"""Frame sequences, segments to render, and head tracks.

A person keeps one synthetic identity while it is visible: the heads of
consecutive frames of one camera are linked into tracks, and every head
of a track gets the track's seed (:func:`blur_data.generate.routing.
track_seed`). Heads are linked on their person boxes (``link_on:
person``; the head region blinks with the facing decision, the person
box does not) or on the head regions themselves (``link_on: head``).

The frames to render are grouped into *segments*: runs of consecutive
frames of one sequence (camera, or PP4AV city) that cover every output
frame plus ``--window BEFORE AFTER`` frames of context, so a track can
reach over the output frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from blur_data.metrics.reid import box_iou

Box = tuple[float, float, float, float]
#: one track: (frame index, box index in that frame) per frame
Track = list[tuple[int, int]]
#: a frame of a person track without a generated head (context only)
NO_HEAD = -1


def sequences(
    dataset: str, data: Path, regions_dir: Path
) -> dict[str, list[str]]:
    """Sequence (camera, or PP4AV city) -> its keys in time order.

    Our recordings: every camera image of ``data``; PP4AV: the region
    files of the regions run, by city folder.
    """
    out: dict[str, list[str]] = {}
    if dataset == "v2b":
        from blur_data.data.v2b import list_camera_images

        for im in list_camera_images(data):
            out.setdefault(f"{im.scene}/{im.channel}", []).append(im.key)
        return out
    for p in sorted((regions_dir / "regions").glob("*/*.json")):
        out.setdefault(p.parent.name, []).append(f"{p.parent.name}/{p.stem}")
    return out


def segments(
    seqs: dict[str, list[str]],
    wanted: list[str],
    before: int,
    after: int,
    active: set[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Runs of consecutive frames covering every wanted key +- window.

    Keys not in ``active`` (no generated region) need no context.
    """
    where = {k: (s, i) for s, keys in seqs.items() for i, k in enumerate(keys)}
    need: dict[str, set[int]] = {}
    for k in wanted:
        s, i = where[k]
        b, a = (before, after) if active is None or k in active else (0, 0)
        lo, hi = max(0, i - b), min(len(seqs[s]) - 1, i + a)
        need.setdefault(s, set()).update(range(lo, hi + 1))
    out = []
    for s, idx in sorted(need.items()):
        run: list[int] = []
        for i in sorted(idx):
            if run and i != run[-1] + 1:
                out.append((s, [seqs[s][j] for j in run]))
                run = []
            run.append(i)
        out.append((s, [seqs[s][j] for j in run]))
    return out


def grown(box: Box, scale: float) -> Box:
    """``box`` scaled by ``scale`` about its centre."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = scale * (box[2] - box[0]) / 2, scale * (box[3] - box[1]) / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def duplicates_in(boxes: Sequence[Box], iou: float = 0.7) -> set[int]:
    """Boxes that repeat an earlier box of the same frame (one head found
    twice, e.g. as head and occupant): left out of the tracks."""
    return {
        j
        for j in range(len(boxes))
        if any(box_iou(boxes[i], boxes[j]) >= iou for i in range(j))
    }


def link(
    frames: Sequence[Sequence[Box]], min_iou: float, scale: float = 1.0
) -> list[Track]:
    """Link boxes of consecutive frames into tracks by IoU.

    A box continues the track whose box in the previous frame overlaps
    it best (>= ``min_iou``, at least 2x the runner-up unless that is a
    duplicate of the best, each track and box used once, best pairs
    first); otherwise it starts a new track. ``scale`` > 1 compares
    boxes grown about their centres (a search region for fast motion,
    e.g. PP4AV's 5-frame steps). A box that repeats an earlier one of
    its frame (:func:`duplicates_in`) is left out.
    """
    dup = [duplicates_in(boxes) for boxes in frames]
    frames = [[grown(b, scale) for b in boxes] for boxes in frames]
    tracks: list[Track] = []
    open_: dict[int, int] = {}  # box index in the previous frame -> track
    for f, boxes in enumerate(frames):
        prev = frames[f - 1] if f else []
        pairs = []
        for b, box in enumerate(boxes):
            if b in dup[f]:
                continue
            ious = sorted(
                ((box_iou(p, box), j) for j, p in enumerate(prev)),
                reverse=True,
            )
            # a runner-up that duplicates the best box (one head found
            # twice, e.g. as head and occupant) is no ambiguity
            clear = (
                len(ious) <= 1
                or ious[1][0] * 2 <= ious[0][0]
                or box_iou(prev[ious[0][1]], prev[ious[1][1]]) >= 0.5
            )
            if ious and ious[0][0] >= min_iou and clear:
                pairs.append((ious[0][0], ious[0][1], b))
        now: dict[int, int] = {}
        used = set()
        for _, j, b in sorted(pairs, reverse=True):
            if j in open_ and j not in used and b not in now:
                used.add(j)
                now[b] = open_[j]
                tracks[open_[j]].append((f, b))
        for b in range(len(boxes)):
            if b not in now and b not in dup[f]:
                now[b] = len(tracks)
                tracks.append([(f, b)])
        open_ = now
    return tracks


def person_tracks(
    persons: Sequence[Sequence[Box]],
    heads: Sequence[Sequence[Box | None]],
    min_iou: float,
    scale: float = 1.0,
) -> list[Track]:
    """Head tracks linked through their PERSON boxes.

    Persons are linked (:func:`link`) and each person frame holds the
    index of its head (the head whose ``person_box`` matches, IoU >=
    0.5) or :data:`NO_HEAD`. Frames without a head at either end are
    dropped; heads without a person box (occupants) are left out.
    """
    out = []
    for tr in link(persons, min_iou, scale):
        entries = []
        for fi, pi in tr:
            best, ri = 0.5, NO_HEAD
            for j, pb in enumerate(heads[fi]):
                if pb is not None:
                    iou = box_iou(persons[fi][pi], pb)
                    if iou >= best:
                        best, ri = iou, j
            entries.append((fi, ri))
        has = [i for i, (_, ri) in enumerate(entries) if ri != NO_HEAD]
        if has:
            out.append(entries[has[0] : has[-1] + 1])
    return out


def head_tracks(
    heads: Sequence[Sequence[Box]],
    persons: Sequence[Sequence[Box]],
    head_persons: Sequence[Sequence[Box | None]],
    link_on: str,
    min_iou: float,
    scale: float = 1.0,
) -> list[Track]:
    """The head tracks of a segment.

    ``heads``: per frame the boxes of the generated heads;
    ``persons``: the person detections; ``head_persons``: per generated
    head its person box (None for occupants). With ``link_on: person``
    the heads are linked through their persons, and the heads left over
    (occupants, heads without a person track) on their own boxes.
    """
    if link_on != "person":
        return link(heads, min_iou, scale)
    tracks = person_tracks(persons, head_persons, min_iou, scale)
    used = {(fi, ri) for t in tracks for fi, ri in t if ri != NO_HEAD}
    far = (-1e9, -1e9, -1e9 + 1.0, -1e9 + 1.0)  # out of reach: not linked
    rest = [
        [far if (fi, ri) in used else b for ri, b in enumerate(bs)]
        for fi, bs in enumerate(heads)
    ]
    for tr in link(rest, min_iou, scale):
        left = [(fi, ri) for fi, ri in tr if (fi, ri) not in used]
        if left:
            tracks.append(left)
    return tracks
