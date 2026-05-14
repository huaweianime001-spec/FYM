"""Cheap motion descriptors from a reference video (optical flow statistics)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def motion_description_from_video(
    video_path: str | Path,
    *,
    max_frames: int = 32,
    resize_width: int = 320,
) -> str:
    """
    Build an English suffix describing coarse motion (camera / subject movement).
    This is *not* the closed-form Follow-Your-Motion LoRA objective; it bootstraps a
    stronger teacher text embedding when the user motion prompt is vague.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    flows: list[float] = []
    angles: list[float] = []
    prev_gray = None
    count = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if count >= max_frames:
            break
        h, w = bgr.shape[:2]
        scale = resize_width / max(w, 1)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            flows.append(float(mag.mean()))
            angles.append(float(ang.mean()))
        prev_gray = gray
        count += 1

    cap.release()

    if not flows:
        return (
            "Motion cue: single frame or unreadable sequence; assume subtle idle motion and "
            "gentle camera drift."
        )

    mean_mag = float(np.mean(flows))
    std_mag = float(np.std(flows))
    mean_deg = float(np.degrees(np.mean(angles)))
    pan_hint = "horizontal-ish" if abs(np.cos(np.radians(mean_deg))) > abs(np.sin(np.radians(mean_deg))) else "vertical-ish"

    intensity = "very subtle"
    if mean_mag > 2.5:
        intensity = "strong"
    elif mean_mag > 1.2:
        intensity = "moderate"
    elif mean_mag > 0.5:
        intensity = "noticeable"

    return (
        f"Motion cue from reference clip: {intensity} optical-flow magnitude "
        f"(mean={mean_mag:.3f}, std={std_mag:.3f}), coarse direction bias {pan_hint}, "
        f"dominant flow angle ~ {mean_deg:.1f} deg. Match this pacing and directionality."
    )
