#!/usr/bin/env python3
"""
main_with_side_lid_fixed.py

Based on your main.py:
- Ribbed sphere with bending rib-axis field (cap_axis -> down_axis)
- Optional top cap suppression (for the white insert at north pole)
- NEW: Optional side lid seat placed on the "morph-start side" BUT lifted toward the pole
       (so it matches your small-circle location, not the equator).

Key NEW idea:
  morph_dir = normalize(cross(k, cap_axis))     # direction on the correct "side"
  lid_axis  = normalize(cos(gamma)*cap_axis + sin(gamma)*morph_dir)
where gamma = lid_center_from_cap_deg (e.g., 20–35 degrees).

Dependencies:
  - numpy
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ----------------------------
# Helpers
# ----------------------------

def normalize_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n if n != 0 else 1.0)

def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def axis_from_tilt_azimuth(tilt_deg: float, az_deg: float) -> np.ndarray:
    """
    tilt_deg: 0 => +Z. az_deg: direction around Z (0 => towards +X, 90 => towards +Y).
    """
    tilt = math.radians(tilt_deg)
    az = math.radians(az_deg)
    return normalize(np.array([
        math.sin(tilt) * math.cos(az),
        math.sin(tilt) * math.sin(az),
        math.cos(tilt)
    ], dtype=np.float64))

def choose_orthogonal_axis(a: np.ndarray) -> np.ndarray:
    """Pick any unit vector orthogonal to a (robust even if a ~ [0,0,1])."""
    a = normalize(a)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(a, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    k = np.cross(a, ref)
    return normalize(k)

def rotate_about_axis(v: np.ndarray, k: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Rodrigues rotation of a single vector v about fixed unit axis k by per-sample angle[].
    v,k are (3,), angle is (N,).
    Returns (N,3).
    """
    v = v.astype(np.float64)
    k = normalize(k.astype(np.float64))
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    kv = np.cross(k, v)  # (3,)
    kdotv = float(np.dot(k, v))
    return v[None, :] * c + kv[None, :] * s + k[None, :] * (kdotv * (1.0 - c))

def interp_axis_cap_to_down(cap_axis: np.ndarray, down_axis: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Interpolate cap_axis -> down_axis with a stable rotation.
    Handles antipodal case with choose_orthogonal_axis().
    """
    a = normalize(cap_axis)
    b = normalize(down_axis)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle_total = math.acos(dot)

    if angle_total < 1e-9:
        return np.repeat(a[None, :], len(s), axis=0)

    if abs(math.pi - angle_total) < 1e-6:
        k = choose_orthogonal_axis(a)
        angle = (math.pi * s).astype(np.float64)
        out = rotate_about_axis(a, k, angle)
        return normalize_rows(out)

    k = normalize(np.cross(a, b))
    angle = (angle_total * s).astype(np.float64)
    out = rotate_about_axis(a, k, angle)
    return normalize_rows(out)

def rib_profile(alpha: np.ndarray, ribs: int, sharpness: float) -> np.ndarray:
    """alpha azimuth; returns [0..1] (1=rib peak, 0=groove valley)."""
    return np.abs(np.cos((ribs * alpha) / 2.0)) ** sharpness


def morph_rotation_axis(cap_axis: np.ndarray, down_axis: np.ndarray) -> np.ndarray:
    """
    Reconstruct the SAME 'k' logic as interp_axis_cap_to_down() uses,
    so our lid ends up on the correct "morph-start side".
    """
    a = normalize(cap_axis)
    b = normalize(down_axis)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle_total = math.acos(dot)

    if angle_total < 1e-9:
        return choose_orthogonal_axis(a)

    if abs(math.pi - angle_total) < 1e-6:
        return choose_orthogonal_axis(a)

    cross_ab = np.cross(a, b)
    cn = float(np.linalg.norm(cross_ab))
    if cn < 1e-12:
        return choose_orthogonal_axis(a)
    return cross_ab / cn


# ----------------------------
# Parameters
# ----------------------------

@dataclass
class Params:
    radius: float = 20.0
    ribs: int = 32
    depth: float = 2.0
    sharpness: float = 3.2
    fade_power: float = 1.0
    squash_z: float = 1.0

    # Cap axis (white insert normal)
    cap_tilt_deg: float = 0.0
    cap_az_deg: float = 0.0

    # Smooth cap region around top insert
    cap_angle_deg: float = 0.0
    cap_blend_deg: float = 0.0
    cap_recess: float = 0.0

    # Axis bending control (0..1 from top to bottom)
    axis_blend_start: float = 0.0
    axis_blend_end: float = 1.0
    axis_blend_power: float = 0.1

    # NEW: Side lid seat near the morph-start side, but near the top
    lid_enable: bool = False
    lid_flip: bool = False
    lid_center_from_cap_deg: float = 25.0   # <-- THIS moves lid up/down (small circle = ~20..35)
    lid_angle_deg: float = 25.0             # patch radius
    lid_blend_deg: float = 7.0              # soft fade width
    lid_recess: float = 0.6                 # recess depth (0 disables)

    # Mesh resolution
    n_theta: int = 360
    n_phi: int = 180


# ----------------------------
# Mesh generation
# ----------------------------

def generate_mesh(p: Params) -> Tuple[np.ndarray, np.ndarray]:
    R = float(p.radius)
    ribs = int(p.ribs)
    depth = float(p.depth)
    sharpness = float(p.sharpness)
    fade_power = float(p.fade_power)
    squash = float(p.squash_z)

    n_theta = int(p.n_theta)
    n_phi = int(p.n_phi)

    cap_axis = axis_from_tilt_azimuth(p.cap_tilt_deg, p.cap_az_deg)
    down_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Sphere sampling (global)
    phi = np.linspace(0.0, math.pi, n_phi + 1)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    phi_rings = phi[1:-1]  # exclude poles
    th, ph = np.meshgrid(theta, phi_rings)

    # Unit directions d on the sphere
    dx = np.sin(ph) * np.cos(th)
    dy = np.sin(ph) * np.sin(th)
    dz = np.cos(ph)
    d = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3).astype(np.float64)

    # Blend parameter s based on global height z (top->bottom)
    t = (1.0 - d[:, 2]) * 0.5
    s = smoothstep(float(p.axis_blend_start), float(p.axis_blend_end), t)
    s = np.clip(s, 0.0, 1.0) ** float(p.axis_blend_power)

    # Interpolated axis per point
    axis_i = interp_axis_cap_to_down(cap_axis, down_axis, s)  # (N,3)

    # Build per-point orthonormal basis around axis_i to define azimuth alpha
    ref = np.zeros_like(axis_i)
    ref[:, 2] = 1.0
    near_parallel = np.abs(axis_i[:, 2]) > 0.95
    ref[near_parallel] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    v = np.cross(ref, axis_i)
    v = normalize_rows(v)
    w = np.cross(axis_i, v)

    dv = np.einsum("ij,ij->i", d, v)
    dw = np.einsum("ij,ij->i", d, w)
    alpha = np.arctan2(dw, dv)

    # Fade ribs near the local poles
    du = np.einsum("ij,ij->i", d, axis_i)
    beta = np.arccos(np.clip(du, -1.0, 1.0))
    fade = (np.sin(beta)) ** fade_power

    # Top cap suppression (white insert area)
    if p.cap_angle_deg > 0.0:
        cap_angle = math.radians(p.cap_angle_deg)
        cap_blend = math.radians(max(p.cap_blend_deg, 1e-6))
        beta_cap = np.arccos(np.clip(d @ cap_axis, -1.0, 1.0))
        cap_mask = smoothstep(cap_angle, cap_angle + cap_blend, beta_cap)  # 0 inside cap, 1 outside
        fade *= cap_mask

    # Side lid suppression (FIXED placement: same side as morph start, but near the pole)
    lid_mask: Optional[np.ndarray] = None
    if p.lid_enable and p.lid_angle_deg > 0.0:
        k = morph_rotation_axis(cap_axis, down_axis)          # interpolation rotation axis
        morph_dir = np.cross(k, normalize(cap_axis))          # tangent direction of rotation at the cap
        if float(np.linalg.norm(morph_dir)) < 1e-12:
            morph_dir = choose_orthogonal_axis(cap_axis)
        morph_dir = normalize(morph_dir)

        if p.lid_flip:
            morph_dir = -morph_dir

        # Move lid center UP toward the pole by gamma (instead of sitting on the equator).
        gamma = math.radians(p.lid_center_from_cap_deg)
        lid_axis = normalize(math.cos(gamma) * normalize(cap_axis) + math.sin(gamma) * morph_dir)

        lid_angle = math.radians(p.lid_angle_deg)
        lid_blend = math.radians(max(p.lid_blend_deg, 1e-6))
        beta_lid = np.arccos(np.clip(d @ lid_axis, -1.0, 1.0))
        lid_mask = smoothstep(lid_angle, lid_angle + lid_blend, beta_lid)  # 0 inside, 1 outside
        fade *= lid_mask

    prof = rib_profile(alpha, ribs=ribs, sharpness=sharpness)

    # Radius displacement: carve grooves inward
    r = R - depth * (1.0 - prof) * fade

    # Optional recess in the top cap area
    if p.cap_recess > 0.0 and p.cap_angle_deg > 0.0:
        cap_angle = math.radians(p.cap_angle_deg)
        cap_blend = math.radians(max(p.cap_blend_deg, 1e-6))
        beta_cap = np.arccos(np.clip(d @ cap_axis, -1.0, 1.0))
        recess_w = 1.0 - smoothstep(cap_angle, cap_angle + cap_blend, beta_cap)
        r -= float(p.cap_recess) * recess_w

    # Optional recess for side lid
    if p.lid_enable and p.lid_recess > 0.0 and lid_mask is not None:
        r -= float(p.lid_recess) * (1.0 - lid_mask)

    # Build vertices using displaced radius
    verts = (d * r[:, None]).astype(np.float32)
    verts[:, 2] *= squash

    # Add global poles
    north = np.array([[0.0, 0.0, R * squash]], dtype=np.float32)
    south = np.array([[0.0, 0.0, -R * squash]], dtype=np.float32)
    verts = np.vstack([verts, north, south])

    n_rings = len(phi_rings)
    north_i = n_rings * n_theta
    south_i = north_i + 1

    faces = []

    # Ring quads -> triangles
    for i in range(n_rings - 1):
        row0 = i * n_theta
        row1 = (i + 1) * n_theta
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            a = row0 + j
            b = row1 + j
            c = row1 + j1
            d0 = row0 + j1
            faces.append((a, b, c))
            faces.append((a, c, d0))

    # Caps
    first_row = 0
    for j in range(n_theta):
        j1 = (j + 1) % n_theta
        a = first_row + j
        b = first_row + j1
        faces.append((north_i, b, a))

    last_row = (n_rings - 1) * n_theta
    for j in range(n_theta):
        j1 = (j + 1) % n_theta
        a = last_row + j
        b = last_row + j1
        faces.append((south_i, a, b))

    return verts, np.asarray(faces, dtype=np.int32)


# ----------------------------
# STL export
# ----------------------------

def write_binary_stl(path: str, verts: np.ndarray, faces: np.ndarray, header_text: str = "ribbed_cactus_ball") -> None:
    header = header_text.encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    tris = verts[faces]

    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    n = np.cross(v1, v2)
    lens = np.linalg.norm(n, axis=1)
    lens[lens == 0] = 1.0
    n = (n.T / lens).T

    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", tris.shape[0]))
        for ni, tri in zip(n.astype(np.float32), tris.astype(np.float32)):
            f.write(struct.pack("<3f", float(ni[0]), float(ni[1]), float(ni[2])))
            f.write(struct.pack("<9f", *map(float, tri.reshape(-1))))
            f.write(struct.pack("<H", 0))


# ----------------------------
# CLI
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate a bending-rib cactus sphere and export STL.")
    ap.add_argument("--out", type=str, default="cactus_ball.stl")
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--ribs", type=int, default=32)
    ap.add_argument("--depth", type=float, default=2.0)
    ap.add_argument("--sharpness", type=float, default=3.2)
    ap.add_argument("--fade_power", type=float, default=1.0)
    ap.add_argument("--squash_z", type=float, default=1.0)
    ap.add_argument("--n_theta", type=int, default=360)
    ap.add_argument("--n_phi", type=int, default=180)

    ap.add_argument("--cap_tilt_deg", type=float, default=0.0)
    ap.add_argument("--cap_az_deg", type=float, default=0.0)

    ap.add_argument("--cap_angle_deg", type=float, default=0.0)
    ap.add_argument("--cap_blend_deg", type=float, default=0.0)
    ap.add_argument("--cap_recess", type=float, default=0.0)

    ap.add_argument("--axis_blend_start", type=float, default=0.0, help="0..1: where bending starts (top->bottom)")
    ap.add_argument("--axis_blend_end", type=float, default=1.0, help="0..1: where bending finishes")
    ap.add_argument("--axis_blend_power", type=float, default=0.1, help=">1 pushes bending later; <1 earlier")

    # NEW: Side lid seat controls (fixed placement)
    ap.add_argument("--lid_enable", action="store_true", default=True,
                    help="Enable a smooth/recessed side lid seat on the morph-start side (near the top).")
    ap.add_argument("--lid_flip", action="store_true", default=True,
                    help="Flip lid seat to the opposite side.")
    ap.add_argument("--lid_center_from_cap_deg", type=float, default=30.0,
                    help="How far down from the cap (deg) the lid center sits. (20..35 is typical)")
    ap.add_argument("--lid_angle_deg", type=float, default=30.0,
                    help="Angular radius of the lid seat patch (deg).")
    ap.add_argument("--lid_blend_deg", type=float, default=7.0,
                    help="Blend width for fading ribs into the lid seat patch (deg).")
    ap.add_argument("--lid_recess", type=float, default=0.2,
                    help="Recess depth for the lid seat (0 disables).")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    p = Params(
        radius=args.radius,
        ribs=args.ribs,
        depth=args.depth,
        sharpness=args.sharpness,
        fade_power=args.fade_power,
        squash_z=args.squash_z,
        cap_tilt_deg=args.cap_tilt_deg,
        cap_az_deg=args.cap_az_deg,
        cap_angle_deg=args.cap_angle_deg,
        cap_blend_deg=args.cap_blend_deg,
        cap_recess=args.cap_recess,
        axis_blend_start=args.axis_blend_start,
        axis_blend_end=args.axis_blend_end,
        axis_blend_power=args.axis_blend_power,
        lid_enable=args.lid_enable,
        lid_flip=args.lid_flip,
        lid_center_from_cap_deg=args.lid_center_from_cap_deg,
        lid_angle_deg=args.lid_angle_deg,
        lid_blend_deg=args.lid_blend_deg,
        lid_recess=args.lid_recess,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
    )

    verts, faces = generate_mesh(p)
    write_binary_stl(args.out, verts, faces)
    print(f"Wrote STL: {args.out}  (verts={len(verts)}, tris={len(faces)})")


if __name__ == "__main__":
    main()
