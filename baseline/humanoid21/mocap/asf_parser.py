"""Parse CMU ASF (Acclaim Skeleton File) format.

ASF defines the skeleton: bone names, directions, lengths, axes, DOFs, hierarchy.
Each bone has:
  - direction: unit vector along bone in parent's local frame (before axis rotation)
  - length: bone length (in ASF units)
  - axis: rotation axis for the bone's local frame (Euler XYZ in degrees)
  - dof: degrees of freedom (e.g. "rx ry rz")
  - limits: angle limits per DOF
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Bone:
    name: str
    direction: np.ndarray       # (3,) unit vector
    length: float
    axis: np.ndarray            # (3,) Euler XYZ in degrees
    id: int = 0
    dof: List[str] = field(default_factory=list)
    limits: List[Tuple[float, float]] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    axis_mat: Optional[np.ndarray] = None  # (3,3) rotation from axis Euler

    @property
    def has_dof(self) -> bool:
        return len(self.dof) > 0


@dataclass
class Skeleton:
    bones: Dict[str, Bone] = field(default_factory=dict)
    root_order: str = "TX TY TZ RX RY RZ"
    root_axis: str = "XYZ"
    units_length: float = 0.45
    units_angle: str = "deg"

    def get_bone(self, name: str) -> Bone:
        return self.bones[name]


def _euler_xyz_to_mat(rx: float, ry: float, rz: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


def _parse_bone_block(lines: List[str], start: int) -> Tuple[dict, int]:
    """Parse a single bone 'begin ... end' block. Returns (bone_dict, next_index)."""
    bone = {}
    i = start
    while i < len(lines) and lines[i].strip() != "end":
        bline = lines[i].strip()
        if bline == "begin":
            i += 1
            continue
        parts = bline.split()
        key = parts[0]
        if key == "id":
            bone["id"] = int(parts[1])
        elif key == "name":
            bone["name"] = parts[1]
        elif key == "direction":
            bone["direction"] = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
        elif key == "length":
            bone["length"] = float(parts[1])
        elif key == "axis":
            bone["axis"] = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
        elif key == "dof":
            bone["dof"] = parts[1:]
        elif key == "limits":
            raw = bline
            while raw.count("(") != raw.count(")"):
                i += 1
                raw += " " + lines[i].strip()
            pairs = re.findall(r"\(([-\d.]+)\s+([-\d.]+)\)", raw)
            bone["limits"] = [(float(a), float(b)) for a, b in pairs]
        i += 1
    return bone, i + 1  # skip "end"


def parse_asf(path: str) -> Skeleton:
    """Parse an ASF file into a Skeleton."""
    skel = Skeleton()

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith(":units"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(":"):
                uline = lines[i].strip()
                if uline.startswith("length"):
                    skel.units_length = float(uline.split()[1])
                elif uline.startswith("angle"):
                    skel.units_angle = uline.split()[1]
                i += 1
            continue

        if line.startswith(":root"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(":"):
                rline = lines[i].strip()
                if rline.startswith("order"):
                    skel.root_order = " ".join(rline.split()[1:])
                elif rline.startswith("axis"):
                    skel.root_axis = rline.split()[1]
                i += 1
            continue

        if line == ":bonedata":
            i += 1
            while i < len(lines) and lines[i].strip() == "begin":
                bone_dict, i = _parse_bone_block(lines, i)
                axis_mat = _euler_xyz_to_mat(*bone_dict["axis"])
                bone = Bone(
                    name=bone_dict["name"],
                    direction=bone_dict["direction"],
                    length=bone_dict["length"],
                    axis=bone_dict["axis"],
                    id=bone_dict.get("id", 0),
                    dof=bone_dict.get("dof", []),
                    limits=bone_dict.get("limits", []),
                    axis_mat=axis_mat,
                )
                skel.bones[bone.name] = bone
            continue

        if line == ":hierarchy":
            i += 1
            if i < len(lines) and lines[i].strip() == "begin":
                i += 1
            while i < len(lines) and lines[i].strip() != "end":
                hline = lines[i].strip()
                parts = hline.split()
                parent_name = parts[0]
                child_names = parts[1:]
                if parent_name not in skel.bones:
                    skel.bones[parent_name] = Bone(
                        name=parent_name, direction=np.zeros(3), length=0, axis=np.zeros(3)
                    )
                for cn in child_names:
                    if cn not in skel.bones:
                        skel.bones[cn] = Bone(
                            name=cn, direction=np.zeros(3), length=0, axis=np.zeros(3)
                        )
                    skel.bones[cn].parent = parent_name
                    if cn not in skel.bones[parent_name].children:
                        skel.bones[parent_name].children.append(cn)
                i += 1
            i += 1
            continue

        i += 1

    if "root" not in skel.bones:
        skel.bones["root"] = Bone(name="root", direction=np.zeros(3), length=0, axis=np.zeros(3))

    return skel
