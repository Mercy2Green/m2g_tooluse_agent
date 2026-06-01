from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdPhysics


M2G_TOOLUSE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USD_PATH = M2G_TOOLUSE_ROOT / "assets" / "robots" / "Go2Piper" / "go2_piper_v1.usd"
DEFAULT_SANITIZED_USD_PATH = (
    M2G_TOOLUSE_ROOT / "assets" / "robots" / "Go2Piper" / "go2_piper_v1_train_sanitized.usd"
)
DEFAULT_REPORT_DIR = M2G_TOOLUSE_ROOT / "logs" / "go2_piper_asset_sanity"


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value]


def _is_non_finite(values: list[float] | None) -> bool:
    return values is not None and any(not math.isfinite(x) for x in values)


def _is_zero_vec(values: list[float] | None, eps: float = 0.0) -> bool:
    return values is not None and len(values) == 3 and all(abs(x) <= eps for x in values)


def _body_name(root_path: str, prim_path: str) -> str:
    if "/piper/" in prim_path:
        return "piper/" + prim_path.split("/piper/", 1)[1]
    prefix = root_path.rstrip("/") + "/"
    rel = prim_path.removeprefix(prefix)
    return rel.split("/")[-1]


def _mass_api(prim):
    if prim.HasAPI(UsdPhysics.MassAPI):
        return UsdPhysics.MassAPI(prim)
    return None


def _get_mass(prim) -> float | None:
    api = _mass_api(prim)
    if api is None:
        return None
    value = api.GetMassAttr().Get()
    return None if value is None else float(value)


def _get_vec_attr(api, attr_name: str) -> list[float] | None:
    value = getattr(api, attr_name)().Get()
    return _as_float_list(value)


def _classify_body(body_name: str, prim_path: str) -> str:
    lowered = f"{body_name} {prim_path}".lower()
    if "/piper/" in lowered or body_name.startswith("piper/"):
        return "piper"
    return "go2"


def build_usd_report(usd_path: str | Path) -> dict[str, Any]:
    usd_path = Path(usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    default_path = str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else ""
    articulation_roots: list[str] = []
    rigid_bodies: list[dict[str, Any]] = []
    joints: list[dict[str, str]] = []
    fixed_joints: list[dict[str, Any]] = []

    root_path = f"{default_path}/go2" if default_path else ""

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim_path)

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = _mass_api(prim)
            mass = _get_mass(prim)
            body_name = _body_name(root_path, prim_path) if root_path else prim.GetName()
            center_of_mass = _get_vec_attr(api, "GetCenterOfMassAttr") if api else None
            diagonal_inertia = _get_vec_attr(api, "GetDiagonalInertiaAttr") if api else None
            rigid_bodies.append(
                {
                    "path": prim_path,
                    "name": prim.GetName(),
                    "body_name": body_name,
                    "group": _classify_body(body_name, prim_path),
                    "mass": mass,
                    "centerOfMass": center_of_mass,
                    "diagonalInertia": diagonal_inertia,
                }
            )

        if prim.IsA(UsdPhysics.Joint):
            joints.append({"path": prim_path, "name": prim.GetName(), "type": prim.GetTypeName()})

        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.Joint(prim)
            body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            fixed_joints.append({"path": prim_path, "name": prim.GetName(), "body0": body0, "body1": body1})

    body_names = [body["body_name"] for body in rigid_bodies]
    joint_names = [joint["name"] for joint in joints]
    actuated_joints = [
        name
        for name in joint_names
        if name.endswith("_hip_joint")
        or name.endswith("_thigh_joint")
        or name.endswith("_calf_joint")
        or name in {f"joint{i}" for i in range(1, 9)}
    ]
    non_finite_com_bodies = [
        body for body in rigid_bodies if _is_non_finite(body.get("centerOfMass"))
    ]
    zero_inertia_bodies = [
        body for body in rigid_bodies if _is_zero_vec(body.get("diagonalInertia"), eps=0.0)
    ]

    piper_mass = sum(body["mass"] or 0.0 for body in rigid_bodies if body["group"] == "piper")
    go2_mass = sum(body["mass"] or 0.0 for body in rigid_bodies if body["group"] == "go2")
    total_mass = piper_mass + go2_mass

    fixed_base_to_arm_base = [
        joint
        for joint in fixed_joints
        if any(str(target).endswith("/go2/base") or str(target).endswith("/base") for target in joint["body0"] + joint["body1"])
        and any("piper/arm_base" in str(target) for target in joint["body0"] + joint["body1"])
    ]

    arm_base = [body for body in rigid_bodies if body["body_name"] == "piper/arm_base" or body["name"] == "arm_base"]
    warnings: list[str] = []
    for body in arm_base:
        if (body["mass"] or 0.0) > 5.0:
            warnings.append(f"piper/arm_base mass is {body['mass']} kg (> 5 kg); confirm this is intentional.")
    if non_finite_com_bodies:
        warnings.append("One or more rigid bodies have non-finite centerOfMass values.")
    if zero_inertia_bodies:
        warnings.append("One or more rigid bodies have exactly [0, 0, 0] diagonalInertia.")

    return {
        "usd_path": str(usd_path),
        "default_prim": default_path,
        "articulation_roots": articulation_roots,
        "rigid_body_count": len(rigid_bodies),
        "joint_count": len(joints),
        "rigid_bodies": rigid_bodies,
        "body_names": body_names,
        "joint_names": joint_names,
        "actuated_joints": actuated_joints,
        "body_names_containing_base": [name for name in body_names if "base" in name],
        "body_names_containing_foot": [name for name in body_names if "foot" in name],
        "body_names_containing_thigh": [name for name in body_names if "thigh" in name],
        "body_names_containing_calflower": [name for name in body_names if "calflower" in name],
        "non_finite_centerOfMass_bodies": non_finite_com_bodies,
        "zero_or_near_zero_inertia_bodies": [
            body for body in rigid_bodies if _is_zero_vec(body.get("diagonalInertia"), eps=1.0e-12)
        ],
        "exact_zero_diagonalInertia_bodies": zero_inertia_bodies,
        "piper_total_mass": piper_mass,
        "go2_total_mass": go2_mass,
        "total_robot_mass": total_mass,
        "fixed_joint_from_go2_base_to_piper_arm_base": fixed_base_to_arm_base,
        "warnings": warnings,
    }


def save_json_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_usd(
    input_path: str | Path,
    output_path: str | Path,
    *,
    scale_piper_mass: float | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input USD does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite the source USD.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)

    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Unable to open copied USD for editing: {output_path}")

    edits: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = _mass_api(prim)
        if api is None:
            continue
        prim_path = str(prim.GetPath())
        mass_attr = api.GetMassAttr()
        mass = mass_attr.Get()
        mass_value = 0.0 if mass is None else float(mass)

        com_attr = api.GetCenterOfMassAttr()
        com = _as_float_list(com_attr.Get())
        if _is_non_finite(com):
            com_attr.Set((0.0, 0.0, 0.0))
            edits.append({"path": prim_path, "field": "centerOfMass", "old": com, "new": [0.0, 0.0, 0.0]})

        inertia_attr = api.GetDiagonalInertiaAttr()
        inertia = _as_float_list(inertia_attr.Get())
        if mass_value > 0.0 and _is_zero_vec(inertia, eps=0.0):
            inertia_floor = max(1.0e-6, mass_value * 1.0e-4)
            new_inertia = [inertia_floor, inertia_floor, inertia_floor]
            inertia_attr.Set(tuple(new_inertia))
            edits.append({"path": prim_path, "field": "diagonalInertia", "old": inertia, "new": new_inertia})

        if scale_piper_mass is not None and "/piper/" in prim_path:
            old_mass = mass_value
            new_mass = old_mass * scale_piper_mass
            mass_attr.Set(new_mass)
            edits.append(
                {
                    "path": prim_path,
                    "field": "mass",
                    "old": old_mass,
                    "new": new_mass,
                    "scale_piper_mass": scale_piper_mass,
                }
            )

    stage.GetRootLayer().Save()
    patched_report = build_usd_report(output_path)
    patch_report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "scale_piper_mass": scale_piper_mass,
        "edit_count": len(edits),
        "edits": edits,
        "sanitized_report": patched_report,
    }
    return patch_report
