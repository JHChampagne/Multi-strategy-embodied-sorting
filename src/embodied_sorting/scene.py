"""Static station configuration and a parameterized MJCF scene, with one robot asset."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .types import COLORS, TRAYS

OBJECT_IDS = tuple(f"object_{i:03d}" for i in range(7))
# Three inset corners in 120 mm bins; 46 mm spacing leaves finger clearance.
BASIC_TRAYS = {**TRAYS, "yellow": (0.22, -0.005)}
BASIC_TRAY_HALF = 0.052
GRASP_OPEN = 0.4
STANDOFF_DISTANCE = 0.010
SLOT_OFFSETS = ((-0.023, 0.023), (0.023, -0.023), (-0.023, -0.023))


def tray_slots():
    return {color: tuple(np.asarray(xy) + off for off in SLOT_OFFSETS) for color, xy in BASIC_TRAYS.items()}


def build_basic_model(assets: Path):
    root = ET.parse(assets / "sorting.xml").getroot()
    root.set("model", "seven_material_workstation")
    world = root.find("worldbody")
    world.find("camera[@name='rgbd']").set("fovy", "60")
    for color, xy in BASIC_TRAYS.items():
        tray = world.find(f"body[@name='tray_{color}']")
        tray.set("pos", f"{xy[0]} {xy[1]} 0")
        tray.find(f"geom[@name='tray_floor_{color}']").set("size", "0.060 0.060 0.003")
        for direction in "ns":
            wall = tray.find(f"geom[@name='tray_wall_{direction}_{color}']")
            wall.set("size", "0.060 0.004 0.012")
            wall.set("pos", f"0 {0.056 if direction == 'n' else -0.056} 0.012")
        for direction in "ew":
            wall = tray.find(f"geom[@name='tray_wall_{direction}_{color}']")
            wall.set("size", "0.004 0.060 0.012")
            wall.set("pos", f"{0.056 if direction == 'e' else -0.056} 0 0.012")
    template = deepcopy(world.find("body[@name='block_red']"))
    for body in list(world.findall("body")):
        if body.get("name", "").startswith("block_"):
            world.remove(body)
    for name in OBJECT_IDS:
        body = deepcopy(template)
        body.set("name", name)
        body.find("freejoint").set("name", name + "_free")
        body.find("geom").set("name", name + "_geom")
        world.append(body)
    files = {str(p.relative_to(assets)): p.read_bytes() for p in assets.rglob("*") if p.is_file()}
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), files)


def random_colors(rng):
    counts = rng.permutation([3, 2, 2])
    colors = np.repeat(COLORS, counts)
    rng.shuffle(colors)
    return dict(zip(OBJECT_IDS, colors.tolist()))
