#!/usr/bin/env python3
"""Export an actual task timeline as standalone SVG, using only the standard library."""

import argparse
from collections import OrderedDict
from html import escape
import json
from pathlib import Path


def export(trace, output):
    groups = OrderedDict()
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["record_type"] == "action" and row["goal_id"]:
                groups.setdefault(row["goal_id"], []).append(row)
    palette = dict(
        zip(
            [
                "approach",
                "descend",
                "align",
                "close",
                "lift",
                "transport",
                "place",
                "release",
                "retreat",
                "observe",
            ],
            [
                "#60a5fa",
                "#818cf8",
                "#c084fc",
                "#f59e0b",
                "#fb7185",
                "#34d399",
                "#10b981",
                "#eab308",
                "#64748b",
                "#d1d5db",
            ],
        )
    )
    width, left, top, row_height = 1200, 340, 115, 35
    height = top + row_height * len(groups) + 65
    duration = max(r["sim_time"] for rows in groups.values() for r in rows) + 0.066
    scale = (width - left - 30) / duration
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Measured seven-object execution and recovery timeline</title>",
        "<desc>Each row is one goal attempt. An aborted grasp is followed by a fresh goal for the same visual track.</desc>",
        "<style>text{font:13px sans-serif;fill:#172033}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="26" style="font-size:19px;font-weight:bold">Actual geometry + NoOp trace: seed {escape(trace.stem)}</text>',
    ]
    for i, (phase, color) in enumerate(palette.items()):
        x, y = 20 + (i % 5) * 225, 46 + (i // 5) * 24
        svg += [
            f'<rect x="{x}" y="{y}" width="15" height="13" fill="{color}"/>',
            f'<text x="{x + 21}" y="{y + 12}">{phase}</text>',
        ]
    seen = set()
    for index, (goal_id, rows) in enumerate(groups.items()):
        y = top + index * row_height
        goal = rows[0]["goal"]
        aborted = not any(r["phase"] == "release" for r in rows)
        suffix = " (aborted)" if aborted else " (retry)" if goal["track_id"] in seen else ""
        label = f"{goal_id} / {goal['track_id']} / {goal['color']}{suffix}"
        seen.add(goal["track_id"])
        svg.append(f'<text x="20" y="{y + 16}">{escape(label)}</text>')
        for row in rows:
            x = left + row["sim_time"] * scale
            svg.append(
                f'<rect x="{x:.2f}" y="{y}" width="{0.066 * scale + 0.1:.3f}" height="23" fill="{palette.get(row["phase"], "#e5e7eb")}"/>'
            )
    axis_y = top + len(groups) * row_height + 5
    svg.append(f'<path d="M {left} {axis_y} H {width - 30}" stroke="#64748b"/>')
    for tick in range(0, int(duration) + 1, 10):
        x = left + tick * scale
        svg.append(f'<text x="{x:.2f}" y="{axis_y + 20}" text-anchor="middle">{tick}</text>')
    svg.append(
        f'<text x="{left + 280}" y="{axis_y + 43}">Simulation time (s); no expert fallback</text></svg>'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.trace, args.output)
