"""Reusable data export helpers for the paper figures and audit checklist.

Every plotting function can call :func:`export_figure_data` before closing its
figure.  The helper extracts the numeric artist data from Matplotlib and writes
both JSON (machine-readable) and Markdown (easy to paste into a paper note).
The baseline audit writer additionally emits the full A1/A5/A6 tables from the
ADMM state rather than trying to reconstruct them from pixels.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


def _root(output_dir=None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return Path(os.environ.get("EXPERIMENT_DATA_DIR", "experiment_data"))


def jsonable(value):
    """Convert NumPy/Matplotlib-friendly values to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_json(name, payload, output_dir=None):
    root = _root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return lines


def _scalar(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _series_markdown(series):
    kind = series.get("kind", "series")
    title = series.get("label") or kind
    lines = [f"#### {title} ({kind})", ""]
    if "x" in series and "y" in series:
        x, y = series["x"], series["y"]
        rows = [(_scalar(a), _scalar(b)) for a, b in zip(x, y)]
        lines.extend(_table(["x", "y"], rows))
    elif "offsets" in series:
        rows = [(_scalar(p[0]), _scalar(p[1])) for p in series["offsets"]]
        lines.extend(_table(["x", "y"], rows))
    elif "patches" in series:
        rows = []
        for patch in series["patches"]:
            rows.append(
                (
                    _scalar(patch.get("x")),
                    _scalar(patch.get("y")),
                    _scalar(patch.get("width")),
                    _scalar(patch.get("height")),
                )
            )
        lines.extend(_table(["x", "y", "width", "height"], rows))
    elif "vertices" in series:
        for idx, vertices in enumerate(series["vertices"]):
            lines.append(f"vertices[{idx}]")
            lines.extend(_table(["x", "y"], [(_scalar(p[0]), _scalar(p[1])) for p in vertices]))
    else:
        lines.append("```json")
        lines.append(json.dumps(jsonable(series), ensure_ascii=False, indent=2))
        lines.append("```")
    lines.append("")
    return lines


def export_figure_data(fig, figure_name, output_dir=None, print_data=None):
    """Extract and export the numeric data represented by a Matplotlib figure."""
    payload = {"figure": figure_name, "axes": []}

    for axis_index, ax in enumerate(fig.axes):
        axis = {
            "index": axis_index,
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
            "series": [],
        }

        for line in ax.get_lines():
            x, y = line.get_data()
            axis["series"].append(
                {
                    "kind": "line",
                    "label": line.get_label(),
                    "x": np.asarray(x).tolist(),
                    "y": np.asarray(y).tolist(),
                }
            )

        for container in ax.containers:
            patches = []
            for patch in getattr(container, "patches", []):
                if not hasattr(patch, "get_height"):
                    continue
                patches.append(
                    {
                        "x": patch.get_x(),
                        "y": patch.get_y(),
                        "width": patch.get_width(),
                        "height": patch.get_height(),
                    }
                )
            if patches:
                axis["series"].append(
                    {
                        "kind": "bars",
                        "label": getattr(container, "get_label", lambda: "")(),
                        "patches": patches,
                    }
                )

            # ErrorbarContainer stores its center line, caps, and vertical
            # segments in ``lines`` rather than ``patches``.  Preserve those
            # numeric segments so error bars are also re-plottable.
            error_lines = getattr(container, "lines", ())
            if error_lines:
                for child in error_lines:
                    children = child if isinstance(child, (list, tuple)) else (child,)
                    for item in children:
                        if hasattr(item, "get_data"):
                            x, y = item.get_data()
                            axis["series"].append(
                                {
                                    "kind": "errorbar_line",
                                    "label": container.get_label(),
                                    "x": np.asarray(x).tolist(),
                                    "y": np.asarray(y).tolist(),
                                }
                            )
                        elif hasattr(item, "get_segments"):
                            segments = [
                                np.asarray(segment).tolist()
                                for segment in item.get_segments()
                            ]
                            axis["series"].append(
                                {
                                    "kind": "errorbar_segments",
                                    "label": container.get_label(),
                                    "segments": segments,
                                }
                            )

        for collection in ax.collections:
            offsets = getattr(collection, "get_offsets", lambda: np.empty((0, 2)))()
            offsets = np.asarray(offsets)
            if offsets.size:
                axis["series"].append(
                    {
                        "kind": "collection_offsets",
                        "label": collection.get_label(),
                        "offsets": offsets.tolist(),
                    }
                )
            paths = getattr(collection, "get_paths", lambda: [])()
            vertices = [np.asarray(path.vertices).tolist() for path in paths]
            if vertices:
                axis["series"].append(
                    {
                        "kind": "collection_vertices",
                        "label": collection.get_label(),
                        "vertices": vertices,
                    }
                )

        payload["axes"].append(axis)

    root = _root(output_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{figure_name}.json"
    md_path = root / f"{figure_name}.md"
    json_path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [f"# {figure_name} plotted data", ""]
    for axis in payload["axes"]:
        md_lines.extend(
            [
                f"## Axes {axis['index']}",
                f"- title: {axis['title']}",
                f"- xlabel: {axis['xlabel']}",
                f"- ylabel: {axis['ylabel']}",
                f"- scale: x={axis['xscale']}, y={axis['yscale']}",
                "",
            ]
        )
        for series in axis["series"]:
            md_lines.extend(_series_markdown(series))
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    if print_data is None:
        print_data = os.environ.get("PRINT_FIGURE_DATA", "1").lower() not in {
            "0",
            "false",
            "no",
        }
    print(f"[数据] {figure_name}: {json_path} / {md_path}")
    if print_data:
        print(md_path.read_text(encoding="utf-8"))
    return payload


def export_baseline_audit(microgrids, solver, results, output_dir=None, print_data=True):
    """Export full A1/A5/A6 audit tables from one solved baseline case."""
    root = _root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    history = solver.history
    lines = ["# Baseline ADMM audit data", ""]

    lines += ["## A1 ADMM residual trajectory", ""]
    lines += _table(
        ["iteration", "primal_residual", "dual_residual", "rho"],
        [
            (k, f"{p:.10f}", f"{d:.10f}", f"{rho:.10f}")
            for k, (p, d, rho) in enumerate(
                zip(history["primal_residual"], history["dual_residual"], history["rho"])
            )
        ],
    )
    rho_changes = []
    for k, (before, after) in enumerate(zip(history["rho"], history["rho"][1:]), start=1):
        if abs(before - after) > 1e-12:
            rho_changes.append((k, f"{before:.10f}", f"{after:.10f}"))
    lines += ["", "rho changes:", ""]
    lines += _table(["iteration", "rho_before", "rho_after"], rho_changes or [("none", "", "")])
    lines += [
        "",
        f"converged = {solver.converged}",
        f"final_iteration = {solver.final_iteration}",
        f"final_primal_residual = {solver.final_primal_residual:.10f}",
        f"final_dual_residual = {solver.final_dual_residual:.10f}",
        "",
    ]

    lines += ["## A5 Figure 2 dispatch data", ""]
    rows = []
    p_trade = np.asarray(results["P_trade"])
    for i, mg in enumerate(microgrids):
        local = solver.get_local_result(i)
        for t in range(solver.T):
            p2p_buy = float(sum(max(p_trade[i, j, t], 0.0) for j in range(solver.N) if j != i))
            p2p_sell = float(sum(max(-p_trade[i, j, t], 0.0) for j in range(solver.N) if j != i))
            rows.append(
                (
                    f"MG{i + 1}",
                    t,
                    f"{local['P_grid'][t]:.10f}",
                    f"{p2p_buy:.10f}",
                    f"{p2p_sell:.10f}",
                    f"{local['P_ch'][t]:.10f}",
                    f"{local['P_dis'][t]:.10f}",
                    f"{local['SOC'][t] / mg.battery_capacity:.10f}",
                )
            )
    lines += _table(["MG", "t", "grid_purchase", "p2p_buy", "p2p_sell", "ch", "dis", "SOC_pu"], rows)

    # Match the project's existing "one-direction" definition: for each
    # unordered pair, count only the positive-flow direction once.
    total_p2p = float(
        sum(
            np.sum(np.maximum(p_trade[i, j, :], 0.0))
            for i in range(solver.N)
            for j in range(i + 1, solver.N)
        )
    )
    total_grid = float(sum(np.sum(mg.grid_price * solver.get_local_result(i)["P_grid"]) for i, mg in enumerate(microgrids)))
    total_loss = float(np.sum(solver.line_loss))
    soc_values = np.concatenate([solver.get_local_result(i)["SOC"] / mg.battery_capacity for i, mg in enumerate(microgrids)])
    lines += [
        "",
        f"P2P one-direction total (kWh) = {total_p2p:.10f}",
        f"grid purchase cost (yuan) = {total_grid:.10f}",
        f"total line loss (kWh) = {total_loss:.10f}",
        f"SOC per-unit range = [{np.min(soc_values):.10f}, {np.max(soc_values):.10f}]",
        "",
        "## A6 Figure 1 profile data",
        "",
    ]
    profile_rows = [
        (f"MG{i + 1}", t, f"{mg.load[t]:.10f}", f"{mg.pv_generation[t]:.10f}", f"{mg.wind_generation[t]:.10f}")
        for i, mg in enumerate(microgrids)
        for t in range(solver.T)
    ]
    lines += _table(
        ["MG", "t", "load", "pv", "wind"],
        profile_rows,
    )

    path = root / "baseline_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    export_json(
        "baseline_audit",
        {
            "a1": {
                "history": history,
                "rho_changes": rho_changes,
                "converged": solver.converged,
                "final_iteration": solver.final_iteration,
                "final_primal_residual": solver.final_primal_residual,
                "final_dual_residual": solver.final_dual_residual,
            },
            "a5": {"rows": rows, "total_p2p": total_p2p, "total_grid": total_grid, "total_loss": total_loss},
            "a6": {"rows": profile_rows},
        },
        output_dir=root,
    )
    print(f"[数据] baseline audit: {path}")
    if print_data:
        print(path.read_text(encoding="utf-8"))
    return path


def export_result_bundle(name, payload, output_dir=None):
    """Save complete raw experiment results for re-plotting."""
    root = _root(output_dir)
    path = export_json(name, payload, output_dir=root)
    md = root / f"{name}.md"
    md.write_text(
        f"# {name}\n\nRaw JSON: `{path.name}`\n\n```json\n"
        + json.dumps(jsonable(payload), ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"[数据] {name}: {path} / {md}")
    return path
