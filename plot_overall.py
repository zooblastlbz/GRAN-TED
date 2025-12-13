#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot overall_accuracy lines for multiple (prefix, suffix) groups.

For each group defined by prefix/suffix, this scans files named:
  prefix-{var}-suffix.json
in a given directory, where {var} is a numeric value (int/float).
It reads summary.accuracy_statistics.overall_accuracy from each JSON and
plots a line per group over sorted {var}. Each point is annotated with its
rank on that line (1 = highest). Ties share the same rank (dense ranking).
Saves the figure as PDF.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot overall_accuracy lines for multiple prefix/suffix groups.")
    p.add_argument("--dir",  default="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/output_per_layer/", help="Directory containing JSON files.")
    p.add_argument("--groups", default=["qwen3vl_last-:_2layers_1:Instruct","qwen3vl_thinking_last-:_2layers_1:Thinking"],
                   help="One group spec as 'prefix:suffix[:label]'. Repeat for multiple groups.")
    p.add_argument("--output",  default="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/visual/overall.pdf", help="Output PDF path. Default: <dir>/question_type_accuracy.pdf")
    p.add_argument("--title", dest="title", default=None, help="Figure title.")
    p.add_argument("--xlabel", dest="xlabel", default="Variable", help="X-axis label.")
    p.add_argument("--ylabel", dest="ylabel", default="Overall Accuracy", help="Y-axis label.")
    p.add_argument("--ylim01", dest="ylim01", action="store_true", help="Clamp Y axis to [0,1].")
    p.add_argument("--dpi", dest="dpi", type=int, default=200, help="Figure DPI for rasterization inside PDF.")
    p.add_argument("--legend_loc", dest="legend_loc", default="best", help="Matplotlib legend loc.")
    p.add_argument("--annot_fontsize", dest="annot_fontsize", type=int, default=8, help="Annotation font size.")
    return p.parse_args()


def parse_group_spec(spec: str) -> Tuple[str, str, str]:
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid group spec: {spec}. Expected 'prefix:suffix[:label]'")
    prefix, suffix = parts[0], parts[1]
    label = parts[2] if len(parts) == 3 else f"{prefix}-{suffix}"
    return prefix, suffix, label


def find_matching_files(root: Path, prefix: str, suffix: str) -> List[Tuple[float, Path]]:
    # Pattern: ^prefix-(var)-suffix.json$
    pattern = re.compile(rf"^{re.escape(prefix)}(.+){re.escape(suffix)}\.json$")
    matches: List[Tuple[float, Path]] = []
    for fp in root.glob("*.json"):
        m = pattern.match(fp.name)
        if not m:
            continue
        var_str = m.group(1)
        try:
            var_val = float(var_str)
        except ValueError:
            continue
        matches.append((var_val, fp))
    matches.sort(key=lambda x: x[0])
    return matches


def load_overall_accuracy(json_path: Path) -> float | None:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["summary"]["accuracy_statistics"]["overall_accuracy"])
    except Exception:
        return None


def dense_rank(values: List[float]) -> List[int]:
    # Dense ranking: highest value rank=1; ties share rank; ranks increase by 1 for next distinct value.
    uniq_sorted = sorted(set(values), reverse=True)
    rank_map: Dict[float, int] = {v: i + 1 for i, v in enumerate(uniq_sorted)}
    return [rank_map[v] for v in values]


def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    group_specs: List[Tuple[str, str, str]] = []
    for spec in args.groups:
        try:
            group_specs.append(parse_group_spec(spec))
        except ValueError as e:
            raise SystemExit(str(e))

    # Collect series per group
    series: List[Tuple[str, List[float], List[float]]] = []  # (label, xs, ys)

    for prefix, suffix, label in group_specs:
        pairs = find_matching_files(root, prefix, suffix)
        xs: List[float] = []
        ys: List[float] = []
        for xval, fp in pairs:
            acc = load_overall_accuracy(fp)
            if acc is None:
                continue
            xs.append(xval)
            ys.append(acc)
        if xs and ys:
            # Ensure sorted by x (pairs already sorted, but keep safe)
            sorted_xy = sorted(zip(xs, ys), key=lambda t: t[0])
            xs, ys = [t[0] for t in sorted_xy], [t[1] for t in sorted_xy]
            series.append((label, xs, ys))

    if not series:
        raise SystemExit("No matching data for provided groups.")

    # Plot
    plt.figure(figsize=(10, 6), dpi=args.dpi)
    cmap = plt.cm.get_cmap("tab20", max(6, len(series)))

    for idx, (label, xs, ys) in enumerate(series):
        color = cmap(idx)
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=label, color=color)
        ranks = dense_rank(ys)
        for x, y, r in zip(xs, ys, ranks):
            plt.annotate(str(r), (x, y), textcoords="offset points", xytext=(0, 6), ha="center",
                         fontsize=args.annot_fontsize,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7, alpha=0.85))

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.title:
        plt.title(args.title)
    if args.ylim01:
        plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(loc=args.legend_loc, fontsize=9, ncol=2)
    plt.tight_layout()

    out_path = Path(args.output) if args.output else root / "overall_accuracy_groups.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
