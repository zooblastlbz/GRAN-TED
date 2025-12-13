#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot question_type_accuracy across a numeric variable extracted from filenames.

It scans a directory for files named: {prefix}-{var}-{suffix}.json
- prefix/suffix are provided via CLI
- var is parsed as a number (int/float)
For each JSON, it reads summary.accuracy_statistics.question_type_accuracy and
plots one curve per question type over the sorted var values. Saves as PDF.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot question_type_accuracy from JSON files matched by prefix/suffix.")
    p.add_argument("--dir",  default="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/output_per_layer/", help="Directory containing JSON files.")
    p.add_argument("--prefix",  default="qwen3vl_last-", help="Filename prefix before the variable, e.g., 'prefix'.")
    p.add_argument("--suffix",  default="_2layers_1", help="Filename suffix after the variable, e.g., 'suffix'.")
    p.add_argument("--output",  default="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/proj_text_enc/attn_pool_contrastive/visual/question_type_accuracy_instruct.pdf", help="Output PDF path. Default: <dir>/question_type_accuracy.pdf")
    p.add_argument("--title", dest="title", default=None, help="Figure title.")
    p.add_argument("--xlabel", dest="xlabel", default="Variable", help="X-axis label.")
    p.add_argument("--ylabel", dest="ylabel", default="Accuracy", help="Y-axis label.")
    p.add_argument("--ylim01", dest="ylim01", action="store_true", help="Clamp Y axis to [0,1].")
    p.add_argument("--dpi", dest="dpi", type=int, default=200, help="Figure DPI for rasterization inside PDF.")
    p.add_argument("--legend_loc", dest="legend_loc", default="best", help="Matplotlib legend loc.")
    return p.parse_args()



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
            # Skip non-numeric variable parts
            continue
        matches.append((var_val, fp))
    # Sort by numeric var ascending
    matches.sort(key=lambda x: x[0])
    return matches


def load_question_type_accuracy(json_path: Path) -> Dict[str, float]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        qta = data["summary"]["accuracy_statistics"]["question_type_accuracy"]
    except Exception:
        return {}
    result: Dict[str, float] = {}
    for qtype, stats in qta.items():
        # Some files may already store float 0..1
        acc = stats.get("accuracy")
        if isinstance(acc, (int, float)):
            result[qtype] = float(acc)
    return result


def main():
    args = parse_args()
    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    pairs = find_matching_files(root, args.prefix, args.suffix)
    if not pairs:
        raise SystemExit("No matching JSON files found. Check --prefix/--suffix and directory.")

    # Accumulate per question type
    # qtype -> list of (x, y)
    series: Dict[str, List[Tuple[float, float]]] = {}

    for xval, fp in pairs:
        qta = load_question_type_accuracy(fp)
        if not qta:
            continue
        for qtype, acc in qta.items():
            series.setdefault(qtype, []).append((xval, acc))

    if not series:
        raise SystemExit("No question_type_accuracy data found in matched files.")

    # Plot
    plt.figure(figsize=(10, 6), dpi=args.dpi)
    cmap = plt.cm.get_cmap("tab20", max(6, len(series)))

    for idx, (qtype, xy) in enumerate(sorted(series.items(), key=lambda kv: kv[0])):
        # Sort by x
        xy_sorted = sorted(xy, key=lambda t: t[0])
        xs = [t[0] for t in xy_sorted]
        ys = [t[1] for t in xy_sorted]
        line_color = cmap(idx)
        plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=qtype, color=line_color)

        # Mark the highest and second highest points on the curve
        if ys:
            sorted_idx = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
            i1 = sorted_idx[0]
            plt.scatter(xs[i1], ys[i1], marker='*', s=120, c=[line_color], edgecolors='k', zorder=5)
            if len(sorted_idx) > 1:
                i2 = sorted_idx[1]
                if i2 != i1:
                    plt.scatter(xs[i2], ys[i2], marker='^', s=80, c=[line_color], edgecolors='k', zorder=4)

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.title:
        plt.title(args.title)
    if args.ylim01:
        plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(loc=args.legend_loc, fontsize=9, ncol=2)
    plt.tight_layout()

    out_path = Path(args.output) if args.output else root / "question_type_accuracy.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()