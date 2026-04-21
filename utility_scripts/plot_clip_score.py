"""
CLIP Score bar chart across all techniques.
Saves to results/plots/clip_score.png.
"""

import json
from pathlib import Path

import plotly.graph_objects as go

REPORTS_DIR = Path(__file__).parent.parent / "results" / "final_reports"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "plots"


def parse_reports() -> dict:
    data = {}
    for path in sorted(REPORTS_DIR.glob("*_report.json")):
        report = json.loads(path.read_text())
        technique = report["technique_name"]
        results = report["metric_results"]
        if "clip_score" in results:
            data[technique] = results["clip_score"]["value"]
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = parse_reports()

    techniques = list(data.keys())
    values = [data[t] for t in techniques]

    fig = go.Figure(
        go.Bar(
            x=techniques,
            y=values,
            marker_color="#00CC96",
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="CLIP Score by Technique",
        yaxis_title="CLIP Score (higher is better)",
        xaxis_title="Technique",
        template="plotly_white",
        height=500,
        width=900,
    )
    out_path = OUTPUT_DIR / "clip_score.png"
    fig.write_image(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
