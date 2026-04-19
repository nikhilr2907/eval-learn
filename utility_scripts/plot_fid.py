"""
FID bar chart across all techniques.
Saves to results/plots/fid.html.
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
        if "fid" in results:
            data[technique] = results["fid"]["value"]
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
            marker_color="#636EFA",
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="FID Score by Technique",
        yaxis_title="FID (lower is better)",
        xaxis_title="Technique",
        template="plotly_white",
        height=500,
        width=900,
    )
    out_path = OUTPUT_DIR / "fid.html"
    fig.write_html(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
