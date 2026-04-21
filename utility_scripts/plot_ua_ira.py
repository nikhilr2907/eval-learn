"""
UA-IRA grouped bar chart across all techniques.
Saves to results/plots/ua_ira.png.
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
        if "ua_ira" in results:
            entry = results["ua_ira"]
            data[technique] = {
                "ua_ira": entry["value"],
                "ua_score": entry["details"]["ua_score"],
                "ira_score": entry["details"]["ira_score"],
            }
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = parse_reports()

    techniques = list(data.keys())

    fig = go.Figure([
        go.Bar(
            name="UA-IRA (combined)",
            x=techniques,
            y=[data[t]["ua_ira"] for t in techniques],
            marker_color="#636EFA",
            text=[f"{data[t]['ua_ira']:.2f}" for t in techniques],
            textposition="outside",
        ),
        go.Bar(
            name="UA Score",
            x=techniques,
            y=[data[t]["ua_score"] for t in techniques],
            marker_color="#EF553B",
            text=[f"{data[t]['ua_score']:.2f}" for t in techniques],
            textposition="outside",
        ),
        go.Bar(
            name="IRA Score",
            x=techniques,
            y=[data[t]["ira_score"] for t in techniques],
            marker_color="#00CC96",
            text=[f"{data[t]['ira_score']:.2f}" for t in techniques],
            textposition="outside",
        ),
    ])
    fig.update_layout(
        title="UA-IRA by Technique",
        yaxis_title="Score",
        xaxis_title="Technique",
        barmode="group",
        template="plotly_white",
        height=500,
        width=1100,
    )
    out_path = OUTPUT_DIR / "ua_ira.png"
    fig.write_image(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
