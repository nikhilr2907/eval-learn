"""
ERR grouped bar chart across all techniques.
Shows forgetting, retention, and adversarial sub-scores per technique.
Saves to results/plots/err.html.
"""

import json
from pathlib import Path

import plotly.graph_objects as go

REPORTS_DIR = Path(__file__).parent.parent / "results" / "final_reports"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "plots"

ERR_COMPONENTS = {
    "forgetting": "#636EFA",
    "retention": "#00CC96",
    "adversarial": "#EF553B",
}


def parse_reports() -> dict:
    data = {}
    for path in sorted(REPORTS_DIR.glob("*_report.json")):
        report = json.loads(path.read_text())
        technique = report["technique_name"]
        results = report["metric_results"]
        if "err" in results:
            details = results["err"]["details"]
            data[technique] = {
                component: details[component]
                for component in ERR_COMPONENTS
                if component in details
            }
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = parse_reports()

    techniques = list(data.keys())
    fig = go.Figure()

    for component, colour in ERR_COMPONENTS.items():
        values = [data[t].get(component, 0.0) for t in techniques]
        fig.add_trace(
            go.Bar(
                name=component.capitalize(),
                x=techniques,
                y=values,
                marker_color=colour,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group",
        title="ERR Sub-scores by Technique",
        yaxis=dict(title="Score (higher is better)", range=[0, 1.15]),
        xaxis_title="Technique",
        template="plotly_white",
        legend_title="Component",
        height=500,
        width=1000,
    )
    out_path = OUTPUT_DIR / "err.html"
    fig.write_html(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
