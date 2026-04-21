"""
Per-technique ASR bar charts.
Saves one HTML file per technique to results/plots/asr/.
"""

import json
from pathlib import Path

import plotly.graph_objects as go

REPORTS_DIR = Path(__file__).parent.parent / "results" / "final_reports"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "plots" / "asr"

ASR_METRICS = {
    "asr_i2p": "I2P",
    "asr_ring_a_bell": "Ring-A-Bell",
    "asr_mma_diffusion": "MMA-Diffusion",
}


def parse_reports() -> dict:
    data = {}
    for path in sorted(REPORTS_DIR.glob("*_report.json")):
        report = json.loads(path.read_text())
        technique = report["technique_name"]
        results = report["metric_results"]
        data[technique] = {
            key: results[key]["value"]
            for key in ASR_METRICS
            if key in results
        }
    return data


def plot_technique(technique: str, scores: dict, output_dir: Path) -> None:
    labels = [ASR_METRICS[k] for k in ASR_METRICS if k in scores]
    values = [scores[k] for k in ASR_METRICS if k in scores]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#636EFA", "#EF553B", "#00CC96"],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"ASR Scores — {technique}",
        yaxis=dict(title="ASR", range=[0, 1.1]),
        xaxis_title="Attack Method",
        template="plotly_white",
        height=450,
        width=550,
    )
    out_path = output_dir / f"{technique}_asr.png"
    fig.write_image(str(out_path))
    print(f"Saved {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = parse_reports()
    for technique, scores in data.items():
        plot_technique(technique, scores, OUTPUT_DIR)


if __name__ == "__main__":
    main()
