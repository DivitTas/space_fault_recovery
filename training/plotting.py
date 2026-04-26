"""SVG plot generation for training logs."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Iterable


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def moving_average(values: list[float], window: int) -> list[float]:
    averaged: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        averaged.append(running_sum / len(queue))
    return averaged


def _series_points(
    xs: list[float],
    ys: list[float],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    points: list[str] = []
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    for x_value, y_value in zip(xs, ys):
        x = left + ((x_value - x_min) / x_span) * width
        y = top + height - ((y_value - y_min) / y_span) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _nice_bounds(values: Iterable[float]) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 1.0
    low = min(finite_values)
    high = max(finite_values)
    if abs(high - low) < 1e-9:
        pad = 1.0 if abs(high) < 1.0 else abs(high) * 0.1
    else:
        pad = (high - low) * 0.08
    return low - pad, high + pad


def write_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[float], list[float], str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 420
    left = 72
    top = 48
    chart_width = 830
    chart_height = 300

    all_x = [x for _, xs, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _ in series for y in ys]
    x_min, x_max = _nice_bounds(all_x)
    y_min, y_max = _nice_bounds(all_y)

    grid_lines = []
    for idx in range(6):
        ratio = idx / 5
        y = top + chart_height - ratio * chart_height
        value = y_min + ratio * (y_max - y_min)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" '
            'stroke="#e7eaf0" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-size="12" fill="#526071">'
            f"{value:.2f}</text>"
        )

    plotted_series = []
    legend = []
    for idx, (name, xs, ys, color) in enumerate(series):
        if not xs or not ys:
            continue
        points = _series_points(
            xs,
            ys,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=top,
            width=chart_width,
            height=chart_height,
        )
        plotted_series.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
        legend_y = top + 20 + idx * 22
        legend.append(
            f'<line x1="{left + chart_width - 164}" y1="{legend_y}" '
            f'x2="{left + chart_width - 128}" y2="{legend_y}" stroke="{color}" '
            'stroke-width="3" />'
        )
        legend.append(
            f'<text x="{left + chart_width - 120}" y="{legend_y + 4}" '
            'font-size="13" fill="#263241">'
            f"{html.escape(name)}</text>"
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{left}" y="28" font-size="22" font-weight="700" fill="#17202a">{html.escape(title)}</text>
  {"".join(grid_lines)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#2f3a45" stroke-width="1.4" />
  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#2f3a45" stroke-width="1.4" />
  <text x="{left + chart_width / 2}" y="{height - 28}" text-anchor="middle" font-size="14" fill="#344052">{html.escape(x_label)}</text>
  <text x="18" y="{top + chart_height / 2}" text-anchor="middle" font-size="14" fill="#344052" transform="rotate(-90 18 {top + chart_height / 2})">{html.escape(y_label)}</text>
  <text x="{left}" y="{top + chart_height + 22}" text-anchor="middle" font-size="12" fill="#526071">{x_min:.0f}</text>
  <text x="{left + chart_width}" y="{top + chart_height + 22}" text-anchor="middle" font-size="12" fill="#526071">{x_max:.0f}</text>
  {"".join(plotted_series)}
  {"".join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def generate_training_plots(
    metrics: list[dict[str, str]], output_dir: Path, steps_csv: Path | None = None
) -> list[Path]:
    if not metrics:
        return []

    episodes = [float(row["episode"]) for row in metrics]
    rewards = [float(row["total_reward"]) for row in metrics]
    losses = [float(row["mean_loss"]) for row in metrics]
    success_rates = [float(row["rolling_success_rate_20"]) for row in metrics]
    steps = [float(row["steps"]) for row in metrics]

    reward_ma = moving_average(rewards, 20)
    loss_ma = moving_average(losses, 20)

    outputs = [
        output_dir / "reward_curve.svg",
        output_dir / "loss_curve.svg",
        output_dir / "success_rate_curve.svg",
        output_dir / "episode_length_curve.svg",
    ]

    write_line_chart(
        outputs[0],
        title="Training Reward",
        x_label="Episode",
        y_label="Total reward",
        series=[
            ("episode reward", episodes, rewards, "#2f6fed"),
            ("20 episode average", episodes, reward_ma, "#e05d2f"),
        ],
    )
    write_line_chart(
        outputs[1],
        title="Training Loss",
        x_label="Episode",
        y_label="Mean TD loss",
        series=[
            ("episode loss", episodes, losses, "#6f4cc3"),
            ("20 episode average", episodes, loss_ma, "#209a72"),
        ],
    )
    write_line_chart(
        outputs[2],
        title="Recovery Success Rate",
        x_label="Episode",
        y_label="Rolling success rate",
        series=[("20 episode rate", episodes, success_rates, "#1f8a99")],
    )
    write_line_chart(
        outputs[3],
        title="Episode Length",
        x_label="Episode",
        y_label="Steps",
        series=[("steps", episodes, steps, "#c67a00")],
    )

    # Optional per-step reward plot (all steps across all episodes)
    if steps_csv is not None and steps_csv.exists():
        step_rows = read_metrics_csv(steps_csv)
        if step_rows:
            global_step_idx = [float(i + 1) for i in range(len(step_rows))]
            step_rewards = [float(row.get("reward", "0") or 0.0) for row in step_rows]
            out_step = output_dir / "step_reward_curve.svg"
            write_line_chart(
                out_step,
                title="Per-step Reward (all episodes)",
                x_label="Step index",
                y_label="Reward",
                series=[("step reward", global_step_idx, step_rewards, "#2f6fed")],
            )
            outputs.append(out_step)
    return outputs

