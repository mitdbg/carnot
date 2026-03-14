#!/usr/bin/env python3
"""Create visualization graphs from evaluation results.

Usage:
    python create_graph.py [--output-dir results/graphs]

Reads evaluation data and produces:
- Vector index graphs
- Flat index graphs
- Summary comparison graphs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# Evaluation data
VECTOR_INDEX_DATA = {
    "K": [50, 100, 200, 400],
    "precision": [0.9810, 0.9810, 0.9810, 0.9716],
    "recall": [0.6867, 0.7259, 0.7463, 0.7672],
    "f1": [0.7808, 0.8162, 0.8316, 0.8417],
    "time_s": [20.47, 22.44, 54.45, 37.61],
    "cost_usd": [0.2366, 0.4585, 0.8767, 1.6814],
}

JUST_VECTOR_INDEX_DATA = {
    "K": [20, 25, 50, 100, 200, 400],
    "precision": [0.4320, 0.3664, 0.2128, 0.1144, 0.0582, 0.0293],
    "recall": [0.7730, 0.8075, 0.9146, 0.9752, 0.9920, 1.0000],
    "f1": [0.5375, 0.4901, 0.3388, 0.2026, 0.1093, 0.0568],
    "time_s": [0.26, 0.27, 0.32, 0.31, 0.37, 0.37],
    "cost_usd": [0.00000024, 0.00000024, 0.00000024, 0.00000024, 0.00000024, 0.00000024],
}

FLAT_INDEX_DATA = {
    "label": [
        "K=50\nMAX=100",
        "K=50\nMAX=150",
        "K=100\nMAX=150",
    ],
    "precision": [0.9817, 0.9817, 0.9817],
    "recall": [0.6868, 0.7083, 0.7193],
    "f1": [0.7838, 0.7974, 0.8094],
    "time_s": [64.10, 82.93, 85.87],
    "cost_usd": [0.1279, 0.2637, 0.4520],
}

LOTUS = {
    "precision": 0.9829,
    "recall": 0.8042,
    "f1": 0.8762,
    "time_s": 64.2277,
    "cost_usd": 2.182678,
}

PZ = {
    "precision": 0.9606,
    "recall": 0.8451,
    "f1": 0.8891,
    "time_s": 98.3278,
    "cost_usd": 2.8608,
}


def _pareto_frontier(points: list[dict], x_key: str, y_key: str) -> list[dict]:
    """Return Pareto-optimal points minimizing x and maximizing y."""
    frontier = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            if (
                other[x_key] <= point[x_key]
                and other[y_key] >= point[y_key]
                and (other[x_key] < point[x_key] or other[y_key] > point[y_key])
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda p: p[x_key])


def create_vector_quality_graph(output_path: Path) -> None:
    """Precision, Recall, F1 vs K with LOTUS and PZ as flat lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    K = VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    # Origin point
    ax.plot([0], [0], "k+", markersize=10, markeredgewidth=2, label="Origin")

    # Vector Index lines
    ax.plot(K, VECTOR_INDEX_DATA["precision"], "o-", label="Vector Index (Precision)", color="C0")
    ax.plot(K, VECTOR_INDEX_DATA["recall"], "s-", label="Vector Index (Recall)", color="C1")
    ax.plot(K, VECTOR_INDEX_DATA["f1"], "^-", label="Vector Index (F1)", color="C2")

    # LOTUS and PZ as horizontal lines
    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles="--", alpha=0.7, label="LOTUS (Precision)")
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", alpha=0.7, label="LOTUS (Recall)")
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles="--", alpha=0.7, label="LOTUS (F1)")
    ax.hlines(PZ["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.5, label="PZ (Precision)")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.5, label="PZ (Recall)")
    ax.hlines(PZ["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.5, label="PZ (F1)")

    ax.set_xlabel("K (top-k retrieved)")
    ax.set_ylabel("Score")
    ax.set_title("Quality: Precision, Recall, F1 vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "quality_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'quality_vs_k.png'}")


def create_vector_time_graph(output_path: Path) -> None:
    """Latency vs K with LOTUS and PZ as flat lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    K = VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    ax.plot([0], [0], "k+", markersize=10, markeredgewidth=2, label="Origin")
    ax.plot(K, VECTOR_INDEX_DATA["time_s"], "o-", label="Vector Index", color="C0", linewidth=2)

    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")

    ax.set_xlabel("K (top-k retrieved)")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("Latency vs K")
    ax.set_xticks([0] + K)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path / "latency_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'latency_vs_k.png'}")


def create_vector_cost_graph(output_path: Path) -> None:
    """Cost vs K with LOTUS and PZ as flat lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    K = VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    ax.plot([0], [0], "k+", markersize=10, markeredgewidth=2, label="Origin")
    ax.plot(K, VECTOR_INDEX_DATA["cost_usd"], "o-", label="Vector Index", color="C0", linewidth=2)

    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")

    ax.set_xlabel("K (top-k retrieved)")
    ax.set_ylabel("Average Cost ($)")
    ax.set_title("Cost vs K")
    ax.set_xticks([0] + K)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path / "cost_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'cost_vs_k.png'}")


def create_vector_combined_graph(output_path: Path) -> None:
    """All metrics in a 2x2 subplot grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    K = VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    # Quality (Precision, Recall, F1)
    ax = axes[0, 0]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, VECTOR_INDEX_DATA["precision"], "o-", label="Vector (P)", color="C0")
    ax.plot(K, VECTOR_INDEX_DATA["recall"], "s-", label="Vector (R)", color="C1")
    ax.plot(K, VECTOR_INDEX_DATA["f1"], "^-", label="Vector (F1)", color="C2")
    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles="--", alpha=0.6)
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", alpha=0.6)
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles="--", alpha=0.6)
    ax.hlines(PZ["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.4)
    ax.hlines(PZ["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.4)
    ax.hlines(PZ["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.4)
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # Recall only (clearer comparison)
    ax = axes[0, 1]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, VECTOR_INDEX_DATA["recall"], "o-", label="Vector Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time
    ax = axes[1, 0]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, VECTOR_INDEX_DATA["time_s"], "o-", label="Vector Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("K")
    ax.set_ylabel("Time (s)")
    ax.set_title("Latency vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost
    ax = axes[1, 1]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, VECTOR_INDEX_DATA["cost_usd"], "o-", label="Vector Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("K")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Evaluation Results: Vector Index vs LOTUS vs PZ", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "eval_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'eval_comparison.png'}")


def create_just_vector_quality_graph(output_path: Path) -> None:
    """Precision, Recall, F1 vs K for retrieval-only vector index."""
    fig, ax = plt.subplots(figsize=(8, 5))

    K = JUST_VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    ax.plot([0], [0], "k+", markersize=10, markeredgewidth=2, label="Origin")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["precision"], "o-", label="Just Vector (Precision)", color="C0")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["recall"], "s-", label="Just Vector (Recall)", color="C1")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["f1"], "^-", label="Just Vector (F1)", color="C2")

    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles="--", alpha=0.7, label="LOTUS (Precision)")
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", alpha=0.7, label="LOTUS (Recall)")
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles="--", alpha=0.7, label="LOTUS (F1)")
    ax.hlines(PZ["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.5, label="PZ (Precision)")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.5, label="PZ (Recall)")
    ax.hlines(PZ["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.5, label="PZ (F1)")

    ax.set_xlabel("K (top-k retrieved)")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval-Only Vector Quality vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "quality_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'quality_vs_k.png'}")


def create_just_vector_time_graph(output_path: Path) -> None:
    """Latency vs K for retrieval-only vector index."""
    fig, ax = plt.subplots(figsize=(8, 5))

    K = JUST_VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    ax.plot([0], [0], "k+", markersize=10, markeredgewidth=2, label="Origin")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["time_s"], "o-", label="Just Vector", color="C0", linewidth=2)
    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")

    ax.set_xlabel("K (top-k retrieved)")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("Retrieval-Only Vector Latency vs K")
    ax.set_xticks([0] + K)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path / "latency_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'latency_vs_k.png'}")


def create_just_vector_cost_graph(output_path: Path) -> None:
    """Cost vs K for retrieval-only vector index."""
    fig, ax = plt.subplots(figsize=(8, 5))

    K = JUST_VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    ax.plot([0], [0], "k+", markersize=10, markeredgewidth=2, label="Origin")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["cost_usd"], "o-", label="Just Vector", color="C0", linewidth=2)
    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")

    ax.set_xlabel("K (top-k retrieved)")
    ax.set_ylabel("Average Cost ($)")
    ax.set_title("Retrieval-Only Vector Cost vs K")
    ax.set_xticks([0] + K)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path / "cost_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'cost_vs_k.png'}")


def create_just_vector_combined_graph(output_path: Path) -> None:
    """All metrics for retrieval-only vector index in a 2x2 subplot grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    K = JUST_VECTOR_INDEX_DATA["K"]
    x_min, x_max = 0, 400

    ax = axes[0, 0]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, JUST_VECTOR_INDEX_DATA["precision"], "o-", label="Just Vector (P)", color="C0")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["recall"], "s-", label="Just Vector (R)", color="C1")
    ax.plot(K, JUST_VECTOR_INDEX_DATA["f1"], "^-", label="Just Vector (F1)", color="C2")
    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles="--", alpha=0.6)
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", alpha=0.6)
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles="--", alpha=0.6)
    ax.hlines(PZ["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.4)
    ax.hlines(PZ["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.4)
    ax.hlines(PZ["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.4)
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, JUST_VECTOR_INDEX_DATA["recall"], "o-", label="Just Vector", color="C0", linewidth=2)
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("K")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, JUST_VECTOR_INDEX_DATA["time_s"], "o-", label="Just Vector", color="C0", linewidth=2)
    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("K")
    ax.set_ylabel("Time (s)")
    ax.set_title("Latency vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=2)
    ax.plot(K, JUST_VECTOR_INDEX_DATA["cost_usd"], "o-", label="Just Vector", color="C0", linewidth=2)
    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("K")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost vs K")
    ax.set_xticks([0] + K)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Evaluation Results: Retrieval-Only Vector Index vs LOTUS vs PZ", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "eval_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'eval_comparison.png'}")


def create_flat_quality_graph(output_path: Path) -> None:
    """Precision, Recall, F1 across flat-index configurations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(FLAT_INDEX_DATA["label"])))
    x_min, x_max = 0, len(x) - 1

    ax.plot(x, FLAT_INDEX_DATA["precision"], "o-", label="Flat Index (Precision)", color="C0")
    ax.plot(x, FLAT_INDEX_DATA["recall"], "s-", label="Flat Index (Recall)", color="C1")
    ax.plot(x, FLAT_INDEX_DATA["f1"], "^-", label="Flat Index (F1)", color="C2")

    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles="--", alpha=0.7, label="LOTUS (Precision)")
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", alpha=0.7, label="LOTUS (Recall)")
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles="--", alpha=0.7, label="LOTUS (F1)")
    ax.hlines(PZ["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.5, label="PZ (Precision)")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.5, label="PZ (Recall)")
    ax.hlines(PZ["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.5, label="PZ (F1)")

    ax.set_xlabel("Flat Index Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Flat Index Quality by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "quality_by_config.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'quality_by_config.png'}")


def create_flat_time_graph(output_path: Path) -> None:
    """Latency across flat-index configurations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(FLAT_INDEX_DATA["label"])))
    x_min, x_max = 0, len(x) - 1

    ax.plot(x, FLAT_INDEX_DATA["time_s"], "o-", label="Flat Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")

    ax.set_xlabel("Flat Index Configuration")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("Flat Index Latency by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "latency_by_config.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'latency_by_config.png'}")


def create_flat_cost_graph(output_path: Path) -> None:
    """Cost across flat-index configurations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(FLAT_INDEX_DATA["label"])))
    x_min, x_max = 0, len(x) - 1

    ax.plot(x, FLAT_INDEX_DATA["cost_usd"], "o-", label="Flat Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")

    ax.set_xlabel("Flat Index Configuration")
    ax.set_ylabel("Average Cost ($)")
    ax.set_title("Flat Index Cost by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "cost_by_config.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'cost_by_config.png'}")


def create_flat_combined_graph(output_path: Path) -> None:
    """Flat-index quality, recall, time, and cost in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = list(range(len(FLAT_INDEX_DATA["label"])))
    x_min, x_max = 0, len(x) - 1

    ax = axes[0, 0]
    ax.plot(x, FLAT_INDEX_DATA["precision"], "o-", label="Flat (P)", color="C0")
    ax.plot(x, FLAT_INDEX_DATA["recall"], "s-", label="Flat (R)", color="C1")
    ax.plot(x, FLAT_INDEX_DATA["f1"], "^-", label="Flat (F1)", color="C2")
    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles="--", alpha=0.6)
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", alpha=0.6)
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles="--", alpha=0.6)
    ax.hlines(PZ["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.4)
    ax.hlines(PZ["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.4)
    ax.hlines(PZ["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.4)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, FLAT_INDEX_DATA["recall"], "o-", label="Flat Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Recall")
    ax.set_title("Recall by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, FLAT_INDEX_DATA["time_s"], "o-", label="Flat Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Time (s)")
    ax.set_title("Latency by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, FLAT_INDEX_DATA["cost_usd"], "o-", label="Flat Index", color="C0", linewidth=2)
    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(FLAT_INDEX_DATA["label"])
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Evaluation Results: Flat Index vs LOTUS vs PZ", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "eval_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'eval_comparison.png'}")


def create_summary_comparison_graph(output_path: Path) -> None:
    """Compare vector, flat, LOTUS, and PZ on quality, latency, and cost."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    vector_x = VECTOR_INDEX_DATA["K"]
    flat_x = [50, 100]
    flat_precision = [0.9817, 0.9817]
    flat_recall = [0.7083, 0.7193]
    flat_f1 = [0.7974, 0.8094]
    flat_time = [82.93, 85.87]
    flat_cost = [0.2637, 0.4520]
    x_min, x_max = 0, 400

    # Quality panel shows precision, recall, and F1 together.
    ax = axes[0]
    ax.plot(vector_x, VECTOR_INDEX_DATA["precision"], "o-", label="Vector (Precision)", color="C0")
    ax.plot(vector_x, VECTOR_INDEX_DATA["recall"], "s-", label="Vector (Recall)", color="C1")
    ax.plot(vector_x, VECTOR_INDEX_DATA["f1"], "^-", label="Vector (F1)", color="C2")
    ax.plot(flat_x, flat_precision, "o--", label="Flat (Precision)", color="C3")
    ax.plot(flat_x, flat_recall, "s--", label="Flat (Recall)", color="C4")
    ax.plot(flat_x, flat_f1, "^--", label="Flat (F1)", color="C5")
    ax.hlines(LOTUS["precision"], x_min, x_max, colors="C0", linestyles=":", alpha=0.6, label="LOTUS (Precision)")
    ax.hlines(LOTUS["recall"], x_min, x_max, colors="C1", linestyles=":", alpha=0.6, label="LOTUS (Recall)")
    ax.hlines(LOTUS["f1"], x_min, x_max, colors="C2", linestyles=":", alpha=0.6, label="LOTUS (F1)")
    ax.hlines(PZ["precision"], x_min, x_max, colors="C3", linestyles=":", alpha=0.6, label="PZ (Precision)")
    ax.hlines(PZ["recall"], x_min, x_max, colors="C4", linestyles=":", alpha=0.6, label="PZ (Recall)")
    ax.hlines(PZ["f1"], x_min, x_max, colors="C5", linestyles=":", alpha=0.6, label="PZ (F1)")
    ax.set_title("Quality")
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(vector_x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(vector_x, VECTOR_INDEX_DATA["time_s"], "o-", label="Vector Index", color="C0", linewidth=2)
    ax.plot(flat_x, flat_time, "s--", label="Flat Index", color="C3", linewidth=2)
    ax.hlines(LOTUS["time_s"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["time_s"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_title("Latency")
    ax.set_xlabel("K")
    ax.set_ylabel("Time (s)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xticks(vector_x)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.plot(vector_x, VECTOR_INDEX_DATA["cost_usd"], "o-", label="Vector Index", color="C0", linewidth=2)
    ax.plot(flat_x, flat_cost, "s--", label="Flat Index", color="C3", linewidth=2)
    ax.hlines(LOTUS["cost_usd"], x_min, x_max, colors="C1", linestyles="--", linewidth=2, label="LOTUS")
    ax.hlines(PZ["cost_usd"], x_min, x_max, colors="C2", linestyles=":", linewidth=2, label="PZ")
    ax.set_title("Cost")
    ax.set_xlabel("K")
    ax.set_ylabel("Cost ($)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xticks(vector_x)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Summary Comparison: Flat Index, Vector Index, LOTUS, and PZ", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path / "summary_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {output_path / 'summary_comparison.png'}")


def _build_pareto_points() -> list[dict]:
    """Build shared points for Pareto frontier plots."""
    points = []
    for k, recall, f1, time_s, cost_usd in zip(
        VECTOR_INDEX_DATA["K"],
        VECTOR_INDEX_DATA["recall"],
        VECTOR_INDEX_DATA["f1"],
        VECTOR_INDEX_DATA["time_s"],
        VECTOR_INDEX_DATA["cost_usd"],
    ):
        points.append(
            {
                "label": f"Vector K={k}",
                "family": "vector",
                "recall": recall,
                "f1": f1,
                "time_s": time_s,
                "cost_usd": cost_usd,
            }
        )

    points.append(
        {
            "label": "Just Vector K=20",
            "family": "just_vector",
            "recall": JUST_VECTOR_INDEX_DATA["recall"][0],
            "f1": JUST_VECTOR_INDEX_DATA["f1"][0],
            "time_s": JUST_VECTOR_INDEX_DATA["time_s"][0],
            "cost_usd": JUST_VECTOR_INDEX_DATA["cost_usd"][0],
        }
    )

    flat_labels = FLAT_INDEX_DATA["label"]
    for label, recall, f1, time_s, cost_usd in zip(
        flat_labels,
        FLAT_INDEX_DATA["recall"],
        FLAT_INDEX_DATA["f1"],
        FLAT_INDEX_DATA["time_s"],
        FLAT_INDEX_DATA["cost_usd"],
    ):
        points.append(
            {
                "label": f"Flat {label.replace(chr(10), ' ')}",
                "family": "flat",
                "recall": recall,
                "f1": f1,
                "time_s": time_s,
                "cost_usd": cost_usd,
            }
        )

    points.append(
        {
            "label": "LOTUS",
            "family": "lotus",
            "recall": LOTUS["recall"],
            "f1": LOTUS["f1"],
            "time_s": LOTUS["time_s"],
            "cost_usd": LOTUS["cost_usd"],
        }
    )
    points.append(
        {
            "label": "PZ",
            "family": "pz",
            "recall": PZ["recall"],
            "f1": PZ["f1"],
            "time_s": PZ["time_s"],
            "cost_usd": PZ["cost_usd"],
        }
    )
    return points


def _create_pareto_frontier_graph(
    output_path: Path,
    metric_key: str,
    metric_label: str,
    filename: str,
) -> None:
    """Plot Pareto frontiers for a given quality metric vs latency and cost."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    points = _build_pareto_points()

    styles = {
        "vector": {"color": "C0", "marker": "o", "label": "Vector Index"},
        "just_vector": {"color": "C6", "marker": "P", "label": "Just Vector Index"},
        "flat": {"color": "C3", "marker": "s", "label": "Flat Index"},
        "lotus": {"color": "C1", "marker": "^", "label": "LOTUS"},
        "pz": {"color": "C2", "marker": "D", "label": "PZ"},
    }

    def plot_panel(ax, x_key: str, x_label: str, title: str) -> None:
        seen_families = set()
        for point in points:
            style = styles[point["family"]]
            label = style["label"] if point["family"] not in seen_families else None
            seen_families.add(point["family"])
            ax.scatter(point[x_key], point[metric_key], color=style["color"], marker=style["marker"], s=70, label=label)
            ax.annotate(
                point["label"],
                (point[x_key], point[metric_key]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

        frontier = _pareto_frontier(points, x_key=x_key, y_key=metric_key)
        ax.plot(
            [p[x_key] for p in frontier],
            [p[metric_key] for p in frontier],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Pareto frontier",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(title)
        metric_values = [p[metric_key] for p in points]
        min_y = min(metric_values)
        max_y = max(metric_values)
        padding = max(0.02, (max_y - min_y) * 0.08)
        ax.set_ylim(max(0.0, min_y - padding), min(1.0, max_y + padding))
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_panel(axes[0], x_key="time_s", x_label="Latency (s)", title=f"Pareto Frontier: {metric_label} vs Latency")
    plot_panel(axes[1], x_key="cost_usd", x_label="Cost ($)", title=f"Pareto Frontier: {metric_label} vs Cost")

    fig.tight_layout()
    fig.savefig(output_path / filename, dpi=150)
    plt.close()
    print(f"Saved {output_path / filename}")


def create_recall_pareto_frontier_graph(output_path: Path) -> None:
    """Plot Pareto frontiers for recall vs latency and recall vs cost."""
    _create_pareto_frontier_graph(
        output_path,
        metric_key="recall",
        metric_label="Recall",
        filename="pareto_frontier_recall.png",
    )


def create_f1_pareto_frontier_graph(output_path: Path) -> None:
    """Plot Pareto frontiers for F1 vs latency and F1 vs cost."""
    _create_pareto_frontier_graph(
        output_path,
        metric_key="f1",
        metric_label="F1",
        filename="pareto_frontier_f1.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Create evaluation graphs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/graphs",
        help="Output directory for graph images",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vector_output = output_path / "vector_index"
    just_vector_output = output_path / "just_vector_index"
    flat_output = output_path / "flat_index"
    vector_output.mkdir(parents=True, exist_ok=True)
    just_vector_output.mkdir(parents=True, exist_ok=True)
    flat_output.mkdir(parents=True, exist_ok=True)

    create_vector_quality_graph(vector_output)
    create_vector_time_graph(vector_output)
    create_vector_cost_graph(vector_output)
    create_vector_combined_graph(vector_output)

    create_just_vector_quality_graph(just_vector_output)
    create_just_vector_time_graph(just_vector_output)
    create_just_vector_cost_graph(just_vector_output)
    create_just_vector_combined_graph(just_vector_output)

    create_flat_quality_graph(flat_output)
    create_flat_time_graph(flat_output)
    create_flat_cost_graph(flat_output)
    create_flat_combined_graph(flat_output)

    create_summary_comparison_graph(output_path)
    create_recall_pareto_frontier_graph(output_path)
    create_f1_pareto_frontier_graph(output_path)

    print(f"\nAll graphs saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()
