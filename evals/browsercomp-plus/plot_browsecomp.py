from pathlib import Path

import matplotlib.pyplot as plt

N_QUERIES = 25

SYSTEMS = [
    {
        "label": "SemFilter all docs",
        "precision": 0.7055,
        "recall": 0.8722,
        "f1": 0.7145,
        "time_s": 92.34,
        "cost_usd": 1.99610043,  # already average cost/query
        "color": "C0",
        "marker": "o",
    },
    {
        "label": "SemFilter summaries",
        "precision": 0.6733,
        "recall": 0.8396,
        "f1": 0.7082,
        "time_s": 85.59,
        "cost_usd": 0.88522067,  # already average cost/query
        "color": "C1",
        "marker": "s",
    },
    {
        "label": "PZ",
        "precision": 0.8743,
        "recall": 0.5933,
        "f1": 0.6690,
        "time_s": 68.4402,
        "cost_usd": 1.5292 ,  
        "color": "C2",
        "marker": "^",
    },
    {
        "label": "LOTUS",
        "precision": 0.8100,
        "recall": 0.4071,
        "f1": 0.4938,
        "time_s": 65.5679,
        "cost_usd": 1.3725 , 
        "color": "C3",
        "marker": "D",
    },
]


VECTOR_INDEX_DATA = {
    "K": [200, 400, 600, 800, 1000],
    "precision": [0.0050, 0.0047, 0.0053, 0.0055, 0.0055],
    "recall": [0.2243, 0.3705, 0.6113, 0.8177, 1.0000],
    "f1": [0.0097, 0.0093, 0.0106, 0.0109, 0.0109],
    "time_s": [0.49, 0.51, 0.72, 0.58, 0.85],
    "cost_usd": [0.00000241, 0.00000241, 0.00000241, 0.00000241, 0.00000241],
}


def pareto_frontier(points, x_key, y_key):
    """Pareto-optimal points minimizing x and maximizing y."""
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


def plot_pareto(points, x_key, y_key, x_label, y_label, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot all points
    for p in points:
        ax.scatter(
            p[x_key],
            p[y_key],
            s=90,
            color=p["color"],
            marker=p["marker"],
            label=p["label"],
            zorder=3,
        )
        ax.annotate(
            p["label"],
            (p[x_key], p[y_key]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    # Pareto frontier
    frontier = pareto_frontier(points, x_key, y_key)
    ax.plot(
        [p[x_key] for p in frontier],
        [p[y_key] for p in frontier],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Pareto frontier",
        zorder=2,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def create_pareto_frontier_graph(points, metric_key, metric_label, output_path):
    """Plot Pareto frontiers for one metric vs latency and cost."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_panel(ax, x_key, x_label, title):
        for p in points:
            ax.scatter(
                p[x_key],
                p[metric_key],
                s=90,
                color=p["color"],
                marker=p["marker"],
                label=p["label"],
                zorder=3,
            )
            ax.annotate(
                p["label"],
                (p[x_key], p[metric_key]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

        frontier = pareto_frontier(points, x_key, metric_key)
        ax.plot(
            [p[x_key] for p in frontier],
            [p[metric_key] for p in frontier],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Pareto frontier",
            zorder=2,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)

        metric_values = [p[metric_key] for p in points]
        min_y = min(metric_values)
        max_y = max(metric_values)
        padding = max(0.02, (max_y - min_y) * 0.08)
        ax.set_ylim(max(0.0, min_y - padding), min(1.0, max_y + padding))

    plot_panel(
        axes[0],
        x_key="time_s",
        x_label="Latency (s)",
        title=f"Pareto Frontier: {metric_label} vs Latency",
    )
    plot_panel(
        axes[1],
        x_key="cost_usd",
        x_label="Cost ($)",
        title=f"Pareto Frontier: {metric_label} vs Cost",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_vector_recall_vs_k(output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        VECTOR_INDEX_DATA["K"],
        VECTOR_INDEX_DATA["recall"],
        "o-",
        color="C4",
        linewidth=2,
        markersize=7,
        label="Vector index only",
    )

    for k, recall in zip(
        VECTOR_INDEX_DATA["K"], VECTOR_INDEX_DATA["recall"], strict=True
    ):
        ax.annotate(
            f"K={k}",
            (k, recall),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("K")
    ax.set_ylabel("Average Recall")
    ax.set_title("BrowserComp+: Vector Index Recall vs K")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    out_dir = Path("browsecomp_plus_pareto_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto(
        SYSTEMS,
        x_key="time_s",
        y_key="f1",
        x_label="Average Time (s)",
        y_label="Average F1",
        title="BrowserComp+: F1 vs Time",
        output_path=out_dir / "f1_vs_time.png",
    )

    plot_pareto(
        SYSTEMS,
        x_key="cost_usd",
        y_key="f1",
        x_label="Average Cost per Query ($)",
        y_label="Average F1",
        title="BrowserComp+: F1 vs Cost",
        output_path=out_dir / "f1_vs_cost.png",
    )

    plot_pareto(
        SYSTEMS,
        x_key="time_s",
        y_key="recall",
        x_label="Average Time (s)",
        y_label="Average Recall",
        title="BrowserComp+: Recall vs Time",
        output_path=out_dir / "recall_vs_time.png",
    )

    plot_pareto(
        SYSTEMS,
        x_key="cost_usd",
        y_key="recall",
        x_label="Average Cost per Query ($)",
        y_label="Average Recall",
        title="BrowserComp+: Recall vs Cost",
        output_path=out_dir / "recall_vs_cost.png",
    )

    create_pareto_frontier_graph(
        SYSTEMS,
        metric_key="f1",
        metric_label="F1",
        output_path=out_dir / "pareto_frontier_f1.png",
    )

    create_pareto_frontier_graph(
        SYSTEMS,
        metric_key="recall",
        metric_label="Recall",
        output_path=out_dir / "pareto_frontier_recall.png",
    )

    plot_vector_recall_vs_k(
        out_dir / "vector_recall_vs_k.png",
    )


if __name__ == "__main__":
    main()