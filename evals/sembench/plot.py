"""
Created on June 4, 2025

@author: Jiale Lao

An enhanced plotting module with improved visualizations.
"""

import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Set

from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns
import tomli
from matplotlib.lines import Line2D
from scipy.stats import gmean


# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2

latex_full_width = 7.031875  # Width of a figure stretching across the whole page in ACM two-column layout in inches.
latex_single_column_width = 3.349263889  # Width of a figure stretching a single column in ACM two-column layout in inches.


class BenchmarkPlotter:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.files_dir = self.base_dir / "files"
        self.figures_dir = self.base_dir / "figures"

        # Create figures directory if it doesn't exist
        self.figures_dir.mkdir(exist_ok=True)

        # Professional academic color palette optimized for SIGMOD papers
        # Colors chosen for high contrast, print-friendly, and colorblind
        # accessibility
        self.base_colors = [
            "#1f77b4",  # Blue - primary academic blue
            "#d62728",  # Red - strong contrast red
            "#2ca02c",  # Green - academic green
            "#ff7f0e",  # Orange - vibrant orange
            "#9467bd",  # Purple - professional purple
            "#8c564b",  # Brown - warm brown
            "#e377c2",  # Pink - distinctive pink
            "#7f7f7f",  # Gray - neutral gray
            "#bcbd22",  # Olive - unique olive
            "#17becf",  # Cyan - clear cyan
            "#1a55FF",  # Royal Blue - deeper blue
            "#c44e52",  # Brick Red - muted red
            "#55A868",  # Forest Green - deeper green
            "#DD8452",  # Burnt Orange - warmer orange
            "#B3722A",  # Golden Brown - rich brown
            "#CC7EC0",  # Mauve - softer pink
            "#A4A4A4",  # Medium Gray - balanced gray
            "#B8BD26",  # Yellow Green - bright accent
            "#4C72B0",  # Steel Blue - professional blue
            "#DD4477",  # Rose - elegant pink
            "#55A4A6",  # Teal - balanced teal
            "#C49C94",  # Dusty Rose - subtle accent
            "#8172B3",  # Lavender - soft purple
        ]

        self.system_colors = {
            "lotus": "#FF730F",  # Orange
            "bigquery": "#d62728",  # Red
            "bigquery (gemini-2.5-flash)": "#82D219",
            "bigquery (gemini-2.5-pro)": "#0F9C66",
            "ground_truth": "#2ca02c",  # Green
            "snowflake": "#29B5E8",  # Snowflake blue
            "palimpzest": "#9467bd",  # Purple
            "palimpzest (gpt-4o)": "#50138a",
            "palimpzest (gpt-4o-mini)": "#ca92ff",
            "flockmtl": "#8c564b",  # Brown
            "caesura": "#e377c2",  # Pink
            "thalamusdb": "#7f7f7f",  # Gray
            "lotus_exact_SF100": "#bcbd22",  # Olive
            "lotus_optimized_SF100": "#17becf",  # Cyan
            "lotus (gemini-2.5-pro)": "#583200",
            "lotus (gemini-2.5-flash)": "#a1870f",
            "lotus (gemini-2.5-flash-lite)": "#e2c74f",
        }

        # Define hatching patterns for better distinction in print
        self.system_patterns = {
            "lotus": None,
            "bigquery": "///",
            "ground_truth": "\\\\\\",
            "lotus_exact_SF100": "|||",
            "lotus_optimized_SF100": "---",
            "system3": "+++",
            "system4": "xxx",
        }

        # System markers for visual distinction
        self.system_markers = {
            "lotus": "o",  # Circle
            "bigquery": "s",  # Square
            "snowflake": "^",  # Triangle up
            "palimpzest": "D",  # Diamond
            "thalamusdb": "v",  # Triangle down
        }

    def get_system_color(self, system_name):
        """Get color for a system with better distinction for many systems."""
        if system_name not in self.system_colors:
            # Assign colors from the base palette in order
            color_index = len(self.system_colors) % len(self.base_colors)
            self.system_colors[system_name] = self.base_colors[color_index]
        return self.system_colors[system_name]

    def get_system_pattern(self, system_name):
        """Get hatch pattern for a system."""
        if system_name not in self.system_patterns:
            patterns = [
                "...",
                "ooo",
                "***",
                "///",
                "\\\\\\",
                "|||",
                "---",
                "+++",
                "xxx",
            ]
            self.system_patterns[system_name] = patterns[
                len(self.system_patterns) % len(patterns)
            ]
        return self.system_patterns[system_name]

    def unify_accuracy_metric(self, metric):
        if "metric_type" in metric and "accuracy" in metric:
            return metric["metric_type"], metric["accuracy"]
        elif "f1_score" in metric:
            return "f1_score", metric["f1_score"]
        elif "relative_error" in metric:
            return "relative_error", (
                1 / (1 + metric["relative_error"])
                if metric["relative_error"] is not None
                else None
            )
        elif "spearman_correlation" in metric:
            return "spearman_correlation", metric["spearman_correlation"]
        else:
            return None, None

    def max_overlap_of_n_sets(self, sets: List[Set[Any]]) -> Set[Any]:
        """
        Computes the maximum overlap of multiple lists, e.g.:
         list1: [a, b, c, d]
         list2: [b, c, d]
         list3: [b, d, e]
        Then the result should be [b, d], because it appears in all lists.
        """
        intersection = sets[0]
        for s in sets:
            intersection = intersection & s
        return intersection

    def add_legend_with_layout(
        self, ax, systems, location="lower center", bbox_anchor=(0.5, 1.02)
    ):
        """Add legend with proper layout - two rows if systems > 5."""
        ncol = len(systems) if len(systems) <= 5 else (len(systems) + 1) // 2
        ax.legend(
            loc=location,
            bbox_to_anchor=bbox_anchor,
            ncol=ncol,
            frameon=False,
            fontsize=10,
        )

    def add_red_mark(self, ax, x_pos, y_pos, mark_type="×"):
        """Add red mark for missing data or zero values."""
        ax.text(
            x_pos,
            y_pos,
            mark_type,
            ha="center",
            va="center",
            fontsize=14,
            color="red",
            fontweight="bold",
            zorder=10,
        )

    def format_value(self, value, metric_type):
        """Format values for display on bars."""
        if metric_type == "execution_time":
            if value >= 100:
                return f"{value:.0f}s"
            elif value >= 10:
                return f"{value:.1f}s"
            else:
                return f"{value:.2f}s"
        elif metric_type == "token_usage":
            if value >= 1000000:
                return f"{value/1000000:.1f}M"
            elif value >= 1000:
                return f"{value/1000:.0f}K"
            else:
                return f"{int(value)}"
        elif metric_type == "money_cost":
            if value >= 1:
                return f"${value:.2f}"
            elif value >= 0.01:
                return f"${value:.3f}"
            else:
                return f"${value:.4f}"
        elif metric_type in [
            "precision",
            "recall",
            "f1_score",
            "spearman_correlation",
            "kendall_tau",
        ]:
            if value == 1:
                return "1"
            elif value == 0:
                return "0"
            return f"{value:.2f}"
        elif metric_type == "mean_absolute_percentage_error":
            if value == 0:
                return "0%"
            return f"{value:.1f}%"
        elif metric_type in ["relative_error", "absolute_error"]:
            if value == 0:
                return "0"
            elif value >= 1:
                return f"{value:.2f}"
            else:
                return f"{value:.3f}"
        else:
            if value == 0:
                return "0"
            elif value >= 1:
                return f"{value:.2f}"
            else:
                return f"{value:.3f}"

    def get_use_cases(self):
        """Get all use cases from the files directory."""
        use_cases = []
        if not self.files_dir.exists():
            print(f"Files directory {self.files_dir} not found!")
            return use_cases

        for item in self.files_dir.iterdir():
            if item.is_dir() and (item / "metrics").exists():
                use_cases.append(item.name)

        return sorted(use_cases)

    def load_metrics_data(self, use_case):
        """Load metrics JSON files for a given use case (excluding scalability
        files)."""
        metrics_dir = self.files_dir / use_case / "metrics"
        metrics_data = {}

        if not metrics_dir.exists():
            print(f"Metrics directory {metrics_dir} not found!")
            return metrics_data

        for json_file in metrics_dir.glob("*.json"):
            # Skip files with scaling factors
            if "_sf" in json_file.name:
                continue

            system_name = json_file.stem
            try:
                with open(json_file, "r") as f:
                    metrics_data[system_name] = json.load(f)
                print(f"Loaded metrics for {system_name} in {use_case}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return metrics_data

    def get_system_subfolders(self, use_case):
        """Get system subfolders from the metrics directory, excluding 'temp'."""
        metrics_dir = self.files_dir / use_case / "metrics"
        system_folders = []

        if not metrics_dir.exists():
            return system_folders

        for item in metrics_dir.iterdir():
            if item.is_dir() and item.name != "temp":
                system_folders.append(item.name)

        return sorted(system_folders)

    def load_system_metrics_data(self, use_case, system_name):
        """Load metrics JSON files for a specific system subfolder."""
        system_metrics_dir = self.files_dir / use_case / "metrics" / system_name
        metrics_data = {}

        if not system_metrics_dir.exists():
            print(f"System metrics directory {system_metrics_dir} not found!")
            return metrics_data

        for json_file in system_metrics_dir.glob("*.json"):
            # Skip files with scaling factors
            if "_sf" in json_file.name:
                continue

            system_variant = json_file.stem
            try:
                with open(json_file, "r") as f:
                    metrics_data[system_variant] = json.load(f)
                print(
                    f"Loaded metrics for {system_variant} in {use_case}/{system_name}"  # noqa: E501
                )
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return metrics_data

    def plot_execution_time(self, metrics_data, use_case, system_name=None):
        """Plot execution time comparison with the legend between the title and
        the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # --- Data Preparation ---
        all_query_ids = {
            k.replace("Q", "")
            for data in metrics_data.values()
            for k, v in data.items()
            if "execution_time" in v
        }
        query_ids = sorted(all_query_ids, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
        systems = sorted(metrics_data.keys())

        x_pos = np.arange(len(query_ids))
        width = 0.8 / len(systems)

        # --- Plotting Bars ---
        for i, sys_name in enumerate(systems):
            # Gather data and track missing queries
            execution_times = []
            missing_queries = []
            for qid in query_ids:
                system_data = metrics_data.get(sys_name, {})
                query_data = system_data.get(f"Q{qid}", {})
                if (
                    "execution_time" in query_data
                    and query_data["execution_time"] > 0
                ):
                    execution_times.append(query_data["execution_time"])
                    missing_queries.append(False)
                else:
                    execution_times.append(0.001)  # Small value for log scale
                    missing_queries.append(True)

            color = self.get_system_color(sys_name)
            pattern = self.get_system_pattern(sys_name)
            bars = ax.bar(
                x_pos + i * width - (len(systems) - 1) * width / 2,
                execution_times,
                width,
                label=sys_name,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.0,
                hatch=pattern,
            )

            # Add text labels on top of bars or red marks for missing data
            for bar, value, is_missing in zip(
                bars, execution_times, missing_queries
            ):
                height = bar.get_height()
                if is_missing:
                    # Add red mark for missing data
                    self.add_red_mark(
                        ax, bar.get_x() + bar.get_width() / 2.0, height * 1.5
                    )
                elif value > 0.001:  # Only show text for actual values
                    text = self.format_value(value, "execution_time")
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height * 1.02,
                        text,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        ax.set_yscale("log")

        # --- Styling and Labels ---
        ax.set_xlabel("Query ID", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Execution Time (seconds, log scale)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Q{qid}" for qid in query_ids])

        # This combination correctly positions the title and legend
        ax.set_title(
            f'Execution Time Comparison - {use_case.replace("_", " ").title()}',
            fontsize=14,
            fontweight="bold",
            pad=40,
        )  # pad creates space for the legend

        self.add_legend_with_layout(ax, systems, "lower center", (0.5, 1.0))

        ax.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        # --- Save Figure ---
        if system_name:
            output_dir = self.figures_dir / use_case / system_name
        else:
            output_dir = self.figures_dir / use_case
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(
            output_dir / "execution_time.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        if system_name:
            print(f"Saved execution_time.png for {use_case}/{system_name}")
        else:
            print(f"Saved execution_time.png for {use_case}")

    def plot_cost(self, metrics_data, use_case, system_name=None):
        """Plot cost comparison (token usage and money cost) in two rows."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Collect all query IDs first
        all_query_ids = set()
        for sys_name, data in metrics_data.items():
            for query_key, metrics in data.items():
                if "token_usage" in metrics or "money_cost" in metrics:
                    try:
                        query_id = query_key.replace("Q", "")
                        all_query_ids.add(query_id)
                    except ValueError:
                        # Skip queries with non-numeric IDs like Q3a
                        continue

        query_ids = sorted(all_query_ids, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
        systems = sorted(metrics_data.keys())

        x_pos = np.arange(len(query_ids))
        width = 0.8 / len(systems)

        # --- Token Usage subplot ---
        for i, sys_name in enumerate(systems):
            token_usage = []
            missing_or_zero = []
            for qid in query_ids:
                system_data = metrics_data.get(sys_name, {})
                query_data = system_data.get(f"Q{qid}", {})
                token_val = query_data.get("token_usage", 0)
                if "token_usage" not in query_data or token_val == 0:
                    token_usage.append(1)  # Small value for log scale
                    missing_or_zero.append(True)
                else:
                    token_usage.append(token_val)
                    missing_or_zero.append(False)

            color = self.get_system_color(sys_name)
            pattern = self.get_system_pattern(sys_name)
            bars = ax1.bar(
                x_pos + i * width - (len(systems) - 1) * width / 2,
                token_usage,
                width,
                label=sys_name,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.0,
                hatch=pattern,
            )

            for bar, value, is_missing_or_zero in zip(
                bars, token_usage, missing_or_zero
            ):
                height = bar.get_height()
                if is_missing_or_zero:
                    # Add red mark for missing data or zero values
                    self.add_red_mark(
                        ax1, bar.get_x() + bar.get_width() / 2.0, height * 1.5
                    )
                elif value > 1:
                    text = self.format_value(value, "token_usage")
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height * 1.02,
                        text,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        ax1.set_yscale("log")
        ax1.set_xlabel("Query ID", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Token Usage (log scale)", fontsize=12, fontweight="bold"
        )
        ax1.set_title(
            "Token Usage Comparison", fontsize=13, fontweight="bold", y=1.15
        )  # Move title up
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"Q{qid}" for qid in query_ids])
        self.add_legend_with_layout(ax1, systems, "lower center", (0.5, 1.01))
        ax1.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")
        ax1.set_axisbelow(True)
        ax1.spines[["top", "right"]].set_visible(False)

        # --- Money Cost subplot ---
        for i, sys_name in enumerate(systems):
            money_cost = []
            missing_or_zero = []
            for qid in query_ids:
                system_data = metrics_data.get(sys_name, {})
                query_data = system_data.get(f"Q{qid}", {})
                cost_val = query_data.get("money_cost", 0)
                if "money_cost" not in query_data or cost_val == 0:
                    money_cost.append(0.0001)  # Small value for log scale
                    missing_or_zero.append(True)
                else:
                    money_cost.append(cost_val)
                    missing_or_zero.append(False)

            color = self.get_system_color(sys_name)
            pattern = self.get_system_pattern(sys_name)
            bars = ax2.bar(
                x_pos + i * width - (len(systems) - 1) * width / 2,
                money_cost,
                width,
                label=sys_name,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.0,
                hatch=pattern,
            )

            for bar, value, is_missing_or_zero in zip(
                bars, money_cost, missing_or_zero
            ):
                height = bar.get_height()
                if is_missing_or_zero:
                    # Add red mark for missing data or zero values
                    self.add_red_mark(
                        ax2, bar.get_x() + bar.get_width() / 2.0, height * 1.5
                    )
                elif value > 0.0001:
                    text = self.format_value(value, "money_cost")
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height * 1.02,
                        text,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        ax2.set_yscale("log")
        ax2.set_xlabel("Query ID", fontsize=12, fontweight="bold")
        ax2.set_ylabel(
            "Money Cost ($, log scale)", fontsize=12, fontweight="bold"
        )
        ax2.set_title(
            "Money Cost Comparison", fontsize=13, fontweight="bold", y=1.15
        )  # Move title up
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"Q{qid}" for qid in query_ids])
        self.add_legend_with_layout(ax2, systems, "lower center", (0.5, 1.01))
        ax2.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")
        ax2.set_axisbelow(True)
        ax2.spines[["top", "right"]].set_visible(False)

        plt.suptitle(
            f'Cost Analysis - {use_case.replace("_", " ").title()}',
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        # Adjust layout
        plt.subplots_adjust(hspace=0.6)  # Add vertical space between plots
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save the plot
        if system_name:
            output_dir = self.figures_dir / use_case / system_name
        else:
            output_dir = self.figures_dir / use_case
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "cost.png", dpi=300, bbox_inches="tight")
        plt.close()
        if system_name:
            print(f"Saved cost.png for {use_case}/{system_name}")
        else:
            print(f"Saved cost.png for {use_case}")

    def plot_quality(self, metrics_data, use_case, system_name=None):
        """Plot quality metrics with a combined legend for systems and metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 16)
        )  # Increased height for 3 plots
        systems = sorted(metrics_data.keys())

        # --- Plot 1: Retrieval Quality Metrics ---
        retrieval_metrics = ["precision", "recall", "f1_score"]
        metric_patterns = {"precision": "..", "recall": "xx", "f1_score": "//"}

        retrieval_queries = sorted(
            {
                k.replace("Q", "")
                for data in metrics_data.values()
                for k, v in data.items()
                if any(metric in v for metric in retrieval_metrics)
            }
        )

        if retrieval_queries:
            n_groups = len(retrieval_queries)
            n_metrics = len(retrieval_metrics)
            n_systems = len(systems)

            bar_width = 0.9 / (n_metrics * n_systems)
            x_base = np.arange(n_groups)

            # --- Create Legend Handles ---
            system_handles = [
                mpatches.Patch(color=self.get_system_color(s), label=s)
                for s in systems
            ]
            metric_handles = [
                mpatches.Patch(
                    facecolor="white", edgecolor="black", hatch=p, label=m
                )
                for m, p in metric_patterns.items()
            ]

            for i, sys_name in enumerate(systems):
                system_color = self.get_system_color(sys_name)
                for j, metric in enumerate(retrieval_metrics):
                    metric_values = []
                    missing_queries = []
                    for qid in retrieval_queries:
                        system_data = metrics_data.get(sys_name, {})
                        query_data = system_data.get(f"Q{qid}", {})
                        if metric in query_data:
                            metric_values.append(query_data[metric])
                            missing_queries.append(False)
                        else:
                            metric_values.append(0)
                            missing_queries.append(True)

                    offset = (i * n_metrics + j) * bar_width
                    position = x_base - 0.4 + offset + bar_width / 2

                    bars = ax1.bar(
                        position,
                        metric_values,
                        bar_width,
                        color=system_color,
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.8,
                        hatch=metric_patterns[metric],
                    )

                    # Add text labels on top of bars or red marks for missing
                    # data
                    for bar, value, is_missing in zip(
                        bars, metric_values, missing_queries
                    ):
                        height = bar.get_height()
                        if is_missing:
                            # Add red mark for missing data
                            self.add_red_mark(
                                ax1,
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.05,
                            )
                        else:
                            text = self.format_value(value, metric)
                            ax1.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.01,
                                text,
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                fontweight="bold",
                            )

            # --- Styling and Legend ---
            ax1.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax1.set_title(
                "Retrieval Quality Metrics",
                fontsize=13,
                fontweight="bold",
                pad=60,
            )
            ax1.set_xticks(x_base)
            ax1.set_xticklabels([f"Q{qid}" for qid in retrieval_queries])

            # Create two separate legends and place them side-by-side
            system_ncol = (
                len(system_handles)
                if len(system_handles) <= 5
                else (len(system_handles) + 1) // 2
            )
            system_legend = ax1.legend(
                handles=system_handles,
                title="Systems",
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                ncol=system_ncol,
                frameon=False,
            )
            ax1.add_artist(system_legend)
            _ = ax1.legend(
                handles=metric_handles,
                title="Metrics",
                loc="lower right",
                bbox_to_anchor=(1.0, 1.02),
                ncol=len(metric_handles),
                frameon=False,
            )

            ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax1.set_axisbelow(True)
            ax1.set_ylim(0, 1.1)
            ax1.spines[["top", "right"]].set_visible(False)
        else:
            ax1.text(
                0.5,
                0.5,
                "No Retrieval Quality Data Available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=12,
            )

        # --- Plot 2: Aggregation Quality Metrics ---
        agg_metrics = [
            "relative_error",
            "absolute_error",
            "mean_absolute_percentage_error",
        ]
        agg_patterns = {
            "relative_error": "..",
            "absolute_error": "xx",
            "mean_absolute_percentage_error": "//",
        }

        agg_queries = sorted(
            {
                k.replace("Q", "")
                for data in metrics_data.values()
                for k, v in data.items()
                if any(metric in v for metric in agg_metrics)
            }
        )

        if agg_queries:
            n_groups = len(agg_queries)
            n_metrics = len(agg_metrics)
            n_systems = len(systems)

            bar_width = 0.8 / (n_metrics * n_systems)
            x_base = np.arange(n_groups)

            system_handles = [
                mpatches.Patch(color=self.get_system_color(s), label=s)
                for s in systems
            ]
            metric_handles = [
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=p,
                    label=m.replace("_", " "),
                )
                for m, p in agg_patterns.items()
            ]

            for i, sys_name in enumerate(systems):
                system_color = self.get_system_color(sys_name)
                for j, metric in enumerate(agg_metrics):
                    metric_values = []
                    missing_queries = []
                    for qid in agg_queries:
                        system_data = metrics_data.get(sys_name, {})
                        query_data = system_data.get(f"Q{qid}", {})
                        if metric in query_data:
                            value = query_data[metric]
                            metric_values.append(
                                value if value > 0 else 1e-9
                            )  # Handle 0 values for log scale
                            missing_queries.append(False)
                        else:
                            metric_values.append(
                                1e-9
                            )  # Small value for log scale
                            missing_queries.append(True)

                    offset = (i * n_metrics + j) * bar_width
                    position = x_base - 0.4 + offset + bar_width / 2

                    bars = ax2.bar(
                        position,
                        metric_values,
                        bar_width,
                        color=system_color,
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.8,
                        hatch=agg_patterns[metric],
                    )

                    # Add text labels on top of bars or red marks for missing
                    # data
                    for k, (bar, value, is_missing) in enumerate(
                        zip(bars, metric_values, missing_queries)
                    ):
                        height = bar.get_height()
                        if is_missing:
                            # Add red mark for missing data
                            self.add_red_mark(
                                ax2,
                                bar.get_x() + bar.get_width() / 2.0,
                                height * 1.5,
                            )
                        else:
                            # Get original value from the query data for display
                            qid = agg_queries[k]
                            original_value = (
                                metrics_data[sys_name]
                                .get(f"Q{qid}", {})
                                .get(metric, value)
                            )
                            text = self.format_value(original_value, metric)
                            ax2.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height * 1.05,
                                text,
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                fontweight="bold",
                            )

            # --- Styling and Legend ---
            ax2.set_yscale("log")
            ax2.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax2.set_ylabel(
                "Error Value (log scale)", fontsize=12, fontweight="bold"
            )
            ax2.set_title(
                "Aggregation Quality Metrics",
                fontsize=13,
                fontweight="bold",
                pad=60,
            )
            ax2.set_xticks(x_base)
            ax2.set_xticklabels([f"Q{qid}" for qid in agg_queries])

            system_ncol = (
                len(system_handles)
                if len(system_handles) <= 5
                else (len(system_handles) + 1) // 2
            )
            system_legend = ax2.legend(
                handles=system_handles,
                title="Systems",
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                ncol=system_ncol,
                frameon=False,
            )
            ax2.add_artist(system_legend)
            _ = ax2.legend(
                handles=metric_handles,
                title="Metrics",
                loc="lower right",
                bbox_to_anchor=(1.0, 1.02),
                ncol=len(metric_handles),
                frameon=False,
            )

            ax2.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")
            ax2.set_axisbelow(True)
            ax2.spines[["top", "right"]].set_visible(False)
        else:
            ax2.text(
                0.5,
                0.5,
                "No Aggregation Quality Data Available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )

        plt.suptitle(
            f'Quality Analysis - {use_case.replace("_", " ").title()}',
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.subplots_adjust(hspace=0.5)  # Add space between subplots

        # --- Plot 3: Ranking Quality Metrics ---
        ranking_metrics = ["spearman_correlation", "kendall_tau"]
        ranking_patterns = {"spearman_correlation": "..", "kendall_tau": "xx"}

        ranking_queries = sorted(
            {
                k.replace("Q", "")
                for data in metrics_data.values()
                for k, v in data.items()
                if any(metric in v for metric in ranking_metrics)
            }
        )

        if ranking_queries:
            n_groups = len(ranking_queries)
            n_metrics = len(ranking_metrics)
            n_systems = len(systems)

            bar_width = 0.9 / (n_metrics * n_systems)
            x_base = np.arange(n_groups)

            system_handles = [
                mpatches.Patch(color=self.get_system_color(s), label=s)
                for s in systems
            ]
            metric_handles = [
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=p,
                    label=m.replace("_", " "),
                )
                for m, p in ranking_patterns.items()
            ]

            for i, sys_name in enumerate(systems):
                system_color = self.get_system_color(sys_name)
                for j, metric in enumerate(ranking_metrics):
                    metric_values = []
                    missing_queries = []
                    for qid in ranking_queries:
                        system_data = metrics_data.get(sys_name, {})
                        query_data = system_data.get(f"Q{qid}", {})
                        if metric in query_data:
                            metric_values.append(query_data[metric])
                            missing_queries.append(False)
                        else:
                            metric_values.append(0)
                            missing_queries.append(True)

                    offset = (i * n_metrics + j) * bar_width
                    position = x_base - 0.4 + offset + bar_width / 2

                    bars = ax3.bar(
                        position,
                        metric_values,
                        bar_width,
                        color=system_color,
                        alpha=0.85,
                        edgecolor="black",
                        linewidth=0.8,
                        hatch=ranking_patterns[metric],
                    )

                    # Add text labels on top of bars or red marks for missing
                    # data
                    for bar, value, is_missing in zip(
                        bars, metric_values, missing_queries
                    ):
                        height = bar.get_height()
                        if is_missing:
                            # Add red mark for missing data
                            self.add_red_mark(
                                ax3,
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.05,
                            )
                        else:
                            text = self.format_value(value, metric)
                            ax3.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height + 0.01,
                                text,
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                fontweight="bold",
                            )

            # --- Styling and Legend ---
            ax3.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax3.set_ylabel("Correlation Score", fontsize=12, fontweight="bold")
            ax3.set_title(
                "Ranking Quality Metrics",
                fontsize=13,
                fontweight="bold",
                pad=60,
            )
            ax3.set_xticks(x_base)
            ax3.set_xticklabels([f"Q{qid}" for qid in ranking_queries])

            # Create two separate legends and place them side-by-side
            system_ncol = (
                len(system_handles)
                if len(system_handles) <= 5
                else (len(system_handles) + 1) // 2
            )
            system_legend = ax3.legend(
                handles=system_handles,
                title="Systems",
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                ncol=system_ncol,
                frameon=False,
            )
            ax3.add_artist(system_legend)
            _ = ax3.legend(
                handles=metric_handles,
                title="Metrics",
                loc="lower right",
                bbox_to_anchor=(1.0, 1.02),
                ncol=len(metric_handles),
                frameon=False,
            )

            ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax3.set_axisbelow(True)
            ax3.set_ylim(
                -0.1, 1.1
            )  # Correlation values typically range from -1 to 1
            ax3.spines[["top", "right"]].set_visible(False)
        else:
            ax3.text(
                0.5,
                0.5,
                "No Ranking Quality Data Available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )

        plt.suptitle(
            f'Quality Analysis - {use_case.replace("_", " ").title()}',
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.subplots_adjust(hspace=0.5)  # Add space between subplots

        if system_name:
            output_dir = self.figures_dir / use_case / system_name
        else:
            output_dir = self.figures_dir / use_case
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "quality.png", dpi=300, bbox_inches="tight")
        plt.close()
        if system_name:
            print(f"Saved quality.png for {use_case}/{system_name}")
        else:
            print(f"Saved quality.png for {use_case}")

    def plot_quality_one_metric(
        self, metrics_data, use_case, target_metrics=None, system_name=None
    ):
        """
        Plots specified quality metrics for different systems side-by-side.

        Args:
            metrics_data (dict): A dictionary containing metrics for each
            system. Example: {'SystemA': {'Q1': {'f1_score': 0.9,
            'relative_error': 0.05}, ...}, ...}
            use_case (str): The name of the use case being analyzed (e.g.,
            'financial_summary').
            target_metrics (dict, optional): A dictionary specifying the
            metrics to plot. Defaults to {'retrieval': 'f1_score',
            'aggregation': 'relative_error', 'ranking': 'spearman_correlation'}.
            Example: {'retrieval': 'precision', 'aggregation':
            'absolute_error', 'ranking': 'kendall_tau'}
        """
        # --- 1. Setup and Configuration ---
        if target_metrics is None:
            target_metrics = {}
        retrieval_metric = target_metrics.get("retrieval", "f1_score")
        agg_metric = target_metrics.get("aggregation", "relative_error")
        ranking_metric = target_metrics.get("ranking", "spearman_correlation")

        # Always create 3 subplots for retrieval, aggregation, and ranking
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        systems = sorted(metrics_data.keys())
        n_systems = len(systems)

        # Define retrieval and ranking metrics to avoid overlap
        retrieval_metrics_set = {"precision", "recall", "f1_score"}
        aggregation_metrics_set = {
            "relative_error",
            "absolute_error",
            "mean_absolute_percentage_error",
        }
        ranking_metrics_set = {"spearman_correlation", "kendall_tau"}

        # --- 2. Plot 1: Retrieval Quality Metric ---
        retrieval_queries = sorted(
            {
                k.replace("Q", "")
                for data in metrics_data.values()
                for k, v in data.items()
                if retrieval_metric in v
                and any(metric in v for metric in retrieval_metrics_set)
            }
        )

        if retrieval_queries:
            n_groups = len(retrieval_queries)
            bar_width = 0.8 / n_systems
            x_base = np.arange(n_groups)

            for i, sys_name in enumerate(systems):
                system_color = self.get_system_color(sys_name)

                metric_values = []
                is_missing = []
                for qid in retrieval_queries:
                    query_data = metrics_data.get(sys_name, {}).get(
                        f"Q{qid}", {}
                    )
                    if retrieval_metric in query_data:
                        metric_values.append(query_data[retrieval_metric])
                        is_missing.append(False)
                    else:
                        metric_values.append(
                            0
                        )  # Use 0 for missing data bar height
                        is_missing.append(True)

                offset = i * bar_width
                position = x_base - (0.8 / 2) + offset + bar_width / 2
                bars = ax1.bar(
                    position,
                    metric_values,
                    bar_width,
                    color=system_color,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                    label=sys_name,
                )

                for bar, value, missing_flag in zip(
                    bars, metric_values, is_missing
                ):
                    height = bar.get_height()
                    if missing_flag:
                        # Mark missing data with a red 'x'
                        ax1.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02,
                            "×",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="red",
                            fontweight="bold",
                        )
                    else:
                        # Add text labels on top of each bar, including for 0
                        text = self.format_value(value, retrieval_metric)
                        ax1.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            text,
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

            # --- Styling and Legend for Plot 1 ---
            ax1.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax1.set_title(
                f'Retrieval Quality - {retrieval_metric.replace("_", " ").title()}',  # noqa: E501
                fontsize=14,
                fontweight="bold",
                pad=50,
            )  # Increased pad for legend
            ax1.set_xticks(x_base)
            ax1.set_xticklabels([f"Q{qid}" for qid in retrieval_queries])
            self.add_legend_with_layout(
                ax1, systems, "lower center", (0.5, 1.02)
            )
            ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax1.set_axisbelow(True)
            ax1.set_ylim(0, 1.15)  # Increased ylim for text
            ax1.spines[["top", "right"]].set_visible(False)
        else:
            ax1.text(
                0.5,
                0.5,
                f'No {retrieval_metric.replace("_", " ")} Data Available',
                ha="center",
                va="center",
                transform=ax1.transAxes,
                fontsize=12,
            )

        # --- 3. Plot 2: Aggregation Quality Metric ---
        agg_queries = sorted(
            {
                k.replace("Q", "")
                for data in metrics_data.values()
                for k, v in data.items()
                if agg_metric in v
                and any(metric in v for metric in aggregation_metrics_set)
            }
        )

        if agg_queries:
            n_groups = len(agg_queries)
            bar_width = 0.8 / n_systems
            x_base = np.arange(n_groups)

            for i, sys_name in enumerate(systems):
                system_color = self.get_system_color(sys_name)

                metric_values = []
                is_missing = []
                for qid in agg_queries:
                    query_data = metrics_data.get(sys_name, {}).get(
                        f"Q{qid}", {}
                    )
                    if agg_metric in query_data:
                        # Ensure value is not zero for log scale, plot very
                        # small instead
                        value = query_data[agg_metric]
                        metric_values.append(value if value > 0 else 1e-9)
                        is_missing.append(False)
                    else:
                        metric_values.append(
                            1e-9
                        )  # Use small value for missing data
                        is_missing.append(True)

                offset = i * bar_width
                position = x_base - (0.8 / 2) + offset + bar_width / 2
                bars = ax2.bar(
                    position,
                    metric_values,
                    bar_width,
                    color=system_color,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                    label=sys_name,
                )

                for bar, value, missing_flag in zip(
                    bars, metric_values, is_missing
                ):
                    height = bar.get_height()
                    if missing_flag:
                        # Mark missing data with a red 'x'
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height * 1.5,
                            "×",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="red",
                            fontweight="bold",
                        )
                    else:
                        # Add text labels on top of each bar
                        text = self.format_value(value, agg_metric)
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height * 1.05,
                            text,
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

            # --- Styling and Legend for Plot 2 ---
            ax2.set_yscale("log")
            ax2.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Error (log scale)", fontsize=12, fontweight="bold")
            ax2.set_title(
                f'Aggregation Quality - {agg_metric.replace("_", " ").title()}',
                fontsize=14,
                fontweight="bold",
                pad=50,
            )  # Increased pad for legend
            ax2.set_xticks(x_base)
            ax2.set_xticklabels([f"Q{qid}" for qid in agg_queries])
            self.add_legend_with_layout(
                ax2, systems, "lower center", (0.5, 1.02)
            )
            ax2.grid(True, alpha=0.3, axis="y", which="both", linestyle="--")
            ax2.set_axisbelow(True)
            ax2.spines[["top", "right"]].set_visible(False)
        else:
            ax2.text(
                0.5,
                0.5,
                f'No {agg_metric.replace("_", " ")} Data Available',
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )

        # --- 3. Plot 3: Ranking Quality Metric ---
        ranking_queries = sorted(
            {
                k.replace("Q", "")
                for data in metrics_data.values()
                for k, v in data.items()
                if ranking_metric in v
                and any(metric in v for metric in ranking_metrics_set)
            }
        )

        if ranking_queries:
            n_groups = len(ranking_queries)
            bar_width = 0.8 / n_systems
            x_base = np.arange(n_groups)

            for i, sys_name in enumerate(systems):
                system_color = self.get_system_color(sys_name)

                metric_values = []
                is_missing = []
                for qid in ranking_queries:
                    query_data = metrics_data.get(sys_name, {}).get(
                        f"Q{qid}", {}
                    )
                    if ranking_metric in query_data:
                        metric_values.append(query_data[ranking_metric])
                        is_missing.append(False)
                    else:
                        metric_values.append(
                            0
                        )  # Use 0 for missing data bar height
                        is_missing.append(True)

                offset = i * bar_width
                position = x_base - (0.8 / 2) + offset + bar_width / 2
                bars = ax3.bar(
                    position,
                    metric_values,
                    bar_width,
                    color=system_color,
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.5,
                    label=sys_name,
                )

                for bar, value, missing_flag in zip(
                    bars, metric_values, is_missing
                ):
                    height = bar.get_height()
                    if missing_flag:
                        # Mark missing data with a red 'x'
                        ax3.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.02,
                            "×",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            color="red",
                            fontweight="bold",
                        )
                    else:
                        # Add text labels on top of each bar, including for 0
                        text = self.format_value(value, ranking_metric)
                        ax3.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.01,
                            text,
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

            # --- Styling and Legend for Plot 3 ---
            ax3.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax3.set_ylabel("Correlation Score", fontsize=12, fontweight="bold")
            ax3.set_title(
                f'Ranking Quality - {ranking_metric.replace("_", " ").title()}',
                fontsize=14,
                fontweight="bold",
                pad=50,
            )  # Increased pad for legend
            ax3.set_xticks(x_base)
            ax3.set_xticklabels([f"Q{qid}" for qid in ranking_queries])
            self.add_legend_with_layout(
                ax3, systems, "lower center", (0.5, 1.02)
            )
            ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
            ax3.set_axisbelow(True)
            ax3.set_ylim(
                -0.1, 1.1
            )  # Correlation values typically range from -1 to 1
            ax3.spines[["top", "right"]].set_visible(False)
        else:
            ax3.text(
                0.5,
                0.5,
                f'No {ranking_metric.replace("_", " ")} Data Available',
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )

        # --- 4. Final Figure Adjustments and Saving ---
        plt.suptitle(
            f'Quality Analysis - {use_case.replace("_", " ").title()}',
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout for suptitle

        if system_name:
            output_dir = self.figures_dir / use_case / system_name
        else:
            output_dir = self.figures_dir / use_case
        output_dir.mkdir(exist_ok=True, parents=True)

        filename = (
            f"quality_{retrieval_metric}_{agg_metric}_{ranking_metric}.png"
        )
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Successfully saved chart to {output_dir / filename}")

    def plot_pareto(self, metrics_data, use_case, system_name=None):
        """
        Plots a compact, publication-quality Pareto analysis.

        This version features:
        - A two-column layout.
        - A separate, unobtrusive legend for each subplot, located below the
        title.
        - A light grey background for each subplot for visual distinction.
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        systems = sorted(metrics_data.keys())

        # 1. --- Data Preparation ---
        queries = []
        all_query_ids = set()
        for sys_name in systems:
            for query_key, metrics in metrics_data[sys_name].items():
                try:
                    # Skip non-numeric query IDs
                    query_id_str = query_key.replace("Q", "")
                    query_id = (query_id_str)
                except ValueError:
                    # Skip queries with non-numeric IDs like Q3a
                    continue

                if query_id not in all_query_ids:
                    all_query_ids.add(query_id)
                    query_info = {"id": query_id}
                    if "f1_score" in metrics:
                        query_info["type"] = "retrieval"
                    elif "relative_error" in metrics:
                        query_info["type"] = "aggregation"
                    elif (
                        "spearman_correlation" in metrics
                        or "kendall_tau" in metrics
                    ):
                        query_info["type"] = "ranking"
                    else:
                        # Skip queries without identifiable type
                        continue
                    queries.append(query_info)

        queries.sort(key=lambda q: q["id"])

        if not queries:
            print(f"No queries found for Pareto analysis in {use_case}")
            return

        # 2. --- Figure and Axes Setup ---
        n_queries = len(queries)
        n_cols = 2
        n_rows = (n_queries + n_cols - 1) // n_cols

        # Increased vertical space in figsize to accommodate per-subplot legends
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(12, 4.5 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        # 3. --- Legend Handles and Marker Mapping ---
        # Define different markers for each system to handle overlapping points
        marker_styles = ["o", "s", "^", "D", "v", "p", "*", "X", "H", "<", ">"]
        system_markers = {
            sys: marker_styles[i % len(marker_styles)]
            for i, sys in enumerate(systems)
        }

        system_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=system_markers[s],
                color="w",
                markerfacecolor=self.get_system_color(s),
                markersize=9,
                label=s,
            )
            for s in systems
        ]

        # 4. --- Plotting Loop for Each Query ---
        for idx, query in enumerate(queries):
            ax = axes[idx]
            query_id, query_type = query["id"], query.get("type", "")
            query_key = f"Q{query_id}"

            # Set light grey background for the subplot
            ax.set_facecolor("#f5f5f5")

            plot_data = []
            missing_systems = []
            for sys_name in systems:
                if query_key in metrics_data.get(sys_name, {}):
                    metrics = metrics_data[sys_name][query_key]
                    cost = metrics.get("money_cost")

                    if query_type == "retrieval":
                        quality = metrics.get("f1_score")
                        x_label = "F1 Score"
                    elif query_type == "aggregation":
                        rel_err = metrics.get("relative_error")
                        # For aggregation queries, relative error can exceed 1,
                        # so we need to handle this properly
                        # Instead of using 1 - rel_err (which can be negative),
                        # use 1 / (1 + rel_err) for quality
                        quality = (
                            1 / (1 + rel_err) if rel_err is not None else None
                        )
                        x_label = "Quality (1 / (1 + Relative Error))"
                    else:  # ranking
                        quality = metrics.get("spearman_correlation")
                        x_label = "Spearman Correlation"

                    if cost is not None and quality is not None and cost > 0:
                        plot_data.append(
                            {
                                "system": sys_name,
                                "quality": quality,
                                "cost": cost,
                            }
                        )
                    else:
                        missing_systems.append(sys_name)
                else:
                    missing_systems.append(sys_name)

            if plot_data:
                # Plot points using different markers for each system (no data
                # distortion)
                for data_point in plot_data:
                    ax.scatter(
                        data_point["quality"],
                        data_point["cost"],
                        color=self.get_system_color(data_point["system"]),
                        marker=system_markers[data_point["system"]],
                        s=150,
                        edgecolor="black",
                        linewidth=1.5,
                        alpha=0.9,
                        zorder=3,
                    )

                # Set Y-axis limits based on actual data
                costs = [d["cost"] for d in plot_data]
                padding = (
                    (max(costs) - min(costs)) * 0.2
                    if max(costs) > min(costs)
                    else max(costs) * 0.1
                )
                ax.set_ylim(max(0, min(costs) - padding), max(costs) + padding)

            # Add red marks for missing systems
            if missing_systems:
                for i, missing_sys_name in enumerate(missing_systems):
                    # Place missing marks at consistent positions
                    x_pos = 0.1 + (i % 3) * 0.3  # Spread across x-axis
                    y_pos = ax.get_ylim()[0] + 0.1 * (
                        ax.get_ylim()[1] - ax.get_ylim()[0]
                    )
                    self.add_red_mark(ax, x_pos, y_pos)

            # 5. --- Subplot Styling and Legend ---
            ax.set_title(
                f"Query {query_id} ({query_type.capitalize()})",
                fontsize=12,
                fontweight="bold",
                pad=50,
            )  # Increased pad for legend spacing

            # Add the legend for each subplot between the title and the plot
            # with better spacing.
            # When there are more than 6 methods, use 3 rows
            if len(systems) <= 5:
                legend_ncol = len(systems)
            elif len(systems) <= 6:
                legend_ncol = (len(systems) + 1) // 2
            else:
                legend_ncol = (
                    len(systems) + 2
                ) // 3  # 3 rows for more than 6 systems
            ax.legend(
                handles=system_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=legend_ncol,
                frameon=False,
                fontsize=9,
                columnspacing=1.5,
                handletextpad=0.5,
            )

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel("Cost ($)", fontsize=10)

            ax.set_xlim(-0.05, 1.05)
            # Y-axis limits are now set after jitter calculation above

            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"${x:.2f}")
            )

        # Hide any unused subplots
        for i in range(n_queries, len(axes)):
            axes[i].set_visible(False)

        # 6. --- Final Figure Layout ---
        fig.suptitle(
            f'Pareto Analysis: Cost vs. Quality for {use_case.replace("_", " ").title()}',  # noqa: E501
            fontsize=16,
            fontweight="bold",
        )

        # Use tight_layout with rect to make space for the main title and
        # legends
        fig.tight_layout(
            rect=[0, 0, 1, 0.94]
        )  # More space for titles and legends

        if system_name:
            output_dir = self.figures_dir / use_case / system_name
        else:
            output_dir = self.figures_dir / use_case
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / "pareto.png"  # Filename is now pareto.png
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved compact Pareto plot to {save_path}")

    def plot_pareto_curve(self, metrics_data, use_case, system_name=None):
        plotting_data = []

        # Collect necessary plotting data
        for sys_name, data in metrics_data.items():
            for query_key, metrics in data.items():
                try:
                    query_id = query_key.replace("Q", "")
                except ValueError:
                    # Skip queries with non-numeric IDs like Q3a
                    continue
                if "accuracy" in metrics and "money_cost" in metrics:
                    plotting_data.append(
                        {
                            "system": sys_name,
                            "query_id": query_id,
                            "accuracy": metrics["accuracy"],
                            "money_cost": metrics["money_cost"],
                        }
                    )

        if not plotting_data:
            print(f"No plotting data found for {use_case}")
            return

        # --- Combined Pareto Plot (normalized costs) ---
        query_to_max_costs = defaultdict(lambda: 1.0)
        for entry in plotting_data:
            qid = entry["query_id"]
            query_to_max_costs[qid] = max(
                query_to_max_costs[qid], entry["money_cost"]
            )

        marker_styles = ["o", "s", "^", "D", "v", "p", "*", "X", "H", "<", ">"]
        unique_query_ids = sorted(
            set(entry["query_id"] for entry in plotting_data),
            key=lambda x: (int(x) if str(x).isdigit() else float('inf'), str(x))
        )
        query_id_to_marker = {
            qid: marker_styles[i % len(marker_styles)]
            for i, qid in enumerate(unique_query_ids)
        }

        fig, ax = plt.subplots(figsize=(7, 4))

        for entry in plotting_data:
            system = entry["system"]
            query_id = entry["query_id"]
            accuracy = entry["accuracy"]
            money_cost = entry["money_cost"]
            normalized_cost = money_cost / query_to_max_costs[query_id]

            color = sns.color_palette("tab10")[hash(system) % 10]
            marker = query_id_to_marker[query_id]

            ax.scatter(
                accuracy,
                normalized_cost,
                label=f"{system}_Q{query_id}",
                color=color,
                marker=marker,
                s=80,
                alpha=0.8,
                edgecolors="w",
                linewidths=0.5,
                clip_on=False,
            )

        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_ylabel("Normalized Cost in $", fontsize=12)
        ax.set_title(
            f"Pareto Curve - {use_case.title()}", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        system_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=system,
                markerfacecolor=sns.color_palette("tab10")[hash(system) % 10],
                markersize=10,
            )
            for system in set(entry["system"] for entry in plotting_data)
        ]

        query_handles = [
            Line2D(
                [0],
                [0],
                marker=query_id_to_marker[qid],
                color="k",
                label=f"Q{qid}",
                linestyle="None",
                markersize=8,
            )
            for qid in unique_query_ids
        ]

        legend1 = ax.legend(
            handles=system_handles,
            title="System",
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
        )
        legend2 = ax.legend(
            handles=query_handles,
            title="Query ID",
            loc="lower left",
            bbox_to_anchor=(1.0, 0.0),
        )
        ax.add_artist(legend1)

        if system_name:
            output_dir = self.figures_dir / use_case / system_name
        else:
            output_dir = self.figures_dir / use_case
        output_dir.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            output_dir / "pareto.png",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[legend1, legend2],
        )
        plt.close()
        if system_name:
            print(f"Saved pareto.png for {use_case}/{system_name}")
        else:
            print(f"Saved pareto.png for {use_case}")

        # --- Individual Pareto Plots (per query, unnormalized costs) ---
        for query_id in unique_query_ids:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            filtered_data = [
                entry
                for entry in plotting_data
                if entry["query_id"] == query_id
            ]

            for entry in filtered_data:
                system = entry["system"]
                accuracy = entry["accuracy"]
                money_cost = entry["money_cost"]
                color = self.system_colors.get(
                    system,
                    np.random.rand(
                        3,
                    ),
                )

                ax.scatter(
                    accuracy,
                    money_cost,
                    label=system,
                    color=color,
                    marker="o",
                    s=80,
                    alpha=0.8,
                    edgecolors="w",
                    linewidths=0.5,
                    clip_on=False,
                )

            ax.set_xlabel("Accuracy", fontsize=12)
            ax.set_ylabel("Cost in $", fontsize=12)
            ax.set_title(
                f"Cost/Accuracy - Q{query_id} - {use_case.title()}",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, None)

            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=4,
                frameon=False,
            )
            plt.tight_layout()
            plt.savefig(
                output_dir / f"pareto_q{query_id}.png",
                dpi=300,
                bbox_inches="tight",
                bbox_extra_artists=[legend],
            )
            plt.close()
            if system_name:
                print(
                    f"Saved pareto_q{query_id}.png for {use_case}/{system_name}"
                )
            else:
                print(f"Saved pareto_q{query_id}.png for {use_case}")

    def plot_avg_cost_accuracy_ratio_per_use_case(self, all_metrics):
        all_metrics = pd.DataFrame.from_records(all_metrics)
        all_use_cases = all_metrics["use_case"].unique()
        fig, axes = plt.subplots(
            figsize=(4.5 * 2, 5), ncols=len(all_use_cases), nrows=1
        )

        for ax_idx, use_case in enumerate(all_use_cases):
            ax = axes[ax_idx]
            # restrict to only the columns we need and drop any NaNs
            plot_df = all_metrics[
                [
                    "use_case",
                    "system",
                    "model_name",
                    "query_id_str",
                    "money_cost",
                    "accuracy",
                ]
            ]
            plot_df = plot_df.dropna()

            # Filter down to the use-case/system/model combinations we are
            # interested in
            plot_df = plot_df.query(
                f"""
                use_case == '{use_case}' and
                (
                    (model_name in ('gemini-2.5-flash', 'gpt-5-mini', 'gpt-4o-mini'))
                    or
                    (system == 'snowflake')
                )
                and accuracy > 0  # to prevent inf in geomean
            """.replace(
                    "\n", " "
                )
            )

            # Make sure that we have an equal set of queries for each system in
            # this use case
            queries_implemented_in_all_systems = self.max_overlap_of_n_sets(
                plot_df.groupby("system").agg({"query_id_str": set})[
                    "query_id_str"
                ]
            )
            plot_df = plot_df[
                plot_df["query_id_str"].isin(queries_implemented_in_all_systems)
            ]

            plot_df["cost_accuracy_tradeoff"] = (
                plot_df["money_cost"] / plot_df["accuracy"]
            )
            print(
                plot_df[
                    [
                        "system",
                        "money_cost",
                        "accuracy",
                        "cost_accuracy_tradeoff",
                    ]
                ]
            )
            plot_df = plot_df.groupby("system").agg(
                {"cost_accuracy_tradeoff": lambda s: gmean(s)}
            )
            plot_df = plot_df.reset_index()
            print(plot_df)

            ax.bar(plot_df["system"], plot_df["cost_accuracy_tradeoff"])

            ax.set_xticklabels(plot_df["system"], rotation=40, ha="right")

            covered_queries = list(queries_implemented_in_all_systems)
            covered_queries.sort()
            ax.set_title(f"{use_case}\n({','.join(covered_queries)})")

        fig.supylabel(
            "Avg. Cost/Accuracy-Ratio\nacross Queries within a Use Case"
        )
        fig.tight_layout()

        save_path = self.figures_dir / "avg_cost_accuracy_tradeoff.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved compact Pareto plot to {save_path}")

    def load_all_query_metrics(self):
        """
        Attempts to unify query metric loading across various different query
        storage formats.

        Challenges:
        * The movie, animals, and detective use case currently uses three
        different folder structures for reporting metrics.
        * ecomm and medical use the same format for reporting metrics. movie
        also uses that format but just for a small subset of metrics.
        * mmqa uses the same folder structure, but uses a different json
        structure inside the file
        """
        all_metrics = []
        for use_case in ["medical", "ecomm"]:
            # Classical way of loading metrics
            metrics_data = self.load_metrics_data(use_case)
            for system_name, metrics in metrics_data.items():
                for query_id, metric in metrics.items():
                    metric["use_case"] = use_case
                    metric["system"] = system_name
                    metric["query_id_str"] = query_id
                    metric_type, accuracy = self.unify_accuracy_metric(metric)
                    metric["metric_type"] = metric_type
                    metric["accuracy"] = accuracy
                    all_metrics.append(metric)
        for use_case in ["detective", "movie", "animals"]:
            # We just pick any of the folders and hope they all contain
            # duplicate data:
            metrics_files = glob.glob(
                os.path.join(
                    self.files_dir,
                    use_case,
                    "metrics",
                    "across_system**",  # animals uses 'across_system_*', detective and movie use 'across_systems_*' # noqa: E501
                    "*.json",
                ),
                recursive=True,
            )
            for json_file in metrics_files:
                with open(json_file, "r") as f:
                    content = json.load(f)
                    for query_id, metric in content.items():
                        metric["use_case"] = use_case
                        metric["system"] = Path(json_file).stem
                        metric["query_id_str"] = query_id
                        metric_type, accuracy = self.unify_accuracy_metric(
                            metric
                        )
                        metric["metric_type"] = metric_type
                        metric["accuracy"] = accuracy
                        all_metrics.append(metric)
        return pd.DataFrame.from_records(all_metrics)

    def plot_pareto_across_all_use_cases(self):
        # All queries that only contain a semantic filter on textual data
        queries_with_only_sem_filter_on_text = [
            ("ecomm", "Q1"),
            ("mmqa", "q3a"),
            # ("mmqa", "q3b"),
            # ("mmqa", "q3c"),
            # ("mmqa", "q3d"),
            # ("mmqa", "q3e"),
            ("mmqa", "q3f"),
            # ("mmqa", "q3g"),
            ("mmqa", "q6a"),
            ("mmqa", "q6b"),
            ("mmqa", "q6c"),
            ("movie", "Q1"),
            ("movie", "Q2"),
            ("movie", "Q3"),
            ("movie", "Q4"),
            ("medical", "Q1"),
            ("medical", "Q4"),
        ]

        all_metrics = self.load_all_query_metrics()
        all_metrics = all_metrics[
            all_metrics.apply(
                lambda row: any(
                    [
                        (
                            True
                            if row["use_case"] == q[0]
                            and row["query_id_str"] == q[1]
                            else False
                        )
                        for q in queries_with_only_sem_filter_on_text
                    ]
                ),
                axis=1,
            )
        ]
        # Now only filter for gemini-2.5-flash or snowflake
        all_metrics = all_metrics[
            (
                all_metrics["model_name"].isin(
                    ["gemini-2.5-flash", "vertex_ai/gemini-2.5-flash"]
                )
            )
            | (all_metrics["system"] == "snowflake")
        ]

        # Remove ThalamusDB, because it is implemented in too few use cases
        all_metrics = all_metrics[all_metrics["system"] != "thalamusdb"]

        # Unify system names
        all_metrics["system"] = all_metrics["system"].apply(
            lambda row: row.split(" ")[0]
        )

        # TODO: 'movie' use case contain duplicate metrics entries here. we
        # should filter them.

        # Make sure that we have an equal set of queries for each system in
        # this use case. Otherwise, we would introduce unfair skew.
        all_metrics["use_case_query_id"] = (
            all_metrics["use_case"] + "-" + all_metrics["query_id_str"]
        )
        queries_implemented_in_all_systems = self.max_overlap_of_n_sets(
            all_metrics.groupby("system").agg({"use_case_query_id": set})[
                "use_case_query_id"
            ]
        )
        all_metrics = all_metrics[
            all_metrics["use_case_query_id"].isin(
                queries_implemented_in_all_systems
            )
        ]

        all_metrics["cost_accuracy_ratio"] = (
            all_metrics["money_cost"] / all_metrics["accuracy"]
        )

        # Data is prepared. Now start plotting. #
        fig, ax = plt.subplots(figsize=(4.9, 3))

        sns.boxplot(
            data=all_metrics,
            x="system",
            y="cost_accuracy_ratio",
            orient="v",
            ax=ax,
            width=0.5,
            color="#64c9ee",
            saturation=1.0,
            linecolor="#000000",
            linewidth=1.0,
            fliersize=1,
        )

        fig.suptitle(
            "Cost/Accuracy for Queries containing only SEM_FILTER on Text"
        )
        ax.set_xlabel(
            f"included queries: {','.join(queries_implemented_in_all_systems)}"
        )
        ax.set_ylabel("Cost/Accuracy")

        fig.tight_layout()
        save_path = self.figures_dir / "operator_cost_accuracy_tradeoff.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved compact Pareto plot to {save_path}")

    def plot_summary_across_systems(self, use_case, across_system_folder):
        """
        Create a 3-column summary figure for across-system data for any use
        case.
        Includes money cost, latency (execution time), and quality metrics.
        Designed for double-column SIGMOD paper format.

        Args:
            use_case (str): The use case name (e.g., 'movie', 'detective',
            'animals')
            across_system_folder (str): The across_system folder name (e.g.,
            'across_system_2.5flash')
        """
        data_dir = self.files_dir / use_case / "metrics" / across_system_folder

        if not data_dir.exists():
            print(f"Data directory {data_dir} not found!")
            return

        # Load metrics data from the specific directory
        metrics_data = {}
        for json_file in data_dir.glob("*.json"):
            system_name = json_file.stem
            try:
                with open(json_file, "r") as f:
                    metrics_data[system_name] = json.load(f)
                print(f"Loaded metrics for {system_name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        if not metrics_data:
            print("No metrics data found!")
            return

        # Collect all query IDs and systems
        all_query_ids = set()
        for data in metrics_data.values():
            for query_key in data.keys():
                try:
                    query_id = query_key.replace("Q", "")
                    all_query_ids.add(query_id)
                except ValueError:
                    continue

        query_ids = sorted(all_query_ids, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
        systems = sorted(metrics_data.keys())

        if not query_ids or not systems:
            print("No valid queries or systems found!")
            return

        # Create figure with 3 subplots (1 row, 3 columns)
        # Narrower figure with higher width-to-height ratio for publication
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.2))

        # Define plot parameters
        x_pos = np.arange(len(query_ids))
        width = 0.8 / len(systems)

        # Helper function to collect actual data and find minimums for red mar
        #  positioning
        def collect_data_and_plot(
            ax, metric_name, ylabel, title, use_log=False, ylim_max=None
        ):
            # First pass: collect all actual values
            all_actual_values = []
            data_matrix = {}  # {system: {query: value or None}}

            for sys_name in systems:
                data_matrix[sys_name] = {}
                for qid in query_ids:
                    query_data = metrics_data.get(sys_name, {}).get(
                        f"Q{qid}", {}
                    )

                    # Check if this system supports this query (money_cost > 0)
                    cost = query_data.get("money_cost", 0)
                    system_supports_query = cost > 0

                    if metric_name == "money_cost":
                        if system_supports_query:
                            data_matrix[sys_name][qid] = cost
                            all_actual_values.append(cost)
                        else:
                            data_matrix[sys_name][qid] = None
                    elif metric_name == "execution_time":
                        if system_supports_query:
                            latency = query_data.get("execution_time", 0)
                            if latency > 0:
                                data_matrix[sys_name][qid] = latency
                                all_actual_values.append(latency)
                            else:
                                # System supports query but no execution time
                                # recorded
                                data_matrix[sys_name][qid] = None
                        else:
                            data_matrix[sys_name][qid] = None
                    elif metric_name == "quality":
                        if system_supports_query:
                            # Determine quality metric based on available data
                            if "f1_score" in query_data:
                                value = query_data["f1_score"]
                            elif "relative_error" in query_data:
                                rel_err = query_data["relative_error"]
                                value = (
                                    1 / (1 + rel_err)
                                    if rel_err is not None
                                    else None
                                )
                            elif "spearman_correlation" in query_data:
                                value = query_data["spearman_correlation"]
                            else:
                                value = None

                            # Include quality values even if they are 0 (poor
                            # performance is still data)
                            if value is not None:
                                data_matrix[sys_name][qid] = value
                                all_actual_values.append(value)
                            else:
                                data_matrix[sys_name][qid] = None
                        else:
                            data_matrix[sys_name][qid] = None

            # Second pass: plot only actual bars
            for i, sys_name in enumerate(systems):
                actual_values = []
                actual_positions = []
                missing_positions = []

                for j, qid in enumerate(query_ids):
                    x_position = (
                        x_pos[j] + i * width - (len(systems) - 1) * width / 2
                    )
                    if data_matrix[sys_name][qid] is not None:
                        actual_values.append(data_matrix[sys_name][qid])
                        actual_positions.append(x_position)
                    else:
                        missing_positions.append(x_position)

                # Plot bars only for actual data
                if actual_values:
                    color = self.get_system_color(sys_name)
                    ax.bar(
                        actual_positions,
                        actual_values,
                        width,
                        color=color,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                    )

            # Apply styling first to get proper axis limits
            if use_log:
                ax.set_yscale("log")
            ax.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"Q{qid}" for qid in query_ids])
            ax.tick_params(axis="both", which="major", labelsize=11)
            if ylim_max:
                ax.set_ylim(0, ylim_max)
            ax.grid(
                True,
                alpha=0.3,
                axis="y",
                which="both" if use_log else "major",
                linestyle="--",
            )
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            # Red marks will be added separately after all subplots are created
            # to ensure consistent height across all figures

        # First, plot all subplots to establish their scales
        collect_data_and_plot(
            ax1,
            "money_cost",
            "Cost ($)",
            "Money Cost",
            use_log=True,
        )
        collect_data_and_plot(
            ax2,
            "execution_time",
            "Time (s)",
            "Latency",
            use_log=True,
        )
        collect_data_and_plot(
            ax3, "quality", "Quality", "Quality", ylim_max=1.1
        )

        # Now add red marks at consistent height across all subplots
        # Define a consistent red mark height relative to each subplot's scale
        consistent_red_mark_height_ratio = 0.02  # 2% above x-axis

        # Collect all missing data positions for each subplot
        # Red marks should only appear when money_cost is 0 (system doesn't
        # support the query)
        # Quality of 0 just means poor performance, not missing support
        all_missing_data = {ax1: [], ax2: [], ax3: []}

        for i, sys_name in enumerate(systems):
            for j, qid in enumerate(query_ids):
                x_position = (
                    x_pos[j] + i * width - (len(systems) - 1) * width / 2
                )
                query_data = metrics_data.get(sys_name, {}).get(f"Q{qid}", {})

                # Check money_cost - if 0, system doesn't support this query
                cost = query_data.get("money_cost", 0)
                if cost <= 0:
                    # System doesn't support this query - add red marks to all
                    # subplots
                    all_missing_data[ax1].append(x_position)  # Money cost
                    all_missing_data[ax2].append(x_position)  # Latency
                    all_missing_data[ax3].append(x_position)  # Quality

        # Add red marks at consistent heights for each subplot
        # For log scale plots (ax1, ax2), position just above the bottom
        # For linear scale plots (ax3), position as percentage above bottom

        for ax, missing_positions in all_missing_data.items():
            if missing_positions:
                y_min, y_max = ax.get_ylim()

                # Check if this is a log scale subplot
                if ax.get_yscale() == "log":
                    # For log scale: position red marks at a small multiplier
                    # above y_min
                    red_mark_height = (
                        y_min * 1.3
                    )  # 30% above the minimum in log space
                else:
                    # For linear scale: position as percentage of range above
                    # y_min
                    red_mark_height = (
                        y_min
                        + (y_max - y_min) * consistent_red_mark_height_ratio
                    )

                for x_pos_missing in missing_positions:
                    self.add_red_mark(ax, x_pos_missing, red_mark_height)

        # Add single legend at the top of the figure with larger font and
        # closer positioning
        handles = [
            mpatches.Patch(color=self.get_system_color(s), label=s)
            for s in systems
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(systems),
            frameon=False,
            fontsize=12,
        )

        # Add scenario-specific title at bottom with compact spacing
        scenario_titles = {
            "movie": "(a) Movie Scenario",
            "detective": "(b) Detective Scenario",
            "animals": "(c) Wildlife Scenario",
            "ecomm": "(d) Ecomm Scenario",
            "mmqa": "(e) MMQA Scenario",
            "medical": "(f) Medical Scenario",
        }
        title_text = scenario_titles.get(
            use_case, f"({use_case.title()} Scenario)"
        )

        fig.suptitle(
            title_text, fontsize=13, fontweight="bold", y=0.04, ha="center"
        )

        # Adjust layout with minimal spacing for legend and title
        plt.tight_layout(
            rect=[0, 0.06, 1, 0.99]
        )  # Reserve minimal space for legend (1%) and title (6%)

        # Extract model name from across_system folder (e.g.,
        # "across_system_2.5flash" -> "2.5flash")
        model_name = (
            across_system_folder.split("_", 2)[-1]
            if "_" in across_system_folder
            else across_system_folder
        )

        # Create filename: {scenario}_{model}_performance.pdf
        filename = f"{use_case}_{model_name}_performance.pdf"

        # Save to the across_system folder
        output_dir = self.figures_dir / use_case / across_system_folder
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save both PDF and PNG formats
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

        # Save PNG version
        png_filename = filename.replace(".pdf", ".png")
        plt.savefig(output_dir / png_filename, dpi=300, bbox_inches="tight")

        plt.close()
        print(f"✅ Saved {filename} and {png_filename} to {output_dir}")

    def plot_summary_across_systems_with_error_bar(self, use_case, model_tag):
        """
        Create error bar plots by aggregating data from multiple round folders.
        Auto-detects folders with pattern: across_system_{model_tag}_{round_number}
        Saves to: figures/{use_case}/across_system_{model_tag}/{use_case}_{model_tag}_performance_error_bar.png(pdf)

        Args:
            use_case (str): The use case name (e.g., 'movie', 'detective', 'animals')
            model_tag (str): The model tag (e.g., '2.5flash')
        """
        # Auto-detect round folders with pattern across_system_{model_tag}_{round_number}
        base_dir = self.files_dir / use_case / "metrics"
        round_folders = []

        for i in range(1, 6):  # rounds 1-5
            round_folder = f"across_system_{model_tag}_{i}"
            round_path = base_dir / round_folder
            if round_path.exists():
                round_folders.append(round_folder)

        if not round_folders:
            print(f"No round folders found for across_system_{model_tag}")
            return

        print(f"Found {len(round_folders)} round folders: {round_folders}")

        # Load metrics data from all rounds
        all_round_data = {}  # {round: {system: metrics}}
        systems = set()

        for round_folder in round_folders:
            data_dir = base_dir / round_folder
            round_data = {}

            for json_file in data_dir.glob("*.json"):
                system_name = json_file.stem
                systems.add(system_name)
                try:
                    with open(json_file, "r") as f:
                        round_data[system_name] = json.load(f)
                    print(
                        f"Loaded metrics for {system_name} from {round_folder}"
                    )
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

            all_round_data[round_folder] = round_data

        if not all_round_data or not systems:
            print("No valid data found across rounds!")
            return

        systems = sorted(systems)

        # Collect all query IDs
        all_query_ids = set()
        for round_data in all_round_data.values():
            for system_data in round_data.values():
                for query_key in system_data.keys():
                    try:
                        query_id = (query_key.replace("Q", ""))
                        all_query_ids.add(query_id)
                    except ValueError:
                        continue

        query_ids = sorted(all_query_ids, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
        print(f"Found {len(query_ids)} queries: {query_ids}")
        print(f"Found {len(systems)} systems: {systems}")

        if not query_ids or not systems:
            print("No valid queries or systems found!")
            return

        # Create figure with 3 subplots (1 row, 3 columns)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3.2))

        # Define plot parameters
        x_pos = np.arange(len(query_ids))
        width = 0.8 / len(systems)

        # Helper function to aggregate data across rounds and plot with error bars
        def collect_and_aggregate_data(
            ax, metric_name, ylabel, title, use_log=False, ylim_max=None
        ):
            # Aggregate data across rounds
            aggregated_data = {}  # {system: {query: [values_across_rounds]}}

            for sys_name in systems:
                aggregated_data[sys_name] = {}
                for qid in query_ids:
                    aggregated_data[sys_name][qid] = []

                    # Collect values from all rounds
                    for round_folder, round_data in all_round_data.items():
                        if sys_name in round_data:
                            query_data = round_data[sys_name].get(f"Q{qid}", {})

                            # Check if this system supports this query (money_cost > 0)
                            cost = query_data.get("money_cost", 0)
                            system_supports_query = cost > 0

                            if metric_name == "money_cost":
                                if system_supports_query:
                                    aggregated_data[sys_name][qid].append(cost)
                            elif metric_name == "execution_time":
                                if system_supports_query:
                                    latency = query_data.get(
                                        "execution_time", 0
                                    )
                                    if latency > 0:
                                        aggregated_data[sys_name][qid].append(
                                            latency
                                        )
                            elif metric_name == "quality":
                                if system_supports_query:
                                    # Determine quality metric based on available data
                                    value = None
                                    # Check for metric_type + accuracy first (for adjusted-rand-index in ecomm)
                                    if "metric_type" in query_data and "accuracy" in query_data:
                                        value = query_data["accuracy"]
                                    elif "f1_score" in query_data:
                                        value = query_data["f1_score"]
                                    elif "relative_error" in query_data:
                                        rel_err = query_data["relative_error"]
                                        value = (
                                            1 / (1 + rel_err)
                                            if rel_err is not None
                                            else None
                                        )
                                    elif "spearman_correlation" in query_data:
                                        value = query_data[
                                            "spearman_correlation"
                                        ]

                                    if value is not None:
                                        aggregated_data[sys_name][qid].append(
                                            value
                                        )

            # Calculate means and standard deviations, then plot
            for i, sys_name in enumerate(systems):
                means = []
                stds = []
                positions = []
                missing_positions = []

                for j, qid in enumerate(query_ids):
                    x_position = (
                        x_pos[j] + i * width - (len(systems) - 1) * width / 2
                    )
                    values = aggregated_data[sys_name][qid]

                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values) if len(values) > 1 else 0
                        means.append(mean_val)
                        stds.append(std_val)
                        positions.append(x_position)
                    else:
                        missing_positions.append(x_position)

                # Plot bars with error bars
                if means:
                    color = self.get_system_color(sys_name)
                    bars = ax.bar(
                        positions,
                        means,
                        width,
                        yerr=stds,  # Add error bars
                        color=color,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                        capsize=3,  # Error bar cap size
                        error_kw={
                            "linewidth": 1,
                            "ecolor": "black",
                        },  # Error bar styling
                    )

            # Apply styling
            if use_log:
                ax.set_yscale("log")
            ax.set_xlabel("Query ID", fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"Q{qid}" for qid in query_ids])
            ax.tick_params(axis="both", which="major", labelsize=11)
            if ylim_max:
                ax.set_ylim(0, ylim_max)
            ax.grid(
                True,
                alpha=0.3,
                axis="y",
                which="both" if use_log else "major",
                linestyle="--",
            )
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            return aggregated_data

        # Plot all subplots with error bars
        money_data = collect_and_aggregate_data(
            ax1, "money_cost", "Cost ($)", "Money Cost", use_log=True
        )
        time_data = collect_and_aggregate_data(
            ax2, "execution_time", "Time (s)", "Latency", use_log=True
        )
        quality_data = collect_and_aggregate_data(
            ax3, "quality", "Quality", "Quality", ylim_max=1.1
        )

        # Add red marks for missing data (when no system supports a query across any round)
        consistent_red_mark_height_ratio = 0.02
        all_missing_data = {ax1: [], ax2: [], ax3: []}

        for i, sys_name in enumerate(systems):
            for j, qid in enumerate(query_ids):
                x_position = (
                    x_pos[j] + i * width - (len(systems) - 1) * width / 2
                )

                # Check if system supports this query in any round
                has_support = False
                for round_data in all_round_data.values():
                    if sys_name in round_data:
                        query_data = round_data[sys_name].get(f"Q{qid}", {})
                        cost = query_data.get("money_cost", 0)
                        if cost > 0:
                            has_support = True
                            break

                if not has_support:
                    all_missing_data[ax1].append(x_position)
                    all_missing_data[ax2].append(x_position)
                    all_missing_data[ax3].append(x_position)

        # Add red marks
        for ax, missing_positions in all_missing_data.items():
            if missing_positions:
                y_min, y_max = ax.get_ylim()
                if ax.get_yscale() == "log":
                    red_mark_height = y_min * 1.3
                else:
                    red_mark_height = (
                        y_min
                        + (y_max - y_min) * consistent_red_mark_height_ratio
                    )

                for x_pos_missing in missing_positions:
                    self.add_red_mark(ax, x_pos_missing, red_mark_height)

        # Add legend
        handles = [
            mpatches.Patch(color=self.get_system_color(s), label=s)
            for s in systems
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=len(systems),
            frameon=False,
            fontsize=12,
        )

        # Add scenario-specific title
        scenario_titles = {
            "movie": "(a) Movie Scenario",
            "detective": "(b) Detective Scenario",
            "animals": "(c) Wildlife Scenario",
            "ecomm": "(d) Ecomm Scenario",
            "mmqa": "(e) MMQA Scenario",
            "medical": "(f) Medical Scenario",
        }
        title_text = scenario_titles.get(
            use_case, f"({use_case.title()} Scenario)"
        )

        fig.suptitle(
            title_text, fontsize=13, fontweight="bold", y=0.04, ha="center"
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.06, 1, 0.99])

        # Create filename with error_bar suffix using the model_tag
        filename = f"{use_case}_{model_tag}_performance_error_bar.pdf"

        # Save to the base across_system folder (not the round folders)
        output_dir = self.figures_dir / use_case / f"across_system_{model_tag}"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save both PDF and PNG formats
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

        # Save PNG version
        png_filename = filename.replace(".pdf", ".png")
        plt.savefig(output_dir / png_filename, dpi=300, bbox_inches="tight")

        plt.close()
        print(f"✅ Saved {filename} and {png_filename} to {output_dir}")

        # Generate a tabular format in TSV for easy copy-pasting into spreadsheets
        out_file_path = output_dir / f"table_{use_case}.tsv"
        with open(out_file_path, "w") as table_out:
            table_out.write(
                f"Summary table for use case '{use_case}' with model '{model_tag}'\n"
            )
            table_out.write("\t" + "\t\t\t\t".join(systems) + "\n")
            table_out.write(
                "\t"
                + "\t".join(
                    ["cost\tSTDDEV(cost)\taccuracy\tSTDDEV(accuracy)"]
                    * len(systems)
                )
                + "\n"
            )
            for qid in query_ids:
                row = [f"Q{qid}"]
                for system_name in systems:
                    cost_vals = money_data.get(system_name, {}).get(qid, [])
                    acc_vals = quality_data.get(system_name, {}).get(qid, [])
                    if cost_vals and acc_vals:
                        avg_cost = np.mean(cost_vals)
                        avg_acc = np.mean(acc_vals)
                        std_cost = (
                            np.std(cost_vals) if len(cost_vals) > 1 else 0
                        )
                        std_acc = np.std(acc_vals) if len(acc_vals) > 1 else 0
                        row.append(
                            f"{avg_cost}\t{std_cost}\t{avg_acc}\t{std_acc}"
                        )
                    else:
                        # row.append("N/A\tN/A\tN/A\tN/A")
                        row.append("\t\t\t")  # empty cells if there is no value
                table_out.write("\t".join(row) + "\n")

        # Generate pareto plots with aggregated data (using means, no error bars)
        self._generate_pareto_with_aggregated_data(
            use_case, model_tag, all_round_data, systems, query_ids, output_dir
        )

    def _generate_pareto_with_aggregated_data(
        self,
        use_case,
        model_tag,
        all_round_data,
        systems,
        query_ids,
        output_dir,
    ):
        """
        Generate pareto plots using aggregated data from multiple rounds (means only).
        This will overwrite the original pareto files.
        """
        print(
            f"Generating pareto plots with aggregated data for {use_case} {model_tag}..."
        )

        # Aggregate data across rounds to create mean-based metrics_data
        aggregated_metrics_data = {}

        for sys_name in systems:
            aggregated_metrics_data[sys_name] = {}

            for qid in query_ids:
                query_key = f"Q{qid}"

                # Collect values from all rounds for this system-query combination
                money_costs = []
                execution_times = []
                f1_scores = []
                relative_errors = []
                spearman_correlations = []
                accuracies = []  # For metric_type + accuracy (e.g., adjusted-rand-index)
                metric_type = None  # Track the metric type

                for round_folder, round_data in all_round_data.items():
                    if sys_name in round_data:
                        query_data = round_data[sys_name].get(query_key, {})

                        # Check if this system supports this query (money_cost > 0)
                        cost = query_data.get("money_cost", 0)
                        if cost > 0:
                            money_costs.append(cost)

                            if (
                                "execution_time" in query_data
                                and query_data["execution_time"] is not None
                            ):
                                execution_times.append(
                                    query_data["execution_time"]
                                )

                            # Check for metric_type + accuracy first (for adjusted-rand-index in ecomm)
                            if "metric_type" in query_data and "accuracy" in query_data:
                                if query_data["accuracy"] is not None:
                                    accuracies.append(query_data["accuracy"])
                                    metric_type = query_data["metric_type"]
                            elif (
                                "f1_score" in query_data
                                and query_data["f1_score"] is not None
                            ):
                                f1_scores.append(query_data["f1_score"])

                            if (
                                "relative_error" in query_data
                                and query_data["relative_error"] is not None
                            ):
                                relative_errors.append(
                                    query_data["relative_error"]
                                )

                            if (
                                "spearman_correlation" in query_data
                                and query_data["spearman_correlation"]
                                is not None
                            ):
                                spearman_correlations.append(
                                    query_data["spearman_correlation"]
                                )

                # Calculate means and create aggregated query data
                if (
                    money_costs
                ):  # Only create entry if system supports this query
                    aggregated_query_data = {
                        "query_id": qid,
                        "money_cost": np.mean(money_costs),
                    }

                    if execution_times:
                        aggregated_query_data["execution_time"] = np.mean(
                            execution_times
                        )

                    # Add metric_type + accuracy first (for adjusted-rand-index in ecomm)
                    if accuracies:
                        aggregated_query_data["accuracy"] = np.mean(accuracies)
                        if metric_type:
                            aggregated_query_data["metric_type"] = metric_type

                    if f1_scores:
                        aggregated_query_data["f1_score"] = np.mean(f1_scores)

                    if relative_errors:
                        aggregated_query_data["relative_error"] = np.mean(
                            relative_errors
                        )

                    if spearman_correlations:
                        aggregated_query_data["spearman_correlation"] = np.mean(
                            spearman_correlations
                        )

                    aggregated_metrics_data[sys_name][
                        query_key
                    ] = aggregated_query_data

        # Now use the same pareto plotting logic as plot_pareto_across_systems but with aggregated data
        self._plot_pareto_from_metrics_data(
            use_case,
            model_tag,
            aggregated_metrics_data,
            systems,
            query_ids,
            output_dir,
        )

    def _plot_pareto_from_metrics_data(
        self, use_case, model_tag, metrics_data, systems, query_ids, output_dir
    ):
        """
        Generate pareto plots from given metrics data. This is the core pareto plotting logic.
        """

        # Helper: pick whichever quality metric is present (same as original function)
        def get_quality_metric(query_data):
            # Check for metric_type + accuracy first (for adjusted-rand-index in ecomm)
            if "metric_type" in query_data and "accuracy" in query_data and query_data["accuracy"] is not None:
                return query_data["accuracy"], query_data["metric_type"]
            # Prefer F1 if present
            if "f1_score" in query_data and query_data["f1_score"] is not None:
                return query_data["f1_score"], "F1 Score"
            # Otherwise use transformed relative error if present
            if (
                "relative_error" in query_data
                and query_data["relative_error"] is not None
            ):
                rel_err = query_data["relative_error"]
                return 1 / (1 + rel_err), "Quality (1/(1+RelErr))"
            # Otherwise Spearman if present
            if (
                "spearman_correlation" in query_data
                and query_data["spearman_correlation"] is not None
            ):
                return (
                    query_data["spearman_correlation"],
                    "Spearman Correlation",
                )
            # Nothing found
            return None, "Quality"

        # Create combined pareto figure (2x5 grid)
        fig = plt.figure(figsize=(15, 3))

        # Create GridSpec: 2 rows, 5 columns for pareto plots only
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.2)

        # Create subplots for cost-quality pareto (2x5)
        pareto_axes = []
        for row in range(2):
            row_axes = []
            for col in range(5):
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
            pareto_axes.append(row_axes)

        # Plot cost-quality pareto for each query
        for idx, query_id in enumerate(
            query_ids[:10]
        ):  # Limit to first 10 queries
            row = idx // 5  # 0 for Q1-Q5, 1 for Q6-Q10
            col = idx % 5  # 0-4 for columns

            ax = pareto_axes[row][col]
            query_key = f"Q{query_id}"

            # Collect data for this query
            plot_data = []
            missing_systems = []

            for sys_name in systems:
                if query_key in metrics_data.get(sys_name, {}):
                    query_data = metrics_data[sys_name][query_key]
                    cost = query_data.get("money_cost")
                    quality, quality_label = get_quality_metric(query_data)

                    # Keep only strictly positive costs (zero-cost points are excluded)
                    if cost is not None and quality is not None and cost > 0:
                        plot_data.append(
                            {
                                "system": sys_name,
                                "quality": quality,
                                "cost": cost,
                            }
                        )
                    else:
                        missing_systems.append(sys_name)
                else:
                    missing_systems.append(sys_name)

            # Plot scatter points (same logic as original function)
            if plot_data:
                for data_point in plot_data:
                    sys_name = data_point["system"]
                    ax.scatter(
                        data_point["quality"],
                        data_point["cost"],
                        color=self.get_system_color(sys_name),
                        marker=self.system_markers.get(sys_name, "o"),
                        s=80,  # Larger markers for better visibility
                        edgecolor="black",
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=3,
                    )

                # Extract quality and cost ranges from actual data
                qualities = [d["quality"] for d in plot_data]
                costs = [d["cost"] for d in plot_data]

                # ---- X-axis (Quality) handling ----
                q_min, q_max = min(qualities), max(qualities)

                # Determine padding needed near boundaries
                left_pad = (
                    0.05 if q_min < 0.1 else 0.02
                )  # More padding if data near 0
                right_pad = (
                    0.05 if q_max > 0.9 else 0.02
                )  # More padding if data near 1

                x_low = -left_pad
                x_high = 1.0 + right_pad

                ax.set_xlim(x_low, x_high)

                # Always show standard x-axis ticks
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0.0", "0.5", "1.0"])

                # ---- Y-axis (Cost) handling ----
                c_min, c_max = min(costs), max(costs)
                if c_max == c_min:
                    c_pad = max(0.1 * c_max, 0.02)
                    y_low, y_high = c_min - c_pad, c_max + c_pad
                else:
                    c_range = c_max - c_min
                    c_pad = max(0.15 * c_range, 0.02)
                    y_low, y_high = c_min - c_pad, c_max + c_pad

                # Ensure minimum range for visibility
                if y_high - y_low < 1e-6:
                    y_high = y_low + 0.05

                ax.set_ylim(y_low, y_high)

                # Set y-axis ticks to only show non-negative values
                import numpy as np

                # Generate reasonable tick positions
                y_range = y_high - max(
                    0, y_low
                )  # Only consider positive range for ticks
                if y_range > 0:
                    tick_start = max(
                        0.01, c_min
                    )  # Ensure minimum tick is at least 0.01
                    num_ticks = min(
                        4, max(2, int(y_range / (y_range / 3)))
                    )  # Reduce to 3-4 ticks
                    tick_step = (
                        y_range / (num_ticks - 1) if num_ticks > 1 else y_range
                    )

                    valid_y_ticks = []
                    for i in range(num_ticks):
                        tick = tick_start + i * tick_step
                        if (
                            tick >= 0.01 and y_low <= tick <= y_high
                        ):  # Ensure tick >= 0.01
                            valid_y_ticks.append(tick)

                    # Ensure we have at least one tick and it's not 0.00
                    if not valid_y_ticks and c_max >= 0.01:
                        valid_y_ticks = [c_max]
                    elif not valid_y_ticks:
                        valid_y_ticks = [0.01]

                    if valid_y_ticks:
                        ax.set_yticks(valid_y_ticks)
                        ax.set_yticklabels(
                            [f"{tick:.2f}" for tick in valid_y_ticks]
                        )
                # ---- end axis handling ----
            else:
                # Even with no data, set reasonable range
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(
                    -0.02, 0.1
                )  # Small negative padding for visual space

                # Set default ticks for no-data case
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0.00", "0.50", "1.00"])
                ax.set_yticks([0.01, 0.05, 0.10])
                ax.set_yticklabels(["0.01", "0.05", "0.10"])

            # Subplot styling - larger fonts for presentation quality
            ax.set_title(f"Q{query_id}", fontsize=12, fontweight="bold", pad=4)

            # X-axis: show numbers on both rows, but only label on bottom row
            if row == 1:  # Bottom row gets the "Quality" label
                ax.set_xlabel("Quality", fontsize=11)
            else:  # Top row gets no label, just numbers
                ax.set_xlabel("")
            ax.tick_params(axis="x", which="major", labelsize=10, pad=2)

            # Y-axis: only left column gets labels
            ax.set_ylabel("Cost ($)" if col == 0 else "", fontsize=11)
            ax.tick_params(axis="y", which="major", labelsize=10, pad=2)

            # Adjust y-axis to prevent overlap with x-axis
            ax.tick_params(axis="y", which="major", pad=4)

            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add single unified legend for all systems across all plots
        all_handles = []
        for sys_name in systems:
            handle = mpatches.Patch(
                color=self.get_system_color(sys_name), label=sys_name
            )
            all_handles.append(handle)

        # Position legend outside the figure area at the top with more margin
        fig.legend(
            handles=all_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),  # Further outside to avoid overlap
            ncol=len(systems),
            frameon=False,
            fontsize=11,  # Increased for presentation quality
            columnspacing=1.8,  # Much more space between system names
            handletextpad=0.6,  # More space between patch and text
        )

        # Adjust layout with space for legend and title
        plt.tight_layout(
            rect=[0, 0.08, 1, 0.92]
        )  # Reserve space for legend and title

        # Create filename: {scenario}_{model}_pareto.pdf (overwrites original)
        filename = f"{use_case}_{model_tag}_pareto.pdf"

        # Save both PDF and PNG formats (overwrites original files)
        plt.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        # Save PNG version
        png_filename = filename.replace(".pdf", ".png")
        plt.savefig(
            output_dir / png_filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close()
        print(
            f"✅ Saved aggregated pareto plots: {filename} and {png_filename}"
        )

        # Create query_pareto folder and generate individual query plots (overwrites originals)
        query_pareto_dir = output_dir / "query_pareto"
        query_pareto_dir.mkdir(exist_ok=True, parents=True)

        # Generate individual pareto plot for each query
        for query_id in query_ids:
            query_key = f"Q{query_id}"

            # Create individual figure with moderate rectangular aspect ratio (8:5 ratio)
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

            # Collect data for this query (same logic as above)
            plot_data = []
            missing_systems = []

            for sys_name in systems:
                if query_key in metrics_data.get(sys_name, {}):
                    query_data = metrics_data[sys_name][query_key]
                    cost = query_data.get("money_cost")
                    quality, quality_label = get_quality_metric(query_data)

                    # Keep only strictly positive costs
                    if cost is not None and quality is not None and cost > 0:
                        plot_data.append(
                            {
                                "system": sys_name,
                                "quality": quality,
                                "cost": cost,
                            }
                        )
                    else:
                        missing_systems.append(sys_name)
                else:
                    missing_systems.append(sys_name)

            # Plot individual query pareto (same styling as original)
            if plot_data:
                for data_point in plot_data:
                    sys_name = data_point["system"]
                    ax.scatter(
                        data_point["quality"],
                        data_point["cost"],
                        color=self.get_system_color(sys_name),
                        marker=self.system_markers.get(sys_name, "o"),
                        s=120,  # Larger markers for individual plots
                        edgecolor="black",
                        linewidth=1.0,
                        alpha=0.9,
                        zorder=3,
                    )

                # Set axis ranges (same logic as combined plot)
                qualities = [d["quality"] for d in plot_data]
                costs = [d["cost"] for d in plot_data]

                q_min, q_max = min(qualities), max(qualities)
                left_pad = 0.05 if q_min < 0.1 else 0.02
                right_pad = 0.05 if q_max > 0.9 else 0.02

                ax.set_xlim(-left_pad, 1.0 + right_pad)
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0.0", "0.5", "1.0"])

                c_min, c_max = min(costs), max(costs)
                if c_max == c_min:
                    c_pad = max(0.1 * c_max, 0.02)
                    y_low, y_high = c_min - c_pad, c_max + c_pad
                else:
                    c_range = c_max - c_min
                    c_pad = max(0.15 * c_range, 0.02)
                    y_low, y_high = c_min - c_pad, c_max + c_pad

                if y_high - y_low < 1e-6:
                    y_high = y_low + 0.05

                ax.set_ylim(y_low, y_high)
            else:
                # No data case
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.02, 0.1)
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0.00", "0.50", "1.00"])

            # Styling for individual plots
            ax.set_title(
                f"Query {query_id} - Cost vs Quality",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Quality", fontsize=12)
            ax.set_ylabel("Cost ($)", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=11)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Legend for individual plots
            if plot_data:
                legend_handles = []
                for sys_name in systems:
                    if any(d["system"] == sys_name for d in plot_data):
                        handle = mpatches.Patch(
                            color=self.get_system_color(sys_name),
                            label=sys_name,
                        )
                        legend_handles.append(handle)

                if legend_handles:
                    ax.legend(handles=legend_handles, loc="best", fontsize=10)

            plt.tight_layout()

            # Save individual query plot (overwrites original)
            query_filename = f"Q{query_id}.png"
            plt.savefig(
                query_pareto_dir / query_filename,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

        print(f"✅ Saved individual query pareto plots to {query_pareto_dir}")

    def plot_pareto_across_systems(self, use_case, across_system_folder):
        """
        Create pareto plots figure optimized for double-column SIGMOD paper:
        - 2x5 cost-quality pareto plots for 10 queries only

        Args:
            use_case (str): The use case name (e.g., 'movie', 'detective',
            'animals')
            across_system_folder (str): The across_system folder name (e.g.,
            'across_system_2.5flash')
        """
        data_dir = self.files_dir / use_case / "metrics" / across_system_folder

        if not data_dir.exists():
            print(f"Data directory {data_dir} not found!")
            return

        # Load metrics data from the specific directory
        metrics_data = {}
        for json_file in data_dir.glob("*.json"):
            system_name = json_file.stem
            try:
                with open(json_file, "r") as f:
                    metrics_data[system_name] = json.load(f)
                print(f"Loaded metrics for {system_name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        if not metrics_data:
            print("No metrics data found!")
            return

        # Collect all query IDs and systems
        all_query_ids = set()
        for data in metrics_data.values():
            for query_key in data.keys():
                try:
                    query_id = query_key.replace("Q", "")
                    all_query_ids.add(query_id)
                except ValueError:
                    continue

        query_ids = sorted(all_query_ids, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
        systems = sorted(metrics_data.keys())

        if not query_ids or not systems:
            print("No valid queries or systems found!")
            return

        print(f"Found {len(query_ids)} queries: {query_ids}")
        print(f"Found {len(systems)} systems: {systems}")

        # Helper: pick whichever quality metric is present (without tying to
        # query_id)
        def get_quality_metric(query_data):
            # Check for metric_type + accuracy first (for adjusted-rand-index in ecomm)
            if "metric_type" in query_data and "accuracy" in query_data and query_data["accuracy"] is not None:
                return query_data["accuracy"], query_data["metric_type"]
            # Prefer F1 if present
            if "f1_score" in query_data and query_data["f1_score"] is not None:
                return query_data["f1_score"], "F1 Score"
            # Otherwise use transformed relative error if present
            if (
                "relative_error" in query_data
                and query_data["relative_error"] is not None
            ):
                rel_err = query_data["relative_error"]
                return 1 / (1 + rel_err), "Quality (1/(1+RelErr))"
            # Otherwise Spearman if present
            if (
                "spearman_correlation" in query_data
                and query_data["spearman_correlation"] is not None
            ):
                return (
                    query_data["spearman_correlation"],
                    "Spearman Correlation",
                )
            # Nothing found
            return None, "Quality"

        # Create figure optimized for double-column SIGMOD paper - pareto plots
        #  only
        fig = plt.figure(figsize=(15, 3))

        # Create GridSpec: 2 rows, 5 columns for pareto plots only
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.2)

        # Create subplots for cost-quality pareto (2x5)
        pareto_axes = []
        for row in range(2):
            row_axes = []
            for col in range(5):
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
            pareto_axes.append(row_axes)

        # Plot cost-quality pareto for each query
        for idx, query_id in enumerate(
            query_ids[:10]
        ):  # Limit to first 10 queries
            row = idx // 5  # 0 for Q1-Q5, 1 for Q6-Q10
            col = idx % 5  # 0-4 for columns

            ax = pareto_axes[row][col]
            query_key = f"Q{query_id}"

            # Collect data for this query
            plot_data = []
            missing_systems = []

            for sys_name in systems:
                if query_key in metrics_data.get(sys_name, {}):
                    query_data = metrics_data[sys_name][query_key]
                    cost = query_data.get("money_cost")
                    quality, quality_label = get_quality_metric(query_data)

                    # Keep only strictly positive costs (zero-cost points are
                    # excluded)
                    if cost is not None and quality is not None and cost > 0:
                        plot_data.append(
                            {
                                "system": sys_name,
                                "quality": quality,
                                "cost": cost,
                            }
                        )
                    else:
                        missing_systems.append(sys_name)
                else:
                    missing_systems.append(sys_name)

            # Plot scatter points
            if plot_data:
                for data_point in plot_data:
                    sys_name = data_point["system"]
                    ax.scatter(
                        data_point["quality"],
                        data_point["cost"],
                        color=self.get_system_color(sys_name),
                        marker=self.system_markers.get(sys_name, "o"),
                        s=80,  # Larger markers for better visibility
                        edgecolor="black",
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=3,
                    )

                # Extract quality and cost ranges from actual data
                qualities = [d["quality"] for d in plot_data]
                costs = [d["cost"] for d in plot_data]

                # ---- X-axis (Quality) handling ----
                q_min, q_max = min(qualities), max(qualities)

                # Determine padding needed near boundaries
                left_pad = (
                    0.05 if q_min < 0.1 else 0.02
                )  # More padding if data near 0
                right_pad = (
                    0.05 if q_max > 0.9 else 0.02
                )  # More padding if data near 1

                x_low = -left_pad
                x_high = 1.0 + right_pad

                ax.set_xlim(x_low, x_high)

                # Always show standard x-axis ticks
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0.0", "0.5", "1.0"])

                # ---- Y-axis (Cost) handling ----
                c_min, c_max = min(costs), max(costs)
                if c_max == c_min:
                    c_pad = max(0.1 * c_max, 0.02)
                    y_low, y_high = c_min - c_pad, c_max + c_pad
                else:
                    c_range = c_max - c_min
                    c_pad = max(0.15 * c_range, 0.02)
                    y_low, y_high = c_min - c_pad, c_max + c_pad

                # Ensure minimum range for visibility
                if y_high - y_low < 1e-6:
                    y_high = y_low + 0.05

                ax.set_ylim(y_low, y_high)

                # Set y-axis ticks to only show non-negative values

                # Generate reasonable tick positions
                y_range = y_high - max(
                    0, y_low
                )  # Only consider positive range for ticks
                if y_range > 0:
                    tick_start = max(
                        0.01, c_min
                    )  # Ensure minimum tick is at least 0.01
                    num_ticks = min(
                        4, max(2, int(y_range / (y_range / 3)))
                    )  # Reduce to 3-4 ticks
                    tick_step = (
                        y_range / (num_ticks - 1) if num_ticks > 1 else y_range
                    )

                    valid_y_ticks = []
                    for i in range(num_ticks):
                        tick = tick_start + i * tick_step
                        if (
                            tick >= 0.01 and y_low <= tick <= y_high
                        ):  # Ensure tick >= 0.01
                            valid_y_ticks.append(tick)

                    # Ensure we have at least one tick and it's not 0.00
                    if not valid_y_ticks and c_max >= 0.01:
                        valid_y_ticks = [c_max]
                    elif not valid_y_ticks:
                        valid_y_ticks = [0.01]

                    if valid_y_ticks:
                        ax.set_yticks(valid_y_ticks)
                        ax.set_yticklabels(
                            [f"{tick:.2f}" for tick in valid_y_ticks]
                        )
                # ---- end axis handling ----
            else:
                # Even with no data, set reasonable range
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(
                    -0.02, 0.1
                )  # Small negative padding for visual space

                # Set default ticks for no-data case
                ax.set_xticks([0.0, 0.5, 1.0])
                ax.set_xticklabels(["0.00", "0.50", "1.00"])
                ax.set_yticks([0.01, 0.05, 0.10])
                ax.set_yticklabels(["0.01", "0.05", "0.10"])

            # Subplot styling - larger fonts for presentation quality
            ax.set_title(f"Q{query_id}", fontsize=12, fontweight="bold", pad=4)

            # X-axis: show numbers on both rows, but only label on bottom row
            if row == 1:  # Bottom row gets the "Quality" label
                ax.set_xlabel("Quality", fontsize=11)
            else:  # Top row gets no label, just numbers
                ax.set_xlabel("")
            ax.tick_params(axis="x", which="major", labelsize=10, pad=2)

            # Y-axis: only left column gets labels
            ax.set_ylabel("Cost ($)" if col == 0 else "", fontsize=11)
            ax.tick_params(axis="y", which="major", labelsize=10, pad=2)

            # Adjust y-axis to prevent overlap with x-axis
            ax.tick_params(axis="y", which="major", pad=4)

            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add single unified legend for all systems across all plots
        all_handles = []
        for sys_name in systems:
            handle = mpatches.Patch(
                color=self.get_system_color(sys_name), label=sys_name
            )
            all_handles.append(handle)

        # Position legend outside the figure area at the top with more margin
        fig.legend(
            handles=all_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),  # Further outside to avoid overlap
            ncol=len(systems),
            frameon=False,
            fontsize=11,  # Increased for presentation quality
            columnspacing=1.8,  # Much more space between system names
            handletextpad=0.6,  # More space between patch and text
        )

        # Title following the same style as before
        # scenario_titles = {
        #     "movie": "(a) Movie Scenario",
        #     "detective": "(b) Detective Scenario",
        #     "animals": "(c) Wildlife Scenario",
        #     "ecomm": "(d) Ecomm Scenario",
        #     "mmqa": "(e) MMQA Scenario",
        #     "medical": "(f) Medical Scenario"
        # }
        # title_text = scenario_titles.get(
        # use_case, f"({use_case.title()} Scenario)")
        # fig.suptitle(title_text, fontsize=12, fontweight="bold", y=-0.08)

        # Adjust layout with space for legend and title
        plt.tight_layout(
            rect=[0, 0.08, 1, 0.92]
        )  # Reserve space for legend and title

        # Extract model name from across_system folder
        model_name = (
            across_system_folder.split("_", 2)[-1]
            if "_" in across_system_folder
            else across_system_folder
        )

        # Create filename: {scenario}_{model}_pareto.pdf
        filename = f"{use_case}_{model_name}_pareto.pdf"

        # Save to the across_system folder
        output_dir = self.figures_dir / use_case / across_system_folder
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save both PDF and PNG formats
        plt.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        # Save PNG version
        png_filename = filename.replace(".pdf", ".png")
        plt.savefig(
            output_dir / png_filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close()

        # Create query_pareto folder and generate individual query plots
        query_pareto_dir = output_dir / "query_pareto"
        query_pareto_dir.mkdir(exist_ok=True, parents=True)

        # Generate individual pareto plot for each query
        for query_id in query_ids:
            query_key = f"Q{query_id}"

            # Create individual figure with moderate rectangular aspect ratio
            # (8:5 ratio)
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

            # Collect data for this query
            plot_data = []
            missing_systems = []

            for sys_name in systems:
                if query_key in metrics_data.get(sys_name, {}):
                    query_data = metrics_data[sys_name][query_key]
                    cost = query_data.get("money_cost")
                    quality, quality_label = get_quality_metric(query_data)

                    # Keep only strictly positive costs (zero-cost points are
                    # excluded)
                    if cost is not None and quality is not None and cost > 0:
                        plot_data.append(
                            {
                                "system": sys_name,
                                "quality": quality,
                                "cost": cost,
                            }
                        )
                    else:
                        missing_systems.append(sys_name)
                else:
                    missing_systems.append(sys_name)

            # Plot scatter points
            if plot_data:
                for data_point in plot_data:
                    sys_name = data_point["system"]
                    ax.scatter(
                        data_point["quality"],
                        data_point["cost"],
                        color=self.get_system_color(sys_name),
                        marker=self.system_markers.get(sys_name, "o"),
                        s=200,  # Large markers for dense narrow plots
                        edgecolor="black",
                        linewidth=1.0,
                        alpha=0.9,
                        zorder=3,
                        label=sys_name,
                    )

                # Set axis limits
                costs = [d["cost"] for d in plot_data]
                ax.set_xlim(-0.05, 1.05)

                # Improved Y-axis handling for individual plots - avoid
                # negative numbers
                c_min, c_max = min(costs), max(costs)

                # Calculate padding based on cost range
                if c_max == c_min:
                    base = c_max
                    pad_above = max(
                        0.1 * base, 0.1
                    )  # At least 0.1 padding above
                    y_low, y_high = 0, base + pad_above
                else:
                    c_range = c_max - c_min
                    pad_above = 0.15 * c_range
                    y_high = c_max + pad_above

                    # For y_low: use small positive value to show points near
                    # zero without clipping but avoid negative axis values
                    if c_min <= 0.1:  # If minimum cost is very small
                        y_low = 0
                    else:
                        pad_below = min(
                            0.1 * c_range, 0.05 * c_min
                        )  # Conservative padding below
                        y_low = max(0, c_min - pad_below)  # Never go below 0

                # Ensure reasonable range
                if y_high - y_low < 0.1:
                    y_high = y_low + 0.1

                ax.set_ylim(y_low, y_high)
            else:
                # Even with no data, set reasonable range (no negative y-axis)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(0, 0.1)

            # Styling for individual plots
            ax.set_title(
                f"Query {query_id} - Cost vs Quality",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            ax.set_xlabel("Quality", fontsize=12, fontweight="bold")
            ax.set_ylabel("Cost ($)", fontsize=12, fontweight="bold")

            # Set custom x-axis ticks
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
            ax.tick_params(axis="both", which="major", labelsize=12)

            # Format y-axis labels
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x:.2f}")
            )

            # Grid and spines
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add legend between title and figure to avoid overlap
            if plot_data:
                # Get unique systems that appear in this query's data
                legend_systems = list(set([d["system"] for d in plot_data]))
                legend_handles = []
                for sys_name in legend_systems:
                    # Create legend entry with same marker style as plot
                    handle = plt.Line2D(
                        [0],
                        [0],
                        marker=self.system_markers.get(sys_name, "o"),
                        color="w",
                        markerfacecolor=self.get_system_color(sys_name),
                        markersize=12,
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        label=sys_name,
                        linestyle="None",
                    )
                    legend_handles.append(handle)

                # Position legend between title and plot
                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.85),
                    ncol=len(legend_systems),
                    fontsize=11,
                    frameon=False,
                    columnspacing=1.5,
                )

            # Adjust layout to position legend between title and figure
            plt.tight_layout(
                rect=[0, 0, 1, 0.92]
            )  # Reserve top space for title
            plt.subplots_adjust(
                top=0.82
            )  # Create space between title and plot for legend

            # Save individual query plot
            query_filename = f"Q{query_id}.png"
            plt.savefig(
                query_pareto_dir / query_filename,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

        print(f"✅ Saved {filename} and {png_filename} to {output_dir}")
        print(f"✅ Saved individual query plots to {query_pareto_dir}")

        ### HEATMAP ###
        fig, ax = plt.subplots(figsize=(latex_full_width, 7))
        plotting_data = []
        for system, metrics in metrics_data.items():
            for query_id, metric in metrics.items():
                metric_type, accuracy = self.unify_accuracy_metric(metric)
                if "money_cost" in metric and accuracy is not None:
                    money_cost = metric["money_cost"]
                    plotting_data.append(
                        {
                            "system": system,
                            "query_id": query_id,
                            "money_cost": money_cost,
                            "accuracy": accuracy,
                            "cost_accuracy_tradeoff": (
                                money_cost / accuracy
                                if accuracy > 0
                                else float("nan")
                            ),
                        }
                    )
        plotting_data = pd.DataFrame.from_records(plotting_data)

        # Bring data into a matrix form
        plotting_data_pivot = plotting_data.pivot(
            index="system",
            columns="query_id",
            values="cost_accuracy_tradeoff",
        )
        plotting_data_pivot = plotting_data_pivot.reindex(
            natsorted(plotting_data_pivot.columns), axis=1
        )
        # Normalize columns so that we get column-wise colors
        plotting_data_pivot = plotting_data_pivot.sub(
            plotting_data_pivot.min(axis=0), axis=1
        ).div(plotting_data_pivot.max(axis=0), axis=1)

        # Create custom heatmap
        im = ax.imshow(plotting_data_pivot, cmap="inferno_r")

        # Add labels
        ax.set_xticks(
            range(len(plotting_data_pivot.columns)),
            labels=plotting_data_pivot.columns,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(
            range(len(plotting_data_pivot.index)),
            labels=plotting_data_pivot.index,
        )

        cost = (
            plotting_data.dropna(subset=["system", "query_id", "money_cost"])
            .set_index(["system", "query_id"])["money_cost"]
            .to_dict()
        )
        accuracy = (
            plotting_data.dropna(subset=["system", "query_id", "accuracy"])
            .set_index(["system", "query_id"])["accuracy"]
            .to_dict()
        )

        # Annotate each visible cell
        for i, r in enumerate(plotting_data_pivot.index):
            for j, c in enumerate(plotting_data_pivot.columns):
                mc = cost.get((r, c))
                ac = accuracy.get((r, c))
                if (
                    pd.notna(plotting_data_pivot.loc[r, c])
                    and mc is not None
                    and ac is not None
                ):
                    ax.text(
                        j,
                        i,
                        f"${mc:.2f}\n{ac:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="gray",
                    )

        # Optional: add colorbar
        plt.colorbar(im, ax=ax)

        # sns.heatmap(data=plotting_data, annot=True)
        plt.savefig(
            output_dir / "heatmap_table.pdf", dpi=300, bbox_inches="tight"
        )

        # Generate latex table:
        systems = [
            "LOTUS",
            "Palimpzest",
            "ThalamusDB",
            # "CAESURA",
            # "BigQuery",
            # "Snowflake",
        ]
        query_ids = natsorted(plotting_data["query_id"].unique())
        out_file_path = output_dir / f"heatmap_table_{use_case}.tex"
        with open(out_file_path, "w") as tex_file:
            tex_system_cols = " c " * len(systems)
            tex_file.write(
                f"%% This table is generated by plot.py. Do not edit manually. %%\n"
            )
            tex_file.write(
                """
\\providecommand{\\tikzcell}[4]{%
  \\begingroup
    \\definecolor{ULcolor}{HTML}{#3}%
    \\definecolor{LRcolor}{HTML}{#4}%
    % row metrics (respect \\arraystretch)
    \\dimen0=\\dimexpr\\arraystretch\\ht\\strutbox\\relax
    \\dimen2=\\dimexpr\\arraystretch\\dp\\strutbox\\relax
    \\dimen4=\\dimexpr\\dimen0+\\dimen2\\relax % total height
    \\begin{tikzpicture}[x=1cm,y=1pt,baseline=(rowbase)]
      % baseline of the row is at depth above bottom
      \\coordinate (rowbase) at (0,\\the\\dimen2);
      % your geometry, with explicit expanded lengths
      \\path[fill=LRcolor!80] (0.8,0) -- (2,0) -- (2,\\the\\dimen4) -- (1.2,\\the\\dimen4) -- cycle; % lower-right
      \\path[fill=ULcolor!80] (0,0) -- (0,\\the\\dimen4) -- (1.2,\\the\\dimen4) -- (0.8,0) -- cycle; % upper-left
      % place labels exactly on the row baseline
      \\node[anchor=base west,inner xsep=4pt, inner ysep=0pt] at (0,\\the\\dimen2) {#1}; % UL text
      \\node[anchor=base east,inner xsep=4pt, inner ysep=0pt] at (2,\\the\\dimen2) {#2}; % LR text
    \\end{tikzpicture}%
  \\endgroup
}
"""
            )
            tex_file.write(f"\\begin{{table}}[t]\n")
            tex_file.write(f"  \\centering\n")
            tex_file.write(
                f"  \\caption{{Statistics of the SemBench for {use_case}}}\n"
            )
            tex_file.write(f"  \\setlength{{\\tabcolsep}}{{0pt}}\n")
            tex_file.write(
                f"  \\begin{{tabular}}{{p{{.8cm}} {tex_system_cols}}}\n"
            )
            tex_file.write(f"    \\toprule\n")
            tex_file.write(f"      & {' & '.join(systems)} \\\\\n")
            tex_file.write(f"    \\midrule\n")

            # Store each cell's content in a 2D array that will be written out at the end
            table_cells = [
                ["n/a\\qquad n/a"] * (1 + len(systems))
                for i in range(len(query_ids))
            ]

            # First column contains query ids
            for query_idx, query_id in enumerate(query_ids):
                table_cells[query_idx][0] = query_id

            # TODOs:
            # * try out how it looks with two different color scales for cost and accuracy or one continuous color scale?
            # * try out discretizing to $$$$$ and ***** instead of exact numbers

            def latex_float(f):
                float_str = "{0:.0e}".format(f)
                if "e" in float_str:
                    base, exponent = float_str.split("e")
                    return r"{0}{{\cdot\scriptstyle 10^{{{1}}}}}".format(
                        base, int(exponent)
                    )
                else:
                    return float_str

            for sys_idx, system in enumerate(systems):
                if system.lower() not in metrics_data:
                    continue  # there are no metrics for this system
                sys_metrics = metrics_data[system.lower()]
                for query_idx, query_id in enumerate(query_ids):
                    if query_id not in sys_metrics:
                        continue  # there are no metrics for this query
                    query_metrics = sys_metrics[query_id]
                    if (
                        "money_cost" in query_metrics
                        and not "error" in query_metrics
                    ):
                        cost = query_metrics["money_cost"]
                        _, accuracy = self.unify_accuracy_metric(query_metrics)
                        cost_accuracy = (
                            cost / accuracy if accuracy > 0 else float("nan")
                        )

                        # Normalize cost/accuracy tradeoff for coloring
                        min_cost_accuracy = plotting_data[
                            plotting_data["query_id"] == query_id
                        ]["cost_accuracy_tradeoff"].min()
                        max_cost_accuracy = plotting_data[
                            plotting_data["query_id"] == query_id
                        ]["cost_accuracy_tradeoff"].max()
                        cost_accuracy_norm = (
                            (cost_accuracy - min_cost_accuracy)
                            / (max_cost_accuracy - min_cost_accuracy)
                            if max_cost_accuracy > min_cost_accuracy
                            else 0.0
                        )

                        # Normalize cost for coloring
                        min_cost = plotting_data[
                            plotting_data["query_id"] == query_id
                        ]["money_cost"].min()
                        max_cost = plotting_data[
                            plotting_data["query_id"] == query_id
                        ]["money_cost"].max()
                        cost_norm = (
                            (cost - min_cost) / (max_cost - min_cost)
                            if max_cost > min_cost
                            else 0.0
                        )

                        # Normalize accuracy for coloring
                        min_accuracy = plotting_data[
                            plotting_data["query_id"] == query_id
                        ]["accuracy"].min()
                        max_accuracy = plotting_data[
                            plotting_data["query_id"] == query_id
                        ]["accuracy"].max()
                        accuracy_norm = (
                            (accuracy - min_accuracy)
                            / (max_accuracy - min_accuracy)
                            if max_accuracy > min_accuracy
                            else 1.0
                        )

                        cmap = colors.LinearSegmentedColormap.from_list(
                            "green_red", ["limegreen", "khaki", "firebrick"]
                        )
                        cell_color = colors.to_hex(cmap(cost_accuracy_norm))
                        cost_color = colors.to_hex(cmap(cost_norm))
                        accuracy_color = colors.to_hex(
                            cmap.reversed()(accuracy * 2 - 1)
                        )
                        cost_text = (
                            f"\\$${cost:.2f}$"
                            if round(cost, 2) > 0.0
                            else f"\\$${latex_float(cost)}$"
                        )
                        accuracy_text = f"${accuracy:.2f}$"
                        cell = f"\\tikzcell{{{cost_text}}}{{{accuracy_text}}}{{{cost_color.strip('#')}}}{{{accuracy_color.strip('#')}}}"
                        table_cells[query_idx][1 + sys_idx] = cell

            # Write out table cells
            for row in table_cells:
                tex_file.write(f"    {' & '.join(row)} \\\\\n")

            tex_file.write(f"    \\bottomrule\n")
            tex_file.write(f"  \\end{{tabular}}\n")
            tex_file.write(f"\\end{{table}}\n")
        print(f"✅ Saved heatmap_table to {out_file_path}")

    def plot_summary_2_5_flash(self):
        """
        Create a 3-column summary figure for 2.5flash data across movie use
        case.

        Backward compatibility wrapper for the generalized function.
        """
        self.plot_summary_across_systems("movie", "across_system_2.5flash")

    def generate_all_plots(self):
        """Generate all plots for all use cases."""
        use_cases = self.get_use_cases()

        if not use_cases:
            print("No use cases found!")
            return

        print(f"Found use cases: {use_cases}")

        for use_case in use_cases:
            # if use_case != "ecomm":
            #     continue
            print(f"\nProcessing use case: {use_case}")

            # 1. Load and process regular metrics data (excluding scalability
            # files)
            metrics_data = self.load_metrics_data(use_case)

            if not metrics_data:
                print(f"No regular metrics data found for {use_case}")
            else:
                print(f"Found systems: {list(metrics_data.keys())}")

                # Generate regular plots
                try:
                    # self.plot_execution_time(metrics_data, use_case)
                    # self.plot_cost(metrics_data, use_case)
                    # self.plot_quality(metrics_data, use_case)
                    # self.plot_quality_one_metric(metrics_data, use_case)
                    # self.plot_pareto(metrics_data, use_case)
                    print(f"Successfully generated plots for {use_case}")
                except Exception as e:
                    print(f"Error generating plots for {use_case}: {e}")
                    import traceback

                    traceback.print_exc()

            # 2. Load and process system-specific metrics data
            system_folders = self.get_system_subfolders(use_case)
            if system_folders:
                print(f"Found system subfolders: {system_folders}")

                for system_folder in system_folders:
                    print(
                        f"\nProcessing system subfolder: {use_case}/{system_folder}"  # noqa: E501
                    )

                    system_metrics_data = self.load_system_metrics_data(
                        use_case, system_folder
                    )

                    if not system_metrics_data:
                        print(
                            f"No metrics data found for {use_case}/{system_folder}"  # noqa: E501
                        )
                    else:
                        print(
                            f"Found system variants: {list(system_metrics_data.keys())}"  # noqa: E501
                        )

                        # Generate system-specific plots
                        try:
                            # self.plot_execution_time(
                            #     system_metrics_data, use_case, system_folder
                            # )
                            # self.plot_cost(
                            #     system_metrics_data, use_case, system_folder
                            # )
                            # self.plot_quality(
                            #     system_metrics_data, use_case, system_folder
                            # )
                            # self.plot_quality_one_metric(
                            #     system_metrics_data,
                            #     use_case,
                            #     system_name=system_folder,
                            # )
                            # self.plot_pareto(
                            #     system_metrics_data, use_case, system_folder
                            # )
                            # self.plot_pareto_curve(
                            #     system_metrics_data, use_case, system_folder
                            # )

                            # Generate summary plots for across_system folders
                            if system_folder.startswith("across_system"):
                                print(
                                    f"Generating summary plot for {use_case}/{system_folder}"  # noqa: E501
                                )
                                self.plot_summary_across_systems(
                                    use_case, system_folder
                                )
                                self.plot_pareto_across_systems(
                                    use_case, system_folder
                                )

                                # Also generate error bar version if round folders exist
                                # Extract model tag from system_folder (e.g., "across_system_2.5flash" -> "2.5flash")
                                if system_folder.startswith("across_system_"):
                                    model_tag = system_folder.replace(
                                        "across_system_", ""
                                    )
                                    print(
                                        f"Generating error bar plot for {use_case} with model_tag {model_tag}"
                                    )
                                    self.plot_summary_across_systems_with_error_bar(
                                        use_case, model_tag
                                    )

                            print(
                                f"Successfully generated plots for {use_case}/{system_folder}"  # noqa: E501
                            )
                        except Exception as e:
                            print(
                                f"Error generating plots for {use_case}/{system_folder}: {e}"  # noqa: E501
                            )
                            import traceback

                            traceback.print_exc()
            else:
                print(f"No system subfolders found for {use_case}")

        # Plotting across use cases
        self.plot_pareto_across_all_use_cases()

        all_metrics = []
        for use_case in ["animals", "detective", "movie", "ecomm", "mmqa"]:
            metrics_data = self.load_metrics_data(use_case)
            for system_name, metrics in metrics_data.items():
                for query_id, metric in metrics.items():
                    metric["use_case"] = use_case
                    metric["system"] = system_name
                    metric["query_id_str"] = query_id
                    metric_type, accuracy = self.unify_accuracy_metric(metric)
                    metric["metric_type"] = metric_type
                    metric["accuracy"] = accuracy
                    all_metrics.append(metric)
        self.plot_avg_cost_accuracy_ratio_per_use_case(all_metrics)


def plot_llm_model_scatter_plot():
    toml_file_path = os.path.join("src", "models.toml")
    output_image_path = os.path.join("figures", "llm-price-quality.png")

    with open(toml_file_path, "rb") as f:
        data = tomli.load(f)

    # Extract relevant data
    prices = []
    qualities = []
    names = []

    for model in data.get("model", []):
        price_input = model.get("price", {}).get("input")
        quality = model.get("quality", {}).get("livebench")

        if price_input is not None and quality is not None:
            prices.append(price_input)
            qualities.append(quality)
            names.append(model.get("display_name", "Unnamed"))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(prices, qualities, c="blue", alpha=0.7)

    for i, name in enumerate(names):
        plt.annotate(
            name,
            (prices[i], qualities[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    plt.title("Model Input Price vs. Livebench Quality")
    plt.xlabel("Input Price (USD / 1M tokens)")
    plt.ylabel("Livebench Quality")
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()
    print(f"Plot saved to {output_image_path}")

    def plot_palimpzest_pareto_evaluation(self):
        """
        Create cost-quality Pareto plots for Palimpzest evaluation data.
        Generates a 2x5 subplot grid showing cost-quality trade-offs for each
        query, with different markers for models and colors for optimization
        objectives.
        """
        # --- Constants and Style Configuration ---
        data_dir = (
            self.files_dir / "movie" / "metrics" / "palimpzest" / "evaluate"
        )
        output_dir = self.figures_dir

        # Model and objective mappings for consistent plotting
        model_markers = {
            "gemini-2.0-flash": "o",
            "gemini-2.5-flash": "s",
            "gpt-4o-mini": "^",
            "gpt-5-mini": "D",
            "mixed-models": "*",
        }

        # Special styling for mixed-models to make them more prominent
        mixed_models_style = {
            "marker_size": 200,
            "edge_color": "black",
            "edge_width": 1.2,
            "alpha": 0.9,
            "zorder": 3,
        }

        regular_models_style = {
            "marker_size": 100,
            "edge_color": "black",
            "edge_width": 0.8,
            "alpha": 0.7,
            "zorder": 2,
        }

        objective_colors = {"maxquality": "#1f77b4", "mincost": "#d62728"}
        quality_metric_priority = [
            "f1_score",
            "relative_error",
            "spearman_correlation",
            "kendall_tau",
        ]

        # --- Helper Functions ---
        def load_evaluation_data(data_dir: Path):
            """Load all JSON evaluation files from a directory."""
            all_data = {}
            print(f"🔍 Loading palimpzest evaluation data from {data_dir}...")
            for json_file in sorted(data_dir.glob("*.json")):
                try:
                    with open(json_file, "r") as f:
                        all_data[json_file.stem] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️ Error loading {json_file.name}: {e}")
            return all_data

        def extract_model_and_objective(filename: str):
            """Extract model name and optimization objective from a filename."""
            if "mixed-models" in filename:
                model = "mixed-models"
                objective = filename.split("-")[-1]
                return model, objective

            for objective_name in objective_colors.keys():
                suffix = f"-{objective_name}"
                if filename.endswith(suffix):
                    model = filename.removesuffix(suffix)
                    return model, objective_name

            parts = filename.split("-")
            return "-".join(parts[:-1]), parts[-1]

        def calculate_quality_score(query_data):
            """Calculate a unified quality score from available metrics based
            on priority."""
            for metric in quality_metric_priority:
                if metric in query_data:
                    value = query_data[metric]
                    if value is None:
                        continue
                    if metric == "relative_error":
                        return 1 / (1 + value)
                    return value
            return None

        # --- Main Plotting Function ---
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return

        # Load data
        data = load_evaluation_data(data_dir)
        if not data:
            print("❌ No evaluation data found!")
            return

        # Create figure
        fig = plt.figure(figsize=(15, 4.5))
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(2, 5, figure=fig)
        query_ids = list(range(1, 11))

        # Displacement configuration for overlapping points
        displace_radius_x = 0.025
        displace_radius_y = 0.035

        for idx, query_id in enumerate(query_ids):
            ax = fig.add_subplot(gs[idx // 5, idx % 5])
            query_key = f"Q{query_id}"

            # Collect all points for the subplot
            points_by_coord = defaultdict(list)
            points_by_model = defaultdict(dict)  # For drawing trade-off lines
            all_costs = []

            for filename, file_data in data.items():
                if (
                    query_key in file_data
                    and file_data[query_key].get("status") == "success"
                ):
                    query_data = file_data[query_key]
                    cost = query_data.get("money_cost")
                    quality = calculate_quality_score(query_data)
                    if cost is not None and quality is not None and cost >= 0:
                        model, objective = extract_model_and_objective(filename)

                        # Store for dodging
                        coord_key = (round(quality, 2), round(cost, 5))
                        points_by_coord[coord_key].append(
                            {"filename": filename}
                        )

                        # Store for drawing trade-off lines
                        points_by_model[model][objective] = {
                            "quality": quality,
                            "cost": cost,
                        }

                        all_costs.append(cost)

            # Draw trade-off lines connecting mincost and maxquality points for
            # each model
            for model, objectives in points_by_model.items():
                if "mincost" in objectives and "maxquality" in objectives:
                    p_mc = objectives["mincost"]
                    p_mq = objectives["maxquality"]
                    ax.plot(
                        [p_mc["quality"], p_mq["quality"]],
                        [p_mc["cost"], p_mq["cost"]],
                        color="gray",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.7,
                        zorder=1,  # Draw lines behind points
                    )

            # Plot points with dodging for overlapping coordinates
            cost_range = max(all_costs) - min(all_costs) if all_costs else 0
            y_radius = (
                cost_range * displace_radius_y
                if cost_range > 0
                else (np.mean(all_costs) * 0.1 if all_costs else 0.0001)
            )

            for (quality, cost), points in points_by_coord.items():
                num_points = len(points)
                if num_points == 1:
                    point = points[0]
                    model, objective = extract_model_and_objective(
                        point["filename"]
                    )

                    # Apply special styling for mixed-models
                    style = (
                        mixed_models_style
                        if model == "mixed-models"
                        else regular_models_style
                    )

                    ax.scatter(
                        quality,
                        cost,
                        marker=model_markers.get(model, "x"),
                        color=objective_colors.get(objective, "gray"),
                        s=style["marker_size"],
                        edgecolor=style["edge_color"],
                        linewidth=style["edge_width"],
                        alpha=style["alpha"],
                        zorder=style["zorder"],
                    )
                else:
                    # Dodge overlapping points in circular pattern
                    for i, point in enumerate(points):
                        angle = 2 * np.pi * i / num_points
                        dx = displace_radius_x * np.cos(angle)
                        dy = y_radius * np.sin(angle)
                        model, objective = extract_model_and_objective(
                            point["filename"]
                        )

                        # Apply special styling for mixed-models
                        style = (
                            mixed_models_style
                            if model == "mixed-models"
                            else regular_models_style
                        )

                        ax.scatter(
                            quality + dx,
                            cost + dy,
                            marker=model_markers.get(model, "x"),
                            color=objective_colors.get(objective, "gray"),
                            s=style["marker_size"],
                            edgecolor=style["edge_color"],
                            linewidth=style["edge_width"],
                            alpha=style["alpha"],
                            zorder=style["zorder"],
                        )

            # Subplot formatting
            ax.set_title(f"Query {query_id}", fontsize=12, fontweight="bold")
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks([0, 0.5, 1.0])
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda val, pos: f"${val:.3f}")
            )
            if idx % 5 == 0:
                ax.set_ylabel("Cost ($)", fontsize=14)
            if idx // 5 == 1:
                ax.set_xlabel("Quality Score", fontsize=14)

        # Figure-level title and two-row legend
        fig.suptitle(
            "Palimpzest: Cost-Quality Pareto Front by Query",
            fontsize=15,
            fontweight="bold",
            y=0.99,
        )

        # Create model legend handles
        model_handles = []
        for model_name, marker in model_markers.items():
            handle = plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                linestyle="None",
                markersize=9,
                markeredgewidth=1.0,
                markeredgecolor="black",
                label=model_name,
            )
            model_handles.append(handle)

        # Create objective legend handles
        objective_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=c,
                linestyle="None",
                markersize=9,
                label=n,
            )
            for n, c in objective_colors.items()
        ]

        # Add two-row legend
        fig.legend(
            handles=model_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=len(model_handles),
            frameon=False,
            fontsize=12,
        )

        fig.legend(
            handles=objective_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=len(objective_handles),
            frameon=False,
            fontsize=12,
        )

        fig.subplots_adjust(
            left=0.06, right=0.99, top=0.75, bottom=0.12, hspace=0.5, wspace=0.3
        )

        # Save figures
        output_dir.mkdir(exist_ok=True)
        png_path = output_dir / "palimpzest_cost_quality_pareto.png"
        pdf_path = output_dir / "palimpzest_cost_quality_pareto.pdf"

        plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"✅ Palimpzest pareto plots saved to {png_path} and {pdf_path}")


def main():
    """Main function to run the benchmark plotter."""
    plotter = BenchmarkPlotter()
    plotter.generate_all_plots()
    plot_llm_model_scatter_plot()
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
