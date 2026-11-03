from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency for lightweight environments
    np = None
import torch


@dataclass
class CompressionResult:
    retained_tokens: torch.Tensor
    retained_indices: torch.Tensor
    selected_mask: torch.Tensor
    importance_scores: torch.Tensor
    method_name: str
    retention_ratio: float
    extra_info: dict = field(default_factory=dict)


TASK_CATEGORIES = {
    "general_vqa": ["What is", "Describe", "What do you see"],
    "counting": ["How many", "Count", "Number of"],
    "spatial": ["left of", "right of", "above", "below", "between", "Where is"],
    "ocr": ["What does it say", "Read", "text in", "written"],
    "attribute": ["What color", "What shape", "What material", "How big"],
    "yes_no": ["Is there", "Does the", "Are there", "Can you see"],
}


def classify_task(question: str) -> str:
    q_lower = question.lower()
    for category, keywords in TASK_CATEGORIES.items():
        if any(keyword.lower() in q_lower for keyword in keywords):
            return category
    return "general_vqa"


def _to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np is not None and isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(data), handle, indent=2, ensure_ascii=False)


class PerTaskEvaluator:
    def __init__(self, methods: Dict[str, Callable]):
        self.methods = methods

    def evaluate_sample(
        self,
        model_forward_fn: Callable,
        visual_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
        query_tokens: torch.Tensor,
        question: str,
        ground_truth: str,
        ratios: Optional[List[float]] = None,
    ) -> Dict:
        ratios = ratios or [0.25, 0.5, 0.75]
        task = classify_task(question)
        results = {"question": question, "task": task, "gt": ground_truth}
        full_answer = model_forward_fn(visual_tokens, query_tokens)
        results["full_answer"] = full_answer

        for method_name, compress_fn in self.methods.items():
            results[method_name] = {}
            for ratio in ratios:
                compression_result = compress_fn(visual_tokens, attention_weights, query_tokens, ratio)
                compressed_answer = model_forward_fn(compression_result.retained_tokens, query_tokens)
                is_correct = compressed_answer.strip().lower() == ground_truth.strip().lower()
                full_correct = full_answer.strip().lower() == ground_truth.strip().lower()
                results[method_name][ratio] = {
                    "answer": compressed_answer,
                    "correct": is_correct,
                    "full_correct": full_correct,
                    "degradation": bool(full_correct and not is_correct),
                }
        return results

    def aggregate_per_task(self, all_results: List[Dict]) -> Dict[str, Dict[str, Dict[float, Dict[str, float]]]]:
        by_task: Dict[str, List[Dict]] = {}
        for result in all_results:
            by_task.setdefault(result["task"], []).append(result)

        aggregated: Dict[str, Dict[str, Dict[float, Dict[str, float]]]] = {}
        observed_ratios = sorted({ratio for result in all_results for method in self.methods for ratio in result.get(method, {}).keys()})
        for task, task_results in by_task.items():
            aggregated[task] = {}
            for method_name in self.methods:
                aggregated[task][method_name] = {}
                for ratio in observed_ratios:
                    valid_results = [result for result in task_results if ratio in result.get(method_name, {})]
                    correct_count = sum(1 for result in valid_results if result[method_name][ratio]["correct"])
                    degradation_count = sum(1 for result in valid_results if result[method_name][ratio]["degradation"])
                    total = len(valid_results)
                    aggregated[task][method_name][ratio] = {
                        "accuracy": correct_count / max(1, total),
                        "degradation_rate": degradation_count / max(1, total),
                        "total_samples": total,
                    }
        return aggregated


def generate_figure1_task_sensitivity(aggregated_results: Dict, save_path: str) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.size"] = 12
    except ImportError:
        save_json(save_path.replace(".pdf", ".json"), aggregated_results)
        return

    ratio = 0.5
    tasks = list(aggregated_results.keys())
    methods = list(next(iter(aggregated_results.values())).keys()) if aggregated_results else []
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = list(range(len(tasks)))
    width = 0.8 / max(1, len(methods))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for idx, method in enumerate(methods):
        degradations = []
        for task in tasks:
            degradation = aggregated_results[task][method][ratio]["degradation_rate"] * 100
            degradations.append(degradation)
        ax.bar(
            [value + idx * width - 0.4 + width / 2 for value in x],
            degradations,
            width * 0.9,
            label=method,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Task Category", fontsize=14)
    ax.set_ylabel("Accuracy Degradation (%)", fontsize=14)
    ax.set_title(f"Per-Task Accuracy Degradation at {int(ratio * 100)}% Token Retention", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15, ha="right")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure2_distributional_shift(
    shift_results: Dict[str, Dict[float, Dict[str, float]]],
    accuracy_results: Dict[str, Dict[float, float]],
    save_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_json(save_path.replace(".pdf", ".json"), {"shift_results": shift_results, "accuracy_results": accuracy_results})
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"FastV": "#2196F3", "DivPrune": "#FF5722", "DPTC": "#4CAF50", "Random": "#9E9E9E", "TATC": "#9C27B0", "HATP": "#FF9800"}
    ratios = sorted(list(next(iter(shift_results.values())).keys())) if shift_results else []

    for method, ratio_data in shift_results.items():
        kl_values = [ratio_data[ratio]["kl_divergence"] for ratio in ratios]
        ax1.plot(ratios, kl_values, "o-", label=method, color=colors.get(method, "#000000"), linewidth=2, markersize=8)

    ax1.set_xlabel("Retention Ratio", fontsize=13)
    ax1.set_ylabel("KL Divergence (Hidden States)", fontsize=13)
    ax1.set_title("Distributional Shift vs Compression", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    for method in shift_results:
        for ratio in ratios:
            kl_value = shift_results[method][ratio]["kl_divergence"]
            accuracy = accuracy_results.get(method, {}).get(ratio, 0.0)
            ax2.scatter(
                kl_value,
                accuracy,
                c=colors.get(method, "#000000"),
                s=100,
                label=method if ratio == ratios[0] else "",
                zorder=5,
            )

    ax2.set_xlabel("KL Divergence", fontsize=13)
    ax2.set_ylabel("Accuracy", fontsize=13)
    ax2.set_title("KL Shift Predicts Accuracy Better Than Ratio", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure3_hallucination_profile(hall_profiles: Dict[str, Dict[str, float]], save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_json(save_path.replace(".pdf", ".json"), hall_profiles)
        return

    categories = ["object", "attribute", "relation", "count"]
    num_categories = len(categories)
    angles = [index / float(num_categories) * 2 * math.pi for index in range(num_categories)]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection="polar"))
    colors = {"FastV": "#2196F3", "DivPrune": "#FF5722", "HATP": "#4CAF50", "Random": "#9E9E9E", "DPTC": "#9C27B0", "TATC": "#FF9800"}

    for method, profile in hall_profiles.items():
        values = [profile.get(category, 0.0) * 100 for category in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors.get(method, "#000000"))
        ax.fill(angles, values, alpha=0.1, color=colors.get(method, "#000000"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([category.capitalize() for category in categories], fontsize=13)
    ax.set_title("Compression-Induced Hallucination Profile", fontsize=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class EfficiencyProfiler:
    @staticmethod
    def measure_latency(
        compress_fn: Callable,
        visual_tokens: torch.Tensor,
        attention_weights: torch.Tensor,
        ratio: float,
        query_tokens: Optional[torch.Tensor] = None,
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> Dict[str, float]:
        import time

        device = visual_tokens.device
        for _ in range(num_warmup):
            _ = compress_fn(visual_tokens, attention_weights, query_tokens, ratio)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = compress_fn(visual_tokens, attention_weights, query_tokens, ratio)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean_ms": float((sum(times) / max(1, len(times))) * 1000),
            "std_ms": float((torch.tensor(times).std(unbiased=False).item() if len(times) > 1 else 0.0) * 1000),
            "median_ms": float(sorted(times)[len(times) // 2] * 1000),
        }

    @staticmethod
    def estimate_flops_saved(
        num_visual_tokens: int,
        num_retained: int,
        num_text_tokens: int,
        hidden_dim: int,
        num_layers_remaining: int,
    ) -> Dict[str, float]:
        num_visual = num_visual_tokens
        num_kept = num_retained
        num_text = num_text_tokens
        hidden = hidden_dim
        seq_full = num_visual + num_text
        seq_compressed = num_kept + num_text

        attn_full = 2 * seq_full * seq_full * hidden
        attn_compressed = 2 * seq_compressed * seq_compressed * hidden
        ffn_full = 2 * seq_full * hidden * 4 * hidden
        ffn_compressed = 2 * seq_compressed * hidden * 4 * hidden

        total_full = (attn_full + ffn_full) * num_layers_remaining
        total_compressed = (attn_compressed + ffn_compressed) * num_layers_remaining

        return {
            "flops_full_G": total_full / 1e9,
            "flops_compressed_G": total_compressed / 1e9,
            "flops_saved_G": (total_full - total_compressed) / 1e9,
            "speedup_ratio": total_full / max(total_compressed, 1e-8),
            "flops_reduction_pct": (1 - total_compressed / max(total_full, 1e-8)) * 100,
        }
