"""Algorithm comparison for ranking multiple eye tracking algorithms."""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from tabulate import tabulate
from tqdm import tqdm

from .gaze_accuracy import evaluate_gaze_accuracy
from ..core import EyeTracker
from ..types import Position3D, Point3D
from ..geometry.conversions import calculate_angular_error_degrees


@dataclass
class AlgorithmRanking:
    """Algorithm ranking results."""

    rankings: Dict[str, int]
    error_stats: Dict[str, Dict[str, float]]  # Full error statistics per algorithm
    success_rates: Dict[str, float]
    pairwise_angular_diff: Dict[str, Dict[str, float]]  # Algorithm vs algorithm angular differences
    pairwise_cosine_sim: Dict[str, Dict[str, float]]  # Algorithm vs algorithm cosine similarities
    pairwise_amplitude: Dict[str, Dict[str, Dict[str, float]]]  # Algorithm vs algorithm amplitude differences

    def pprint(self, title: str = "Algorithm Ranking"):
        """Print comprehensive ranking and comparison analysis."""
        # Main ranking table
        print(f"\n{title}")
        print("-" * len(title))

        data = []
        for algo in sorted(self.rankings.keys(), key=lambda x: self.rankings[x]):
            rank = self.rankings[algo]
            stats = self.error_stats[algo]["deg"]
            success = self.success_rates[algo]

            mean_error = stats["mean"]
            max_error = stats["max"]
            std_error = stats["std"]

            data.append(
                [
                    rank,
                    algo,
                    f"{mean_error:.3f}°" if not np.isnan(mean_error) else "FAILED",
                    f"{max_error:.3f}°" if not np.isnan(max_error) else "FAILED",
                    f"{std_error:.3f}°" if not np.isnan(std_error) else "FAILED",
                    f"{success:.1f}%",
                ]
            )

        headers = ["Rank", "Algorithm", "Mean Error (°)", "Max Error (°)", "Std Error (°)", "Success Rate"]
        print(tabulate(data, headers=headers, tablefmt="grid"))

        # Pairwise comparison tables
        self._print_pairwise_comparisons()

    def get_best_algorithm(self) -> str:
        """Return best algorithm name."""
        return min(self.rankings.keys(), key=lambda x: self.rankings[x])

    def get_top_n(self, n: int) -> List[str]:
        """Return top N algorithm names."""
        sorted_algos = sorted(self.rankings.keys(), key=lambda x: self.rankings[x])
        return sorted_algos[:n]

    def _print_pairwise_comparisons(self):
        """Print pairwise algorithm comparison tables."""
        algorithms = list(self.rankings.keys())

        # Angular Difference Table (lower is better - more similar)
        print("\nPairwise Angular Differences (°) - Lower = More Similar")
        print("-" * 60)
        self._print_symmetric_matrix(self.pairwise_angular_diff, algorithms, "3f", "°")

        # Cosine Similarity Table (higher is better - more similar)
        print("\nPairwise Cosine Similarities - Higher = More Similar")
        print("-" * 55)
        self._print_symmetric_matrix(self.pairwise_cosine_sim, algorithms, "6f", "")

        # Amplitude Difference Table (mean values, lower is better)
        print("\nPairwise Amplitude Differences (°) - Lower = More Similar")
        print("-" * 62)
        amplitude_means = {}

        # Add progress bar for amplitude calculation
        with tqdm(total=len(algorithms) ** 2, desc="Calculating amplitude differences", leave=False) as pbar:
            for algo1 in algorithms:
                amplitude_means[algo1] = {}
                for algo2 in algorithms:
                    if algo1 in self.pairwise_amplitude and algo2 in self.pairwise_amplitude[algo1]:
                        amplitude_means[algo1][algo2] = self.pairwise_amplitude[algo1][algo2]["mean"]
                    else:
                        amplitude_means[algo1][algo2] = np.nan
                    pbar.update(1)
        self._print_symmetric_matrix(amplitude_means, algorithms, "6f", "°")

    def _print_symmetric_matrix(self, matrix: Dict, algorithms: List[str], fmt: str, unit: str):
        """Print a symmetric comparison matrix."""
        # Create header with abbreviated algorithm names for readability
        abbrev_names = [name[:12] + "..." if len(name) > 15 else name for name in algorithms]

        # Print data matrix
        data = []
        for i, algo1 in enumerate(tqdm(algorithms, desc="Formatting table", leave=False)):
            row = [abbrev_names[i]]
            for algo2 in algorithms:
                if algo1 == algo2:
                    row.append("-")
                elif algo1 in matrix and algo2 in matrix[algo1]:
                    value = matrix[algo1][algo2]
                    row.append(f"{value:.{fmt[0]}f}{unit}" if not np.isnan(value) else "N/A")
                else:
                    row.append("N/A")
            data.append(row)

        headers = ["Algorithm"] + abbrev_names
        print(tabulate(data, headers=headers, tablefmt="grid"))


def compare_algorithms(
    algorithms: Dict[str, EyeTracker], dataset: Dict, description: str = "Comparing algorithms"
) -> AlgorithmRanking:
    """Compare multiple algorithms on same dataset."""

    # Evaluate each algorithm
    results = {}
    for name, algorithm in algorithms.items():
        results[name] = evaluate_gaze_accuracy(algorithm, dataset, description=f"{description} - {name}")

    # Extract error statistics and success rates
    error_stats = {name: r.error_stats for name, r in results.items()}
    success_rates = {name: (r.successful_predictions / r.total_measurements) * 100 for name, r in results.items()}

    # Rank by mean error (lower is better)
    mean_errors = {name: stats["deg"]["mean"] for name, stats in error_stats.items()}
    valid_errors = {name: error for name, error in mean_errors.items() if not np.isnan(error)}
    sorted_algos = sorted(valid_errors.keys(), key=lambda x: valid_errors[x])
    rankings = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}

    # Add failed algorithms at the end
    failed_algos = [name for name in mean_errors.keys() if np.isnan(mean_errors[name])]
    for i, algo in enumerate(failed_algos):
        rankings[algo] = len(sorted_algos) + i + 1

    # Calculate pairwise comparisons
    pairwise_angular_diff = _calculate_pairwise_angular_differences(results)
    pairwise_cosine_sim = _calculate_pairwise_cosine_similarities(results)
    pairwise_amplitude = _calculate_pairwise_amplitude_differences(results)

    return AlgorithmRanking(
        rankings=rankings,
        error_stats=error_stats,
        success_rates=success_rates,
        pairwise_angular_diff=pairwise_angular_diff,
        pairwise_cosine_sim=pairwise_cosine_sim,
        pairwise_amplitude=pairwise_amplitude,
    )


def _calculate_pairwise_angular_differences(results: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate angular differences between all algorithm pairs."""
    algorithms = list(results.keys())
    pairwise_diffs = {}

    with tqdm(total=len(algorithms) ** 2, desc="Computing angular differences", leave=False) as pbar:
        for algo1 in algorithms:
            pairwise_diffs[algo1] = {}
            for algo2 in algorithms:
                if algo1 == algo2:
                    pairwise_diffs[algo1][algo2] = 0.0
                else:
                    diff = point_wise_angular_difference(
                        results[algo1].predicted_points,
                        results[algo2].predicted_points,
                        results[algo1].observer_positions,
                    )
                    pairwise_diffs[algo1][algo2] = diff
                pbar.update(1)

    return pairwise_diffs


def _calculate_pairwise_cosine_similarities(results: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate cosine similarities between all algorithm pairs."""
    algorithms = list(results.keys())
    pairwise_sims = {}

    with tqdm(total=len(algorithms) ** 2, desc="Computing cosine similarities", leave=False) as pbar:
        for algo1 in algorithms:
            pairwise_sims[algo1] = {}
            for algo2 in algorithms:
                if algo1 == algo2:
                    pairwise_sims[algo1][algo2] = 1.0
                else:
                    sim = cosine_similarity_average(results[algo1].predicted_points, results[algo2].predicted_points)
                    pairwise_sims[algo1][algo2] = sim
                pbar.update(1)

    return pairwise_sims


def _calculate_pairwise_amplitude_differences(results: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate amplitude differences between all algorithm pairs."""
    algorithms = list(results.keys())
    pairwise_amps = {}

    with tqdm(total=len(algorithms) ** 2, desc="Computing amplitude differences", leave=False) as pbar:
        for algo1 in algorithms:
            pairwise_amps[algo1] = {}
            for algo2 in algorithms:
                if algo1 == algo2:
                    pairwise_amps[algo1][algo2] = {"mean": 0.0, "std": 0.0, "max": 0.0}
                else:
                    amp_stats = amplitude_agreement(results[algo1].predicted_points, results[algo2].predicted_points)
                    pairwise_amps[algo1][algo2] = amp_stats
                pbar.update(1)

    return pairwise_amps


def point_wise_angular_difference(
    predictions1: List[Position3D], predictions2: List[Position3D], eye_positions: List[Position3D]
) -> float:
    """Point-wise mean angular error between two algorithms."""
    if len(predictions1) != len(predictions2) or len(predictions1) != len(eye_positions):
        return np.nan

    differences = []
    for p1, p2, eye_pos in zip(predictions1, predictions2, eye_positions):
        if p1 is None or p2 is None:
            continue

        point1 = Point3D(p1.x, p1.y, p1.z)
        point2 = Point3D(p2.x, p2.y, p2.z)
        diff = calculate_angular_error_degrees(point1, point2, eye_pos)
        differences.append(diff)

    return np.mean(differences) if differences else np.nan


def cosine_similarity_average(predictions1: List[Position3D], predictions2: List[Position3D]) -> float:
    """Average cosine similarity between prediction vectors."""
    if len(predictions1) != len(predictions2):
        return np.nan

    similarities = []
    for p1, p2 in zip(predictions1, predictions2):
        if p1 is None or p2 is None:
            continue

        vec1 = np.array([p1.x, p1.y, p1.z])
        vec2 = np.array([p2.x, p2.y, p2.z])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            continue

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        similarities.append(similarity)

    return np.mean(similarities) if similarities else np.nan


def amplitude_agreement(predictions1: List[Position3D], predictions2: List[Position3D]) -> Dict[str, float]:
    """Compare prediction magnitudes."""
    if len(predictions1) != len(predictions2):
        return {"mean": np.nan, "std": np.nan, "max": np.nan}

    amp_differences = []
    for p1, p2 in zip(predictions1, predictions2):
        if p1 is None or p2 is None:
            continue

        amp1 = np.sqrt(p1.x**2 + p1.y**2 + p1.z**2)
        amp2 = np.sqrt(p2.x**2 + p2.y**2 + p2.z**2)
        amp_differences.append(abs(amp1 - amp2))

    if not amp_differences:
        return {"mean": np.nan, "std": np.nan, "max": np.nan}

    amp_array = np.array(amp_differences)
    return {"mean": np.mean(amp_array), "std": np.std(amp_array), "max": np.max(amp_array)}
