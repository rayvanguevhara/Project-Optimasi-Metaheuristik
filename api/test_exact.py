import json
import math
from pathlib import Path

import pytest

from benchmark_exact import run_benchmark, solve_tsp_exact

DATASETS = Path(__file__).with_name("benchmark_datasets.json")
MATRICES = json.loads(DATASETS.read_text(encoding="utf-8"))["matrices"]


def test_held_karp_8_city_exact_known_optimum():
    result = solve_tsp_exact(MATRICES["8"], start_node=0)
    assert result["best_distance"] == 111.0
    assert result["best_tour"][0] == 0
    assert len(result["best_tour"]) == 8
    assert len(set(result["best_tour"])) == 8


def test_held_karp_15_city_tour_is_valid_and_distance_matches():
    matrix = MATRICES["15"]
    result = solve_tsp_exact(matrix, start_node=0)
    assert result["best_distance"] == 256.0
    tour = result["best_tour"]
    distance = sum(matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    assert len(tour) == 15
    assert len(set(tour)) == 15
    assert set(tour) == set(range(15))
    assert math.isclose(distance, result["best_distance"])


def test_held_karp_nonzero_start_node():
    matrix = [[0, 2, 9, 10], [2, 0, 6, 4], [9, 6, 0, 8], [10, 4, 8, 0]]
    result = solve_tsp_exact(matrix, start_node=2)
    assert result["best_tour"][0] == 2
    assert len(set(result["best_tour"])) == 4
    assert result["best_distance"] == 23.0


def test_held_karp_rejects_oversize_instance():
    with pytest.raises(ValueError, match="dibatasi"):
        solve_tsp_exact(MATRICES["15"], max_exact_cities=14)


def test_held_karp_rejects_negative_or_nonfinite_distance():
    bad = [[0, 1, 2], [1, 0, -1], [2, -1, 0]]
    with pytest.raises(ValueError, match="non-negatif"):
        solve_tsp_exact(bad)


def test_stage5_benchmark_all_tours_valid_and_exact_gaps_nonnegative():
    result = run_benchmark([8], multi_start_runs=1, adaptive_modes=True)
    assert result["dataset_summaries"][0]["exact_optimum"] == 111.0
    assert result["dataset_summaries"][0]["all_tours_valid"] is True
    assert result["dataset_summaries"][0]["all_tour_distances_match"] is True
    for row in result["results"]:
        assert row["valid_tour"] is True
        assert row["tour_distance_matches"] is True
        assert row["optimality_gap_percent"] >= -1e-9
