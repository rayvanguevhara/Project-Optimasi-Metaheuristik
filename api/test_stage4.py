"""Regression tests for RouteOptima Stage 4 adaptive search and stopping."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

import optimasi
import index as api_index

DATASETS = json.loads((API_DIR / "benchmark_datasets.json").read_text(encoding="utf-8"))["matrices"]


def assert_valid(result, matrix, start=0):
    tour = result["best_tour"]
    n = len(matrix)
    assert len(tour) == n
    assert len(set(tour)) == n
    assert set(tour) == set(range(n))
    assert tour[0] == start
    assert abs(optimasi.calculate_total_distance(tour, matrix) - result["best_distance"]) < 1e-7
    assert all(result["history"][i] <= result["history"][i - 1] for i in range(1, len(result["history"])))


def run_method(method_name, matrix, **overrides):
    common = dict(distance_matrix=matrix, init_method="NN", ls_method="2OPT", start_node=0, random_seed=42)
    if method_name == "simulated_annealing":
        common.update(initial_temp=200.0, alpha=0.95, max_iter=30)
    elif method_name == "tabu_search":
        common.update(tabu_tenure=5, max_iter=30)
    else:
        common.update(max_iter=30)
    common.update(overrides)
    return getattr(optimasi, method_name)(**common)


@pytest.mark.parametrize("method", ["simulated_annealing", "tabu_search", "iterated_local_search"])
def test_adaptive_search_preserves_valid_tour_and_reports_diagnostics(method):
    matrix = DATASETS["15"]
    result = run_method(method, matrix, adaptive_search=True)
    assert_valid(result, matrix)
    assert result["iterations_completed"] == 30
    assert result["stop_reason"] == "max_iter"
    assert result["diversification_events"] >= 0


@pytest.mark.parametrize("method", ["simulated_annealing", "tabu_search", "iterated_local_search"])
def test_early_stop_triggers_after_patience_on_tied_matrix(method):
    n = 8
    matrix = [[0 if i == j else 1 for j in range(n)] for i in range(n)]
    result = run_method(method, matrix, max_iter=30, early_stop_patience=3, adaptive_search=True)
    assert_valid(result, matrix)
    assert result["iterations_completed"] == 3
    assert len(result["history"]) == 4
    assert result["stop_reason"] == "stagnation_patience"


@pytest.mark.parametrize("method", ["simulated_annealing", "tabu_search", "iterated_local_search"])
def test_disabling_adaptive_search_disables_diversification(method):
    matrix = [[0 if i == j else 1 for j in range(8)] for i in range(8)]
    result = run_method(method, matrix, max_iter=30, adaptive_search=False)
    assert result["diversification_events"] == 0


def test_adaptive_results_are_reproducible_for_seeded_algorithms():
    matrix = DATASETS["30"]
    first = run_method("simulated_annealing", matrix, max_iter=80, adaptive_search=True)
    second = run_method("simulated_annealing", matrix, max_iter=80, adaptive_search=True)
    assert first["best_tour"] == second["best_tour"]
    assert first["best_distance"] == second["best_distance"]
    assert first["diversification_events"] == second["diversification_events"]


def test_multistart_includes_diagnostics_and_early_stop():
    matrix = [[0 if i == j else 1 for j in range(8)] for i in range(8)]
    result = optimasi.multi_start_simulated_annealing(
        matrix, "NN", "2OPT", initial_temp=200, alpha=0.95, max_iter=30,
        start_node=0, random_seed=42, multi_start_runs=2,
        adaptive_search=True, early_stop_patience=3,
    )
    assert_valid(result, matrix)
    assert result["num_starts"] == 2
    assert result["total_iterations_completed"] == 6
    assert len(result["runs"]) == 2
    assert all(run["stop_reason"] == "stagnation_patience" for run in result["runs"])


def test_api_accepts_stage4_controls():
    matrix = DATASETS["8"]

    async def call_api():
        return await api_index.run_optimization(
            file=None, data_mode="random", num_cities=8, random_matrix=json.dumps(matrix),
            init_method="NN", ls_method="2OPT", sa_initial_temp=200.0,
            sa_alpha=0.95, sa_max_iter=30, ts_tabu_tenure=5, ts_max_iter=30,
            ils_max_iter=30, start_node=0, multi_start_runs=2, multi_start_seed=42,
            adaptive_search=True, early_stop_patience=3,
        )

    response = asyncio.run(call_api())
    assert response["status"] == "success"
    assert response["data"]["adaptive_search"] is True
    assert response["data"]["early_stop_patience"] == 3
    for key in ("sa", "ts", "ils"):
        assert response["data"][key]["total_iterations_completed"] > 0
        assert len(response["data"][key]["runs"]) == 2


def test_negative_patience_is_rejected():
    with pytest.raises(ValueError, match="early_stop_patience"):
        optimasi.multi_start_tabu_search(DATASETS["8"], early_stop_patience=-1)
