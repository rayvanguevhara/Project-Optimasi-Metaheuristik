"""Tests for RouteOptima Stage 3 multi-start optimization."""
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


def assert_valid_tour(tour, n, start=0):
    assert len(tour) == n
    assert len(set(tour)) == n
    assert set(tour) == set(range(n))
    assert tour[0] == start


@pytest.mark.parametrize("method", ["multi_start_simulated_annealing", "multi_start_tabu_search", "multi_start_iterated_local_search"])
def test_multistart_returns_valid_best_tour_and_metadata(method):
    matrix = DATASETS["8"]
    fn = getattr(optimasi, method)
    kwargs = {
        "distance_matrix": matrix,
        "init_method": "NN",
        "ls_method": "2OPT",
        "start_node": 0,
        "random_seed": 42,
        "multi_start_runs": 3,
    }
    if method == "multi_start_simulated_annealing":
        kwargs.update(initial_temp=200.0, alpha=0.95, max_iter=30)
    elif method == "multi_start_tabu_search":
        kwargs.update(tabu_tenure=5, max_iter=20)
    else:
        kwargs.update(max_iter=5)

    result = fn(**kwargs)
    assert result["num_starts"] == 3
    assert 1 <= result["best_run"] <= 3
    assert len(result["runs"]) == 3
    assert result["time_seconds"] >= 0
    assert_valid_tour(result["best_tour"], len(matrix), 0)
    assert abs(optimasi.calculate_total_distance(result["best_tour"], matrix) - result["best_distance"]) < 1e-9
    assert all(result["history"][i] <= result["history"][i - 1] for i in range(1, len(result["history"])))


def test_multistart_cannot_be_worse_than_same_first_start():
    matrix = DATASETS["15"]
    cases = [
        (optimasi.multi_start_simulated_annealing, {"initial_temp": 200.0, "alpha": 0.95, "max_iter": 100}),
        (optimasi.multi_start_tabu_search, {"tabu_tenure": 5, "max_iter": 100}),
        (optimasi.multi_start_iterated_local_search, {"max_iter": 10}),
    ]
    for fn, extras in cases:
        one = fn(matrix, "NN", "2OPT", start_node=0, random_seed=42, multi_start_runs=1, **extras)
        three = fn(matrix, "NN", "2OPT", start_node=0, random_seed=42, multi_start_runs=3, **extras)
        assert three["best_distance"] <= one["best_distance"] + 1e-9
        assert three["num_starts"] == 3


def test_multistart_results_are_reproducible():
    matrix = DATASETS["8"]
    first = optimasi.multi_start_tabu_search(
        matrix, "NN", "2OPT", tabu_tenure=5, max_iter=25,
        start_node=0, random_seed=42, multi_start_runs=3,
    )
    second = optimasi.multi_start_tabu_search(
        matrix, "NN", "2OPT", tabu_tenure=5, max_iter=25,
        start_node=0, random_seed=42, multi_start_runs=3,
    )
    assert first["best_tour"] == second["best_tour"]
    assert first["best_distance"] == second["best_distance"]
    assert [r["best_distance"] for r in first["runs"]] == [r["best_distance"] for r in second["runs"]]


def test_api_accepts_multistart_fields_and_returns_metadata():
    matrix = DATASETS["8"]

    async def call_api():
        return await api_index.run_optimization(
            file=None,
            data_mode="random",
            num_cities=8,
            random_matrix=json.dumps(matrix),
            init_method="NN",
            ls_method="2OPT",
            sa_initial_temp=200.0,
            sa_alpha=0.95,
            sa_max_iter=30,
            ts_tabu_tenure=5,
            ts_max_iter=20,
            ils_max_iter=5,
            start_node=0,
            multi_start_runs=2,
            multi_start_seed=42,
        )

    response = asyncio.run(call_api())
    assert response["status"] == "success"
    assert response["data"]["multi_start_runs"] == 2
    for key in ("sa", "ts", "ils"):
        result = response["data"][key]
        assert result["num_starts"] == 2
        assert len(result["runs"]) == 2
        assert result["best_tour"][0] == result["best_tour"][-1] == 0
        assert_valid_tour(result["best_tour"][:-1], 8, 0)


def test_multistart_run_count_is_validated():
    with pytest.raises(ValueError, match="antara 1 dan 5"):
        optimasi.multi_start_tabu_search(DATASETS["8"], multi_start_runs=6)


def test_multistart_works_with_3opt_on_small_dataset():
    matrix = DATASETS["8"]
    sa = optimasi.multi_start_simulated_annealing(
        matrix, "NN", "3OPT", initial_temp=200.0, alpha=0.95, max_iter=20,
        start_node=0, random_seed=42, multi_start_runs=2,
    )
    ts = optimasi.multi_start_tabu_search(
        matrix, "NN", "3OPT", tabu_tenure=5, max_iter=10,
        start_node=0, random_seed=42, multi_start_runs=2,
    )
    ils = optimasi.multi_start_iterated_local_search(
        matrix, "NN", "3OPT", max_iter=2,
        start_node=0, random_seed=42, multi_start_runs=2,
    )
    for result in (sa, ts, ils):
        assert result["num_starts"] == 2
        assert_valid_tour(result["best_tour"], len(matrix), 0)
        assert abs(optimasi.calculate_total_distance(result["best_tour"], matrix) - result["best_distance"]) < 1e-9
