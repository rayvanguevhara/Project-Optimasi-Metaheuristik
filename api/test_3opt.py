"""Fast integration tests for the RouteOptima 3-Opt addition."""
import asyncio
import json
import random
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


def test_three_opt_neighborhood_is_valid_and_anchored():
    matrix = DATASETS["8"]
    tour = optimasi.generate_initial_tour(matrix, 0, "NN")
    neighbors, moves = optimasi.generate_all_neighbors(tour, "3OPT")
    assert neighbors
    assert len(neighbors) == len(moves)
    for neighbor in neighbors:
        assert_valid_tour(neighbor, len(matrix), 0)
    assert len({optimasi._canonical_cycle_key(t) for t in neighbors}) == len(neighbors)


def test_three_opt_local_search_is_not_worse_than_two_opt_descent():
    for key in ("8", "30", "60"):
        matrix = DATASETS[key]
        initial = optimasi.generate_initial_tour(matrix, 0, "NN")
        tour_2opt, distance_2opt = optimasi.run_local_search(initial, matrix, "2OPT")
        tour_3opt, distance_3opt = optimasi.run_local_search(initial, matrix, "3OPT")
        assert_valid_tour(tour_2opt, len(matrix), 0)
        assert_valid_tour(tour_3opt, len(matrix), 0)
        assert distance_3opt <= distance_2opt + 1e-9
        assert abs(optimasi.calculate_total_distance(tour_3opt, matrix) - distance_3opt) < 1e-9


def test_three_opt_algorithms_return_valid_routes():
    matrix = DATASETS["8"]
    for seed in (42, 43):
        sa = optimasi.simulated_annealing(
            matrix, "NN", "3OPT", initial_temp=200.0,
            alpha=0.95, max_iter=40, start_node=0, random_seed=seed
        )
        assert_valid_tour(sa["best_tour"], len(matrix), 0)
        assert abs(optimasi.calculate_total_distance(sa["best_tour"], matrix) - sa["best_distance"]) < 1e-9

    ts = optimasi.tabu_search(matrix, "NN", "3OPT", tabu_tenure=5, max_iter=20, start_node=0)
    assert_valid_tour(ts["best_tour"], len(matrix), 0)
    ts_again = optimasi.tabu_search(matrix, "NN", "3OPT", tabu_tenure=5, max_iter=20, start_node=0)
    assert ts["best_tour"] == ts_again["best_tour"]
    assert ts["best_distance"] == ts_again["best_distance"]

    random.seed(42)
    ils = optimasi.iterated_local_search(matrix, "NN", "3OPT", max_iter=5, start_node=0)
    assert_valid_tour(ils["best_tour"], len(matrix), 0)
    assert abs(optimasi.calculate_total_distance(ils["best_tour"], matrix) - ils["best_distance"]) < 1e-9


def test_three_opt_rejects_asymmetric_distance_matrix():
    asymmetric = [
        [0, 2, 3],
        [4, 0, 5],
        [6, 7, 0],
    ]
    with pytest.raises(ValueError, match="matriks jarak simetris"):
        optimasi.simulated_annealing(asymmetric, "NN", "3OPT", max_iter=10)


def test_api_accepts_3opt_for_random_matrix():
    matrix = DATASETS["8"]

    async def call_api():
        return await api_index.run_optimization(
            file=None,
            data_mode="random",
            num_cities=8,
            random_matrix=json.dumps(matrix),
            init_method="NN",
            ls_method="3OPT",
            sa_initial_temp=200.0,
            sa_alpha=0.95,
            sa_max_iter=30,
            ts_tabu_tenure=5,
            ts_max_iter=20,
            ils_max_iter=3,
            start_node=0,
            multi_start_runs=1,
            multi_start_seed=42,
        )

    response = asyncio.run(call_api())
    assert response["status"] == "success"
    assert response["data"]["num_cities"] == 8
    for method in ("sa", "ts", "ils"):
        result = response["data"][method]
        assert_valid_tour(result["best_tour"][:-1], 8, 0)
        assert result["best_tour"][0] == result["best_tour"][-1] == 0
