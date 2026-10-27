import sys
import os
import io
import json
import math
import random
from typing import Optional

# Tambahkan path folder api/ agar modul optimasi.py terbaca di Vercel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import optimasi

app = FastAPI(title="RouteOptima TSP API")

# Izinkan React (frontend) untuk mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_random_symmetric_matrix(n: int) -> list[list[float]]:
    """Membuat matriks jarak simetris berbasis koordinat Euclidean 2D untuk N kota."""
    coords = [(random.uniform(5, 95), random.uniform(5, 95)) for _ in range(n)]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))
            dist = max(5, dist)
            matrix[i][j] = float(dist)
            matrix[j][i] = float(dist)
    return matrix


@app.post("/api/optimize")
async def run_optimization(
    file: Optional[UploadFile] = File(None),
    data_mode: str = Form("upload"),
    num_cities: int = Form(30),
    random_matrix: Optional[str] = Form(None),
    init_method: str = Form("NN"),
    ls_method: str = Form("2OPT"),
    sa_initial_temp: float = Form(10000.0),
    sa_alpha: float = Form(0.99),
    sa_max_iter: int = Form(15000),
    ts_tabu_tenure: int = Form(5),
    ts_max_iter: int = Form(100),
    ils_max_iter: int = Form(50),
    start_node: int = Form(0),
):
    try:
        # 1. Tentukan Matriks Jarak (Dari Mode Random atau Upload Excel)
        if data_mode == "random" or file is None:
            if random_matrix:
                distance_matrix = json.loads(random_matrix)
            else:
                n_cities = max(3, min(int(num_cities), 60))
                distance_matrix = generate_random_symmetric_matrix(n_cities)
        else:
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))

            # Jika kolom pertama adalah nama/indeks kota, abaikan
            df_numeric = df.select_dtypes(include=["number"])
            if df_numeric.shape[1] == df_numeric.shape[0] + 1:
                df_numeric = df_numeric.iloc[:, 1:]

            distance_matrix = df_numeric.values.tolist()

        total_cities = len(distance_matrix)
        if start_node < 0 or start_node >= total_cities:
            start_node = 0

        # Penyesuaian iterasi ILS agar tetap responsif untuk jumlah kota besar (>= 25 kota)
        effective_ils_iter = min(ils_max_iter, 20) if total_cities >= 25 else ils_max_iter

        # 2. Eksekusi Simulated Annealing
        sa_result = optimasi.simulated_annealing(
            distance_matrix=distance_matrix,
            init_method=init_method,
            ls_method=ls_method,
            initial_temp=sa_initial_temp,
            alpha=sa_alpha,
            max_iter=sa_max_iter,
            start_node=start_node,
        )

        # 3. Eksekusi Tabu Search
        ts_result = optimasi.tabu_search(
            distance_matrix=distance_matrix,
            init_method=init_method,
            ls_method=ls_method,
            tabu_tenure=ts_tabu_tenure,
            max_iter=ts_max_iter,
            start_node=start_node,
        )

        # 4. Eksekusi Iterated Local Search (Bonus)
        ils_result = optimasi.iterated_local_search(
            distance_matrix=distance_matrix,
            init_method=init_method,
            ls_method=ls_method,
            max_iter=effective_ils_iter,
            start_node=start_node,
        )

        # Format rute agar kembali ke kota awal (menjadi sebuah siklus/loop penuh)
        sa_tour_full = sa_result["best_tour"] + [sa_result["best_tour"][0]]
        ts_tour_full = ts_result["best_tour"] + [ts_result["best_tour"][0]]
        ils_tour_full = ils_result["best_tour"] + [ils_result["best_tour"][0]]

        return {
            "status": "success",
            "data": {
                "num_cities": total_cities,
                "sa": {
                    "best_distance": sa_result["best_distance"],
                    "best_tour": sa_tour_full,
                    "history": sa_result["history"],
                    "time_seconds": sa_result["time_seconds"],
                },
                "ts": {
                    "best_distance": ts_result["best_distance"],
                    "best_tour": ts_tour_full,
                    "history": ts_result["history"],
                    "time_seconds": ts_result["time_seconds"],
                },
                "ils": {
                    "best_distance": ils_result["best_distance"],
                    "best_tour": ils_tour_full,
                    "history": ils_result["history"],
                    "time_seconds": ils_result["time_seconds"],
                },
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))