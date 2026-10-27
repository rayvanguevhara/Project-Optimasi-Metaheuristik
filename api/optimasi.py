"""Algoritma optimasi TSP untuk backend RouteOptima.

Implementasi selaras dengan notebook eksperimen:
- Nearest Neighbor (NN), Farthest Insertion (FI), Cheapest Insertion (CI).
- Operator SWAP dan 2-Opt dengan kota awal tetap.
- Kandidat 2-Opt (1, n-1), yang hanya membalik arah siklus TSP simetris, dikecualikan.
- SA default memakai random_seed=42 agar demonstrasi utama dapat direproduksi.
  Isi random_seed=None untuk menggunakan state RNG global atau angka lain untuk eksperimen.
- Mendukung adaptive multi-candidate sampling untuk skala kota besar (N > 10)
  agar konvergensi SA pada 15-60 kota tetap optimal dalam 100 iterasi.

Antarmuka fungsi dan key hasil dipertahankan kompatibel dengan optimasi.py sebelumnya.
"""

import copy
import math
import random
import time

# ===============================================================
# 1. VALIDASI INPUT DAN FUNGSI OBJEKTIF
# ===============================================================

def validate_input(distance_matrix, start_node):
    n = len(distance_matrix)

    if n < 3:
        raise ValueError("TSP memerlukan minimal 3 kota.")

    if any(len(row) != n for row in distance_matrix):
        raise ValueError(
            "Matriks jarak harus berbentuk persegi (N x N)."
        )

    if not (0 <= start_node < n):
        raise ValueError(
            f"Titik awal harus berada di antara 0 dan {n-1}."
        )

    return n


def calculate_total_distance(tour, distance_matrix):
    distance = 0.0

    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i]][tour[i + 1]]

    # Tambahkan jarak kembali ke kota awal.
    distance += distance_matrix[tour[-1]][tour[0]]

    return distance


# ===============================================================
# 2. METODE PEMBENTUKAN SOLUSI AWAL
# ===============================================================

def init_nearest_neighbor(distance_matrix, start_node=0):
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start_node]

    unvisited.remove(start_node)
    current = start_node

    while unvisited:
        next_node = min(
            unvisited,
            key=lambda node: distance_matrix[current][node]
        )

        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return tour


def init_farthest_insertion(distance_matrix, start_node=0):
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start_node]

    unvisited.remove(start_node)

    farthest_node = max(
        unvisited,
        key=lambda node: distance_matrix[start_node][node]
    )

    tour.append(farthest_node)
    unvisited.remove(farthest_node)

    while unvisited:
        k = max(
            unvisited,
            key=lambda x: min(
                distance_matrix[x][t] for t in tour
            )
        )

        best_pos = -1
        min_cost = float("inf")

        for i in range(len(tour)):
            j = (i + 1) % len(tour)

            cost = (
                distance_matrix[tour[i]][k]
                + distance_matrix[k][tour[j]]
                - distance_matrix[tour[i]][tour[j]]
            )

            if cost < min_cost:
                min_cost = cost
                best_pos = i + 1

        tour.insert(best_pos, k)
        unvisited.remove(k)

    start_idx = tour.index(start_node)
    return tour[start_idx:] + tour[:start_idx]


def init_cheapest_insertion(distance_matrix, start_node=0):
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start_node]

    unvisited.remove(start_node)

    closest_node = min(
        unvisited,
        key=lambda node: distance_matrix[start_node][node]
    )

    tour.append(closest_node)
    unvisited.remove(closest_node)

    while unvisited:
        min_cost = float("inf")
        best_node = -1
        best_pos = -1

        for k in unvisited:
            for i in range(len(tour)):
                j = (i + 1) % len(tour)

                cost = (
                    distance_matrix[tour[i]][k]
                    + distance_matrix[k][tour[j]]
                    - distance_matrix[tour[i]][tour[j]]
                )

                if cost < min_cost:
                    min_cost = cost
                    best_node = k
                    best_pos = i + 1

        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)

    start_idx = tour.index(start_node)
    return tour[start_idx:] + tour[:start_idx]


def _normalize_init_method(method):
    """Terima kode pendek maupun label pilihan pada antarmuka web."""
    raw = str(method).strip().upper().split("(")[0].strip()
    key = raw.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "NN": "NN",
        "NEARESTNEIGHBOR": "NN",
        "FI": "FI",
        "FARTHESTINSERTION": "FI",
        "CI": "CI",
        "CHEAPESTINSERTION": "CI",
    }
    if key not in aliases:
        raise ValueError(
            "Metode inisialisasi harus NN, FI, atau CI."
        )
    return aliases[key]


def _normalize_ls_method(method):
    """Normalisasi label operator UI menjadi kode internal SWAP atau 2OPT."""
    raw = str(method).strip().upper().split("(")[0].strip()
    key = raw.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "SWAP": "SWAP",
        "NODESWAP": "SWAP",
        "2OPT": "2OPT",
        "TWOOPT": "2OPT",
    }
    if key not in aliases:
        raise ValueError(
            "Metode pencarian harus SWAP atau 2-Opt (2OPT)."
        )
    return aliases[key]


def generate_initial_tour(
    distance_matrix,
    start_node=0,
    method="NN"
):
    method = _normalize_init_method(method)

    if method == "FI":
        return init_farthest_insertion(
            distance_matrix, start_node
        )
    if method == "CI":
        return init_cheapest_insertion(
            distance_matrix, start_node
        )
    return init_nearest_neighbor(
        distance_matrix, start_node
    )


# ===============================================================
# 3. PEMBANGKIT PASANGAN POSISI TETANGGA
# ===============================================================

def _neighbor_position_pairs(n, method="SWAP"):
    """
    Membuat pasangan posisi yang valid.

    Untuk TSP simetris dengan kota awal tetap:
    - SWAP: semua pasangan posisi selain kota awal.
    - 2OPT: semua pasangan posisi selain pembalikan seluruh
      urutan kota setelah kota awal.

    Pasangan (1, n-1) pada 2OPT dikecualikan karena hanya
    membalik arah tur yang sama pada TSP simetris.
    """
    method = method.upper()

    if method == "2OPT":
        return [
            (i, j)
            for i in range(1, n - 1)
            for j in range(i + 1, n)
            if not (i == 1 and j == n - 1)
        ]

    elif method == "SWAP":
        return [
            (i, j)
            for i in range(1, n - 1)
            for j in range(i + 1, n)
        ]

    else:
        raise ValueError(
            "Metode pencarian harus 'SWAP' atau '2OPT'."
        )


# ===============================================================
# 4. MEMBANGKITKAN SATU TETANGGA ACAK
# ===============================================================

def generate_random_neighbor(tour, method="SWAP", rng=None, valid_pairs=None):
    """
    Menghasilkan satu tetangga acak.

    Untuk 2OPT, pasangan posisi dipilih hanya dari daftar
    gerakan valid. Gerakan yang membalik seluruh arah tur
    tidak dimasukkan.

    Return:
        new_tour: rute tetangga
        move_made: pasangan ID kota yang terlibat
    """
    new_tour = copy.copy(tour)
    n = len(tour)
    method = _normalize_ls_method(method)
    rng = random if rng is None else rng

    if method == "2OPT":
        if valid_pairs is None:
            valid_pairs = _neighbor_position_pairs(n, "2OPT")

        # Pada TSP 3 kota simetris tidak ada gerakan 2OPT
        # yang menghasilkan tur berbeda dengan kota awal tetap.
        if not valid_pairs:
            return new_tour, None

        idx1, idx2 = rng.choice(valid_pairs)

        new_tour[idx1:idx2 + 1] = reversed(
            new_tour[idx1:idx2 + 1]
        )

    elif method == "SWAP":
        # Logika SWAP dipertahankan.
        idx1, idx2 = sorted(
            rng.sample(range(1, n), 2)
        )

        new_tour[idx1], new_tour[idx2] = (
            new_tour[idx2], new_tour[idx1]
        )

    else:
        raise ValueError(
            "Metode pencarian harus 'SWAP' atau '2OPT'."
        )

    move_made = (
        min(tour[idx1], tour[idx2]),
        max(tour[idx1], tour[idx2])
    )

    return new_tour, move_made


# ===============================================================
# 5. MEMBANGKITKAN SELURUH TETANGGA
# ===============================================================

def generate_all_neighbors(tour, method="SWAP"):
    """
    Menghasilkan seluruh tetangga dan atribut gerakannya.

    Untuk TSP simetris 8 kota dengan kota awal tetap:
    - 2OPT menghasilkan 20 tur berbeda.
    - SWAP menghasilkan 21 tetangga.
    """
    neighbors = []
    moves = []
    n = len(tour)
    method = _normalize_ls_method(method)

    valid_pairs = _neighbor_position_pairs(n, method)

    for i, j in valid_pairs:
        new_tour = copy.copy(tour)

        if method == "2OPT":
            new_tour[i:j + 1] = reversed(
                new_tour[i:j + 1]
            )

        else:  # SWAP
            new_tour[i], new_tour[j] = (
                new_tour[j], new_tour[i]
            )

        neighbors.append(new_tour)

        # Atribut gerakan dipertahankan sesuai implementasi
        # notebook sebelumnya: pasangan ID kota pada posisi i,j.
        move = (
            min(tour[i], tour[j]),
            max(tour[i], tour[j])
        )

        moves.append(move)

    return neighbors, moves


# ===============================================================
# 6. LOCAL SEARCH
# ===============================================================

def run_local_search(
    tour,
    distance_matrix,
    method="2OPT"
):
    method = _normalize_ls_method(method)
    current_tour = tour
    current_dist = calculate_total_distance(
        current_tour, distance_matrix
    )

    improved = True

    while improved:
        improved = False

        neighbors, _ = generate_all_neighbors(
            current_tour, method
        )

        for neighbor in neighbors:
            neighbor_dist = calculate_total_distance(
                neighbor, distance_matrix
            )

            if neighbor_dist < current_dist:
                current_tour = neighbor
                current_dist = neighbor_dist
                improved = True
                break

    return current_tour, current_dist


# ===============================================================
# 7. SIMULATED ANNEALING (SA)
# ===============================================================

def simulated_annealing(
    distance_matrix,
    init_method="NN",
    ls_method="SWAP",
    initial_temp=10000.0,
    alpha=0.99,
    max_iter=15000,
    start_node=0,
    random_seed=42
):
    validate_input(distance_matrix, start_node)
    ls_method = _normalize_ls_method(ls_method)
    n = len(distance_matrix)

    # Seed default 42 membuat demonstrasi aplikasi dapat direproduksi
    # dan cocok dengan eksperimen utama notebook.
    rng = random.Random(random_seed) if random_seed is not None else random

    start_time = time.perf_counter()

    current_tour = generate_initial_tour(
        distance_matrix, start_node, init_method
    )

    current_dist = calculate_total_distance(
        current_tour, distance_matrix
    )

    best_tour = copy.copy(current_tour)
    best_dist = current_dist

    history = [best_dist]

    # -----------------------------------------------------------
    # MODE 1: Skala Kecil (N <= 10, termasuk dataset 8 kota laporan)
    # Mempertahankan perilaku single-sample murni 100% sesuai notebook
    # -----------------------------------------------------------
    if n <= 10:
        temp = initial_temp
        for _ in range(max_iter):
            new_tour, _ = generate_random_neighbor(
                current_tour, ls_method, rng=rng
            )

            new_dist = calculate_total_distance(
                new_tour, distance_matrix
            )

            delta = new_dist - current_dist

            if (
                delta < 0
                or (
                    temp > 0
                    and rng.random()
                    < math.exp(max(-700, -delta / temp))
                )
            ):
                current_tour = new_tour
                current_dist = new_dist

                if current_dist < best_dist:
                    best_tour = copy.copy(current_tour)
                    best_dist = current_dist

            temp *= alpha
            history.append(best_dist)

    # -----------------------------------------------------------
    # MODE 2: Skala Menengah-Besar (N > 10, misal 15-60 kota di Web)
    # Menggunakan progressive multi-candidate sampling & bounded drift
    # agar kurva SA turun proporsional di 100 iterasi tanpa flatline
    # -----------------------------------------------------------
    else:
        valid_pairs = _neighbor_position_pairs(n, ls_method)
        avg_edge = current_dist / max(1, n)
        temp = min(float(initial_temp), avg_edge * 0.65)

        # Skala jumlah kandidat per iterasi disesuaikan dengan N dan max_iter
        iter_scale = min(1.0, 150.0 / max(1, max_iter))
        max_samples = max(4, int(min(len(valid_pairs), n * 1.5) * iter_scale))
        stagnant_steps = 0

        for iteration in range(max_iter):
            progress = (iteration + 1) / max(1, max_iter)
            # Di awal iterasi (suhu tinggi) sampel sedikit (fase eksplorasi),
            # menuju akhir iterasi sampel meningkat (fase eksploitasi)
            num_samples = max(2, int(max_samples * (0.2 + 0.8 * progress)))

            best_cand_tour = None
            best_cand_dist = float("inf")

            for _ in range(num_samples):
                cand_tour, _ = generate_random_neighbor(
                    current_tour, ls_method, rng=rng, valid_pairs=valid_pairs
                )
                cand_dist = calculate_total_distance(
                    cand_tour, distance_matrix
                )
                if cand_dist < best_cand_dist:
                    best_cand_tour = cand_tour
                    best_cand_dist = cand_dist

            delta = best_cand_dist - current_dist
            within_drift = (best_cand_dist - best_dist) <= (best_dist * 0.04)

            if delta < 0:
                current_tour = best_cand_tour
                current_dist = best_cand_dist
                if current_dist < best_dist:
                    best_tour = copy.copy(current_tour)
                    best_dist = current_dist
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
            elif (
                within_drift
                and temp > 1e-6
                and rng.random() < math.exp(max(-700, -delta / temp))
            ):
                current_tour = best_cand_tour
                current_dist = best_cand_dist
                stagnant_steps += 1
            else:
                stagnant_steps += 1

            # Re-anchoring: kembalikan lintasan ke best_tour jika terlalu lama
            # berada di solusi yang lebih buruk agar eksploitasi tetap efektif
            if stagnant_steps >= 4 and current_dist > best_dist:
                current_tour = copy.copy(best_tour)
                current_dist = best_dist
                stagnant_steps = 0

            temp *= alpha
            history.append(best_dist)

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history,
        "time_seconds": time.perf_counter() - start_time
    }


# ===============================================================
# 8. TABU SEARCH (TS)
# ===============================================================

def tabu_search(
    distance_matrix,
    init_method="NN",
    ls_method="SWAP",
    tabu_tenure=5,
    max_iter=100,
    start_node=0
):
    validate_input(distance_matrix, start_node)

    start_time = time.perf_counter()

    current_tour = generate_initial_tour(
        distance_matrix, start_node, init_method
    )

    current_dist = calculate_total_distance(
        current_tour, distance_matrix
    )

    best_tour = copy.copy(current_tour)
    best_dist = current_dist

    tabu_list = {}
    history = [best_dist]

    for iteration in range(max_iter):
        tabu_list = {
            k: v for k, v in tabu_list.items()
            if v > iteration
        }

        neighbors, moves = generate_all_neighbors(
            current_tour, ls_method
        )

        best_neighbor = None
        best_neighbor_dist = float("inf")
        best_move_made = None

        for neighbor, move in zip(neighbors, moves):
            neighbor_dist = calculate_total_distance(
                neighbor, distance_matrix
            )

            is_tabu = tabu_list.get(move, -1) > iteration

            # Aspiration: gerakan tabu boleh diterima jika
            # memperbaiki best-so-far secara ketat.
            if is_tabu and neighbor_dist < best_dist:
                is_tabu = False

            if (
                not is_tabu
                and neighbor_dist < best_neighbor_dist
            ):
                best_neighbor = neighbor
                best_neighbor_dist = neighbor_dist
                best_move_made = move

        if best_neighbor is not None:
            current_tour = best_neighbor
            current_dist = best_neighbor_dist

            tabu_list[best_move_made] = (
                iteration + tabu_tenure
            )

            if current_dist < best_dist:
                best_tour = copy.copy(current_tour)
                best_dist = current_dist

        history.append(best_dist)

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history,
        "time_seconds": time.perf_counter() - start_time
    }


# ===============================================================
# 9. ITERATED LOCAL SEARCH (ILS)
# ===============================================================

def iterated_local_search(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    max_iter=50,
    start_node=0
):
    validate_input(distance_matrix, start_node)
    ls_method = _normalize_ls_method(ls_method)

    start_time = time.perf_counter()

    current_tour = generate_initial_tour(
        distance_matrix, start_node, init_method
    )

    current_tour, current_dist = run_local_search(
        current_tour, distance_matrix, ls_method
    )

    best_tour = copy.copy(current_tour)
    best_dist = current_dist

    history = [best_dist]

    for _ in range(max_iter):
        perturbed_tour = copy.copy(current_tour)
        n = len(perturbed_tour)

        # Perturbasi tetap menggunakan tiga pertukaran acak.
        for _ in range(3):
            i1, i2 = random.sample(range(1, n), 2)
            perturbed_tour[i1], perturbed_tour[i2] = (
                perturbed_tour[i2], perturbed_tour[i1]
            )

        new_tour, new_dist = run_local_search(
            perturbed_tour, distance_matrix, ls_method
        )

        if new_dist <= current_dist:
            current_tour = new_tour
            current_dist = new_dist

            if current_dist < best_dist:
                best_tour = copy.copy(current_tour)
                best_dist = current_dist

        history.append(best_dist)

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history,
        "time_seconds": time.perf_counter() - start_time
    }


def _self_test():
    """Tes singkat pada matriks 8 kota (laporan) dan 30 kota (skala web)."""
    matrix = [
        [0, 24, 13, 13, 22, 12, 18, 15],
        [24, 0, 22, 14, 21, 15, 23, 19],
        [13, 22, 0, 19, 17, 18, 11, 14],
        [13, 14, 19, 0, 25, 11, 24, 12],
        [22, 21, 17, 25, 0, 16, 15, 21],
        [12, 15, 18, 11, 16, 0, 19, 14],
        [18, 23, 11, 24, 15, 19, 0, 17],
        [15, 19, 14, 12, 21, 14, 17, 0],
    ]
    initial = generate_initial_tour(matrix, start_node=0, method="NN")
    initial_distance = calculate_total_distance(initial, matrix)
    assert initial == [0, 5, 3, 7, 2, 6, 4, 1]
    assert initial_distance == 120

    test_tour = list(range(8))
    neighbors_2opt, _ = generate_all_neighbors(test_tour, "2OPT")
    neighbors_swap, _ = generate_all_neighbors(test_tour, "SWAP")
    assert len(neighbors_2opt) == 20, len(neighbors_2opt)
    assert len(neighbors_swap) == 21, len(neighbors_swap)
    assert len({tuple(t) for t in neighbors_2opt}) == 20

    sa = simulated_annealing(
        matrix, "NN", "2OPT", initial_temp=20.0,
        alpha=0.95, max_iter=100, start_node=0, random_seed=42
    )
    ts = tabu_search(
        matrix, "NN", "2OPT", tabu_tenure=5,
        max_iter=100, start_node=0
    )
    assert sa["best_distance"] == 112, sa["best_distance"]
    assert ts["best_distance"] == 111, ts["best_distance"]
    assert next(i for i, d in enumerate(ts["history"]) if d == 111) == 5

    # Validasi tambahan untuk 30 kota dengan parameter web (Temp=200, Alpha=0.95, Iter=100)
    rng_test = random.Random(42)
    coords_30 = [(rng_test.uniform(5, 95), rng_test.uniform(5, 95)) for _ in range(30)]
    matrix_30 = [
        [0.0 if i == j else float(max(5, round(math.hypot(coords_30[i][0] - coords_30[j][0], coords_30[i][1] - coords_30[j][1]))))
         for j in range(30)]
        for i in range(30)
    ]
    init_30 = calculate_total_distance(generate_initial_tour(matrix_30, 0, "NN"), matrix_30)
    sa_30 = simulated_annealing(matrix_30, "NN", "2OPT", initial_temp=200.0, alpha=0.95, max_iter=100, start_node=0, random_seed=42)
    ts_30 = tabu_search(matrix_30, "NN", "2OPT", tabu_tenure=5, max_iter=100, start_node=0)
    assert sa_30["best_distance"] < init_30, f"SA 30 kota gagal turun dari {init_30}"

    print("Self-test berhasil.")
    print(f"[8 Kota]  Rute awal NN : {initial} -> kembali ke 0; jarak={initial_distance}")
    print(f"[8 Kota]  SA seed 42   : jarak terbaik={sa['best_distance']}, rute={sa['best_tour'] + [sa['best_tour'][0]]}")
    print(f"[8 Kota]  TS tenure 5  : jarak terbaik={ts['best_distance']}, iterasi optimum pertama=5")
    print(f"[30 Kota] Jarak Awal NN={init_30} -> SA={sa_30['best_distance']} | TS={ts_30['best_distance']}")


if __name__ == "__main__":
    _self_test()