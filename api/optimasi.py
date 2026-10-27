"""Implementasi optimasi TSP untuk backend RouteOptima (versi final).

Fokus utama tugas adalah Simulated Annealing (SA) dan Tabu Search (TS).
Iterated Local Search (ILS) tetap disediakan sebagai fitur tambahan demi
kompatibilitas dengan API RouteOptima yang sudah ada.

Implementasi selaras dengan notebook eksperimen:
- Nearest Neighbor (NN), Farthest Insertion (FI), Cheapest Insertion (CI).
- Operator SWAP, 2-Opt, dan 3-Opt dengan kota awal tetap.
- Kandidat 2-Opt (1, n-1), yang hanya membalik arah siklus TSP simetris, dikecualikan.
- 3-Opt menggunakan tujuh reconnect berbeda; pada dataset besar, pencarian lokal
  menggunakan sampel neighborhood berbatas agar tetap responsif.
- SA default memakai random_seed=42 agar demonstrasi utama dapat direproduksi.
  Isi random_seed=None untuk menggunakan state RNG global atau angka lain untuk eksperimen.
- Mendukung adaptive multi-candidate sampling untuk skala kota besar (N > 10)
  agar SA dapat mengevaluasi beberapa kandidat per iterasi secara terkontrol.

Fungsi algoritma satu-run tetap kompatibel; wrapper Multi-Start menggabungkan beberapa
percobaan deterministik dan mengembalikan rute terbaik beserta diagnostik tiap run.
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


def validate_symmetric_matrix(distance_matrix, tolerance=1e-9):
    """3-Opt versi ini memakai representasi sisi tak berarah, jadi perlu TSP simetris."""
    n = len(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(distance_matrix[i][j]) - float(distance_matrix[j][i])) > tolerance:
                raise ValueError(
                    "Metode 3-Opt pada versi ini memerlukan matriks jarak simetris. "
                    f"Nilai [{i}][{j}] dan [{j}][{i}] berbeda."
                )


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
    """Normalisasi label UI menjadi kode internal SWAP, 2OPT, atau 3OPT."""
    raw = str(method).strip().upper().split("(")[0].strip()
    key = raw.replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "SWAP": "SWAP",
        "NODESWAP": "SWAP",
        "2OPT": "2OPT",
        "TWOOPT": "2OPT",
        "3OPT": "3OPT",
        "THREEOPT": "3OPT",
    }
    if key not in aliases:
        raise ValueError(
            "Metode pencarian harus SWAP, 2-Opt (2OPT), atau 3-Opt (3OPT)."
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


def _validated_initial_tour(initial_tour, distance_matrix, start_node):
    """Validate a supplied tour and keep the configured start city anchored."""
    if initial_tour is None:
        return None
    n = len(distance_matrix)
    tour = list(initial_tour)
    if len(tour) != n or set(tour) != set(range(n)):
        raise ValueError("Rute awal harus memuat setiap kota tepat satu kali.")
    if not tour or tour[0] != start_node:
        raise ValueError("Rute awal harus dimulai dari start_node yang dipilih.")
    return tour


def _build_multistart_initial_tours(distance_matrix, start_node, init_method, runs, seed):
    """Keep the heuristic start, then create seeded perturbations of that route.

    Perturbing a good construction instead of starting from a completely random
    permutation makes restarts diverse without making local search prohibitively
    expensive on medium/large instances.
    """
    n = len(distance_matrix)
    base_tour = generate_initial_tour(distance_matrix, start_node, init_method)
    tours = [list(base_tour)]
    seen = {tuple(base_tour)}
    base_swap_count = max(2, min(n - 1, math.ceil(0.15 * (n - 1))))

    for run_index in range(1, runs):
        rng = random.Random(int(seed) + run_index * 104729)
        candidate = None
        for attempt in range(100):
            proposed = list(base_tour)
            swap_count = min(n - 1, base_swap_count + (attempt // 10))
            for _ in range(swap_count):
                i, j = rng.sample(range(1, n), 2)
                proposed[i], proposed[j] = proposed[j], proposed[i]
            if tuple(proposed) not in seen:
                candidate = proposed
                seen.add(tuple(proposed))
                break
        if candidate is None:
            # For tiny TSPs there may be fewer distinct tours than requested starts.
            tail = list(base_tour[1:])
            rng.shuffle(tail)
            candidate = [start_node] + tail
        tours.append(candidate)
    return tours


# ===============================================================
# 3. PEMBANGKIT TETANGGA: SWAP, 2-OPT, DAN 3-OPT
# ===============================================================

def _neighbor_position_pairs(n, method="SWAP"):
    """Membuat pasangan posisi untuk operator SWAP dan 2-Opt."""
    method = _normalize_ls_method(method)

    if method == "2OPT":
        return [
            (i, j)
            for i in range(1, n - 1)
            for j in range(i + 1, n)
            if not (i == 1 and j == n - 1)
        ]

    if method == "SWAP":
        return [
            (i, j)
            for i in range(1, n - 1)
            for j in range(i + 1, n)
        ]

    raise ValueError("Pasangan posisi hanya berlaku untuk SWAP atau 2OPT.")


def _valid_three_opt_edge_triples(n):
    """Tiga sisi potong non-adjacent dalam siklus TSP."""
    return [
        (i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n)
        if n - k + i >= 2
    ]


def _rotate_tour_to_start(tour, start_node):
    """Rotasi representasi siklus agar kota awal selalu berada di indeks 0."""
    index = tour.index(start_node)
    return tour[index:] + tour[:index]


def _canonical_cycle_key(tour):
    """Kunci siklus tanpa membedakan dua arah rute pada TSP simetris."""
    forward = tuple(tour)
    reverse = (tour[0], *reversed(tour[1:]))
    return min(forward, reverse)


def _three_opt_candidate_records(tour, triple):
    """Bangkitkan reconnect 3-Opt beserta signature sisi yang berubah.

    Perubahan sisi dihitung dari tiga sisi potong dan tiga sisi sambungan saja
    (bukan membangun ulang himpunan seluruh sisi untuk setiap kandidat).
    """
    n = len(tour)
    i, j, k = triple

    # Putar siklus sehingga segmen pertama dimulai tepat setelah sisi i.
    rotated = list(tour[i + 1:]) + list(tour[:i + 1])
    p = j - i - 1
    q = k - i - 1
    a = rotated[:p + 1]
    b = rotated[p + 1:q + 1]
    c = rotated[q + 1:]

    if not a or not b or not c:
        return []

    br = list(reversed(b))
    cr = list(reversed(c))
    reconnects = [
        (a, br, c),
        (a, b, cr),
        (a, br, cr),
        (a, c, b),
        (a, cr, b),
        (a, c, br),
        (a, cr, br),
    ]

    def edge(u, v):
        return (min(u, v), max(u, v))

    old_cut_edges = {
        edge(tour[e], tour[(e + 1) % n]) for e in (i, j, k)
    }
    start_node = tour[0]
    original_key = _canonical_cycle_key(tour)
    seen = set()
    records = []

    for first, second, third in reconnects:
        candidate = list(first) + list(second) + list(third)
        if len(candidate) != n or len(set(candidate)) != n:
            continue
        candidate = _rotate_tour_to_start(candidate, start_node)
        cycle_key = _canonical_cycle_key(candidate)
        if cycle_key == original_key or cycle_key in seen:
            continue
        seen.add(cycle_key)

        new_boundary_edges = {
            edge(first[-1], second[0]),
            edge(second[-1], third[0]),
            edge(third[-1], first[0]),
        }
        removed_edges = tuple(sorted(old_cut_edges - new_boundary_edges))
        added_edges = tuple(sorted(new_boundary_edges - old_cut_edges))
        records.append((candidate, (removed_edges, added_edges)))

    return records


def _three_opt_candidates_for_triple(tour, triple):
    """Compatibility helper: return candidate tours only."""
    return [candidate for candidate, _ in _three_opt_candidate_records(tour, triple)]


def _stable_tour_seed(tour):
    """Seed deterministik untuk mengambil sampel 3-Opt pada tur yang sama."""
    return len(tour) * 1_000_003 + sum((i + 1) * (int(city) + 1) for i, city in enumerate(tour))


def _three_opt_neighborhood_limit(n):
    """Batas kandidat 3-Opt untuk menjaga waktu komputasi pada dataset besar."""
    if n <= 12:
        return None  # Neighborhood lengkap untuk dataset kecil.
    if n <= 20:
        return 300
    if n <= 30:
        return 220
    if n <= 40:
        return 180
    return 140


def generate_random_neighbor(
    tour, method="SWAP", rng=None, valid_pairs=None, valid_triples=None
):
    """Menghasilkan satu tetangga acak beserta identitas gerakannya."""
    new_tour = copy.copy(tour)
    n = len(tour)
    method = _normalize_ls_method(method)
    rng = random if rng is None else rng

    if method == "3OPT":
        # Gunakan gabungan neighborhood: sebagian langkah tetap mengeksplorasi
        # seluruh jenis gerakan 2-Opt, sisanya menggunakan reconnect 3-Opt.
        if rng.random() < 0.35:
            return generate_random_neighbor(
                tour, "2OPT", rng=rng, valid_pairs=valid_pairs
            )
        if valid_triples is None:
            valid_triples = _valid_three_opt_edge_triples(n)
        if valid_triples:
            triple = rng.choice(valid_triples)
            records = _three_opt_candidate_records(tour, triple)
            if records:
                return rng.choice(records)
        # 3-Opt belum memiliki tiga sisi potong yang valid untuk N < 6.
        # Fallback ke 2-Opt agar fungsi tetap menghasilkan tetangga jika tersedia.
        return generate_random_neighbor(tour, "2OPT", rng=rng, valid_pairs=valid_pairs)

    if method == "2OPT":
        if valid_pairs is None:
            valid_pairs = _neighbor_position_pairs(n, "2OPT")
        if not valid_pairs:
            return new_tour, None
        idx1, idx2 = rng.choice(valid_pairs)
        new_tour[idx1:idx2 + 1] = reversed(new_tour[idx1:idx2 + 1])
    elif method == "SWAP":
        idx1, idx2 = sorted(rng.sample(range(1, n), 2))
        new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
    else:
        raise ValueError("Metode pencarian harus SWAP, 2OPT, atau 3OPT.")

    move_made = (min(tour[idx1], tour[idx2]), max(tour[idx1], tour[idx2]))
    return new_tour, move_made


def generate_all_neighbors(tour, method="SWAP", max_neighbors=None):
    """Menghasilkan tetangga dan identitas gerakan.

    Untuk 3-Opt, fungsi menggabungkan semua tetangga 2-Opt dengan reconnect 3-Opt.
    Jika max_neighbors ditetapkan, batas itu hanya membatasi tetangga tambahan 3-Opt;
    neighborhood 2-Opt tetap lengkap. Ini menjaga kualitas baseline sembari
    mengendalikan biaya pencarian pada dataset yang lebih besar.
    """
    n = len(tour)
    method = _normalize_ls_method(method)
    neighbors = []
    moves = []

    if method == "3OPT":
        triples = _valid_three_opt_edge_triples(n)
        if not triples:
            return generate_all_neighbors(tour, "2OPT", max_neighbors=max_neighbors)

        # Dahulukan reconnect 3-Opt, lalu tambahkan seluruh neighborhood 2-Opt.
        # Urutan ini memberi kesempatan ke reconnect 3-Opt lebih dulu pada
        # first-improvement local search, tetapi 2-Opt tetap selalu tersedia.
        seen = {_canonical_cycle_key(tour)}

        # Sampling deterministik 3-Opt pada dataset besar. max_neighbors hanya
        # membatasi tambahan kandidat 3-Opt; kandidat 2-Opt tetap lengkap.
        if max_neighbors is not None and len(triples) * 7 > max_neighbors:
            sample_count = max(1, min(len(triples), (max_neighbors + 2) // 3))
            rng = random.Random(_stable_tour_seed(tour))
            triples = rng.sample(triples, sample_count)

        added_three_opt = 0
        stop_sampling = False
        for triple in triples:
            for candidate, move_signature in _three_opt_candidate_records(tour, triple):
                candidate_key = _canonical_cycle_key(candidate)
                if candidate_key in seen:
                    continue
                seen.add(candidate_key)
                neighbors.append(candidate)
                moves.append(move_signature)
                added_three_opt += 1
                if max_neighbors is not None and added_three_opt >= max_neighbors:
                    stop_sampling = True
                    break
            if stop_sampling:
                break

        # Semua gerakan 2-Opt dimasukkan (kecuali siklus yang sudah tersedia
        # dari 3-Opt) agar mode 3-Opt tidak kehilangan neighborhood 2-Opt.
        two_opt_neighbors, two_opt_moves = generate_all_neighbors(tour, "2OPT")
        for candidate, move in zip(two_opt_neighbors, two_opt_moves):
            candidate_key = _canonical_cycle_key(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            neighbors.append(candidate)
            moves.append(move)
        return neighbors, moves

    valid_pairs = _neighbor_position_pairs(n, method)
    for i, j in valid_pairs:
        new_tour = copy.copy(tour)
        if method == "2OPT":
            new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
        else:  # SWAP
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        neighbors.append(new_tour)
        moves.append((min(tour[i], tour[j]), max(tour[i], tour[j])))
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
    if method == "3OPT":
        validate_symmetric_matrix(distance_matrix)
    current_tour = copy.copy(tour)
    current_dist = calculate_total_distance(current_tour, distance_matrix)

    if method == "3OPT":
        # Variable-neighborhood descent: mulai dari optimum lokal 2-Opt,
        # lalu terima reconnect 3-Opt hanya jika setelah dipoles lagi dengan
        # 2-Opt hasilnya benar-benar lebih baik. Jadi 3-Opt tidak menurunkan
        # kualitas dibandingkan hasil local-search 2-Opt pada rute awal sama.
        current_tour, current_dist = run_local_search(
            current_tour, distance_matrix, "2OPT"
        )
        max_neighbors = _three_opt_neighborhood_limit(len(current_tour))

        while True:
            neighbors, _ = generate_all_neighbors(
                current_tour, "3OPT", max_neighbors=max_neighbors
            )
            accepted_tour = None
            accepted_dist = current_dist

            for neighbor in neighbors:
                neighbor_dist = calculate_total_distance(
                    neighbor, distance_matrix
                )
                if neighbor_dist >= current_dist:
                    continue

                # Setelah reconnect 3-Opt, jalankan descent 2-Opt agar
                # neighborhood 2-Opt dieksploitasi hingga local optimum lagi.
                polished_tour, polished_dist = run_local_search(
                    neighbor, distance_matrix, "2OPT"
                )
                if polished_dist < current_dist:
                    accepted_tour = polished_tour
                    accepted_dist = polished_dist
                    break

            if accepted_tour is None:
                return current_tour, current_dist

            current_tour = accepted_tour
            current_dist = accepted_dist

    # Jalur legacy SWAP / 2-Opt dipertahankan tanpa perubahan perilaku.
    improved = True
    while improved:
        improved = False
        neighbors, _ = generate_all_neighbors(current_tour, method)
        for neighbor in neighbors:
            neighbor_dist = calculate_total_distance(neighbor, distance_matrix)
            if neighbor_dist < current_dist:
                current_tour = neighbor
                current_dist = neighbor_dist
                improved = True
                break

    return current_tour, current_dist


def _validate_early_stop_patience(early_stop_patience):
    """0 disables early stopping; a positive value is the no-improvement limit."""
    if not isinstance(early_stop_patience, int) or early_stop_patience < 0:
        raise ValueError("early_stop_patience harus berupa bilangan bulat >= 0.")
    return early_stop_patience


def _perturb_tour(tour, swap_count, rng):
    """Perturb a tour while keeping its configured start city anchored at index 0."""
    result = list(tour)
    n = len(result)
    if n <= 2:
        return result
    for _ in range(max(0, int(swap_count))):
        i, j = rng.sample(range(1, n), 2)
        result[i], result[j] = result[j], result[i]
    return result


# ===============================================================
# 7. SIMULATED ANNEALING (SA)
# ===============================================================

def simulated_annealing(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    initial_temp=10000.0,
    alpha=0.99,
    max_iter=15000,
    start_node=0,
    random_seed=42,
    initial_tour=None,
    adaptive_search=False,
    early_stop_patience=0,
):
    """Simulated Annealing with optional adaptive reheating and early stopping.

    Stage 4 uses the standard Metropolis acceptance rule for all N. On long
    plateaus, adaptive mode reheats and perturbs the incumbent current route;
    the best-so-far route is retained separately and can never be lost.
    """
    validate_input(distance_matrix, start_node)
    ls_method = _normalize_ls_method(ls_method)
    if ls_method == "3OPT":
        validate_symmetric_matrix(distance_matrix)
    _validate_early_stop_patience(early_stop_patience)
    if max_iter < 0:
        raise ValueError("max_iter tidak boleh negatif.")
    n = len(distance_matrix)
    rng = random.Random(random_seed) if random_seed is not None else random
    started_at = time.perf_counter()

    supplied_tour = _validated_initial_tour(initial_tour, distance_matrix, start_node)
    current_tour = supplied_tour if supplied_tour is not None else generate_initial_tour(
        distance_matrix, start_node, init_method
    )
    current_dist = calculate_total_distance(current_tour, distance_matrix)
    best_tour = copy.copy(current_tour)
    best_dist = current_dist
    history = [best_dist]
    no_improve = 0
    stagnant_since_diversification = 0
    diversification_events = 0
    iterations_completed = 0
    stop_reason = "max_iter"

    # The temperature ceiling is scale-aware for larger datasets, preventing
    # a user-entered very large initial temperature from overwhelming deltas.
    if n > 10:
        avg_edge = current_dist / max(1, n)
        temp_ceiling = min(float(initial_temp), avg_edge * 0.65)
        valid_pairs = None
        valid_triples = None
        if ls_method == "3OPT":
            valid_triples = _valid_three_opt_edge_triples(n)
            neighborhood_size = max(1, len(valid_triples) * 7)
        else:
            valid_pairs = _neighbor_position_pairs(n, ls_method)
            neighborhood_size = len(valid_pairs)
        iter_scale = min(1.0, 150.0 / max(1, max_iter))
        max_samples = max(4, int(min(neighborhood_size, n * 1.5) * iter_scale))
    else:
        temp_ceiling = float(initial_temp)
        valid_pairs = None
        valid_triples = None
        max_samples = 1

    temp = max(1e-12, temp_ceiling)
    reheat_patience = max(10, min(40, max_iter // 8 if max_iter >= 80 else 10))

    for iteration in range(max_iter):
        if n <= 10:
            candidate_tour, _ = generate_random_neighbor(
                current_tour, ls_method, rng=rng,
                valid_pairs=valid_pairs, valid_triples=valid_triples,
            )
            candidate_dist = calculate_total_distance(candidate_tour, distance_matrix)
        else:
            progress = (iteration + 1) / max(1, max_iter)
            num_samples = max(2, int(max_samples * (0.2 + 0.8 * progress)))
            candidate_tour = None
            candidate_dist = float("inf")
            for _ in range(num_samples):
                proposed_tour, _ = generate_random_neighbor(
                    current_tour, ls_method, rng=rng,
                    valid_pairs=valid_pairs, valid_triples=valid_triples,
                )
                proposed_dist = calculate_total_distance(proposed_tour, distance_matrix)
                if proposed_dist < candidate_dist:
                    candidate_tour, candidate_dist = proposed_tour, proposed_dist
            if candidate_tour is None:
                candidate_tour = list(current_tour)
                candidate_dist = current_dist

        delta = candidate_dist - current_dist
        if adaptive_search:
            # Adaptive mode uses the standard Metropolis rule without the
            # historical distance-to-best drift cap.
            accept = delta <= 0 or (
                temp > 0 and rng.random() < math.exp(max(-700, -delta / temp))
            )
        elif n <= 10:
            # Keep the Stage 3 small-instance behavior reproducible by default.
            accept = delta < 0 or (
                temp > 0 and rng.random() < math.exp(max(-700, -delta / temp))
            )
        else:
            # Baseline mode retains Stage 3's bounded-drift acceptance for
            # medium/large instances; benchmark evidence showed it was more
            # reliable on the fixed 30-city dataset than unrestricted uphill moves.
            within_drift = (candidate_dist - best_dist) <= (best_dist * 0.04)
            accept = delta < 0 or (
                within_drift and temp > 1e-6
                and rng.random() < math.exp(max(-700, -delta / temp))
            )

        if accept:
            current_tour = list(candidate_tour)
            current_dist = candidate_dist

        if current_dist < best_dist:
            best_tour = list(current_tour)
            best_dist = current_dist
            no_improve = 0
            stagnant_since_diversification = 0
        else:
            no_improve += 1
            stagnant_since_diversification += 1

        # Stage 3's bounded-drift re-anchoring remains active in baseline mode.
        if n > 10 and not adaptive_search and stagnant_since_diversification >= 4 and current_dist > best_dist:
            current_tour = list(best_tour)
            current_dist = best_dist
            stagnant_since_diversification = 0

        iterations_completed = iteration + 1
        history.append(best_dist)

        if early_stop_patience and no_improve >= early_stop_patience:
            stop_reason = "stagnation_patience"
            break

        if adaptive_search and stagnant_since_diversification >= reheat_patience:
            diversification_events += 1
            temp = max(temp, temp_ceiling * 0.35)
            perturb_count = min(n - 1, 1 + (diversification_events - 1) % 4)
            current_tour = _perturb_tour(current_tour, perturb_count, rng)
            current_dist = calculate_total_distance(current_tour, distance_matrix)
            stagnant_since_diversification = 0

        temp *= alpha

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history,
        "time_seconds": time.perf_counter() - started_at,
        "iterations_completed": iterations_completed,
        "stop_reason": stop_reason,
        "diversification_events": diversification_events,
    }


# ===============================================================
# 8. TABU SEARCH (TS)
# ===============================================================

def tabu_search(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    tabu_tenure=5,
    max_iter=100,
    start_node=0,
    initial_tour=None,
    random_seed=42,
    adaptive_search=False,
    early_stop_patience=0,
):
    """Tabu Search with adaptive diversification and optional early stop."""
    validate_input(distance_matrix, start_node)
    ls_method = _normalize_ls_method(ls_method)
    if ls_method == "3OPT":
        validate_symmetric_matrix(distance_matrix)
    _validate_early_stop_patience(early_stop_patience)
    if max_iter < 0:
        raise ValueError("max_iter tidak boleh negatif.")
    if tabu_tenure < 0:
        raise ValueError("tabu_tenure tidak boleh negatif.")

    rng = random.Random(random_seed) if random_seed is not None else random
    started_at = time.perf_counter()
    supplied_tour = _validated_initial_tour(initial_tour, distance_matrix, start_node)
    current_tour = supplied_tour if supplied_tour is not None else generate_initial_tour(
        distance_matrix, start_node, init_method
    )
    current_dist = calculate_total_distance(current_tour, distance_matrix)
    best_tour = list(current_tour)
    best_dist = current_dist
    tabu_list = {}
    history = [best_dist]
    no_improve = 0
    stagnant_since_diversification = 0
    diversification_events = 0
    iterations_completed = 0
    stop_reason = "max_iter"
    diversification_patience = max(10, int(tabu_tenure) * 3)

    for iteration in range(max_iter):
        tabu_list = {key: expiry for key, expiry in tabu_list.items() if expiry > iteration}
        max_neighbors = _three_opt_neighborhood_limit(len(current_tour)) if ls_method == "3OPT" else None
        neighbors, moves = generate_all_neighbors(current_tour, ls_method, max_neighbors=max_neighbors)
        best_neighbor = None
        best_neighbor_dist = float("inf")
        best_move_made = None

        for neighbor, move in zip(neighbors, moves):
            neighbor_dist = calculate_total_distance(neighbor, distance_matrix)
            if ls_method == "3OPT":
                _removed_edges, added_edges = move
                is_tabu = tabu_list.get(added_edges, -1) > iteration
            else:
                is_tabu = tabu_list.get(move, -1) > iteration
            if is_tabu and neighbor_dist < best_dist:
                is_tabu = False
            if not is_tabu and neighbor_dist < best_neighbor_dist:
                best_neighbor = neighbor
                best_neighbor_dist = neighbor_dist
                best_move_made = move

        if best_neighbor is not None:
            current_tour = list(best_neighbor)
            current_dist = best_neighbor_dist
            tabu_key = best_move_made[0] if ls_method == "3OPT" else best_move_made
            tabu_list[tabu_key] = iteration + int(tabu_tenure)
            if current_dist < best_dist:
                best_tour = list(current_tour)
                best_dist = current_dist
                no_improve = 0
                stagnant_since_diversification = 0
            else:
                no_improve += 1
                stagnant_since_diversification += 1
        else:
            no_improve += 1
            stagnant_since_diversification += 1

        iterations_completed = iteration + 1
        history.append(best_dist)

        if early_stop_patience and no_improve >= early_stop_patience:
            stop_reason = "stagnation_patience"
            break

        if adaptive_search and stagnant_since_diversification >= diversification_patience:
            diversification_events += 1
            # Restart near the best-so-far, but with a growing perturbation.
            # Clearing tabu memory prevents an expired search basin from trapping the run.
            swap_count = min(len(best_tour) - 1, 1 + diversification_events)
            current_tour = _perturb_tour(best_tour, swap_count, rng)
            current_dist = calculate_total_distance(current_tour, distance_matrix)
            tabu_list.clear()
            stagnant_since_diversification = 0

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history,
        "time_seconds": time.perf_counter() - started_at,
        "iterations_completed": iterations_completed,
        "stop_reason": stop_reason,
        "diversification_events": diversification_events,
    }


# ===============================================================
# 9. ITERATED LOCAL SEARCH (ILS)
# ===============================================================

def iterated_local_search(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    max_iter=50,
    start_node=0,
    random_seed=None,
    initial_tour=None,
    adaptive_search=False,
    early_stop_patience=0,
):
    """Iterated Local Search with perturbation strength that grows on stagnation."""
    validate_input(distance_matrix, start_node)
    ls_method = _normalize_ls_method(ls_method)
    if ls_method == "3OPT":
        validate_symmetric_matrix(distance_matrix)
    _validate_early_stop_patience(early_stop_patience)
    if max_iter < 0:
        raise ValueError("max_iter tidak boleh negatif.")

    started_at = time.perf_counter()
    rng = random if random_seed is None else random.Random(random_seed)
    supplied_tour = _validated_initial_tour(initial_tour, distance_matrix, start_node)
    current_tour = supplied_tour if supplied_tour is not None else generate_initial_tour(
        distance_matrix, start_node, init_method
    )
    current_tour, current_dist = run_local_search(current_tour, distance_matrix, ls_method)
    best_tour = list(current_tour)
    best_dist = current_dist
    history = [best_dist]
    no_improve = 0
    iterations_completed = 0
    diversification_events = 0
    stop_reason = "max_iter"
    strength_growth_interval = max(3, min(10, max_iter // 5 if max_iter >= 15 else 3))

    for iteration in range(max_iter):
        if adaptive_search:
            perturb_strength = min(len(current_tour) - 1, 3 + no_improve // strength_growth_interval)
        else:
            # Preserve Stage 3 baseline behavior: three swap attempts even for N=3.
            perturb_strength = 3

        perturbed_tour = _perturb_tour(current_tour, perturb_strength, rng)
        new_tour, new_dist = run_local_search(perturbed_tour, distance_matrix, ls_method)

        if new_dist <= current_dist:
            current_tour = list(new_tour)
            current_dist = new_dist

        if current_dist < best_dist:
            best_tour = list(current_tour)
            best_dist = current_dist
            no_improve = 0
        else:
            no_improve += 1
            if adaptive_search and no_improve > 0 and no_improve % strength_growth_interval == 0:
                diversification_events += 1

        iterations_completed = iteration + 1
        history.append(best_dist)

        if early_stop_patience and no_improve >= early_stop_patience:
            stop_reason = "stagnation_patience"
            break

    return {
        "best_tour": best_tour,
        "best_distance": best_dist,
        "history": history,
        "time_seconds": time.perf_counter() - started_at,
        "iterations_completed": iterations_completed,
        "stop_reason": stop_reason,
        "diversification_events": diversification_events,
    }


# ===============================================================
# 10. MULTI-START WRAPPERS
# ===============================================================

def _validate_multi_start_runs(multi_start_runs):
    if not isinstance(multi_start_runs, int) or not 1 <= multi_start_runs <= 5:
        raise ValueError("Jumlah multi-start harus berupa bilangan bulat antara 1 dan 5.")


def _run_summary(result, run_index, run_metadata, initial_tour, distance_matrix):
    return {
        "run": run_index + 1,
        "algorithm_seed": run_metadata["algorithm_seed"],
        "restart_seed": int(run_metadata["restart_seed"]),
        "initial_distance": float(calculate_total_distance(initial_tour, distance_matrix)),
        "best_distance": float(result["best_distance"]),
        "time_seconds": float(result["time_seconds"]),
        "iterations_completed": int(result.get("iterations_completed", len(result.get("history", [])) - 1)),
        "stop_reason": result.get("stop_reason", "max_iter"),
        "diversification_events": int(result.get("diversification_events", 0)),
    }


def _combine_multistart_results(results, initial_tours, run_metadata, started_at, distance_matrix):
    """Return the best run and a monotone global-best history across sequential starts."""
    best_index = min(range(len(results)), key=lambda i: results[i]["best_distance"])
    best = dict(results[best_index])
    combined_history = []
    incumbent = float("inf")
    run_summaries = []

    for index, (result, initial_tour, metadata) in enumerate(zip(results, initial_tours, run_metadata)):
        for value in result.get("history", [result["best_distance"]]):
            incumbent = min(incumbent, float(value))
            combined_history.append(incumbent)
        run_summaries.append(_run_summary(result, index, metadata, initial_tour, distance_matrix))

    best["history"] = combined_history
    best["time_seconds"] = time.perf_counter() - started_at
    best["num_starts"] = len(results)
    best["best_run"] = best_index + 1
    best["runs"] = run_summaries
    best["total_iterations_completed"] = sum(
        int(result.get("iterations_completed", len(result.get("history", [])) - 1))
        for result in results
    )
    best["total_diversification_events"] = sum(
        int(result.get("diversification_events", 0)) for result in results
    )
    return best


def multi_start_simulated_annealing(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    initial_temp=10000.0,
    alpha=0.99,
    max_iter=15000,
    start_node=0,
    random_seed=42,
    multi_start_runs=3,
    adaptive_search=False,
    early_stop_patience=0,
):
    """Run SA from a heuristic start and reproducible randomized starts."""
    validate_input(distance_matrix, start_node)
    _validate_multi_start_runs(multi_start_runs)
    _validate_early_stop_patience(early_stop_patience)
    starts = _build_multistart_initial_tours(distance_matrix, start_node, init_method, multi_start_runs, random_seed)
    started_at = time.perf_counter()
    results, run_metadata = [], []
    for index, initial_tour in enumerate(starts):
        seed = int(random_seed) + index
        run_metadata.append({
            "algorithm_seed": seed,
            "restart_seed": int(random_seed) if index == 0 else int(random_seed) + index * 104729,
        })
        results.append(simulated_annealing(
            distance_matrix=distance_matrix, init_method=init_method, ls_method=ls_method,
            initial_temp=initial_temp, alpha=alpha, max_iter=max_iter, start_node=start_node,
            random_seed=seed, initial_tour=initial_tour,
            adaptive_search=adaptive_search, early_stop_patience=early_stop_patience,
        ))
    return _combine_multistart_results(results, starts, run_metadata, started_at, distance_matrix)


def multi_start_tabu_search(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    tabu_tenure=5,
    max_iter=100,
    start_node=0,
    random_seed=42,
    multi_start_runs=3,
    adaptive_search=False,
    early_stop_patience=0,
):
    """Run TS from distinct, reproducible initial tours."""
    validate_input(distance_matrix, start_node)
    _validate_multi_start_runs(multi_start_runs)
    _validate_early_stop_patience(early_stop_patience)
    starts = _build_multistart_initial_tours(distance_matrix, start_node, init_method, multi_start_runs, random_seed)
    started_at = time.perf_counter()
    results, run_metadata = [], []
    for index, initial_tour in enumerate(starts):
        run_metadata.append({
            "algorithm_seed": None,  # TS does not use a stochastic move generator.
            "restart_seed": int(random_seed) if index == 0 else int(random_seed) + index * 104729,
        })
        results.append(tabu_search(
            distance_matrix=distance_matrix, init_method=init_method, ls_method=ls_method,
            tabu_tenure=tabu_tenure, max_iter=max_iter, start_node=start_node,
            initial_tour=initial_tour, random_seed=int(random_seed) + index,
            adaptive_search=adaptive_search, early_stop_patience=early_stop_patience,
        ))
    return _combine_multistart_results(results, starts, run_metadata, started_at, distance_matrix)


def multi_start_iterated_local_search(
    distance_matrix,
    init_method="NN",
    ls_method="2OPT",
    max_iter=50,
    start_node=0,
    random_seed=42,
    multi_start_runs=3,
    adaptive_search=False,
    early_stop_patience=0,
):
    """Run seeded ILS restarts and return the best solution found."""
    validate_input(distance_matrix, start_node)
    _validate_multi_start_runs(multi_start_runs)
    _validate_early_stop_patience(early_stop_patience)
    starts = _build_multistart_initial_tours(distance_matrix, start_node, init_method, multi_start_runs, random_seed)
    started_at = time.perf_counter()
    results, run_metadata = [], []
    for index, initial_tour in enumerate(starts):
        seed = int(random_seed) + index
        run_metadata.append({
            "algorithm_seed": seed,
            "restart_seed": int(random_seed) if index == 0 else int(random_seed) + index * 104729,
        })
        results.append(iterated_local_search(
            distance_matrix=distance_matrix, init_method=init_method, ls_method=ls_method,
            max_iter=max_iter, start_node=start_node, random_seed=seed, initial_tour=initial_tour,
            adaptive_search=adaptive_search, early_stop_patience=early_stop_patience,
        ))
    return _combine_multistart_results(results, starts, run_metadata, started_at, distance_matrix)

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

    # Uji 3-Opt: rute tetangga valid, kota awal tetap, dan local search tidak memburuk.
    neighbors_3opt, moves_3opt = generate_all_neighbors(test_tour, "3OPT")
    assert neighbors_3opt, "3-Opt harus menghasilkan tetangga untuk 8 kota"
    assert len(neighbors_3opt) == len(moves_3opt)
    assert all(len(t) == 8 and len(set(t)) == 8 and t[0] == 0 for t in neighbors_3opt)
    assert all(set(t) == set(range(8)) for t in neighbors_3opt)
    local_3opt_tour, local_3opt_dist = run_local_search(initial, matrix, "3OPT")
    assert local_3opt_dist <= initial_distance
    assert local_3opt_tour[0] == 0 and set(local_3opt_tour) == set(range(8))

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
    print(f"[8 Kota]  3-Opt tetangga={len(neighbors_3opt)} | local-search distance={local_3opt_dist}")


if __name__ == "__main__":
    _self_test()