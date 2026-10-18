import random
import math
import copy
import time

# ============================================================
# 1. UTILITAS & VALIDASI
# ============================================================

def validate_input(distance_matrix, start_node):
    n = len(distance_matrix)
    if n < 3:
        raise ValueError("TSP memerlukan minimal 3 kota.")
    if any(len(row) != n for row in distance_matrix):
        raise ValueError("Matriks jarak harus berbentuk persegi (N x N).")
    if not (0 <= start_node < n):
        raise ValueError(f"Titik awal harus berada di antara 0 dan {n-1}.")
    return n

def calculate_total_distance(tour, distance_matrix):
    distance = 0.0
    for i in range(len(tour) - 1):
        distance += distance_matrix[tour[i]][tour[i+1]]
    distance += distance_matrix[tour[-1]][tour[0]]
    return distance

# ============================================================
# 2. GREEDY CONSTRUCTION HEURISTICS (INITIALIZATION)
# ============================================================

def init_nearest_neighbor(distance_matrix, start_node=0):
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start_node]
    unvisited.remove(start_node)
    
    current = start_node
    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix[current][node])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour

def init_farthest_insertion(distance_matrix, start_node=0):
    n = len(distance_matrix)
    unvisited = set(range(n))
    tour = [start_node]
    unvisited.remove(start_node)
    
    farthest_node = max(unvisited, key=lambda node: distance_matrix[start_node][node])
    tour.append(farthest_node)
    unvisited.remove(farthest_node)
    
    while unvisited:
        k = max(unvisited, key=lambda x: min(distance_matrix[x][t] for t in tour))
        best_pos = -1
        min_cost = float('inf')
        
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost = distance_matrix[tour[i]][k] + distance_matrix[k][tour[j]] - distance_matrix[tour[i]][tour[j]]
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
    
    closest_node = min(unvisited, key=lambda node: distance_matrix[start_node][node])
    tour.append(closest_node)
    unvisited.remove(closest_node)
    
    while unvisited:
        min_cost = float('inf')
        best_node = -1
        best_pos = -1
        
        for k in unvisited:
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                cost = distance_matrix[tour[i]][k] + distance_matrix[k][tour[j]] - distance_matrix[tour[i]][tour[j]]
                if cost < min_cost:
                    min_cost = cost
                    best_node = k
                    best_pos = i + 1
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
        
    start_idx = tour.index(start_node)
    return tour[start_idx:] + tour[:start_idx]

def generate_initial_tour(distance_matrix, start_node=0, method='NN'):
    method = method.upper()
    if method == 'FI':
        return init_farthest_insertion(distance_matrix, start_node)
    elif method == 'CI':
        return init_cheapest_insertion(distance_matrix, start_node)
    else:
        return init_nearest_neighbor(distance_matrix, start_node)

# ============================================================
# 3. NEIGHBORHOOD GENERATORS (LOCAL SEARCH)
# ============================================================

def generate_random_neighbor(tour, method='SWAP'):
    new_tour = copy.copy(tour)
    n = len(tour)
    idx1, idx2 = sorted(random.sample(range(1, n), 2))
    
    if method.upper() == '2OPT':
        new_tour[idx1:idx2+1] = reversed(new_tour[idx1:idx2+1])
        move_made = (min(tour[idx1], tour[idx2]), max(tour[idx1], tour[idx2]))
    else:
        new_tour[idx1], new_tour[idx2] = new_tour[idx2], new_tour[idx1]
        move_made = (min(tour[idx1], tour[idx2]), max(tour[idx1], tour[idx2]))
    return new_tour, move_made

def generate_all_neighbors(tour, method='SWAP'):
    neighbors = []
    moves = []
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            new_tour = copy.copy(tour)
            if method.upper() == '2OPT':
                new_tour[i:j+1] = reversed(new_tour[i:j+1])
            else:
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            neighbors.append(new_tour)
            moves.append((min(tour[i], tour[j]), max(tour[i], tour[j])))
    return neighbors, moves

def run_local_search(tour, distance_matrix, method='2OPT'):
    """Menjalankan Local Search murni hingga mencapai local optimum (Digunakan untuk ILS)."""
    current_tour = tour
    current_dist = calculate_total_distance(current_tour, distance_matrix)
    improved = True
    while improved:
        improved = False
        neighbors, _ = generate_all_neighbors(current_tour, method)
        for neighbor in neighbors:
            ndist = calculate_total_distance(neighbor, distance_matrix)
            if ndist < current_dist:
                current_tour = neighbor
                current_dist = ndist
                improved = True
                break # First improvement
    return current_tour, current_dist

# ============================================================
# 4. ALGORITMA METAHEURISTIK (SA, TS, ILS)
# ============================================================

def simulated_annealing(distance_matrix, init_method='NN', ls_method='SWAP', 
                        initial_temp=10000.0, alpha=0.99, max_iter=15000, start_node=0):
    validate_input(distance_matrix, start_node)
    start_time = time.perf_counter()
    
    current_tour = generate_initial_tour(distance_matrix, start_node, init_method)
    current_dist = calculate_total_distance(current_tour, distance_matrix)
    best_tour, best_dist = copy.copy(current_tour), current_dist
    temp, history = initial_temp, [best_dist]
    
    for _ in range(max_iter):
        new_tour, _ = generate_random_neighbor(current_tour, ls_method)
        new_dist = calculate_total_distance(new_tour, distance_matrix)
        delta = new_dist - current_dist
        
        if delta < 0 or (temp > 0 and random.random() < math.exp(max(-700, -delta / temp))):
            current_tour, current_dist = new_tour, new_dist
            if current_dist < best_dist:
                best_tour, best_dist = copy.copy(current_tour), current_dist
                
        temp *= alpha
        history.append(best_dist)
        
    return {"best_tour": best_tour, "best_distance": best_dist, "history": history, "time_seconds": time.perf_counter() - start_time}

def tabu_search(distance_matrix, init_method='NN', ls_method='SWAP', 
                tabu_tenure=5, max_iter=100, start_node=0):
    validate_input(distance_matrix, start_node)
    start_time = time.perf_counter()
    
    current_tour = generate_initial_tour(distance_matrix, start_node, init_method)
    current_dist = calculate_total_distance(current_tour, distance_matrix)
    best_tour, best_dist = copy.copy(current_tour), current_dist
    tabu_list, history = {}, [best_dist]
    
    for iteration in range(max_iter):
        tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
        neighbors, moves = generate_all_neighbors(current_tour, ls_method)
        
        best_neighbor, best_neighbor_dist, best_move_made = None, float('inf'), None
        for neighbor, move in zip(neighbors, moves):
            neighbor_dist = calculate_total_distance(neighbor, distance_matrix)
            is_tabu = tabu_list.get(move, -1) > iteration
            
            if is_tabu and neighbor_dist < best_dist: is_tabu = False # Aspiration criteria
                
            if not is_tabu and neighbor_dist < best_neighbor_dist:
                best_neighbor, best_neighbor_dist, best_move_made = neighbor, neighbor_dist, move
                
        if best_neighbor is not None:
            current_tour, current_dist = best_neighbor, best_neighbor_dist
            tabu_list[best_move_made] = iteration + tabu_tenure
            if current_dist < best_dist:
                best_tour, best_dist = copy.copy(current_tour), current_dist
                
        history.append(best_dist)
        
    return {"best_tour": best_tour, "best_distance": best_dist, "history": history, "time_seconds": time.perf_counter() - start_time}

def iterated_local_search(distance_matrix, init_method='NN', ls_method='2OPT', 
                          max_iter=50, start_node=0):
    """Bonus: Iterated Local Search (ILS) sesuai saran dosen."""
    validate_input(distance_matrix, start_node)
    start_time = time.perf_counter()
    
    # 1. Inisialisasi & Base Local Search
    current_tour = generate_initial_tour(distance_matrix, start_node, init_method)
    current_tour, current_dist = run_local_search(current_tour, distance_matrix, ls_method)
    
    best_tour, best_dist = copy.copy(current_tour), current_dist
    history = [best_dist]
    
    for _ in range(max_iter):
        # 2. Perturbation (Goncangan: 3x Swap acak agar keluar dari local optimum)
        perturbed_tour = copy.copy(current_tour)
        n = len(perturbed_tour)
        for _ in range(3):
            i1, i2 = random.sample(range(1, n), 2)
            perturbed_tour[i1], perturbed_tour[i2] = perturbed_tour[i2], perturbed_tour[i1]
            
        # 3. Local Search kembali
        new_tour, new_dist = run_local_search(perturbed_tour, distance_matrix, ls_method)
        
        # 4. Acceptance Criterion
        if new_dist <= current_dist:
            current_tour, current_dist = new_tour, new_dist
            if current_dist < best_dist:
                best_tour, best_dist = copy.copy(current_tour), current_dist
                
        history.append(best_dist)
        
    return {"best_tour": best_tour, "best_distance": best_dist, "history": history, "time_seconds": time.perf_counter() - start_time}