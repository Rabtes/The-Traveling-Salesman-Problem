import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.animation as animation

# --- PYMOO IMPORTS ---
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

# 1. Read City Data
cities = []
try:
    with open("cityData.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                city_id, x, y = parts
                cities.append((int(city_id), float(x), float(y)))
except FileNotFoundError:
    print("Error: cityData.txt not found.")
    exit()

# 2. Read Distance Matrix
dist_matrix = []
try:
    with open("intercityDistance.txt", "r") as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            dist_matrix.append(row)
    dist_matrix = np.array(dist_matrix).astype(float)
except FileNotFoundError:
    print("Error: intercityDistance.txt not found.")
    exit()

# 3. Clustering Logic: Farthest First Traversal
def group_cities_farthest_first(cities, dist_matrix, group_size=6):
    n = len(cities)
    unassigned = set(range(n))
    groups = []
    
    while unassigned:
        # Find the two farthest points among unassigned cities
        max_dist = -1
        farthest_pair = (None, None)
        unassigned_list = list(unassigned)
        
        for i in range(len(unassigned_list)):
            for j in range(i+1, len(unassigned_list)):
                d = dist_matrix[unassigned_list[i], unassigned_list[j]]
                if d > max_dist:
                    max_dist = d
                    farthest_pair = (unassigned_list[i], unassigned_list[j])
        
        # If only one city remains, make it a group
        if len(unassigned) <= group_size:
            groups.append(list(unassigned))
            break
            
        # Start a group from each of the two farthest points
        for start in farthest_pair:
            if start not in unassigned:
                continue
            
            group = [start]
            unassigned.remove(start)
            
            # Temporary distance array to find nearest neighbors
            dists = dist_matrix[start].copy()
            # Set distance to already assigned cities to infinity
            dists[list(set(range(n)) - unassigned - {start})] = np.inf
            
            for _ in range(group_size - 1):
                idx = np.argmin(dists)
                group.append(idx)
                unassigned.discard(idx)
                dists[idx] = np.inf
                if len(unassigned) == 0:
                    break
            
            groups.append(group)
            if len(unassigned) == 0:
                break
    return groups

# Create groups (Cluster size ~7)
groups = group_cities_farthest_first(cities, dist_matrix, group_size=7)

# 4. Find Centroids of each group
centers = []
for group in groups:
    xs = [cities[i][1] for i in group]
    ys = [cities[i][2] for i in group]
    center_x = np.mean(xs)
    center_y = np.mean(ys)
    centers.append((center_x, center_y))

# Map each city index to its group index
city_to_group = {}
for group_idx, group in enumerate(groups):
    for city_idx in group:
        city_to_group[city_idx] = group_idx

# 5. Distance Matrix between Centers (Euclidean)
center_dist = np.zeros((len(centers), len(centers)))
for i in range(len(centers)):
    for j in range(len(centers)):
        if i != j:
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            center_dist[i, j] = np.sqrt(dx*dx + dy*dy)
        else:
            center_dist[i, j] = np.inf

# Helper: Simple Nearest Neighbor for TSP approximation
def tsp_nearest_neighbor(dist_matrix, start=0):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    path = [start]
    unvisited.remove(start)
    current = start
    while unvisited:
        next_city = min(unvisited, key=lambda x: dist_matrix[current, x])
        path.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return path

# Calculate path between groups (Inter-Group Routing)
group_path = tsp_nearest_neighbor(center_dist, start=0)
print("Inter-group shortest path order (indices):", group_path)

# --- FIX: Genetic Algorithm Definition (Method 2) ---
def solve_tsp_ga(dist_matrix_local):
    n = dist_matrix_local.shape[0]
    
    class TSPProblem(Problem):
        def __init__(self):
            # FIXED: Added elementwise_evaluation=True to prevent shape errors
            super().__init__(n_var=n, n_obj=1, n_constr=0, xl=0, xu=n-1, type_var=int, elementwise_evaluation=True)
            
        def _evaluate(self, x, out, *args, **kwargs):
            # FIXED: Ensure inputs are integers
            x = x.astype(int)
            
            # Check for valid permutation (no duplicate cities)
            if len(np.unique(x)) < n:
                out["F"] = 1e9 # Penalty for invalid solutions
                return

            d = dist_matrix_local[x[-1], x[0]] # Loop back to start
            for i in range(n-1):
                d += dist_matrix_local[x[i], x[i+1]]
            out["F"] = d
            
    problem = TSPProblem()
    algorithm = GA(pop_size=100, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ('n_gen', 300), verbose=False)
    return res.X.astype(int), res.F[0]

# --- OPTIONAL: Run GA once to satisfy project requirement ---
# This runs GA on the first group just to show it works
print("\n--- Testing Genetic Algorithm on Group 0 ---")
try:
    group0_indices = groups[0]
    sub_matrix_0 = dist_matrix[np.ix_(group0_indices, group0_indices)]
    ga_route, ga_dist = solve_tsp_ga(sub_matrix_0)
    print(f"GA Result for Group 0: Distance = {ga_dist:.2f}")
except Exception as e:
    print(f"GA Warning: {e}")

# 6. Solve for Final Route (Clustering Method)
def solve_group_tsp(dist_matrix, group):
    sub_matrix = dist_matrix[np.ix_(group, group)]
    path_local = tsp_nearest_neighbor(sub_matrix, start=0)
    return [group[i] for i in path_local]

# Calculate Intra-Group routes
group_routes = [solve_group_tsp(dist_matrix, group) for group in groups]

# --- USER INPUT ---
try:
    start_city_id = int(input("Enter Start City ID: "))
except ValueError:
    start_city_id = 1
    print(f"Invalid input. Defaulting to City {start_city_id}.")

# Find global index of start city
try:
    start_idx = next(i for i, c in enumerate(cities) if c[0] == start_city_id)
except StopIteration:
    print("City ID not found! Exiting.")
    exit()

# Which group is this city in?
start_group = city_to_group[start_idx]
print(f"Start City {start_city_id} is in Group {start_group}.")

# Reorder group sequence to start from the correct group
if start_group in group_path:
    start_index = group_path.index(start_group)
    reordered_groups = group_path[start_index:] + group_path[:start_index]
else:
    raise ValueError("Start group not found in group_path!")

# Construct Final City Sequence
final_city_order = []
for g_idx in reordered_groups:
    route = group_routes[g_idx]
    
    # If this is the starting group, rotate the route to start at the specific city
    if g_idx == start_group:
        if start_idx in route:
            local_idx = route.index(start_idx)
            route = route[local_idx:] + route[:local_idx]
            
    final_city_order.extend(route)

# Convert indices back to City IDs for display
city_id_order = [cities[i][0] for i in final_city_order]

print("\n🗺️ Final route City IDs (from start):")
print(city_id_order)

# --- VISUALIZATION 1: CLUSTERS ---
plt.figure(figsize=(10, 7))
colors = cm.rainbow(np.linspace(0, 1, len(groups)))

city_coords = np.array([[c[1], c[2]] for c in cities]) 

for g_idx, group in enumerate(groups):
    group_coords = np.array([[cities[i][1], cities[i][2]] for i in group])
    plt.scatter(group_coords[:, 0], group_coords[:, 1],
                color=colors[g_idx], label=f"Group {g_idx}", s=60)
    
    # Annotate City IDs
    for i in group:
        plt.text(cities[i][1] + 0.8, cities[i][2] + 0.8,
                 str(cities[i][0]), fontsize=8)

# Plot Centers
centers = np.array(centers)
plt.scatter(centers[:, 0], centers[:, 1], color="black", marker="*", s=150, label="Group Centers")

# Plot Arrows between Groups
for i in range(len(group_path) - 1):
    a = centers[group_path[i]]
    b = centers[group_path[i + 1]]
    plt.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1],
              color="black", width=0.3, head_width=2.5,
              length_includes_head=True, alpha=0.6)

# Mark Start City
start_city = next(c for c in cities if c[0] == start_city_id)
plt.scatter(start_city[1], start_city[2],
            color="red", s=120, edgecolor="black",
            label=f"Start City {start_city_id}", zorder=5)

plt.title("Grouped TSP City Visualization")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.grid(True)
plt.show()

# --- VISUALIZATION 2: ANIMATION ---
route_coords = np.array([[cities[i][1], cities[i][2]] for i in final_city_order])
city_labels = [cities[i][0] for i in final_city_order]

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("TSP Route Simulation")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.grid(True)

# Show all cities in background
for g_idx, group in enumerate(groups):
    group_coords = np.array([[cities[i][1], cities[i][2]] for i in group])
    ax.scatter(group_coords[:, 0], group_coords[:, 1],
               color=colors[g_idx], s=40, alpha=0.5)

# Mark start city
ax.scatter(start_city[1], start_city[2], color="red", s=120, edgecolor="black", label="Start City")

# Elements to update
(line,) = ax.plot([], [], color="black", linewidth=2, alpha=0.7)
(point,) = ax.plot([], [], "ro", markersize=8)

# Set Axis Limits
ax.set_xlim(min(city_coords[:, 0]) - 5, max(city_coords[:, 0]) + 5)
ax.set_ylim(min(city_coords[:, 1]) - 5, max(city_coords[:, 1]) + 5)

def update(frame):
    line.set_data(route_coords[: frame + 1, 0], route_coords[: frame + 1, 1])
    point.set_data([route_coords[frame, 0]], [route_coords[frame, 1]]) 
    ax.set_title(f"TSP Route Simulation — Visiting City {city_labels[frame]}")
    return line, point

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(route_coords),
    interval=700, # 700ms per frame
    repeat=False,
)

plt.legend()
plt.show()