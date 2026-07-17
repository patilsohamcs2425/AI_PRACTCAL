import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. ROAD NETWORK (Marol Naka -> Karjat)
# -------------------------------------------------------------------------

graph = {
    'Marol Naka': {
        'Vikhroli': 5.9,
        'Airport': 4.0
    },

    # Route 1 : Via Vikhroli -> Sion Panvel Highway
    'Vikhroli': {
        'Sion Panvel Hwy': 6.1
    },

    'Sion Panvel Hwy': {
        'Panvel': 44.6
    },

    # Route 2 : Via Airport -> Chembur -> Vashi
    'Airport': {
        'Chembur': 13.8
    },

    'Chembur': {
        'Vashi': 30.9,
        'Atal Setu': 35.7
    },

    'Vashi': {
        'Panvel': 21.6
    },

    # Route 3 : Via Atal Setu
    'Atal Setu': {
        'Panvel': 15.8
    },

    'Panvel': {
        'Karjat': 21.6
    },

    'Karjat': {}
}

# -------------------------------------------------------------------------
# Straight-line heuristic h(n) to Karjat
# -------------------------------------------------------------------------

heuristics = {
    'Marol Naka': 54,
    'Vikhroli': 48,
    'Airport': 55,
    'Airport MTHL': 54,
    'Sion Panvel Hwy': 42,
    'Chembur': 43,
    'Vashi': 35,          # Added missing node to prevent KeyError
    'Atal Setu': 27,
    'Panvel': 18,
    'Karjat': 0
}

# -------------------------------------------------------------------------
# Coordinates for plotting
# -------------------------------------------------------------------------

coords = {
    'Marol Naka': (2, 8),
    'Kurla': (1, 6),
    'Airport': (1, 6),           # Added missing node coordinate mapping
    'Powai': (4, 7),
    'Vikhroli': (4, 7),          # Added missing node coordinate mapping
    'Sion': (6, 6),
    'Sion Panvel Hwy': (6, 6),   # Added missing node coordinate mapping
    'Chembur': (1, 4),
    'Airoli': (4, 5),
    'Vashi': (2, 2),
    'MTHL': (6, 3),
    'Atal Setu': (6, 3),         # Added missing node coordinate mapping
    'Panvel': (5, 1),
    'Chowk': (8, 0),
    'Karjat': (11, -1)
}

# -------------------------------------------------------------------------
# RBFS ALGORITHM
# -------------------------------------------------------------------------

def rbfs_search(start, goal):
    success, path, cost, _ = rbfs(
        start,
        goal,
        g=0,
        f_limit=float('inf'),
        path=[start]
    )
    return path, cost


def rbfs(node, goal, g, f_limit, path):
    if node == goal:
        return True, path, g, g

    neighbors = graph[node]

    if not neighbors:
        return False, [], 0, float('inf')

    successors = []

    for neighbor, distance in neighbors.items():
        if neighbor not in path:
            next_g = g + distance
            next_f = max(
                next_g + heuristics[neighbor],
                g + heuristics[node]
            )
            successors.append(
                [next_f, neighbor, next_g]
            )

    if not successors:
        return False, [], 0, float('inf')

    while True:
        successors.sort(key=lambda x: x[0])
        best = successors[0]

        if best[0] > f_limit:
            return False, [], 0, best[0]

        alternative = successors[1][0] if len(successors) > 1 else float('inf')

        success, result_path, total_g, returned_f = rbfs(
            best[1],
            goal,
            best[2],
            min(f_limit, alternative),
            path + [best[1]]
        )

        best[0] = returned_f

        if success:
            return True, result_path, total_g, returned_f


# -------------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------------

path, total_distance = rbfs_search("Marol Naka", "Karjat")

print("\nOptimal RBFS Path:\n")
print(" -> ".join(path))
print(f"\nTotal Distance = {total_distance} km")

# -------------------------------------------------------------------------
# GRAPH VISUALIZATION
# -------------------------------------------------------------------------

plt.figure(figsize=(12, 7))

# Draw Edges
for node, neighbors in graph.items():
    x1, y1 = coords[node]
    for neighbor, dist in neighbors.items():
        x2, y2 = coords[neighbor]

        is_path = (
            node in path and
            neighbor in path and
            path.index(neighbor) == path.index(node) + 1
        )

        color = '#2ecc71' if is_path else '#b0b0b0'
        width = 3 if is_path else 1.5

        plt.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=width
        )

        plt.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            f"{dist} km",
            color='red',
            fontsize=9
        )

# Draw Nodes
for node, (x, y) in coords.items():
    # Only render nodes that are actively part of the defined route network graph
    if node in graph or any(node in nbrs for nbrs in graph.values()):
        node_color = '#f39c12' if node in path else '#3498db'

        plt.scatter(
            x,
            y,
            s=1200,
            color=node_color,
            edgecolors='black',
            zorder=3
        )

        plt.text(
            x,
            y,
            f"{node}\nh={heuristics[node]}",
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            fontweight='bold'
        )

plt.title(
    f"Recursive Best-First Search\nMarol Naka → Karjat\nTotal Distance = {round(total_distance, 2)} km",
    fontsize=14,
    fontweight='bold'
)

plt.axis('off')
plt.show()