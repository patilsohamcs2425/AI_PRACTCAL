from collections import deque

# 1. Representing Andheri to Lonavala map as an adjacency list
mumbai_map = {

    # Starting Point
    'Andheri': ['Powai', 'Bandra', 'Jogeshwari'],

    # Route 1 (Via Powai - Eastern Route)
    'Powai': ['Vikhroli'],
    'Vikhroli': ['Ghatkopar'],
    'Ghatkopar': ['Vashi'],
    'Vashi': ['Panvel'],

    # Route 2 (Via Bandra - Highway Route)
    'Bandra': ['BKC'],
    'BKC': ['Chembur'],
    'Chembur': ['Panvel'],

    # Route 3 (Via Jogeshwari - Western Route)
    'Jogeshwari': ['Goregaon'],
    'Goregaon': ['Malad'],
    'Malad': ['Panvel'],

    # Common Route
    'Panvel': ['Khopoli'],
    'Khopoli': ['Lonavala'],

    # Destination
    'Lonavala': []
}


# A) Breadth First Search (BFS) Implementation

def bfs_shortest_path(graph, start, destination):

    queue = deque([(start, [start])])
    visited = set([start])
    nodes_explored_count = 0

    while queue:

        current, path = queue.popleft()
        nodes_explored_count += 1

        if current == destination:
            return path, nodes_explored_count

        for neighbor in graph.get(current, []):

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, nodes_explored_count



# B) Iterative Depth First Search (DFS) Implementation

def iterative_dfs_path(graph, start, destination):

    stack = [(start, [start])]
    visited = set()
    nodes_explored_count = 0

    while stack:

        current, path = stack.pop()
        nodes_explored_count += 1

        if current == destination:
            return path, nodes_explored_count

        if current not in visited:

            visited.add(current)

            for neighbor in graph.get(current, []):

                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None, nodes_explored_count



# Execution

start_node = 'Andheri'
end_node = 'Lonavala'


bfs_path, bfs_count = bfs_shortest_path(
    mumbai_map,
    start_node,
    end_node
)


dfs_path, dfs_count = iterative_dfs_path(
    mumbai_map,
    start_node,
    end_node
)



print("--- BFS Results ---")

print(f"Path Found: {' -> '.join(bfs_path)}")

print(f"Total Steps (Edges): {len(bfs_path) - 1}")

print(f"Total Nodes Visited/Checked: {bfs_count}\n")



print("--- Iterative DFS Results ---")

print(f"Path Found: {' -> '.join(dfs_path)}")

print(f"Total Steps (Edges): {len(dfs_path) - 1}")

print(f"Total Nodes Visited/Checked: {dfs_count}")