from collections import defaultdict, deque
import random
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph):
    # створюємо орієнтований граф (Di - directed)
    G = nx.DiGraph()
    for vertex, neighbors in graph.adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)

    pos = nx.spring_layout(G)

    # налаштування для відображення графу
    nx.draw(G, pos, with_labels=True, node_size=600, node_color="skyblue", font_size=10, font_weight="bold", edge_color="black", linewidths=1, alpha=0.7)
    plt.title("Graph Visualization")
    plt.show()

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices

        # вектор суміжності для кожної вершини, створюємо список для кожного ключа (вершини)
        self.adjacency_list = defaultdict(list)
        for i in range(vertices):
            self.adjacency_list[i] = []

    # додаємо вершину до списків суміжності
    def add_edge(self, source, destination):
        self.adjacency_list[source].append(destination)

    # топологічний алгоритм сортування Кана
    def topological_sort(self):
        in_degree = [0] * self.vertices
        for neighbors in self.adjacency_list.values():
            for neighbor in neighbors:
                in_degree[neighbor] += 1

        queue = deque()
        for i in range(self.vertices):
            if in_degree[i] == 0:
                queue.append(i)

        topological_sort = []
        while queue:
            vertex = queue.popleft()
            topological_sort.append(vertex)

            for neighbor in self.adjacency_list[vertex]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topological_sort) != self.vertices:
            raise ValueError("Граф містить цикл ;(")

        return topological_sort

    def print_graph(self):
        for vertex, neighbors in self.adjacency_list.items():
            print(f"Вершина {vertex}: {' '.join(map(str, neighbors))}")

    @staticmethod
    def generate_random_graph(n, delta):
        max_edges = (n * (n - 1)) // 2
        num_edges = int(max_edges * delta)

        graph = Graph(n)
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

        random.shuffle(edge_list)
        for edge in edge_list[:num_edges]:
            graph.add_edge(edge[0], edge[1])

        return graph

if __name__ == "__main__":
    graph = Graph.generate_random_graph(10, 0.5)

    graph.print_graph()
    visualize_graph(graph)
    topological_sort = graph.topological_sort()
    print("Алгоритм Кана:")
    print(' '.join(map(str, topological_sort)))
