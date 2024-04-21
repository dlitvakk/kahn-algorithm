from collections import defaultdict, deque
import random
import networkx as nx
import matplotlib.pyplot as plt
import time

def visualize_graph(graph):
    # створюємо орієнтований граф (Di - directed)
    G = nx.DiGraph()
    for vertex in graph.adjacency_list:
        G.add_node(vertex)

        # Add edges from the graph's adjacency list
    for vertex, neighbors in graph.adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)

    pos = nx.spring_layout(G)

    # налаштування для відображення графу
    nx.draw(G, pos, with_labels=True, node_size=600, node_color="skyblue", font_size=10, font_weight="bold", edge_color="black", linewidths=1, alpha=0.7)
    plt.title("Graph Visualization")
    plt.show()

class Graph:
    def __init__(self, vertices, adjacency_list=None, adjacency_matrix=None):
        self.vertices = vertices
        self.adjacency_list = defaultdict(list)
        self.adjacency_matrix = adjacency_matrix

        if adjacency_list:
            self.adjacency_list = adjacency_list

            if not adjacency_matrix:
                self.adjacency_matrix = [[0] * vertices for _ in range(vertices)]
                for vertex, neighbors in adjacency_list.items():
                    for neighbor in neighbors:
                        self.adjacency_matrix[vertex][neighbor] = 1

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

def adjacency_matrix_to_list(adjacency_matrix):
    adjacency_list = {}
    num_vertices = len(adjacency_matrix)

    for i in range(num_vertices):
        adjacency_list[i] = []

        for j in range(num_vertices):
            if adjacency_matrix[i][j] == 1:
                adjacency_list[i].append(j)

    return adjacency_list

def print_graph_matrix(adjacency_list):
    num_vertices = len(adjacency_list)
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    for vertex, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            adjacency_matrix[vertex][neighbor] = 1

    for row in adjacency_matrix:
        print(row)

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
    # graph = Graph.generate_random_graph(200, 0.9)
    print(
        "Оберіть щось з наступного: \n Введіть \033[1;36m1\033[0m - задати граф матрицею суміжності, \n \033[1;36m2\033[0m - списками суміжності, \n \033[1;36m3\033[0m - згенерувати граф")
    choice = int(input("Ваш варіант: "))

    if choice == 1:
        n = int(input("Введіть кількість вершин: "))
        print("Введіть матрицю суміжності:")
        adjacency_matrix = [list(map(int, input().split())) for _ in range(n)]
        adjacency_list = adjacency_matrix_to_list(adjacency_matrix)
        graph = Graph(n, adjacency_list, adjacency_matrix)
        graph.print_graph()


    elif choice == 2:
        n = int(input("Введіть кількість вершин: "))
        print("Введіть список суміжності (для кожної вершини введіть її сусідів, розділених пробілом): ")
        adjacency_list = {}
        for i in range(n):
            neighbors = list(map(int, input(f"Вершина {i}: ").split()))
            adjacency_list[i] = neighbors
        graph = Graph(n)
        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                graph.add_edge(vertex, neighbor)

        print("Матриця суміжності: ")
        print_graph_matrix(adjacency_list)

    elif choice == 3:
        n = int(input("Введіть кількість вершин: "))
        delta = float(input("Введіть щільність графу (від 0 до 1): "))
        graph = generate_random_graph(n, delta)
        graph.print_graph()


    visualize_graph(graph)
    start_time = time.time_ns()
    topological_sort = graph.topological_sort()
    end_time = time.time_ns()
    print("Алгоритм Кана:")
    print(' '.join(map(str, topological_sort)))
    execution_time = (end_time - start_time) / 1000000000 # з наносекунд в секунди
    print(f"Час виконання:{execution_time:10f} секунд")
