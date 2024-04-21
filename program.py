from collections import defaultdict, deque
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np


def visualize_graph(graph):
    # створюємо орієнтований граф (Di - directed)
    G = nx.DiGraph()
    for vertex in graph.adjacency_list:
        G.add_node(vertex)

        # додаємо вершини зі списку суміжності
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

            # створюємо матрицю суміжності, якщо її не існує
            if not adjacency_matrix:
                self.adjacency_matrix = [[0] * vertices for _ in range(vertices)]
                for vertex, neighbors in adjacency_list.items():
                    for neighbor in neighbors:
                        self.adjacency_matrix[vertex][neighbor] = 1

    # додаємо вершину до списків суміжності
    def add_edge(self, source, destination):
        self.adjacency_list[source].append(destination)

    # топологічний алгоритм сортування Кана
    # за допомогою матриці суміжності
    def topological_sort_adj_matrix(self, adjacency_matrix):
        in_degree = [0] * self.vertices
        for i in range(self.vertices):
            for j in range(self.vertices):
                in_degree[i] += adjacency_matrix[j][i]

        queue = deque()
        for i in range(self.vertices):
            if in_degree[i] == 0:
                queue.append(i)

        topological_sort = []
        while queue:
            vertex = queue.popleft()
            topological_sort.append(vertex)

            for i in range(self.vertices):
                if adjacency_matrix[vertex][i] == 1:
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)

        if len(topological_sort) != self.vertices:
            raise ValueError("Граф містить цикл ;(")

        return topological_sort

    # за допомогою списків суміжності
    def topological_sort_adj_list(self):
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

    # виводимо граф за допомогою списку суміжності
    def print_graph(self):
        # sorted_graph = self.topological_sort()
        for vertex, neighbors in sorted(self.adjacency_list.items()):
            print(f"Вершина {vertex}: {' '.join(map(str, neighbors))}")

# конвертуємо матрицю суміжності в список
def adjacency_matrix_to_list(adjacency_matrix):
    adjacency_list = {}
    num_vertices = len(adjacency_matrix)

    for i in range(num_vertices):
        adjacency_list[i] = []

        for j in range(num_vertices):
            if adjacency_matrix[i][j] == 1:
                adjacency_list[i].append(j)

    return adjacency_list

# функція виводу матриці суміжності
def print_graph_matrix(adjacency_list):
    num_vertices = len(adjacency_list)
    adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]

    for vertex, neighbors in adjacency_list.items():
        for neighbor in neighbors:
            adjacency_matrix[vertex][neighbor] = 1

    for row in adjacency_matrix:
        print(row)

# генеруємо граф за допомогою random
def generate_random_graph(n, delta):
    max_edges = (n * (n - 1)) // 2
    num_edges = int(max_edges * delta)

    graph = Graph(n)
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

    random.shuffle(edge_list)
    for edge in edge_list[:num_edges]:
        graph.add_edge(edge[0], edge[1])

    return graph

# генеруємо рандомну матрицю суміжносі графа
def generate_random_graph_matrix(n, delta):
    graph = Graph(n)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < delta:
                graph.add_edge(i, j)
                adjacency_matrix[i][j] = 1
    return graph, adjacency_matrix

if __name__ == "__main__":

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

        start_time = time.time_ns()
        topological_sort_1 = graph.topological_sort_adj_matrix(adjacency_matrix)
        end_time = time.time_ns()

        print("Алгоритм Кана:")
        print(' '.join(map(str, topological_sort_1)))

        execution_time = (end_time - start_time) / 1000000000  # з наносекунд в секунди
        print(f"Час виконання:{execution_time:10f} секунд")



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

        start_time = time.time_ns()
        topological_sort = graph.topological_sort_adj_list()
        end_time = time.time_ns()
        print("Алгоритм Кана:")
        print(' '.join(map(str, topological_sort)))
        execution_time = (end_time - start_time) / 1000000000
        print(f"Час виконання за допомогою матриці суміжності:{execution_time:10f} секунд")

    elif choice == 3:
        n = int(input("Введіть кількість вершин: "))
        delta = float(input("Введіть щільність графу (від 0 до 1): "))
        graph, adjacency_matrix = generate_random_graph_matrix(n, delta)
        graph.print_graph()

        start_time_1 = time.time_ns()
        topological_sort_1 = graph.topological_sort_adj_matrix(adjacency_matrix)
        end_time_1 = time.time_ns()

        start_time_2 = time.time_ns()
        topological_sort_2 = graph.topological_sort_adj_list()
        end_time_2 = time.time_ns()
        print("Алгоритм Кана:")
        print("Через матрицю суміжності:")
        print(' '.join(map(str, topological_sort_1)))
        print("Через списки суміжності:")
        print(' '.join(map(str, topological_sort_2)))
        execution_time_1 = (end_time_1 - start_time_1) / 1000000000  # з наносекунд в секунди
        execution_time_2 = (end_time_2 - start_time_2) / 1000000000
        print(f"Час виконання за допомогою матриці суміжності:{execution_time_1:10f} секунд")
        print(f"Час виконання за допомогою списків суміжності:{execution_time_2:10f} секунд")

    else:
        print("Something went wrong.")
        exit()

    visualize_graph(graph)


