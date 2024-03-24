from itertools import product, combinations
from typing import List, Set
from pprint import pprint
import argparse
from collections import defaultdict

class Node:
    def __init__(self, value: list[int]) -> None:
        self.value = value

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, list):
            return self.value == __value
        elif isinstance(__value, Node):
            return self.value == __value.value
        else:
            return False

    def __hash__(self) -> int:
        res = 1
        for el in self.value:
            res <<= el
        return res

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)

class Graph:
    def __init__(self) -> None:
        self.graph: dict[Node, set[Node]] = defaultdict(set)

    def __str__(self) -> str:
        return str(self.graph)

    def add_vertex(self, from_node: Node, to_node: Node) -> None:
        self.graph[from_node].add(to_node)

    def find_connected_components(self) -> List[List[Node]]:
        visited = set()
        connected_components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        # Создаем копию ключей перед итерацией
        nodes_to_visit = list(self.graph.keys())

        for node in nodes_to_visit:
            if node not in visited:
                component = []
                dfs(node, component)
                connected_components.append(component)

        return connected_components

    def transpose(self) -> 'Graph':
        transposed_graph = Graph()
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                transposed_graph.add_vertex(neighbor, node)
        return transposed_graph

    def find_strong_connected_components(self) -> List[List[Node]]:
        # Первый обход графа для определения порядка посещения вершин
        visited = set()
        order = []

        def first_dfs(node):
            visited.add(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    first_dfs(neighbor)
            order.append(node)

        nodes_to_visit = list(self.graph.keys())
        for node in nodes_to_visit:
            if node not in visited:
                first_dfs(node)

        # Транспонирование графа
        transposed = self.transpose()

        # Второй обход графа в порядке, определенном первым обходом
        visited.clear()
        scc = []

        def second_dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in transposed.graph[node]:
                if neighbor not in visited:
                    second_dfs(neighbor, component)

        while order:
            node = order.pop()
            if node not in visited:
                component = []
                second_dfs(node, component)
                scc.append(component)

        return scc


def compute_zhegalkin_polynomial(input_x: int, current_state: list[int], coeffs: list[int]) -> None:
    zp = [1]  # initial state, why?
    extended_current_state = current_state + [input_x]

    for i in range(1, len(extended_current_state) + 1):
        combs = combinations(extended_current_state, i)
        for el in combs:
            product = 1
            for num in el:
                product *= num
            zp.append(product)

    for i in range(2 ** len(extended_current_state)):
        zp[i] *= coeffs[i]

    return zp.count(1) % 2

def generate_binary_combinations(n):
    def generate_combinations_helper(current_combination, index):
        if index == n:
            yield current_combination
            return
        current_combination[index] = 0
        yield from generate_combinations_helper(current_combination, index + 1)
        current_combination[index] = 1
        yield from generate_combinations_helper(current_combination, index + 1)

    initial_combination = [0] * n
    yield from generate_combinations_helper(initial_combination, 0)

def generate_tgf_format(graph: Graph) -> None:
    mapping: dict[Node, int] = {}

    for i, key in enumerate(graph.graph.keys()):
        mapping.update({key: i+1})
        print(key, i+1)
    print("#")
    for key, value in graph.graph.items():
        for el in value:
            print(f"{mapping.get(key)} {mapping.get(el)}")

def generate_edges_format(graph: Graph) -> None:
    mapping: dict[Node, int] = {}

    for i, key in enumerate(graph.graph.keys()):
        mapping.update({key: i+1})

    for key, value in graph.graph.items():
        for el in value:
            print(f"{mapping.get(key)} -> {mapping.get(el)}")

def main(n: int, phi: list[int], psi: list[int]) -> None:
    graph = Graph()

    for state in generate_binary_combinations(n):
        state = list(state)

        graph_node = Node(state)
        for x in [0, 1]:
            zp_phi = compute_zhegalkin_polynomial(input_x=x, current_state=state, coeffs=phi)
            zp_psi = compute_zhegalkin_polynomial(input_x=x, current_state=state, coeffs=psi)

            new_state = state[1:] + [zp_phi]
            graph.add_vertex(graph_node, Node(new_state))

    # # TASK 1
    # print("TASK 1")
    # weak_components = graph.find_connected_components()

    # # TASK 2
    # print("TASK 2")
    # strong_compoents = graph.find_strong_connected_components()

    # TASK 3
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="solver")
    parser.add_argument("n", type=int)
    parser.add_argument("phi", type=str)
    parser.add_argument("psi", type=str)

    ns = parser.parse_args()

    main(ns.n, list(map(int, ns.phi)), list(map(int, ns.psi)))
