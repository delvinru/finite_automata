from itertools import product, combinations
from typing import List, Set
from pprint import pprint
import argparse
from collections import defaultdict

class State:
    def __init__(self, key: list[int]) -> None:
        self.key = key

    def __hash__(self):
        res = 1
        for el in self.key:
            res <<= el
        return res

    def __eq__(self, __key: object) -> bool:
        if isinstance(__key, list):
            return self.key == __key
        elif isinstance(__key, State):
            return self.key == __key.key
        else:
            return False

    def __str__(self) -> str:
        return str(self.key)

    def __repr__(self) -> str:
        return ''.join(map(str, self.key))


class Graph:
    def __init__(self) -> None:
        self.graph: dict[State, set[State]] = defaultdict(set)

    def __str__(self) -> str:
        return str(self.graph)

    def add_vertex(self, from_node: State, to_node: State) -> None:
        self.graph[from_node].add(to_node)

    def find_connected_components(self) -> list[list[State]]:
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

    def find_strong_connected_components(self) -> list[list[State]]:
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

class FSM:
    def __init__(self, n: int, phi: list[int], psi: list[int]) -> None:
        self.n = n
        self.phi = phi
        self.psi = psi

        self.graph = Graph()
        self.table: dict[State, tuple[list[State], list[int]]] = dict()

        self._init()

    def get_connected_components(self) -> list[list[State]]:
        return self.graph.find_connected_components()

    def get_strong_connected_components(self) -> list[list[State]]:
        return self.graph.find_strong_connected_components()

    def _get_first_classes(self) -> list[list[State]]:
        dict_classes = {
            State([0, 0]): [],
            State([0, 1]): [],
            State([1, 0]): [],
            State([1, 1]): []
        }

        for key, values in self.table.items():
            first_class_key = State([values[1][0], values[1][1]])
            dict_classes[first_class_key].append(key)

        return  [x for x in dict_classes.values() if x]

    def _is_in_one_equal_class(self, value_1: State, value_2: State, old_classes: list[set[State]]) -> bool:
        perehod_value_1_from_0 = self.table[value_1][0][0]
        perehod_value_1_from_1 = self.table[value_1][0][1]
        perehod_value_2_from_0 = self.table[value_2][0][0]
        perehod_value_2_from_1 = self.table[value_2][0][1]

        check_0 = False
        check_1 = False
        for old_clazz in old_classes:
            if perehod_value_1_from_0 in old_clazz and perehod_value_2_from_0 in old_clazz:
                check_0 = True
                break

        for old_clazz in old_classes:
            if perehod_value_1_from_1 in old_clazz and perehod_value_2_from_1 in old_clazz:
                check_1 = True
                break
        
        return check_0 and check_1

    def _step_split_class(self, current_class: set[State], old_classes: list[set[State]]) -> list[set[State]]:
        new_class = [current_class[0]]
        for i in range(1, len(current_class)):
            if self._is_in_one_equal_class(current_class[0], current_class[i], old_classes):
                new_class.append(current_class[i])
        print(new_class)
        return new_class

    def _split_class(self, clazz: set[State], old_classes: list[set[State]]) -> list[set[State]]:
        split_classes = []
        current_class = clazz
        while current_class != []:
            new_class = self._step_split_class(current_class, old_classes)
            split_classes.append(new_class)
            current_class = list(set(current_class).difference(set(new_class)))
        return split_classes

    def _compute_k_clazzes(self, old_classes: list[set[State]]) -> list[set[State]]:
        k_classes: list[set[State]] = []
        for clazz in old_classes:
            split_classes = self._split_class(clazz, old_classes)
            # print(f"{split_classes=}")
            for split_class in split_classes:
                k_classes.append(split_class)
        return k_classes

    def get_equivalence_classes(self) -> dict[int, list[list[State]]]:
        equivalence_classes: dict[int, list[list[State]]] = defaultdict(list)
        # find 1-classes
        equivalence_classes[1] = self._get_first_classes()
        # find k-classes
        k = 1
        while True:
            # print(f"{equivalence_classes[k]=}")
            new_class = self._compute_k_clazzes(equivalence_classes[k])
            # print(f"{new_class=}")
            if new_class == equivalence_classes[k]:
                break

            k += 1
            equivalence_classes[k] = new_class
        return equivalence_classes

    def _init(self) -> None:
        for state in self._generate_binary_combinations(self.n):
            state = list(state)

            graph_node = State(state)
            phis: list[State]  = []
            psis: list[int] = []
            for x in [0, 1]:
                zp_phi = self._compute_zhegalkin_polynomial(input_x=x, current_state=state, coeffs=self.phi)
                zp_psi = self._compute_zhegalkin_polynomial(input_x=x, current_state=state, coeffs=self.psi)

                new_state = state[1:] + [zp_phi]

                self.graph.add_vertex(graph_node, State(new_state))

                # init table
                phis.append(State(new_state))
                psis.append(zp_psi)

            # print(f"{phis=} {psis=}")
            self.table[graph_node] = (phis, psis)

    def _compute_zhegalkin_polynomial(self, input_x: int, current_state: list[int], coeffs: list[int]) -> None:
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

    def _generate_binary_combinations(self, n: int):
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

def main(n: int, phi: list[int], psi: list[int]) -> None:
    fsm = FSM(n, phi, psi)

    # # TASK 1
    # print("TASK 1")
    # weak_components = fsm.get_connected_components()
    # print("res", weak_components)

    # # TASK 2
    # print("TASK 2")
    # strong_compoents = fsm.get_strong_connected_components()
    # print("res", strong_compoents)


    # # TASK 3
    # print("TASK 3")
    equivalence_classes = fsm.get_equivalence_classes()
    print(f"Степень различимости автомата, delta(A): {len(equivalence_classes.keys())}")
    print(f"mu(A): {len(equivalence_classes[len(equivalence_classes)])}")

    # TASK 4
    # print("TASK 4")
    

# helper functions for graph printing
def generate_tgf_format(graph: Graph) -> None:
    mapping: dict[State, int] = {}

    for i, key in enumerate(graph.graph.keys()):
        mapping.update({key: i+1})
        print(key, i+1)
    print("#")
    for key, value in graph.graph.items():
        for el in value:
            print(f"{mapping.get(key)} {mapping.get(el)}")

def generate_edges_format(graph: Graph) -> None:
    mapping: dict[State, int] = {}

    for i, key in enumerate(graph.graph.keys()):
        mapping.update({key: i+1})

    for key, value in graph.graph.items():
        for el in value:
            print(f"{mapping.get(key)} -> {mapping.get(el)}")
# end helper functions



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="solver")
    parser.add_argument("n", type=int)
    parser.add_argument("phi", type=str)
    parser.add_argument("psi", type=str)

    ns = parser.parse_args()

    main(ns.n, list(map(int, ns.phi)), list(map(int, ns.psi)))
