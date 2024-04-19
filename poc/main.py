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
        
    def __lt__(self, __key: object) -> bool:
        return self.key > __key

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

    # изменил возвращаемое значение с list[set[State]] на list[State]
    # ПО идее это правильнее
    def _step_split_class(self, current_class: set[State], old_classes: list[set[State]]) -> list[State]:
        new_class = [current_class[0]]
        for i in range(1, len(current_class)):
            if self._is_in_one_equal_class(current_class[0], current_class[i], old_classes):
                new_class.append(current_class[i])
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

    # Изменил dict[int, list[list[State]]] на dict[int, list[set[State]]]
    # Вроде так правильнее
    def get_equivalence_classes(self) -> dict[int, list[set[State]]]:
        equivalence_classes: dict[int, list[set[State]]] = defaultdict(list)
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

    def combos(self, iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)

    def _compute_zhegalkin_polynomial(self, input_x: int, current_state: list[int], coeffs: list[int]) -> None:
        zp = [1]  # initial state, why?
        extended_current_state = current_state + [input_x]

        for i in range(1, len(extended_current_state) + 1):
            combs = combinations(extended_current_state, i)
            # combs = self.combos(extended_current_state, i)
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

    # Нужно для определения памяти автомата
    def _is_equal_edges_in_q_s(self, q: list[dict[State, list[dict[State, list[int]]]]]) -> bool:
        unique_edges = set()
        edges_count = 0
        for edges in q.values():
            for edge in edges:
                for pair in edge.values():
                    tuple_pair = tuple(pair[0] + pair[1])
                    unique_edges.add(tuple_pair)
                    edges_count += 1
        return len(unique_edges) != edges_count
    
    def compute_memory_function(self):
        # self.table: dict[State, tuple[list[State], list[int]]] = dict()
        q_1: dict[State, list[dict[State, list[list[int]]]]] = dict()
        # dict[State, ... State - состояние, в которое входят ребра
        # dict[State, list[int]]]
        # State - состояние, из которого выходит ребро
        # list[int] - пара значений. Первое - входное значение, второе - выходное
        # generate q_1
        for key_state in self.table.keys():
            for value_state, edges in self.table.items():
                for i in range(len(edges)):
                    if edges[0][i] == key_state:
                        q_1.setdefault(key_state, list())
                        q_1[key_state].append({value_state: [[i], [edges[1][i]]]})
        q_s = [q_1]
        while not self._is_equal_edges_in_q_s(q_s[-1]):
            # start compute q_2, q_3, ...
            # Еще по идее выход из цикла возможен при каком-то s (большом)
            next_q: dict[State, list[dict[State, list[int]]]] = dict()
            for key_state, states_edges in q_s[-1].items():
                for i in range(len(states_edges)):
                    next_q.setdefault(key_state, list())
                    for state, edge in states_edges[i].items():
                        for another_states_another_edges in q_1[state]:
                            for another_state, another_edge in another_states_another_edges.items():
                                next_q[key_state].append({another_state: [another_edge[0] + edge[0],
                                                                          another_edge[1] + edge[1]]})
            q_s.append(next_q)
        for i, q in enumerate(q_s):
            print(f"q_{i + 1} = {q}\n")

    # Тут вычисляем выходную последовательность автомата
    def _compute_u(self, init_state: list[int]) -> list[int]:
        u: list[int] = []
        current_state = State(init_state)
        for i in range(2 ** self.n):
            # На вход подаем только нули по ТЗ
            u.append(self.table[current_state][1][0])
            current_state = self.table[current_state][0][0]
        return u
    
    # Тут считаем минимальный многочлен
    def _berlekamp_massey(self, u: list[int]) -> list[int]:
        # Список отрезков таблицы
        segments: list[list[int]] = []
        # Список многочленов таблицы
        polynomials: list[list[int]] = []
        # Список количества нулей таблицы
        zeros_count: list[int] = []

        # функция для подсчета количества нулей в начале отрезка
        def count_of_leading_zeros(segment: list[int]) -> int:
            count = 0
            for el in segment:
                if el != 0:
                    return count
                else:
                    count += 1
            return count
        # шаг 0 из алгоритма
        zeros_count.append(count_of_leading_zeros(u))
        polynomials.append([1])
        if zeros_count[-1] == len(u):
            return polynomials[0]
        segments.append(u)

        # Шаги 1 <= s <= (l - 1)
        for i in range(1, 2 ** self.n - 1):
            # Список отрезков u_i, сначала добавляем (u_i)^0
            current_segments: list[list[int]] = [segments[-1][1:]]
            # Список многочленов f_i, сначала добавляем (f_i)^0
            current_polynomials: list[list[int]] = [[0] + polynomials[-1]]
            # Список нулей k_i, сначала добавляем (k_i)^0
            current_zeros_count: list[int] = [zeros_count[-1] - 1] if zeros_count[-1] != 0 else [0]

            # Случай 3
            while count_of_leading_zeros(current_segments[-1]) != len(current_segments[-1]) and \
                                         current_zeros_count[-1] in zeros_count:
                # compute "t" from algorithm
                t = zeros_count.index(current_zeros_count[-1])
                # compute r in GF(2)
                if segments[t][zeros_count[t]] == 0:
                    raise Exception("Alarm! 0 when computing u_t(k_t)^(-1)")
                r = current_segments[-1][zeros_count[t]] * segments[t][zeros_count[t]]

                # Для сложения значений двух отрзков разной длины
                def compute_sum_of_two_lists(first_list: list[int], second_list: list[int]) -> list[int]:
                    length = min(len(first_list), len(second_list))
                    new_list = []
                    for i in range(length):
                        new_list.append(first_list[i] ^ second_list[i])
                    return new_list
                
                # Для сложения значений двух многочленов разной длины
                def compute_sum_of_two_polynomials(first_polynomial: list[int], second_polynomial: list[int]) -> list[int]:
                    length = max(len(first_polynomial), len(second_polynomial))
                    if len(first_polynomial) != max:
                        first_polynomial = first_polynomial + [0 for i in range(length - len(first_polynomial))]
                    if len(second_polynomial) != max:
                        second_polynomial = second_polynomial + [0 for i in range(length - len(second_polynomial))]
                    new_list = []
                    for i in range(length):
                        new_list.append(first_polynomial[i] ^ second_polynomial[i])
                    return new_list
                
                current_segments.append(compute_sum_of_two_lists(current_segments[-1],
                                                                 [i * r for i in segments[t]]))
                current_polynomials.append(compute_sum_of_two_polynomials(current_polynomials[-1],
                                                                    [i * r for i in polynomials[t]]))
                current_zeros_count.append(count_of_leading_zeros(current_segments[-1]))
            
            # Случай 1
            if count_of_leading_zeros(current_segments[-1]) == len(current_segments[-1]):
                zeros_count.append(current_zeros_count[-1])
                polynomials.append(current_polynomials[-1])
                segments.append(current_segments[-1])
                return polynomials[-1]
            
            # Случай 2
            segments.append(current_segments[-1])
            polynomials.append(current_polynomials[-1])
            zeros_count.append(current_zeros_count[-1])

        # Шаг l
        segments.append([])
        polynomials.append([0] + polynomials[-1])
        return polynomials[-1]

    # Вызываем для вычсиления минимального многочлена выходного отрезка
    def compute_min_polynomial(self, init_state: list[int]) -> None:
        u = self._compute_u(init_state)
        min_polynomial = self._berlekamp_massey(u)
        return min_polynomial

def main(n: int, phi: list[int], psi: list[int], init_state: list[int]) -> None:
    fsm = FSM(n, phi, psi)
    # pprint(fsm.table)

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
    # equivalence_classes = fsm.get_equivalence_classes()
    # print(equivalence_classes)
    # print(f"Степень различимости автомата, delta(A): {len(equivalence_classes.keys())}")
    # print(f"mu(A): {len(equivalence_classes[len(equivalence_classes)])}")

    # TASK 4
    # print("TASK 4")
    # fsm.compute_memory_function()

    # TASK 5
    min_polynomial = fsm.compute_min_polynomial(init_state)
    print(f"Min polynomial: {min_polynomial}")
    print(f"Linear complexity: {len(min_polynomial)}")
    

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
    parser.add_argument("init_state", type=str)

    ns = parser.parse_args()

    main(ns.n, list(map(int, ns.phi)), list(map(int, ns.psi)), list(map(int, ns.init_state)))


# example to start program
# python3 poc/main.py 7 0100000000000000010000010000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0110000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0100101
# 7 0110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000