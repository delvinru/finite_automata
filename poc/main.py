from itertools import product, combinations
from typing import List, Set
from pprint import pprint
import argparse
from collections import defaultdict
from copy import deepcopy
import gc
from dataclasses import dataclass

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

class EdgePair:
    first_component: list[int]
    second_component: list[int]

    state: State

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

        # классы эквивалентноси теперь храним в объеке
        self.equivalence_classes: dict[int, list[set[State]]] = dict()
        # степень различимости
        self.delta: int = 0
        # приведенный вес автомата
        self.mu: int = 0

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
    def get_equivalence_classes(self) -> None:
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

        self.equivalence_classes = equivalence_classes
        return None

    def compute_delta(self) -> None:
        # Проверяем, что уже вычислили классы эквивалентности
        if len(self.equivalence_classes.keys()) == 0:
            self.get_equivalence_classes()
        self.delta = len(self.equivalence_classes.keys())

    def compute_mu(self) -> None:
        # Проверяем, что уже вычислили классы эквивалентности
        if len(self.equivalence_classes.keys()) == 0:
            self.get_equivalence_classes()
        self.mu = len(self.equivalence_classes[len(self.equivalence_classes)])

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
    def _is_equal_edges_in_q(self, q: dict[State, list[dict[State, list[list[int]]]]]) -> bool:
        unique_edges: set[tuple[list[int], list[int]]] = set()
        for edges in q.values():
            for edge in edges:
                for pair in edge.values():
                    tuple_pair = tuple(pair[0] + pair[1])
                    if tuple_pair in unique_edges:
                        return True
                    unique_edges.add(tuple_pair)
        return False

    # Добавить проверку на минимальность автомата и построение нового в случае чего
    def compute_memory_function(self, fw):
        # Проверяем, вычисляли мы до этого классы эквивалентности или нет
        if len(self.equivalence_classes.keys()) == 0:
            self.get_equivalence_classes()
            self.compute_mu()
            self.compute_delta()

        fsm: FSM = deepcopy(self)

        # self.table: dict[State, tuple[list[State], list[int]]] = dict()
        q_1: dict[State, list[dict[State, list[list[int]]]]] = defaultdict(list)
        # dict[State, ... State - состояние, в которое входят ребра
        # dict[State, list[int]]]
        # State - состояние, из которого выходит ребро
        # list[int] - пара значений. Первое - входное значение, второе - выходное
        # generate q_1

        # Проверяем, является ли автомат минимальным
        if fsm.mu != len(fsm.table.keys()):
            fsm._minimization()

        # all_edges_list: set[tuple[int]] = set()
        # count = 0
        for key_state in fsm.table.keys():
            edges_list: list[list[int]] = list()
            for value_state, edges in fsm.table.items():
                for i in range(len(edges)):
                    tmp_list = [[i], [edges[1][i]]]
                    if edges[0][i] == key_state and tmp_list not in edges_list:
                        q_1[key_state].append({value_state: [[i], [edges[1][i]]]})
                        edges_list.append(tmp_list)
                        # all_edges_list.add(tmp_list)
                        # count += 1
        q_s = [q_1]

        # Число, при достижении которого мы говорим, что память автомата бесконечна
        max_steps = (fsm.mu * (fsm.mu - 1)) / 2
        # Проверяем, что еще есть дублирующиеся элементы и мы не достигли счетчика

        print(f"[+] Вычислили q_{len(q_s)}")
        while fsm._is_equal_edges_in_q(q_s[-1]) and len(q_s) <= max_steps:
            next_q: dict[State, list[dict[State, list[list[int]]]]] = defaultdict(list)

            for state, edges in q_s[-1].items():
                edges_list: set[tuple[int]] = set()
                
                for edge in edges:
                    from_state, values = list(edge.items())[0]

                    # TODO: SOME MAGIC SHIT, LOOK UP AT ME, DON'T IGNORE, DON'T IGNORE
                    if from_state not in q_1:
                        continue

                    for edges_from_state in q_1[from_state]:
                        for another_state, another_edge in edges_from_state.items():
                            tmp_set = tuple(another_edge[0] + values[0] + another_edge[1] + values[1])
                            if tmp_set not in edges_list:
                                next_q[state].append({another_state: [another_edge[0] + values[0], another_edge[1] + values[1]]})
                                edges_list.add(tmp_set)
                                # count += 1
                                # all_edges_list.add(tmp_set)                

            # sanches code (shit code)
            # for key_state, states_edges in q_s[-1].items():
            #     edges_list: set[tuple[int]] = set()

            #     for i in range(len(states_edges)):
            #         for state, edge in states_edges[i].items():
            #             for another_states_another_edges in q_1[state]:
            #                 for another_state, another_edge in another_states_another_edges.items():
            #                     tmp_set = tuple(another_edge[0] + edge[0] + another_edge[1] + edge[1])
            #                     if tmp_set not in edges_list:
            #                         next_q[key_state].append({another_state: [another_edge[0] + edge[0], another_edge[1] + edge[1]]})
            #                         edges_list.add(tmp_set)                

            q_s.append(next_q)
            print(f"[+] Вычислили q_{len(q_s)}")

        print("[+] Нашли все q_s")

        if len(q_s) > max_steps:
            # print("Память автомата бесконечна")
            fw.write("Память автомата бесконечна\n")
        else:
            for i, q in enumerate(q_s):
                # print(f"q_{i + 1}:")
                # print(q)
                # fw.write(f"q_{i + 1}:\n")
                for s, elements in q.items():
                    fw.write(f"q_{i + 1}({s}): ")
                    fw.write(" |_| ".join([" ".join((str(l) for l in element.values())) for element in elements]))
                    # for element in elements:
                        # for l in element.values():
                        #     fw.write(f"{l} |_| ")
                    fw.write("\n\n")
                fw.write("\n")
                # fw.write(f"{q}\n")
            memory = len(q_s)
            # print(f"Память автомата конечна: m(A)={memory}")
            fw.write(f"Память автомата конечна: m(A)={memory}\n")

            print("[+] Записали q_s в файл")

            gc.collect()

            memory_value_vector = fsm._get_memory_value_vector(q_s[-1], memory)
            print("[+] Вычислили вектор y_i")
            # print(memory_value_vector)
            fsm._convert_memory_vector_to_int(memory_value_vector)
            # print(memory_value_vector)
            print("[+] Подставили вместо * нули в векторе y_i")
            # Может нужно, но не отрабатывает
            memory_function_coefs = fsm._get_memory_function_coefs(memory_value_vector)
            # print("Коэфы",memory_function_coefs)
            print("[+] Нашли коэффициенты перед слагаемыми МЖ функции y_i")
            # print(self._get_memory_function_coefs_str(memory_function_coefs))
            fw.write(f"Функция памяти автомата: {self._get_memory_function_coefs_str(memory_function_coefs)}\n\n")
            print("[+] Записали коэффициенты перед слагаемыми МЖ функции y_i в файл")

    def _minimization(self) -> None:
        # Проверяем, вычисляли мы до этого классы эквивалентности или нет
        if len(self.equivalence_classes.keys()) == 0:
            self.get_equivalence_classes()
            self.compute_delta()

        for set_states in self.equivalence_classes[self.delta]:
            len_set = len(set_states)
            if len_set > 1:
                equivalent_state = set_states[0]
                for i in range(1, len_set):
                    for table_tuple in self.table.values():
                        for j, state in enumerate(table_tuple[0]):
                            if state == set_states[i]:
                                table_tuple[0][j] = equivalent_state
                    del self.table[set_states[i]]

        self.get_equivalence_classes()
        self.compute_delta()
        self.compute_mu()

        # print(self.mu)
        # print(self.delta)
        # print(self.equivalence_classes[self.delta])

    def _get_memory_value_vector(self, q_last: dict[State, list[dict[State, list[list[int]]]]], memory: int) -> list[int]:
        memory_value_vector: list[int] = []
        large_table_dict: dict[tuple[tuple[int]], list[int]] = dict()

        large_table_dict_sub_keys: list[list[int]] = []
        for comb in self._generate_binary_combinations(memory):
            state = list(comb)
            large_table_dict_sub_keys.append(state)

        for comb_i in large_table_dict_sub_keys:
            for comb_j in large_table_dict_sub_keys:
                key = tuple([tuple(comb_i), tuple(comb_j)])
                large_table_dict[key] = [None, None]

        for main_state, elements in q_last.items():
            for element in elements:
                for vectors in element.values():
                    current_vector = tuple([tuple(vectors[0]), tuple(vectors[1])])
                    large_table_dict[current_vector][0] = self.table[main_state][1][0]
                    large_table_dict[current_vector][1] = self.table[main_state][1][1]

        for elements in large_table_dict.values():
            memory_value_vector.append(elements[0])
            memory_value_vector.append(elements[1])

        return memory_value_vector
    
    def _convert_memory_vector_to_int(self, memory_vector: list[int]) -> None:
        for i in range(len(memory_vector)):
            if memory_vector[i] == None:
                memory_vector[i] = 0

    # another zhegalin polynomial implementation
    def _get_memory_function_coefs(self, seq: list[int]) -> list[int]:
        # Ломается на больших значениях, гооооооол!
        # vector_len = len(vector)
        # pascal_triangle: list[list[int]] = [[0 for j in range(vector_len)] for i in range(vector_len)]
        # for i in range(1, vector_len + 1):
        #     pascal_triangle[i - 1][0] = vector[vector_len - i]
        #     for j in range(1, i):
        #         pascal_triangle[i - 1][j] = pascal_triangle[i - 2][j - 1] ^ pascal_triangle[i - 1][j - 1]
        # return pascal_triangle[vector_len - 1]
        # magavales realization

        seq_left = []
        seq_right = []
        seq_out = []
        temp1 = []
        temp2 = []

        seq_left = seq[:len(seq)//2]
        seq_right = [(seq[i] + seq[i + len(seq)//2]) % 2 for i in range(len(seq)//2)]

        if len(seq) == 2:
            seq_out.extend([seq_left[0], seq_right[0]])
            return seq_out

        temp1 = self._get_memory_function_coefs(seq_left)
        temp2 = self._get_memory_function_coefs(seq_right)

        seq_out.extend(temp1)
        seq_out.extend(temp2)

        return seq_out
    
    def _get_memory_function_coefs_str(self, vector: list[int]) -> str:
        import math

        vector_string = ""
        if vector[0] == 1:
           vector_string += "1 + "

        length_of_vector = int(math.log(len(vector), 2))
        print(f"{len(vector)=} {length_of_vector=} {math.log(len(vector), 2)=}")

        # for i in range(1, len(vector)):
        #     if vector[i] == 1:
        #         bin_value = bin(i)[2:].zfill((2 * self.n) + 1)
        #         for coef, bit in enumerate(bin_value):
        #             if int(bit) == 1:
        #                 coef_str = ""

        #                 # compute coef_str
        #                 if coef + 1 < self.n + 1:
        #                     coef_str += "x_(i"
        #                     coef_str += f"-{self.n - coef})"
        #                 elif coef + 1 > self.n + 1:
        #                     coef_str += "y_(i"
        #                     coef_str += f"-{2 * self.n - coef + 1})"
        #                 else:
        #                     coef_str += "x_i"

        #                 vector_string += coef_str
        #         vector_string += " + "

        for i in range(1, len(vector)):
            if vector[i] == 1:
                bin_value = bin(i)[2:].zfill(length_of_vector)
                for coef, bit in enumerate(bin_value):
                    if int(bit) == 1:
                        coef_str = ""

                        # compute coef_str
                        if coef < length_of_vector // 2:
                            coef_str += "x_(i"
                            coef_str += f"-{(length_of_vector // 2) - coef})"
                        elif coef > length_of_vector // 2:
                            coef_str += "y_(i"
                            coef_str += f"-{length_of_vector - coef})"
                        else:
                            coef_str += "x_i"

                        vector_string += coef_str
                vector_string += " + "

        vector_string = vector_string.removesuffix(" + ")
        return vector_string

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
                    if len(first_polynomial) != length:
                        first_polynomial = first_polynomial + [0 for i in range(length - len(first_polynomial))]
                    if len(second_polynomial) != length:
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
    # pprint(fsm.graph.graph)

    return_code = 0

    with open("test.txt", "w") as fw:
        # Add n to file
        fw.write(f"n = {fsm.n}\n")
        print("[+] Записали n в файл")

        # Add phi to file
        fw.write("phi: ")
        for el in fsm.phi:
            fw.write(f"{el}")
        fw.write("\n")
        print("[+] Записали phi в файл")

        # Add psi to file
        fw.write("psi: ")
        for el in fsm.psi:
            fw.write(f"{el}")
        fw.write("\n")
        print("[+] Записали psi в файл")

        # TASK 1
        # print("TASK 1")
        # weak_components = fsm.get_connected_components()
        # for weak_component in weak_components:
        #     print(weak_component)
        # print(f"Количество компонент связности: {len(weak_components)}")
        # print()
        fw.write("\nTASK 1\n")
        weak_components = fsm.get_connected_components()
        print("[+] Сделали 1 лабу")
        for i, weak_component in enumerate(weak_components):
            fw.write(f"Компонента связности {i+1}: ")
            fw.write(f"{weak_component}\n\n")
        fw.write(f"Количество компонент связности: {len(weak_components)}\n\n")
        print("[+] Записали 1 лабу в файл")

        # TASK 2
        # print("TASK 2")
        # strong_components = fsm.get_strong_connected_components()
        # for strong_component in strong_components:
        #     print(strong_component)
        # print(f"Количество компонент сильной связности: {len(strong_components)}")
        # print()
        fw.write("\nTASK 2\n")
        strong_components = fsm.get_strong_connected_components()
        print("[+] Сделали 2 лабу")
        for i, strong_component in enumerate(strong_components):
            fw.write(f"Компонента сильной связности {i+1}: ")
            fw.write(f"{strong_component}\n\n")
        fw.write(f"Количество компонент сильной связности: {len(strong_components)}\n\n")
        print("[+] Записали 2 лабу в файл")

        # TASK 3
        # print("TASK 3")
        # fsm.get_equivalence_classes()
        # fsm.compute_delta()
        # fsm.compute_mu()
        # for c in fsm.equivalence_classes.values():
        #     print(c)
        # print(f"Степень различимости автомата, delta(A): {fsm.delta}")
        # print(f"mu(A): {fsm.mu}")
        fw.write("TASK 3\n")
        fsm.get_equivalence_classes()
        fsm.compute_delta()
        fsm.compute_mu()
        print("[+] Сделали 3 лабу")
        for i, equivalence_class in enumerate(fsm.equivalence_classes.values()):
            fw.write(f"k = {i+1}:\n")
            fw.write(" |_| ".join(str(state) for state in equivalence_class))
            fw.write("\n\n")
        fw.write(f"Степень различимости автомата, delta(A): {fsm.delta}\n")
        fw.write(f"Приведенный вес автомата, mu(A): {fsm.mu}\n\n")
        print("[+] Записали 3 лабу в файл")

        # TASK 4
        # print("TASK 4")
        # fsm.compute_memory_function()
        fw.write("\nTASK 4\n")
        try:
            fsm.compute_memory_function(fw)
            print("[+] Записали 4 лабу в файл")
        except KeyboardInterrupt:
            fw.write("Слишком большое потребление ресурсов, вычисление остановлено\n")
            fw.flush()
            return_code = 1
            print("[!] got keyboard interrupt, skipping 4 number")

        # TASK 5
        # print("TASK 5")
        # min_polynomial_coefs = fsm.compute_min_polynomial(init_state)
        # print(f"Минимальный многочлен: ", end="")

        # if min_polynomial_coefs[0] == 1:
        #     print("1 + ", end='')
        # for i in range(1, len(min_polynomial_coefs)):
        #     if min_polynomial_coefs[i] == 1:
        #         print(f"x^{i} + ", end='')

        # print(f"Линейная сложность: {len(min_polynomial_coefs)}")
        fw.write("\nTASK 5\n")
        fw.write(f"Начальное состояние: ")
        for el in init_state:
            fw.write(f"{el}")
        fw.write("\n")
        min_polynomial_coefs = fsm.compute_min_polynomial(init_state)
        print("[+] Сделали 5 лабу")

        min_polynomial = ""
        if min_polynomial_coefs[0] == 1:
            min_polynomial += "1 + "
        for i in range(1, len(min_polynomial_coefs)):
            if min_polynomial_coefs[i] == 1:
                min_polynomial += f"x^{i} + "

        min_polynomial = min_polynomial.removesuffix(" + ")
        fw.write(f"Минимальный многочлен: {min_polynomial}\n")

        fw.write(f"Линейная сложность: {len(min_polynomial_coefs)}\n\n")
        print("[+] Записали 5 лабу в файл")

        exit(return_code)
    

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

    fd = open("graph.edges", "w")

    for i, key in enumerate(graph.graph.keys()):
        mapping.update({key: i+1})

    for key, value in graph.graph.items():
        for el in value:
            print(f"{mapping.get(key)} -> {mapping.get(el)}", file=fd)
# end helper functions



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="solver")
    parser.add_argument("n", type=int)
    parser.add_argument("phi", type=str)
    parser.add_argument("psi", type=str)
    parser.add_argument("init_state", type=str)

    ns = parser.parse_args()

    main(ns.n, list(map(int, ns.phi)), list(map(int, ns.psi)), list(map(int, ns.init_state)))


# python3 main.py 7 0110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0000000
# python3 main.py 8 01000000010010000000000000000000010000000000000000000000000000000000000000010000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 00001000000100000000000100000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 00000000
# python3 main.py 7 0100000000000000010000010000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0110000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 0110010
# python3 main.py 2 00010000 01000011 00
    
# python3 main.py 7 0000001100000011000000110000001100000011111111000000001111111100000000110000001111111100111111000000001111111100111111000000001111111100111111001111110011111100111111000000001111111100000000111111110011111100000000110000001111111100000000110000001111111100 0101010101010101010101010101010101010101010101010101010101010101101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010100101010101010101010101010101010101010101010101010101010101010101 0001000