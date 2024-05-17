from poc.main import *

def compute_memory_function(self):
    # Проверяем, вычисляли мы до этого классы эквивалентности или нет
    if len(self.equivalence_classes.keys()) == 0:
        self.get_equivalence_classes()
        self.compute_mu()
        self.compute_delta()

    fsm: FSM = deepcopy(self)

    # self.table: dict[State, tuple[list[State], list[int]]] = dict()
    # q_1: dict[State, list[dict[State, list[list[int]]]]] = dict()
    q_1: dict[State, list[dict[list[list[int]]]], State] = dict()
    # dict[State, ... State - состояние, в которое входят ребра
    # dict[State, list[int]]]
    # State - состояние, из которого выходит ребро
    # list[int] - пара значений. Первое - входное значение, второе - выходное
    # generate q_1

    # Проверяем, является ли автомат минимальным
    if fsm.mu != len(fsm.table.keys()):
        fsm._minimization()

    for key_state in fsm.table.keys():
        q_1.setdefault(key_state, list())
        for value_state, edges in fsm.table.items():
            for i in range(len(edges)):
                if edges[0][i] == key_state:
                    dicts = [el for el in q_1[key_state]]
                    if [[i], [edges[1][i]]] not in dicts:
                        q_1[key_state].append({[[i], [edges[1][i]]]: value_state})
                    # q_1[key_state].append({value_state: [[i], [edges[1][i]]]})
    # Теперь чистим повторяющиеся в классах элементы
    
    q_s = [q_1]

    # Число, при достижении которого мы говорим, что память автомата бесконечна
    max_steps = (fsm.mu * (fsm.mu - 1)) / 2
    # Проверяем, что еще есть дублирующиеся элементы и мы не достигли счетчика
    count = 1
    print(f"Вычислили Q_{count}")
    while fsm._is_equal_edges_in_q(q_s[-1]) and len(q_s) <= max_steps:
        # start compute q_2, q_3, ...
        next_q: dict[State, list[dict[list[list[int]]]], State] = dict()

        for key_state, states_edges in q_s[-1].items():
            next_q.setdefault(key_state, list())
            for i in range(len(states_edges)):
                for edge, state in states_edges[i].items():
                    for another_states_another_edges in q_1[state]:
                        for another_edge, another_state in another_states_another_edges.items():
                            dicts = [el for el in next_q[key_state]]
                            if [another_edge[0] + edge[0], another_edge[1] + edge[1]] not in dicts:
                                next_q[key_state].append({[another_edge[0] + edge[0],
                                                                    another_edge[1] + edge[1]]: another_state})
        # Теперь чистим повторяющиеся в классах элементы

        count += 1
        print(f"Вычислили Q_{count}")
        q_s.append(next_q)

# Нужно для определения памяти автомата
def _is_equal_edges_in_q(self, q: dict[State, list[dict[State, list[int]]]]) -> bool:
    unique_edges = set()
    for edges in q.values():
        for edge in edges:
            for pair in edge.keys():
                tuple_pair = tuple(pair[0] + pair[1])
                if tuple_pair in unique_edges:
                    return True
                unique_edges.add(tuple_pair)
    return False