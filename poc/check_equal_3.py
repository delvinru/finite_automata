from collections import defaultdict


# table = {
#     1: [(3, 2), (0, 1)],
#     2: [(4, 1), (0, 1)],
#     3: [(2, 4), (1, 1)],
#     4: [(1, 4), (1, 1)],
#     5: [(6, 1), (0, 1)],
#     6: [(5, 5), (1, 1)],
# }

table = {
    1: [(1, 5), (0, 0)],
    2: [(4, 1), (1, 0)],
    3: [(1, 2), (0, 0)],
    4: [(3, 5), (1, 1)],
    5: [(8, 3), (1, 0)],
    6: [(8, 8), (1, 0)],
    7: [(3, 2), (1, 1)],
    8: [(1, 5), (1, 1)],
    9: [(1, 6), (0, 0)],
}

# table = {
#     1: [(1, 4), (0, 1)],
#     2: [(1, 5), (0, 1)],
#     3: [(5, 1), (0, 1)],
#     4: [(3, 4), (1, 1)],
#     5: [(2, 5), (1, 1)],
# }

def get_equivalence_classes(table: dict[int, list[tuple[int]]]) -> None:
    equivalence_classes: dict[int, list[list[int]]] = defaultdict(list)

    # find 1-classes
    keys = list(table.keys())
    seen = set()
    for i in range(len(keys)):
        key_1 = keys[i]
        first_state = table[key_1]

        check_classes = set()
        for j in range(i + 1, len(keys)):
            key_2 = keys[j]
            if key_2 in seen:
                continue

            second_state = table[key_2]

            first_h, first_f = first_state[0], first_state[1]
            second_h, second_f = second_state[0], second_state[1]

            if first_f == second_f:
                seen.add(key_2)
                check_classes.add(key_1)
                check_classes.add(key_2)

        if check_classes:
            equivalence_classes[1].append(list(check_classes))
    print(equivalence_classes)

    return equivalence_classes

def in_one_equal_clazz(value_1: int, value_2: int, old_clazzes: list[list[int]], table: dict[int, list[tuple[int]]]) -> bool:
    perehod_value_1_from_0 = table[value_1][0][0]
    perehod_value_1_from_1 = table[value_1][0][1]
    perehod_value_2_from_0 = table[value_2][0][0]
    perehod_value_2_from_1 = table[value_2][0][1]

    check_0 = False
    check_1 = False
    for old_clazz in old_clazzes:
        if perehod_value_1_from_0 in old_clazz and perehod_value_2_from_0 in old_clazz:
            check_0 = True
            break

    for old_clazz in old_clazzes:
        if perehod_value_1_from_1 in old_clazz and perehod_value_2_from_1 in old_clazz:
            check_1 = True
            break
    
    return check_0 and check_1

def step_split_clazz(current_clazz: list[int], old_clazzes: list[list[int]], table: dict[int, list[tuple[int]]]) -> list[int]:
    new_clazz = [current_clazz[0]]
    for i in range(1, len(current_clazz)):
        if in_one_equal_clazz(current_clazz[0], current_clazz[i], old_clazzes, table):
            new_clazz.append(current_clazz[i])
    return new_clazz

def split_clazz(clazz: list[int], old_clazzes: list[list[int]], table: dict[int, list[tuple[int]]]) -> list[list[int]]:
    split_clazzes = []
    current_clazz = clazz
    while current_clazz != []:
        new_clazz = step_split_clazz(current_clazz, old_clazzes, table)
        split_clazzes.append(new_clazz)
        current_clazz = list(set(current_clazz).difference(set(new_clazz)))
    return split_clazzes

def compute_k_clazzes(old_clazzes: list[list[int]], table: dict[int, list[tuple[int]]]) -> list[list[int]]:
    k_clazzes: list[list[int]] = []
    for clazz in old_clazzes:
        split_clazzes = split_clazz(clazz, old_clazzes, table)
        for split_clazz_ in split_clazzes:
            k_clazzes.append(split_clazz_)
    return k_clazzes

def compute_equivalence_classes(table) -> list[list[int]]:
    clazzes = []
    first_class = dict(get_equivalence_classes(table))[1]
    clazzes.append(first_class)
    while True:
        new_class = compute_k_clazzes(clazzes[len(clazzes) - 1], table)
        if new_class == clazzes[len(clazzes) - 1]:
            break
        clazzes.append(new_class)
    return clazzes
    
clazzes = compute_equivalence_classes(table)

from pprint import pprint
pprint(clazzes)