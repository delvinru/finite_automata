from collections import defaultdict
from itertools import combinations

# table = {
#     1: [(3, 2), (0, 1)],
#     2: [(4, 1), (0, 1)],
#     3: [(2, 4), (1, 1)],
#     4: [(1, 4), (1, 1)],
#     5: [(6, 1), (0, 1)],
#     6: [(5, 5), (1, 1)],
# }

# table = {
#     1: [(1, 5), (0, 0)],
#     2: [(4, 1), (1, 0)],
#     3: [(1, 2), (0, 0)],
#     4: [(3, 5), (1, 1)],
#     5: [(8, 3), (1, 0)],
#     6: [(8, 8), (1, 0)],
#     7: [(3, 2), (1, 1)],
#     8: [(1, 5), (1, 1)],
#     9: [(1, 6), (0, 0)],
# }

#        0        1
# table = {
#     1: [(1, 4), (0, 1)],
#     2: [(1, 5), (0, 1)],
#     3: [(5, 1), (0, 1)],
#     4: [(3, 4), (1, 1)],
#     5: [(2, 5), (1, 1)],
# }

table = {
    (0, 0): [[(0, 0), (0, 1)], [0, 0]],
    (0, 1): [[(1, 0), (1, 1)], [0, 1]],
}

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

    # find k-classes

    k = 2
    while True:
        prev_classes = equivalence_classes[k - 1]

        for clazz in prev_classes:
            new_classes: list[set] = []

            variants = list(combinations(clazz, 2))
            for pair in variants:
                # print(f"{clazz=} {pair=}")
                key_1, key_2 = pair
                first_h, second_h = table[key_1][0], table[key_2][0]

                # x can be 0 or 1 always
                # check that in same clazzes
                for idx in range(2):
                    check = {first_h[idx], second_h[idx]}
                    # ok, found not equalevent classes, split
                    if not any(check.issubset(x) for x in prev_classes):
                        # create {key_n} if not exists
                        for new_class in new_classes:
                            if key_1 in new_class:
                                break
                        else:
                            new_classes.append({key_1})

                        for new_class in new_classes:
                            if key_2 in new_class:
                                break
                        else:
                            new_classes.append({key_2})
                        break
                else:
                    insert_pair = {key_1, key_2}
                    for new_class in new_classes:
                        if new_class & insert_pair:
                            new_class |= insert_pair
                            break
                    else:
                        new_classes.append(insert_pair)

            if variants == []:
                new_classes.append(clazz)

            tmp = [list(x) for x in new_classes]
            equivalence_classes[k].extend(tmp)

        # print(f"{k=} {equivalence_classes[k]}")
        if equivalence_classes[k] == prev_classes:
            equivalence_classes.pop(k)
            break

        k += 1

    return equivalence_classes

from pprint import pprint
pprint(dict(get_equivalence_classes(table)))