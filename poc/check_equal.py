table = {
    1: [(1, 0), (5, 0)],
    2: [(4, 1), (1, 0)],
    3: [(1, 0), (2, 0)],
    4: [(3, 1), (5, 1)],
    5: [(8, 1), (3, 0)],
    6: [(8, 1), (8, 0)],
    7: [(3, 1), (2, 1)],
    8: [(1, 1), (5, 1)],
    9: [(1, 0), (6, 0)],
}
from collections import defaultdict
equivalence_classes: dict[int, list[list[int]]] = defaultdict(list)

# step 1: find 1 equalency classes
class FirstClassKey:
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
        elif isinstance(__key, FirstClassKey):
            return self.key == __key.key
        else:
            return False

    def __str__(self) -> str:
        return str(self.key)

    def __repr__(self) -> str:
        return str(self.key)

dict_classes: dict[FirstClassKey, list[int]] = {
    FirstClassKey([0, 0]): [],
    FirstClassKey([0, 1]): [],
    FirstClassKey([1, 0]): [],
    FirstClassKey([1, 1]): [],
}

for key, values in table.items():
    first_class_key = FirstClassKey([values[0][1], values[1][1]])
    dict_classes[first_class_key].append(key)

classes = []
for value in dict_classes.values():
    if value != []:
        classes.append(value)

equivalence_classes[1] = classes
# step 2: find 2 classess ..., lookup to 1 classes

k = 2
while True:
    previous_classes = equivalence_classes[k - 1]
    for clazz in previous_classes:

        new_classess: list[list[int]] = []

        if len(clazz) == 1:
            new_classess.append(clazz)
            continue

        for i in range(len(clazz)):
            for j in range(i, len(clazz)):
                if clazz[i] == clazz[j]:
                    continue

                is_ok = True
                # first check
                value1 = table[clazz[i]][0][0]
                value2 = table[clazz[j]][0][0]
                # print(f"{clazz[i]=} {clazz[j]=}", value1, value2)

                result = []
                for check_clazz in previous_classes:
                    result.append(all(x in check_clazz for x in [value1, value2]))

                if not any(result):
                    # print(f"ok, find shit: {clazz[i]=} {clazz[j]}=")
                    is_ok = False

                # second check
                value1 = table[clazz[i]][1][0]
                value2 = table[clazz[j]][1][0]
                # print(f"{clazz[i]=} {clazz[j]=}", value1, value2)

                result = []
                for check_clazz in previous_classes:
                    result.append(all(x in check_clazz for x in [value1, value2]))
                if not any(result):
                    # print(f"ok, find shit: {clazz[i]=} {clazz[j]=}")
                    is_ok = False

                state1 = clazz[i]
                state2 = clazz[j]
                if is_ok:
                    if len(new_classess) > 0:
                        for new in new_classess:
                            if any(x in new for x in [state1, state2]):
                                new.append(state1)
                                new.append(state2)
                    else:
                        new_classess.append([state1, state2])
                else:
                    # print('not good', new_classess)
                    if len(new_classess) > 0:
                        tmp = any(state2 in x for x in new_classess)
                        if not tmp:
                            new_classess.append([state2])
                    else:
                        new_classess.append([state1])
                        new_classess.append([state2])

        tmp = []
        for x in new_classess:
            tmp.append(list(set(x)))

        equivalence_classes[k].extend(tmp)

    current_classess = equivalence_classes[k]
    if previous_classes == current_classess:
        print("ok, stop")
        equivalence_classes.pop(k)
        break

    k += 1
    
print(equivalence_classes)