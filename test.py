from typing import List

def find_combinations(arr, k):
    result = []
    n = len(arr)
    
    # Вспомогательная функция для поиска комбинаций
    def backtrack(start, combination):
        if len(combination) == k:
            result.append(combination[:])
            return
        for i in range(start, n):
            combination.append(arr[i])
            backtrack(i + 1, combination)
            combination.pop()

    backtrack(0, [])
    return result

def compute_ZP(input_x: int,
               current_state: List[int],
               coefs: List[int]) -> List[int]:
    ZP = []

    extended_current_state = []
    for el in current_state:
        extended_current_state.append(el)
    extended_current_state.append(input_x)
    
    ZP.append(1)  # начальное значение
    
    for i in range(1, len(extended_current_state) + 2):
        combinations = find_combinations(extended_current_state, i)
        for el in combinations:
            # Произведение переменных в текущей комбинации
            product = 1
            for num in el:
                product *= num
            ZP.append(product)

    for i in range(2 ** len(extended_current_state)):
        ZP[i] = ZP[i] * coefs[i]
    # print(ZP, ZP.count(1))
    return ZP.count(1) % 2

# Пример использования функции
# input_x = 1
# current_state = [1, 1]
# n = 2
# phi = [0, 1, 0, 1, 0, 0, 0, 0]
# psi = [0, 0, 0, 0, 1, 0, 1, 0]
# phi_value = compute_ZP(input_x, current_state, phi)
# new_state = current_state[1:] + [phi_value]

n = 2
phi = [0, 1, 0, 1, 0, 0, 0, 0]
psi = [0, 0, 0, 0, 1, 0, 1, 0]

from itertools import product
all_states = product(range(n), repeat=n)

for state in all_states:
    for x in range(2):
        t = compute_ZP(input_x=x, current_state=list(state), coefs=phi)
        print(f"{state=} {x=} new_state={list(state)[1:] + [t]}", end="")
        psi_small = compute_ZP(input_x=x, current_state=list(state), coefs=psi)
        print(f"{psi_small=}", end="|")
    print()


# current_state = [0, 0]
# for x in range(2):
#     tmp = compute_ZP(input_x=x, current_state, phi)
#     current_state = 

# print(f"Input x: {input_x}")
# print(f"Current state: {current_state}")
# print(f"Phi value: {phi_value}")
# print(f"Phi: {phi}")
# print(f"New state: {new_state}")

# def compute_ZP(input_x: str, state: List[str]) -> List[List[str]]:
#     ZP = []
#     state.append(input_x)
#     ZP.append([1])  # начальное значение
    
#     for i in range(1, len(state) + 2):
#         combinations = find_combinations(state, i)
#         for el in combinations:
#             # Произведение переменных в текущей комбинации
#             product = '*'.join(el)
#             ZP.append([product])
#     return ZP

# print(compute_ZP("x", ["x_1", "x_2"], 2))
