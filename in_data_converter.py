def get_memory_function_coefs_str(power: int) -> list[str]:
    arr = ["1"]

    for i in range(1, 2**power):
        bin_value = bin(i)[2:].zfill(power)
        vector_string = ""
        for coef, bit in enumerate(bin_value):
            if int(bit) == 1:
                vector_string += f"x_{coef + 1}"
        arr.append(vector_string)

    arr.sort(key=lambda item: (len(item), item))
    result_arr = [el for el in arr]
    result_arr.append("x")
    for i in range(1, len(arr)):
        result_arr.append(arr[i]+"x")
    return result_arr

list_keys = get_memory_function_coefs_str(8)
coefs_dict: dict[str, int] = dict()
for el in list_keys:
    coefs_dict[el] = 0

task_key = ['x_1x_3', 'x_2x_8', 'x_4', 'x_5x_7']
for key in task_key:
    coefs_dict[key] = 1
for el in coefs_dict.values():
    print(el, end='')
print()