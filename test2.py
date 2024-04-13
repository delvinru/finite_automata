def berlekamp_massey(u: list[int]) -> list[int]:
    segments: list[list[int]] = []
    polynomials: list[list[int]] = []
    zeros_count: list[int] = []

    def count_of_leading_zeros(segment: list[int]) -> int:
        count = 0
        for el in segment:
            if el != 0:
                return count
            else:
                count += 1
        return count
    # step 0
    zeros_count.append(count_of_leading_zeros(u))
    polynomials.append([1])
    if zeros_count[-1] == len(u):
        return polynomials[0]
    segments.append(u)

    # steps 1 - (l - 1)
    for i in range(1, 2 ** N - 1):
        # compute (u_i)^0
        current_segments: list[list[int]] = [segments[-1][1:]]
        # compute (f_i)^0
        current_polynomials: list[list[int]] = [[0] + polynomials[-1]]
        # compute (k_i)^0
        current_zeros_count: list[int] = []
        current_zeros_count: list[int] = [zeros_count[-1] - 1] if zeros_count[-1] != 0 else [0]

        # case 3
        while count_of_leading_zeros(current_segments[-1]) != len(current_segments[-1]) and \
                                        current_zeros_count[-1] in zeros_count:
            # compute "t" from algorithm
            t = zeros_count.index(current_zeros_count[-1])
            # compute r in GF(2)
            if segments[t][zeros_count[t]] == 0:
                raise Exception("Alarm! 0 when computing u_t(k_t)^(-1)")
            r = current_segments[-1][zeros_count[t]] * segments[t][zeros_count[t]]

            def compute_sum_of_two_segments(first_list: list[int], second_list: list[int]) -> list[int]:
                length = min(len(first_list), len(second_list))
                new_list = []
                for i in range(length):
                    new_list.append(first_list[i] ^ second_list[i])
                return new_list
            
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
            
            current_segments.append(compute_sum_of_two_segments(current_segments[-1],
                                                                [i * r for i in segments[t]]))
            current_polynomials.append(compute_sum_of_two_polynomials(current_polynomials[-1],
                                                                [i * r for i in polynomials[t]]))
            current_zeros_count.append(count_of_leading_zeros(current_segments[-1]))
        
        # case 1
        if count_of_leading_zeros(current_segments[-1]) == len(current_segments[-1]):
            zeros_count.append(current_zeros_count[-1])
            polynomials.append(current_polynomials[-1])
            segments.append(current_segments[-1])
            return polynomials[-1]
        
        # case 2
        segments.append(current_segments[-1])
        polynomials.append(current_polynomials[-1])
        zeros_count.append(current_zeros_count[-1])

    # step l
    segments.append([])
    polynomials.append([0] + polynomials[-1])
    return polynomials[-1]

def compute_min_polynomial(u: list[int]) -> None:
    min_polynomial = berlekamp_massey(u)
    return min_polynomial

# N = 8
# min_polynomial = compute_min_polynomial([0, 1, 1, 1, 0, 0, 1, 0])
# N = 6
# min_polynomial = compute_min_polynomial([0, 1, 1, 0, 1, 1])
# N = 12
# min_polynomial = compute_min_polynomial([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0])
N = 12
min_polynomial = compute_min_polynomial([0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
print(min_polynomial)

if min_polynomial[0] == 1:
    print("1 + ", end="")
for i in range(1, len(min_polynomial)):
    if min_polynomial[i] == 1:
        print(f"x^{i} + ", end="")