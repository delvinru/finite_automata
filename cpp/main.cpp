#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include "fsm.cpp"

std::vector<int> convert_str_to_vector(const char *str) {
    std::vector<int> result;
    int length = strlen(str);
    for (int i = 0; i < length; ++i)
        result.push_back(str[i] - '0');
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] <<"<n> <phi> <psi> <init_state>\n";
        return -1;
    }

    int n = atoi(argv[1]);
    std::vector<int> phi = convert_str_to_vector(argv[2]);
    std::vector<int> psi = convert_str_to_vector(argv[3]);
    std::vector<int> init_state = convert_str_to_vector(argv[4]);

    FSM fsm(n, phi, psi);

    // for (const auto& entry : fsm.table) {
    //     const State& key = entry.first;
    //     const auto& value = entry.second;

    //     // Выводим ключ
    //     std::cout << key << ": ([";

    //     // Выводим первый вектор из кортежа
    //     const auto& firstVector = std::get<0>(value);
    //     for (const auto& state : firstVector) {
    //         std::cout << state << ", ";
    //     }
    //     std::cout << "], ";

    //     std::cout << "[";
    //     // Выводим второй вектор из кортежа
    //     const auto& secondVector = std::get<1>(value);
    //     for (const auto& intValue : secondVector) {
    //         std::cout << intValue << ", ";
    //     }
    //     std::cout << "])" << std::endl;
    // }

    // TASK 1
    // ================================================================================
    // std::vector<std::vector<State>> weak_components = fsm.get_connected_components();

    // for (const auto& vec : weak_components) {
    //     for (const auto& state : vec) {
    //         std::cout << state << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
    // ================================================================================

    // TASK 2
    // ================================================================================
    // std::vector<std::vector<State>> strong_compoents = fsm.get_strong_connected_components();
    // for (const auto& vec : strong_compoents) {
    //     for (const auto& state : vec) {
    //         std::cout << state << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
    // ================================================================================

    // TASK 3
    // ================================================================================
    // std::map<int, std::vector<std::set<State>>> equivalence_classes = fsm.get_equivalence_classes();
    // for (auto entry : equivalence_classes) {
    //     std::cout << "Классы эквивалентности " << entry.first << std::endl;
    //     for (auto s : entry.second) {
    //         std::cout << "[";
    //         for (auto st : s) {
    //             std::cout << st << " ";
    //         }
    //         std::cout << "] ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Степень различимости автомата, delta(A): " << equivalence_classes.size() << std::endl;
    // std::cout << "mu(A): " << equivalence_classes[equivalence_classes.size()].size() << std::endl;
    // ================================================================================

    // TASK 4
    // ================================================================================
    fsm.compute_memory_function();
    // ================================================================================
    // TASK 5
    // ================================================================================
    // std::vector<int> min_polynomial = fsm.compute_min_polynomial(init_state);
    // std::cout << "Минимальный многочлен: ";
    // for (auto coef : min_polynomial) {
    //     std::cout << coef;
    // }
    // std::cout << std::endl;
    // std::cout << "Линейная сложность: " << min_polynomial.size() << std::endl;
    // ================================================================================
    return 0;
}