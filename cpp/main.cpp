#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <fstream>
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

    std::ofstream file("test.txt", std::ios_base::out);

    if (file.is_open()) {
        int n = atoi(argv[1]);
        std::vector<int> phi = convert_str_to_vector(argv[2]);
        std::vector<int> psi = convert_str_to_vector(argv[3]);
        std::vector<int> init_state = convert_str_to_vector(argv[4]);

        // write n to file
        file << "n = " << n << std::endl;
        std::cout << "[+] Записали n в файл" << std::endl;

        // write phi to file
        file << "phi: ";
        for (auto el: phi)
            file << el;
        file << std::endl;
        std::cout << "[+] Записали phi в файл" << std::endl;

        // write psi to file
        file << "psi: ";
        for (auto el: psi)
            file << el;
        file << std::endl << std::endl;
        std::cout << "[+] Записали psi в файл" << std::endl;

        FSM fsm(n, phi, psi);

        // TASK 1
        std::vector<std::vector<State>> weak_components = fsm.get_connected_components();
        std::cout << "[+] Сделали 1 лабу" << std::endl;
        int count = 1;
        file << "TASK 1" << std::endl;
        for (const auto& vec : weak_components) {
            file << "Компонента связности " << count << ": [";
            for (const auto& state : vec) {
                file << state << ", ";
            }
            file.seekp(file.tellp() - long(2));
            file << "]" << std::endl;
            count += 1;
        }
        file << std::endl;
        std::cout << "[+] Записали 1 лабу в файл" << std::endl;

        // TASK 2
        std::vector<std::vector<State>> strong_compoents = fsm.get_strong_connected_components();
        std::cout << "[+] Сделали 2 лабу" << std::endl;
        count = 1;
        file << "TASK 2" << std::endl;
        for (const auto& vec : strong_compoents) {
            file << "Компонента сильной связности " << count << ": [";
            for (const auto& state : vec) {
                file << state << ", ";
            }
            file.seekp(file.tellp() - long(2));
            file << "]" << std::endl;
            count += 1;
        }
        file << std::endl;
        std::cout << "[+] Записали 2 лабу в файл" << std::endl;

        // TASK 3
        fsm.get_equivalence_classes();
        fsm.compute_delta();
        fsm.compute_mu();

        std::cout << "[+] Сделали 3 лабу в файл" << std::endl;
        file << "TASK 3" << std::endl;
        for (auto entry : fsm.equivalence_classes) {
            file << "k = " << entry.first << ":" << std::endl;
            for (auto s : entry.second) {
                file << "[";
                for (auto st : s) {
                    file << st << ", ";
                }
                file.seekp(file.tellp() - long(2));
                file << "] |_| ";
            }
            file.seekp(file.tellp() - long(5));
            file << std::endl;
        }

        file << "Степень различимости автомата, delta(A): " << fsm.delta << std::endl;
        file << "Приведенный вес автомата, mu(A): " << fsm.mu << std::endl;
        file << std::endl;
        std::cout << "[+] Записали 3 лабу в файл" << std::endl;

        // TASK 4
        file << "TASK 4" << std::endl;
        fsm.compute_memory_function(file);
        file << std::endl;
        std::cout << "[+] Записали 4 лабу в файл" << std::endl;

        // TASK 5
        FSM f(n, phi, psi);
        file << "TASK 5" << std::endl;
        file << "Начальное состояние: ";
        for (auto el: init_state) {
            file << el;
        }
        file << std::endl;
        std::cout << "[+] Записали начальное состояние в файл" << std::endl;
        std::vector<int> min_polynomial_coefs = f.compute_min_polynomial(init_state);
        std::cout << "[+] Сделали 5 лабу" << std::endl;
        file << "Минимальный многочлен: ";
        std::string min_polynomial = "";
        if (min_polynomial_coefs[0] == 1)
            min_polynomial += "1 + ";
        for (int i = 1; i < min_polynomial_coefs.size(); i++) {
            if (min_polynomial_coefs[i] == 1) {
                min_polynomial += "x^"; 
                min_polynomial += std::to_string(i);
                min_polynomial += " + ";
            }
        }
        min_polynomial = min_polynomial.substr(0, min_polynomial.length() - std::string(" + ").length());
        file << min_polynomial << std::endl;
        file << "Линейная сложность: " << min_polynomial_coefs.size() << std::endl;
        std::cout << "[+] Записали 5 лабу в файл" << std::endl;

        // print table
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
    }
    file.close();
    return 0;
}