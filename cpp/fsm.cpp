#include <vector>
#include <fstream>
#include <bitset>
#include <iostream>
#include "graph.cpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>

class FSM {
    private:
        void generate_combinations_helper(std::vector<int>& current_combination, int index, int n, std::vector<std::vector<int>>& result) {
            if (index == n) {
                result.push_back(current_combination);
                return;
            }
            current_combination[index] = 0;
            generate_combinations_helper(current_combination, index + 1, n, result);
            current_combination[index] = 1;
            generate_combinations_helper(current_combination, index + 1, n, result);
        }

        std::vector<std::vector<int>> generate_binary_combinations(int n) {
            std::vector<std::vector<int>> result;
            std::vector<int> initial_combination(n, 0);
            generate_combinations_helper(initial_combination, 0, n, result);
            return result;
        }

        std::vector<std::vector<int>> combinations(const std::vector<int>& iterable, int r) {
            std::vector<std::vector<int>> combinations;
            std::vector<int> pool = iterable;
            int n = pool.size();
            
            if (r > n) {
                return combinations;
            }

            std::vector<int> indices(r);
            for (int i = 0; i < r; ++i) {
                indices[i] = i;
            }

            // Добавляем первую комбинацию
            std::vector<int> first_combination;
            for (int i : indices) {
                first_combination.push_back(pool[i]);
            }
            combinations.push_back(first_combination);

            while (true) {
                int i;
                for (i = r - 1; i >= 0; --i) {
                    if (indices[i] != i + n - r) {
                        break;
                    }
                }
                if (i < 0) {
                    break;
                }
                
                ++indices[i];
                for (int j = i + 1; j < r; ++j) {
                    indices[j] = indices[j - 1] + 1;
                }

                // Добавляем комбинацию
                std::vector<int> combination;
                for (int i : indices) {
                    combination.push_back(pool[i]);
                }
                combinations.push_back(combination);
            }

            return combinations;
        }

        int compute_zhegalkin_polynomial(int input_x, const std::vector<int>& current_state, const std::vector<int>& coeffs) {
            std::vector<int> zp = {1};

            std::vector<int> extended_current_state = current_state;
            extended_current_state.push_back(input_x);

            int new_n = extended_current_state.size();
            for (int i = 1; i < new_n + 1; i++) {
                std::vector<std::vector<int>> combs = combinations(extended_current_state, i);
                for (const auto& el : combs) {
                    int product = 1;
                    for (int num : el) {
                        product *= num;
                    }
                    zp.push_back(product);
                }
            }

            for (int i = 0; i < std::pow(2, new_n); ++i) {
                zp[i] *= coeffs[i];
            }

            int count_ones = 0;
            for (int num : zp) {
                if (num == 1) {
                    ++count_ones;
                }
            }
            return count_ones % 2;
        }

        std::vector<std::set<State>> get_first_classes() {
            std::map<State, std::vector<State>> dict_classes = {
                {State({0, 0}), {}},
                {State({0, 1}), {}},
                {State({1, 0}), {}},
                {State({1, 1}), {}}
            };

            for (auto entry : this->table) {
                std::tuple<std::vector <State>, std::vector<int>> values = entry.second;
                std::vector<int> keys = std::get<1>(values);
                State first_class_key(std::vector<int>{keys[0], keys[1]});
                dict_classes[first_class_key].push_back(entry.first);
            }

            std::vector<std::set<State>> result;
            for (auto entry : dict_classes) {
                std::vector<State> non_correct_value = {};
                if (entry.second != non_correct_value) {
                    result.push_back(std::set<State>(entry.second.begin(), entry.second.end()));
                }
            }

            return result;
        }

        bool is_in_one_equal_class(const State& value_1, const State& value_2, std::vector<std::set<State>> old_classes) {
            std::vector<State> values{value_1, value_2};

            std::vector<State> perehod_values;
            for (auto value{values.begin()}; value != values.end(); value++) {
                std::tuple<std::vector<State>, std::vector<int>> table_value = this->table[*value];
                std::vector<State> table_value_states = std::get<0>(table_value);
                for (int i = 0; i < 2; i++) {
                    perehod_values.push_back(table_value_states[i]);
                }
            }

            bool check_0 = false;
            bool check_1 = false;

            for (auto old_class : old_classes) {
                if ((old_class.find(perehod_values[0]) != old_class.end()) && (old_class.find(perehod_values[2]) != old_class.end())) {
                    check_0 = true;
                }

                if ((old_class.find(perehod_values[1]) != old_class.end()) && (old_class.find(perehod_values[3]) != old_class.end())) {
                    check_1 = true;
                }

                if (check_0 && check_1) {
                    break;
                }
            }

            return check_0 && check_1;
        }

        std::set<State> step_split_class(std::set<State>& current_class, std::vector<std::set<State>>& old_classes) {
            std::set<State> new_class;
            new_class.insert(*current_class.begin()); // Add the first element of current class

            for (auto it = std::next(current_class.begin()); it != current_class.end(); ++it) {
                if (is_in_one_equal_class(*new_class.begin(), *it, old_classes)) {
                    new_class.insert(*it);
                }
            }
            return new_class;
        }

        std::vector<std::set<State>> split_class(std::set<State>& clazz, std::vector<std::set<State>>& old_classes) {
            std::vector<std::set<State>> split_classes;
            std::set<State> current_class = clazz;
            while (!current_class.empty()) {
                std::set<State> new_class = step_split_class(current_class, old_classes);
                split_classes.push_back(new_class);

                std::set<State> set_current_class{current_class.begin(), current_class.end()};
                std::set<State> set_new_class{new_class.begin(), new_class.end()};

                std::set<State> result;
                std::set_difference(set_current_class.begin(), set_current_class.end(), set_new_class.begin(), set_new_class.end(), std::inserter(result, result.end()));
                current_class = result;
            }
            return split_classes;
        }

        std::vector<std::set<State>> compute_k_classes(std::vector<std::set<State>>& old_classes) {
            std::vector<std::set<State>> k_classes;
            for (auto& clazz : old_classes) {
                std::vector<std::set<State>> split_classes = split_class(clazz, old_classes);
                for (auto& split_class : split_classes) {
                    k_classes.push_back(split_class);
                }
            }

            return k_classes;
        }

        std::vector<int> compute_u(std::vector<int> init_state) {
            std::vector<int> u;
            State current_state(init_state);
            for (int i = 0; i < std::pow(2, n); i++) {
                u.push_back(std::get<1>(this->table[current_state])[0]);
                current_state = std::get<0>(this->table[current_state])[0];
            }
            return u;
        }

        int count_of_leading_zeros(std::vector<int>& segment) {
            int count = 0;
            for (auto el : segment) {
                if (el != 0) {
                    return count;
                }
                else
                    count += 1;
            }
            return count;
        }

        std::vector<int> compute_sum_of_two_lists(std::vector<int>& first_list, std::vector<int>& second_list) {
            int length = std::min(first_list.size(), second_list.size());
            std::vector<int> new_list = {};
            for (int i = 0; i < length; i++)
                new_list.push_back(first_list[i] ^ second_list[i]);
            return new_list;
        }

        std::vector<int> compute_sum_of_two_polynomials(std::vector<int>& first_polynomial, std::vector<int>& second_polynomial) {
            int length = std::max(first_polynomial.size(), second_polynomial.size());

            if (first_polynomial.size() != length) {
                int diff = length - first_polynomial.size();
                for (int i = 0; i < diff; i++)
                    first_polynomial.push_back(0);
            }

            if (second_polynomial.size() != length) {
                int diff = length - second_polynomial.size();
                for (int i = 0; i < diff; i++)
                    second_polynomial.push_back(0);
            }

            std::vector<int> new_list = {};
            for (int i = 0; i < length; i++) {
                new_list.push_back(first_polynomial[i] ^ second_polynomial[i]);
            }
            return new_list;
        }

        std::vector<int> berlekamp_massey(std::vector<int>& u) {
            std::vector<std::vector<int>> segments = {};
            std::vector<std::vector<int>> polynomials = {};
            std::vector<int> zeros_count = {};

            zeros_count.push_back(count_of_leading_zeros(u));
            polynomials.push_back({1});
            if (zeros_count.back() == u.size()) {
                return polynomials[0];
            }
            segments.push_back(u);

            for (int i = 1; i < std::pow(2, n - 1); i++) {
                std::vector<std::vector<int>> current_segments;
                
                std::vector<int> temp = {};
                for (int i = 1; i < segments.back().size(); i++) {
                    temp.push_back(segments.back()[i]);
                }
                current_segments.push_back(temp);
                std::vector<std::vector<int>> current_polynomials;
                temp = {0};
                for (auto value : polynomials.back())
                    temp.push_back(value);
                current_polynomials.push_back(temp);

                std::vector<int> current_zeros_count;
                current_zeros_count.push_back({zeros_count.back() != 0 ? zeros_count.back() - 1 : 0});

                while (count_of_leading_zeros(current_segments.back()) != current_segments.back().size() &&
                        std::find(zeros_count.begin(), zeros_count.end(), current_zeros_count.back()) != zeros_count.end()) {
                    int t = std::distance(zeros_count.begin(), std::find(zeros_count.begin(), zeros_count.end(), current_zeros_count.back()));
                    
                    if (segments[t][zeros_count[t]] == 0) {
                        throw std::runtime_error("Alarm! 0 when computing u_t(k_t)^(-1)");
                    }
                    int r = current_segments.back()[zeros_count[t]] * segments[t][zeros_count[t]];

                    std::vector<int> temp = {};
                    for (auto i : segments[t]) {
                        temp.push_back(i * r);
                    }
                    current_segments.push_back(compute_sum_of_two_lists(current_segments.back(), temp));

                    temp = {};
                    for (auto i : polynomials[t]) {
                        temp.push_back(i * r);
                    }

                    current_polynomials.push_back(compute_sum_of_two_polynomials(current_polynomials.back(), temp));

                    current_zeros_count.push_back(count_of_leading_zeros(current_segments.back()));                
                }

                if (count_of_leading_zeros(current_segments.back()) == current_segments.back().size()) {
                    zeros_count.push_back(current_zeros_count.back());
                    polynomials.push_back(current_polynomials.back());
                    segments.push_back(current_segments.back());
                    return polynomials.back();
                }

                segments.push_back(current_segments.back());
                polynomials.push_back(current_polynomials.back());
                zeros_count.push_back(current_zeros_count.back());
            }

            segments.push_back({});
            std::vector<int> temp = {0};
            for (auto value : polynomials.back())
                temp.push_back(value);
            polynomials.push_back(temp);
            return polynomials.back();
        }

        bool is_equal_edges_in_q(std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>>& q) {
            std::set<std::tuple<std::vector<int>, std::vector<int>>> unique_edges;

            // start first for
            for (auto item : q) {
                std::vector<std::map<State, std::vector<std::vector<int>>>> edges = item.second;
                // start second for
                for (auto edge : edges) {
                    // start third for
                    for (auto entry : edge) {
                        auto pair = entry.second;
                        std::tuple<std::vector<int>, std::vector<int>> tuple_pair = std::make_tuple(pair[0], pair[1]);
                        if (unique_edges.find(tuple_pair) != unique_edges.end())
                            return true;
                        unique_edges.insert(tuple_pair);
                    }
                }
            }
            return false;
        }

        void minimization() {
            // Проверяем, вычисляли ли мы до этого классы эквивалентности или нет
            if (this->equivalence_classes.empty()) {
                this->get_equivalence_classes();
                this->compute_delta();
            }

            for (const auto& set_states : this->equivalence_classes[this->delta]) {
                if (set_states.size() > 1) {
                    auto it = set_states.begin();
                    State equivalent_state = *it;
                    ++it;
                    for (; it != set_states.end(); ++it) {
                        const State& state = *it;
                        for (auto& table_pair : table) {
                            auto& table_tuple = table_pair.second;
                            auto& state_vec = std::get<0>(table_tuple);
                            std::replace(state_vec.begin(), state_vec.end(), state, equivalent_state);
                        }
                        this->table.erase(state);
                    }
                }
            }
            this->get_equivalence_classes();
            this->compute_delta();
            this->compute_mu();
        }

        std::vector<int> get_memory_value_vector(std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>> q_last, int memory) {
            std::vector<int> memory_value_vector;
            std::map<std::tuple<std::vector<int>, std::vector<int>>, std::vector<int>> large_table_dict;

            std::vector<std::vector<int>> large_table_dict_sub_keys;
            // for (auto comb: this->generate_binary_combinations(memory)) {
            //     std::vector<int> s;
            //     for (auto el: comb) {
            //         s.push_back(el);
            //     }
            //     large_table_dict_sub_keys.push_back(s);
            // }
            for (int comb = 0; comb < (1 << memory); ++comb) {
                std::vector<int> state;
                for (int j = 0; j < memory; ++j) {
                    state.push_back((comb >> j) & 1);
                }
                large_table_dict_sub_keys.push_back(state);
            }

            for (auto& comb_i : large_table_dict_sub_keys) {
                for (auto& comb_j : large_table_dict_sub_keys) {
                    large_table_dict[{comb_i, comb_j}] = {NULL, NULL};
                }
            }

            for (const auto& [main_state, elements] : q_last) {
                for (const auto& element : elements) {
                    for (const auto& vectors : element) {
                        auto current_vector = std::make_tuple(vectors.second[0], vectors.second[1]);
                        large_table_dict[current_vector][0] = std::get<1>(table[main_state])[0];
                        large_table_dict[current_vector][1] = std::get<1>(table[main_state])[1];
                    }
                }
            }

            const int n = 2 * memory + 1;
            const int max_value = (1 << n); // 2^n

            for (int i = 0; i < max_value; ++i) {
                std::bitset<32> bits(i); // bitset размером 32 для гарантии
                std::vector<int> permutation;

                for (int j = n - 1; j >= 0; --j) {
                    permutation.push_back(bits[j]);
                }

                std::vector<int> left_part;
                for (int i = 0; i < memory; i++) {
                    left_part.push_back(permutation[i]);
                }
                std::vector<int> right_part;
                for (int i = memory + 1; i < permutation.size(); i++) {
                    right_part.push_back(permutation[i]);
                }
                auto current_vector = std::make_tuple(left_part, right_part);
                memory_value_vector.push_back(large_table_dict[current_vector][permutation[memory]]);
            }

            return memory_value_vector;
        }

        std::vector<int> get_memory_function_coefs(std::vector<int>& seq) {
            std::vector<int> seqLeft(seq.begin(), seq.begin() + seq.size() / 2);
            std::vector<int> seqRight(seq.begin() + seq.size() / 2, seq.end());
            std::vector<int> seqOut(seq.size());

            for (size_t i = 0; i < seq.size() / 2; ++i) {
                seqRight[i] = (seq[i] + seq[i + seq.size() / 2]) % 2;
            }

            if (seq.size() == 2) {
                seqOut[0] = seqLeft[0];
                seqOut[1] = seqRight[0];
                return seqOut;
            }

            std::vector<int> temp1 = get_memory_function_coefs(seqLeft);
            std::vector<int> temp2 = get_memory_function_coefs(seqRight);

            for (size_t i = 0; i < seqOut.size() / 2; ++i) {
                seqOut[i] = temp1[i];
                seqOut[i + seqOut.size() / 2] = temp2[i];
            }

            return seqOut;
        }

        std::string get_memory_function_coefs_str(std::vector<int> vector) {
            std::string vector_string = "";
            if (vector[0] == 1) {
                vector_string += "1 + ";
            }

            int length_of_vector = static_cast<int>(std::log2(vector.size()));
            for (int i = 1; i < vector.size(); ++i) {
                if (vector[i] == 1) {
                    std::bitset<32> bin_value(i);
                    std::string bin_str = bin_value.to_string().substr(32 - length_of_vector);
                    for (int coef = 0; coef < bin_str.size(); coef++) {
                        if (bin_str[coef] == '1') {
                            std::string coef_str = "";

                            // compute coef_str
                            if (coef < (int)(length_of_vector / 2)) {
                                coef_str += "x_(i";
                                coef_str += "-" + std::to_string((int)(length_of_vector / 2) - coef) + ")";
                            } else if (coef > (int)(length_of_vector / 2)) {
                                coef_str += "y_(i";
                                coef_str += "-" + std::to_string(length_of_vector - coef) + ")";
                            } else {
                                coef_str += "x_i";
                            }

                            vector_string += coef_str;
                        }
                    }
                    vector_string += " + ";
                }
            }

            vector_string = vector_string.substr(0, vector_string.size() - 3);
            return vector_string;
        }
        // std::string get_memory_function_coefs_str(const std::vector<int>& vector) const {
        //     std::string vector_string;

        //     if (vector[0] == 1) {
        //         vector_string += "1 + ";
        //     }

        //     for (size_t i = 1; i < vector.size(); ++i) {
        //         if (vector[i] == 1) {
        //             std::string bin_value = std::bitset<64>(i).to_string();
        //             bin_value = bin_value.substr(bin_value.find('1'));  // Убираем ведущие нули
        //             bin_value = std::string((2 * n) + 1 - bin_value.size(), '0') + bin_value; // Добиваем нулями до нужной длины

        //             for (size_t coef = 0; coef < bin_value.size(); ++coef) {
        //                 if (bin_value[coef] == '1') {
        //                     std::string coef_str;

        //                     if (coef + 1 < n + 1) {
        //                         coef_str += "x_(i";
        //                         coef_str += "-" + std::to_string(n - coef) + ")";
        //                     } else if (coef + 1 > n + 1) {
        //                         coef_str += "y_(i";
        //                         coef_str += "-" + std::to_string(2 * n - coef + 1) + ")";
        //                     } else {
        //                         coef_str += "x_i";
        //                     }

        //                     vector_string += coef_str;
        //                 }
        //             }
        //             vector_string += " + ";
        //         }
        //     }

        //     if (!vector_string.empty() && vector_string.size() >= 3) {
        //         vector_string.erase(vector_string.size() - 3); // Удаляем последнее " + "
        //     }

        //     return vector_string;
        // }

    public:
        int n;
        std::vector<int> phi;
        std::vector<int> psi;
        Graph graph;
        std::map<State, std::tuple<std::vector<State>, std::vector<int>>> table;

        std::map<int, std::vector<std::set<State>>> equivalence_classes;

        int delta = 0;
        int mu = 0;

        FSM (int n, std::vector<int> phi, std::vector<int> psi) {
            this->n = n;
            this->phi = phi;
            this->psi = psi;

            std::vector<std::vector<int>> binary_combinations = generate_binary_combinations(n);
            for (auto state{binary_combinations.begin()}; state != binary_combinations.end(); state++) {
                State graph_node(*state);
                // std::cout << graph_node << "\n\n";
                std::vector<State> phis;
                std::vector<int> psis;

                std::vector<int> values{0, 1};
                for (auto x{values.begin()}; x != values.end(); x++) {
                    int zp_phi = compute_zhegalkin_polynomial(*x, *state, phi);
                    int zp_psi = compute_zhegalkin_polynomial(*x, *state, psi);

                    std::vector<int> new_state;
                    for (int i = 1; i < (*state).size(); i++)
                        new_state.push_back((*state)[i]);
                    new_state.push_back(zp_phi);

                    State argument_state(new_state);
                    this->graph.add_vertex(graph_node, argument_state);

                    // init table
                    phis.push_back(State(new_state));
                    psis.push_back(zp_psi);
                }
                this->table[graph_node] = std::tuple<std::vector<State>, std::vector<int>>{phis, psis};
            }
        }

        std::vector<std::vector<State>> get_connected_components() {
            return this->graph.find_connected_components();
        }

        std::vector<std::vector<State>> get_strong_connected_components() {
            return this->graph.find_strong_connected_components();
        }

        void get_equivalence_classes() {
            std::map<int, std::vector<std::set<State>>> equivalence_classes;

            // find 1-classes
            equivalence_classes[1] = get_first_classes();
            // find k-classes
            int k = 1;
            while (true) {
                std::vector<std::set<State>> new_class = compute_k_classes(equivalence_classes[k]);
                if (new_class == equivalence_classes[k]) {
                    break;
                }

                k += 1;
                equivalence_classes[k] = new_class;
            }
            this->equivalence_classes = equivalence_classes;
        }

        std::vector<int> compute_min_polynomial(std::vector<int> init_state) {
            std::vector<int> u = compute_u(init_state);
            std::vector<int> min_polynomial = berlekamp_massey(u);
            return min_polynomial;
        }

        void compute_delta() {
            if (this->equivalence_classes.size() == 0)
                this->get_equivalence_classes();
            this->delta = this->equivalence_classes.size();
        }

        void compute_mu() {
            if (this->equivalence_classes.size() == 0)
                this->get_equivalence_classes();
            this->mu = (this->equivalence_classes[this->equivalence_classes.size()]).size();
        }

        void compute_memory_function(std::ofstream& file) {
            if (this->equivalence_classes.size() == 0) {
                this->get_equivalence_classes();
                this->compute_delta();
                this->compute_mu();
            }

            std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>> q_1;

            if (this->mu != this->table.size()) {
                this->minimization();
            }

            for (auto pair : this->table) {
                State key_state = pair.first;

                std::set<std::tuple<int, int>> edges_list;
                for (auto entry : this->table) {
                    State value_state = entry.first;
                    std::tuple<std::vector<State>, std::vector<int>> edges = entry.second;
                    // len(edges) == 2?
                    for (int i = 0; i < 2; i++) {
                        std::tuple<int, int> tmp_set = std::make_tuple(i, std::get<1>(edges)[i]);
                        if ((std::get<0>(edges))[i] == key_state && (std::find(edges_list.begin(), edges_list.end(), tmp_set) == edges_list.end())) {
                            if (q_1.find(key_state) == q_1.end()) {
                                q_1[key_state] = {};
                            }
                            std::map<State, std::vector<std::vector<int>>> temp;
                            temp[value_state] = {{i}, {std::get<1>(edges)[i]}};
                            q_1[key_state].push_back(temp);
                            edges_list.insert(tmp_set);
                        }
                    }
                }
            }

            int max_steps = (this->mu * (this->mu - 1)) / 2;

            std::vector<std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>>> q_s;
            q_s.push_back(q_1);

            std::cout << "[+] Вычислили q_" << q_s.size() << std::endl;
            while (this->is_equal_edges_in_q(q_s.back()) && q_s.size() <= max_steps) {
                std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>> next_q;

                // uint64_t q_s_last_size = q_s.back().size();
                // std::vector<State> vertexes;
                // for (auto const& entry: q_s.back()) {
                //     vertexes.push_back(entry.first);
                // }

                for (auto entry: q_s.back()) {
                // #pragma omp parallel for
                // for (int i = 0; i < q_s_last_size; i++) {
                //     State state = vertexes[i];
                //     std::vector<std::map<State, std::vector<std::vector<int>>>> edges = q_s.back()[state];
                    State state = entry.first;
                    std::vector<std::map<State, std::vector<std::vector<int>>>> edges = entry.second;
                    std::set<std::tuple<std::vector<int>>> edges_list;

                    for (auto edge: edges) {
                        auto from_state = edge.begin()->first;
                        auto values = edge.begin()->second;

                        if (q_1.find(from_state) == q_1.end()) {
                            continue;
                        }

                        for (auto edges_from_state: q_1[from_state]) {
                            for (auto another: edges_from_state) {
                                State another_state = another.first;
                                std::vector<std::vector<int>> another_edge = another.second;
                                std::vector<int> tmp_vector;
                                for (auto el: another_edge[0])
                                    tmp_vector.push_back(el);
                                for (auto el: values[0])
                                    tmp_vector.push_back(el);
                                for (auto el: another_edge[1])
                                    tmp_vector.push_back(el);
                                for (auto el: values[1])
                                    tmp_vector.push_back(el);

                                std::tuple<std::vector<int>> tmp_set(tmp_vector);
                                if (edges_list.find(tmp_set) == edges_list.end()) {
                                    std::map<State, std::vector<std::vector<int>>> tmp;
                                    std::vector<int> temp1;
                                    std::vector<int> temp2;
                                    std::vector<std::vector<int>> temp;
                                    for (auto el: another_edge[0])
                                        temp1.push_back(el);
                                    for (auto el: values[0])
                                        temp1.push_back(el);
                                    temp.push_back(temp1);
                                    for (auto el: another_edge[1])
                                        temp2.push_back(el);
                                    for (auto el: values[1])
                                        temp2.push_back(el);
                                    temp.push_back(temp2);                
                                    tmp[another_state] = temp;
                                    next_q[state].push_back(tmp);
                                    edges_list.insert(tmp_set);
                                }
                            }
                        }
                    }
                }

                q_s.push_back(next_q);
                std::cout << "[+] Вычислили q_" << q_s.size() << std::endl;
            }

            std::cout << "[+] Нашли все q_s" << std::endl;

            int memory = q_s.size();
            if (memory > max_steps) {
                file << "Память автомата бесконечна" << std::endl;
            }
            else {
                int i = 1;
                for (auto q : q_s) {
                    for (auto first_entry : q) {
                        file << "q_" << i << "(";
                        file << first_entry.first << "): ";
                        for (auto second_entry : first_entry.second) {
                            file << "[";
                            for (auto third_entry : second_entry) {
                                for (auto fourth_entry : third_entry.second) {
                                    file << "[";
                                    for (auto el : fourth_entry) {
                                        file << el;
                                    }
                                    file << "], ";
                                }
                                file.seekp(file.tellp() - long(2));
                            }
                            file << "] |_| ";
                        }
                        file.seekp(file.tellp() - long(5));
                        file << std::endl;
                    }
                    i++;
                    file << std::endl;
                }
                file << "Память автомата конечна: m(A) = " << q_s.size() << std::endl;

                std::cout << "[+] Записали q_s в файл" << std::endl;

                std::vector<int> memory_value_vector = this->get_memory_value_vector(q_s.back(), memory);
                std::cout << "[+] Вычислили вектор y_i" << std::endl;
                std::cout << "[+] Подставили вместо * нули в векторе y_i" << std::endl;
                std::vector<int> memory_function_coefs = this->get_memory_function_coefs(memory_value_vector);
                std::cout << "[+] Нашли коэффициенты перед слагаемыми МЖ функции y_i" << std::endl;
                file << "Функция памяти автомата: " << this->get_memory_function_coefs_str(memory_function_coefs) << std::endl;
                std::cout << "[+] Записали коэффициенты перед слагаемыми МЖ функции y_i в файл" << std::endl;
            }
        }
};
