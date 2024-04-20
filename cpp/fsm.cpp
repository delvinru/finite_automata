#include <vector>
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
            int edges_count = 0;

            // get q.values()
            std::vector<std::vector<std::map<State, std::vector<std::vector<int>>>>> q_values;
            for (auto pair : q) {
                q_values.push_back(pair.second);
            }

            // start first for
            for (auto edges : q_values) {
                // start second for
                for (auto edge : edges) {
                    // get edge.values()
                    std::vector<std::vector<std::vector<int>>> edge_values;
                    for (auto pair : edge) {
                        edge_values.push_back(pair.second);
                    }
                    // start third for
                    for (auto entry : edge_values) {
                        std::tuple<std::vector<int>, std::vector<int>> tuple_pair = std::make_tuple(entry[0], entry[1]);
                        unique_edges.insert(tuple_pair);
                        edges_count += 1;
                    }
                }
            }
            return unique_edges.size() != edges_count;
        }

    public:
        int n;
        std::vector<int> phi;
        std::vector<int> psi;
        Graph graph;
        std::map<State, std::tuple<std::vector<State>, std::vector<int>>> table;
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

        std::map<int, std::vector<std::set<State>>> get_equivalence_classes() {
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
            return equivalence_classes;
        }

        std::vector<int> compute_min_polynomial(std::vector<int> init_state) {
            std::vector<int> u = compute_u(init_state);
            std::vector<int> min_polynomial = berlekamp_massey(u);
            return min_polynomial;
        }

        void compute_memory_function() {
            std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>> q_1;

            std::vector<State> key_states;
            for(auto pair : this->table) {
                key_states.push_back(pair.first);
            }
            for (auto key_state : key_states) {
                for (auto entry : this->table) {
                    State value_state = entry.first;
                    std::tuple<std::vector<State>, std::vector<int>> edges = entry.second;
                    // len(edges) == 2?
                    for (int i = 0; i < 2; i++) {
                        if ((std::get<0>(edges))[i] == key_state) {
                            if (q_1.find(key_state) == q_1.end()) {
                                q_1[key_state] = {};
                            }
                            std::map<State, std::vector<std::vector<int>>> temp;
                            temp[value_state] = {{i}, {std::get<1>(edges)[i]}};
                            q_1[key_state].push_back(temp);
                        }
                    }
                }
            }

            std::vector<std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>>> q_s;
            q_s.push_back(q_1);
            while (!is_equal_edges_in_q(q_s.back())) {
                std::map<State, std::vector<std::map<State, std::vector<std::vector<int>>>>> next_q;
                for (auto entry : q_s.back()) {
                    State key_state = entry.first;
                    std::vector<std::map<State, std::vector<std::vector<int>>>> states_edges = entry.second;
                    for (int i = 0; i < states_edges.size(); i++) {
                        if (next_q.find(key_state) == next_q.end()) {
                            next_q[key_state] = {};
                        }
                        for (auto pair : states_edges[i]) {
                            State state = pair.first;
                            std::vector<std::vector<int>> edge = pair.second;
                            for (auto another_states_another_edges : q_1[state]) {
                                for (auto another_entry : another_states_another_edges) {
                                    State another_state = another_entry.first;
                                    std::vector<std::vector<int>> another_edge = another_entry.second;
                                    std::map<State, std::vector<std::vector<int>>> temp;
                                    temp[another_state] = {another_edge[0].insert(another_edge[0].end(), edge[0].begin(), edge[0].end()),
                                                            another_edge[1].insert(another_edge[1].end(), edge[1].begin(), edge[1].end())};
                                    next_q[key_state].push_back(temp);
                                }
                            }
                        }
                    }
                }
                q_s.push_back(next_q);
            }

        // std::vector<std::vector<int>>
            int i = 1;
            for (auto q : q_s) {
                std::cout << "q_" << i << std::endl;
                for (auto first_entry : q) {
                    std::cout << first_entry.first << ": ";
                    for (auto second_entry : first_entry.second) {
                        std::cout << "[";
                        for (auto third_entry : second_entry) {
                            std::cout << "{";
                            std::cout << third_entry.first << ": ";
                            std::cout << "[";
                            for (auto fourth_entry : third_entry.second) {
                                std::cout << "<";
                                for (auto el : fourth_entry) {
                                    std::cout << el;
                                }
                                std::cout << "> ";
                            }
                            std::cout << "] ";
                            std::cout << "} ";
                        }
                        std::cout << "] ";
                    }
                    std::cout << std::endl;
                }
                i++;
            }
        }
};