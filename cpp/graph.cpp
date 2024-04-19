#include <vector>
#include <iostream>
#include <map>
#include <set>
#include "state.cpp"

class Graph {
    private:
        void dfs(const State& node, std::set<State>& visited, std::vector<State>& component) {
            visited.insert(node);
            component.push_back(node);
            for (const auto& neighbor : graph[node]) {
                if (visited.find(neighbor) == visited.end()) {
                    dfs(neighbor, visited, component);
                }
            }
            
        }

        Graph transpose() {
            Graph transposed_graph;
            for (auto& pair : graph) {
                State node = pair.first;
                for (State neighbor : pair.second)
                    transposed_graph.add_vertex(neighbor, node);
            }
            return transposed_graph;
        }

        void first_dfs(State& node, std::set<State>& visited, std::vector<State>& order) {
            visited.insert(node);
            for (State neighbor : graph.at(node)) {
                if (visited.find(neighbor) == visited.end()) {
                    first_dfs(neighbor, visited, order);
                }
            }
            order.push_back(node);
        }

        void second_dfs(State& node, std::set<State>& visited, std::vector<State>& component, Graph& transposed) {
            visited.insert(node);
            component.push_back(node);
            for (State neighbor : transposed.graph.at(node)) {
                if (visited.find(neighbor) == visited.end()) {
                    second_dfs(neighbor, visited, component, transposed);
                }
            }
        }
    public:
        std::map<State, std::set<State>> graph;

        void add_vertex(State& from_node, State& to_node) {
            if (!graph.count(from_node))
                graph[from_node] = std::set<State>();
            graph[from_node].insert(to_node);
        }

        friend std::ostream& operator<<(std::ostream& os, const Graph& graph) {
            for (const auto& pair : graph.graph) {
                os << pair.first << ": ";
                const auto& adjacent_vertices = pair.second;
                for (const auto& vertex : adjacent_vertices) {
                    os << vertex << ", ";
                }
                os << '\n';
            }
            return os;
        }

        std::vector<std::vector<State>> find_connected_components() {
            std::set<State> visited;
            std::vector<std::vector<State>> connected_components;

            for (const auto& pair : graph) {
                const State& node = pair.first;
                if (visited.find(node) == visited.end()) {
                    std::vector<State> component;
                    dfs(node, visited, component);
                    connected_components.push_back(component);
                }
            }

            return connected_components;
        }

        std::vector<std::vector<State>> find_strong_connected_components() {
            std::set<State> visited;
            std::vector<State> order;
            std::vector<std::vector<State>> scc;

            for (auto& pair : graph) {
                State node = pair.first;
                if (visited.find(node) == visited.end()) {
                    first_dfs(node, visited, order);
                }
            }

            Graph transposed = transpose();

            visited.clear();

            while (!order.empty()) {
                State node = order.back();
                order.pop_back();
                if (visited.find(node) == visited.end()) {
                    std::vector<State> component;
                    second_dfs(node, visited, component, transposed);
                    scc.push_back(component);
                }
            }

            return scc;
        }
};