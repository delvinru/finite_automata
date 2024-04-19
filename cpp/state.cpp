#include <vector>
#include <iostream>

class State {
    public:
        std::vector<int> key;
        State(std::vector<int> state_key) {
            key = state_key;
        }

        void print_key() {
            for (int i = 0; i < key.size(); i++)
                std::cout << key[i];
        }

        bool operator<(const State& other) const {
            return key < other.key;
        }

        bool operator==(const State& other) const {
            return key == other.key;
        }

        friend std::ostream& operator<<(std::ostream& os, const State& state) {
            for (int i = 0; i < state.key.size(); i++)
                os << state.key[i];
            return os;
        }
};