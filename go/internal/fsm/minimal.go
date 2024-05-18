package fsm

import (
	"fmt"
	"log/slog"
	"reflect"
	"slices"
	"sync"
)

type Q map[State][]map[State][][]uint8

func (f *FSM) minimization() {
	// get last equivalence class and iterate over it
	for _, setStates := range f.EquivalenceClasses[f.Delta] {
		length := len(setStates)
		if length > 1 {
			equivalentState := setStates[0]
			for i := 1; i < length; i++ {
				for _, tableTuple := range f.table {
					for j, state := range tableTuple.Phi {
						if state == setStates[i] {
							slog.Info("minimization",
								"duplicate", state,
							)
							tableTuple.Phi[j] = equivalentState
						}
					}
				}
				delete(f.table, setStates[i])
			}
		}
	}

	// recompute equivalence classess
	f.GetEquivalenceClasses()
}

func (f *FSM) MemoryFunction() {
	slog.Info("computing memory function")
	isEdgesInArray := func(a, b [][]uint8) bool {
		return reflect.DeepEqual(a, b)
	}

	// create copy of object
	fsm := f

	// check if automate is minimal
	if fsm.Mu != len(fsm.table) {
		slog.Info("memory function, doing minimization")
		fsm.minimization()
	}

	slog.Info("memory func", "mu", fsm.Mu)

	// q_1 - special Q
	q_1 := Q{}
	// q_s - hold all Q_i(S_i)
	q_s := []Q{}

	// compute q_1
	// iterate over all states and find connections
	for keyState := range fsm.table {
		edgesList := [][][]uint8{}
		for valueState, edges := range fsm.table {
			// less than 2 because we store in table Pair of 2 elements
			for i := 0; i < 2; i++ {
				tmpList := [][]uint8{
					{uint8(i)},
					{edges.Psi[i]},
				}

				if edges.Phi[i] == keyState && !slices.ContainsFunc(edgesList, func(elem [][]uint8) bool {
					return isEdgesInArray(elem, tmpList)
				}) {
					q_1[keyState] = append(q_1[keyState], map[State][][]uint8{valueState: tmpList})
					edgesList = append(edgesList, tmpList)
				}
			}
		}
	}
	// append q_1 to q_s
	q_s = append(q_s, q_1)

	maxSteps := (fsm.Mu * (fsm.Mu - 1)) / 2

	// NOTE: this is very slow func that's why we use goroutines
	for f.isEqualEdgesInQ(q_s[len(q_s)-1]) && len(q_s) <= maxSteps {
		nextQ := Q{}
		results := make(chan struct {
			state  State
			result map[State][][]uint8
		}, len(q_s[len(q_s)-1])*len(q_s[len(q_s)-1])*2) // buffered channel

		var wg sync.WaitGroup

		for state, edges := range q_s[len(q_s)-1] {
			for _, edge := range edges {
				wg.Add(1)

				go func(state State, edge map[State][][]uint8) {
					defer wg.Done()

					edgesList := make(map[string]struct{})

					var fromState State
					var values [][]uint8

					// TODO: research, hacky way to mimi list(edge.items)[0]
					for key, value := range edge {
						fromState = key
						values = value
						break
					}

					// TODO: some magic shit (not code) in math logic
					if _, exists := q_1[fromState]; !exists {
						return
					}

					for _, edgesFromState := range q_1[fromState] {
						for anotherState, anotherEdge := range edgesFromState {
							key := fmt.Sprintf("%v", slices.Concat(anotherEdge[0], values[0], anotherEdge[1], values[1]))
							if _, exists := edgesList[key]; !exists {
								result := map[State][][]uint8{anotherState: {append(anotherEdge[0], values[0]...), append(anotherEdge[1], values[1]...)}}
								edgesList[key] = struct{}{}
								results <- struct {
									state  State
									result map[State][][]uint8
								}{state, result}
							}
						}
					}

				}(state, edge)
			}
		}

		go func() {
			wg.Wait()
			close(results)
		}()

		for res := range results {
			nextQ[res.state] = append(nextQ[res.state], res.result)
		}

		q_s = append(q_s, nextQ)
		slog.Info(fmt.Sprintf("compute q_%v", len(q_s)))
	}
}

func (f *FSM) isEqualEdgesInQ(q Q) bool {
	allEdges := [][][]uint8{}

	for _, edges := range q {
		for _, edge := range edges {
			for _, value := range edge {
				allEdges = append(allEdges, value)
			}
		}
	}

	seen := make(map[string]struct{})
	for _, edge := range allEdges {
		// convert edge to string for use as key for map
		key := fmt.Sprintf("%v", edge)
		// check if edge exists in seen
		if _, exists := seen[key]; exists {
			// found duplicate
			return true
		}
		// add current element in set
		seen[key] = struct{}{}
	}

	return false
}

// func (f *FSM) minimization() {
// 	// assume that equivalence classes already precomputed

// 	for _, set_states := range f.EquivalenceClasses[f.Delta] {
// 		if len(set_states) > 1 {
// 			equivalentState := set_states[0]
// 			for i := 1; i < len(set_states); i++ {
// 				for _, table_tuple := range f.table {
// 					for j, state := range table_tuple.Phi {
// 						if state == set_states[i] {
// 							table_tuple.Phi[j] = equivalentState
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}

// 	f.GetEquivalenceClasses()
// }

// func (f *FSM) isEqualEdgesInQ(q map[State][]map[State][][]uint8) {
// 	uniqueEdges := map[[]uint8]bool{}
// 	for _, edges := range q {
// 		for _, edge := range edges {
// 			for _, pair := range edge {
// 				checkPair := [2]uint8{pair[0], pair[1]}
// 			}
// 		}
// 	}
// }

// func (f *FSM) MemoryFunction() error {
// 	if len(f.EquivalenceClasses) == 0 {
// 		slog.Debug("computing equivalence classess")
// 		f.GetEquivalenceClasses()
// 	}

// 	// create copy
// 	fsmCopy := f

// 	if f.Mu != len(f.table) {
// 		slog.Debug("automate not minimal, doing minimization")
// 		fsmCopy.minimization()
// 	}

// 	// compute q_1
// 	q_1 := map[State][]map[State][][]uint8{}

// 	for keyState := range fsmCopy.table {
// 		edgesList := map[[2]uint8]bool{}
// 		for valueState, edges := range fsmCopy.table {
// 			for i := range 2 {
// 				tmpSet := [2]uint8{uint8(i), edges.Psi[i]}
// 				if edges.Phi[i] == keyState && !edgesList[tmpSet] {
// 					q_1[keyState] = append(q_1[keyState], map[State][][]uint8{valueState: {{uint8(i)}, {edges.Psi[i]}}})
// 					edgesList[tmpSet] = true
// 				}
// 			}
// 		}
// 	}

// 	q_s := []map[State][]map[State][][]uint8{q_1}

// 	maxSteps := (fsmCopy.Mu * (fsmCopy.Mu - 1)) / 2

// 	// if f.Mu != len(f.table) {
// 	// 	fmt.Println("make minimization")
// 	// 	f.minimization()
// 	// }

// 	return nil
// }
