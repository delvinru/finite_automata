package fsm

import (
	"log/slog"
)

func (f *FSM) minimization() {
	// assume that equivalence classes already precomputed

	for _, set_states := range f.EquivalenceClasses[f.Delta] {
		if len(set_states) > 1 {
			equivalentState := set_states[0]
			for i := 1; i < len(set_states); i++ {
				for _, table_tuple := range f.table {
					for j, state := range table_tuple.Phi {
						if state == set_states[i] {
							table_tuple.Phi[j] = equivalentState
						}
					}
				}
			}
		}
	}

	f.GetEquivalenceClasses()
}

func (f *FSM) isEqualEdgesInQ(q map[State][]map[State][][]uint8) {
	uniqueEdges := map[[]uint8]bool{}
	for _, edges := range q {
		for _, edge := range edges {
			for _, pair := range edge {
				checkPair := [2]uint8{pair[0], pair[1]}
			}
		}
	}
}

func (f *FSM) MemoryFunction() error {
	if len(f.EquivalenceClasses) == 0 {
		slog.Debug("computing equivalence classess")
		f.GetEquivalenceClasses()
	}

	// create copy
	fsmCopy := f

	if f.Mu != len(f.table) {
		slog.Debug("automate not minimal, doing minimization")
		fsmCopy.minimization()
	}

	// compute q_1
	q_1 := map[State][]map[State][][]uint8{}

	for keyState := range fsmCopy.table {
		edgesList := map[[2]uint8]bool{}
		for valueState, edges := range fsmCopy.table {
			for i := range 2 {
				tmpSet := [2]uint8{uint8(i), edges.Psi[i]}
				if edges.Phi[i] == keyState && !edgesList[tmpSet] {
					q_1[keyState] = append(q_1[keyState], map[State][][]uint8{valueState: {{uint8(i)}, {edges.Psi[i]}}})
					edgesList[tmpSet] = true
				}
			}
		}
	}

	q_s := []map[State][]map[State][][]uint8{q_1}

	maxSteps := (fsmCopy.Mu * (fsmCopy.Mu - 1)) / 2

	// if f.Mu != len(f.table) {
	// 	fmt.Println("make minimization")
	// 	f.minimization()
	// }

	return nil
}
