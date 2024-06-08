package fsm

import (
	"fmt"
	"log/slog"
	"math"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/delvinru/finite_automata/internal/filemanager"

	"golang.org/x/exp/maps"
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
							slog.Debug("minimization",
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

func (f *FSM) MemoryFunction() int {
	// timer for 30 seconds
	timeout := time.After(20 * time.Second)

	// channel to signal
	done := make(chan bool)

	go func() {
		isEdgesInArray := func(a, b [][]uint8) bool {
			return reflect.DeepEqual(a, b)
		}

		// create copy of object
		fsm := f

		// check if automate is minimal
		if fsm.Mu != len(fsm.table) {
			slog.Info("task 4: memory function, doing minimization")
			fsm.minimization()
		}

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
			slog.Info("task 4", "q", len(q_s))
		}
		// Directly use filemanager because this function is HUGE MEMORY CONSUPTION

		if len(q_s) > maxSteps {
			// TODO: print that memory infinity
			slog.Info("task 4: memory infinity")
			return
		}

		for i, q := range q_s {
			for state, elements := range q {
				filemanager.Write(fmt.Sprintf("\nq_%v(%v):\n", i+1, state))

				for j, element := range elements {
					for _, value := range maps.Values(element) {
						filemanager.Write(fmt.Sprintf("%v", value))
					}
					if j != len(elements)-1 {
						filemanager.Write(" |_|")
					}
					filemanager.Write("\n")
				}
			}
			filemanager.Write("\n")
		}

		filemanager.Write(fmt.Sprintf("Память автомата конечна: m(A)=%v\n", len(q_s)))

		// force call garbage collector
		runtime.GC()

		slog.Info("task 4: compute memory value vector")
		memoryValueVector := f.getMemoryValueVector(q_s[len(q_s)-1], len(q_s))

		slog.Info("task 4: convert memory value vector to polynomial")
		polynomial := f.convertToPylonomial(memoryValueVector)

		slog.Info("task 4: get human readable polynomial")
		filemanager.Write(fmt.Sprintf("Функция памяти автомата: %v\n", f.convertPolynomialToString(polynomial)))

		done <- true
	}()

	select {
	case <-done:
		slog.Info("task 4: computation done")
		return 0
	case <-timeout:
		slog.Warn("task 4: hard computation, killed by timeout")
		filemanager.Write("\nВычисления приостановлены из-за большого потребления ресурсов\n")
		return 1
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

func (f *FSM) getMemoryValueVector(q Q, memory int) []uint8 {
	memoryValueVector := []uint8{}

	compareTable := map[[2]State][]uint8{}
	for firstCombination := range generateBinaryCombinations(memory) {
		for secondCombination := range generateBinaryCombinations(memory) {
			key := [2]State{
				NewState(firstCombination),
				NewState(secondCombination),
			}
			compareTable[key] = make([]uint8, 2)
		}
	}

	for state, elements := range q {
		for _, element := range elements {
			for _, vectors := range maps.Values(element) {
				key := [2]State{
					NewState(vectors[0]),
					NewState(vectors[1]),
				}

				compareTable[key][0] = f.table[state].Psi[0]
				compareTable[key][1] = f.table[state].Psi[1]
			}
		}
	}

	// Генерируем специально последовательно от элементов на 1 больше, чтобы корректно собрать "большую" таблицу
	for combination := range generateBinaryCombinations(2*memory + 1) {
		// Игнорируем элемент посередине, т.к. он является входным значением
		key := [2]State{
			NewState(combination[:memory]),
			NewState(combination[memory+1:]),
		}

		// Забираем это значение посередине
		memoryValueVector = append(memoryValueVector, compareTable[key][combination[memory]])
	}

	return memoryValueVector
}

func (f *FSM) convertToPylonomial(sequence []uint8) []uint8 {
	var (
		sequenceLeft  = make([]uint8, len(sequence)/2)
		sequenceRight = make([]uint8, len(sequence)/2)
		sequenceOut   = make([]uint8, len(sequence))

		temp1 []uint8
		temp2 []uint8
	)

	for i := 0; i < len(sequence)/2; i++ {
		sequenceLeft[i] = sequence[i]
		sequenceRight[i] = (sequence[i] + sequence[i+len(sequence)/2]) % 2
	}

	if len(sequence) == 2 {
		sequenceOut[0] = sequenceLeft[0]
		sequenceOut[1] = sequenceRight[0]

		return sequenceOut
	}

	temp1 = f.convertToPylonomial(sequenceLeft)
	temp2 = f.convertToPylonomial(sequenceRight)

	for i := 0; i < len(sequence)/2; i++ {
		sequenceOut[i] = temp1[i]
		sequenceOut[i+len(sequenceOut)/2] = temp2[i]
	}

	return sequenceOut
}

func (f *FSM) convertPolynomialToString(polynomial []uint8) string {
	var vectorString strings.Builder
	vectorString.Grow(len(polynomial))

	if polynomial[0] == 1 {
		vectorString.WriteString("1 ⊕ ")
	}

	lengthOfVector := int(math.Log2(float64(len(polynomial))))

	for i := 1; i < len(polynomial); i++ {
		if polynomial[i] == 1 {
			binValue := fmt.Sprintf("%0*b", lengthOfVector, i)
			for coef, bit := range binValue {
				if bit == '1' {
					coefStr := ""
					if coef < lengthOfVector/2 {
						coefStr += "x_(i"
						coefStr += fmt.Sprintf("-%d)", (lengthOfVector/2)-coef)
					} else if coef > lengthOfVector/2 {
						coefStr += "y_(i"
						coefStr += fmt.Sprintf("-%d)", lengthOfVector-coef)
					} else {
						coefStr += "x_i"
					}
					vectorString.WriteString(coefStr)

				}
			}
			vectorString.WriteString(" ⊕ ")
		}
	}

	return strings.TrimSuffix(vectorString.String(), " ⊕ ")
}
