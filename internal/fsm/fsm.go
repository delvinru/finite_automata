package fsm

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hmdsefi/gograph"
	"github.com/hmdsefi/gograph/traverse"
	"gonum.org/v1/gonum/stat/combin"
	"gonum.org/v1/gonum/graph"
)

type Pair[T, U any] struct {
	First  T
	Second U
}

type FSM struct {
	n   int     // length of polynomial
	phi []uint8 // phi(x_1, ..., x_n, x)
	psi []uint8 // psi(x_1, ..., x_n, x)

	graph gograph.Graph[State]
	table map[State]Pair[[]State, []uint8]
}

func (f *FSM) ConnectedComponents() {
	visited := make(map[*gograph.Vertex[State]]bool, len(f.graph.GetAllVertices()))

	connectedComponents := [][]State{}

	for _, node := range f.graph.GetAllVertices() {
		fmt.Println("visited", visited)
		if !visited[node] {
			iter, _ := traverse.NewDepthFirstIterator(f.graph, node.Label())
			tmp := []State{}
			for iter.Iterate(func(v *gograph.Vertex[State]) error {
				fmt.Println(node.Label(), "->", value.Label())
			})

			connectedComponents = append(connectedComponents, tmp)
		}
	}

	fmt.Println(connectedComponents)

	// fmt.Println(connectivity.Kosaraju(f.graph))
	// fmt.Println(connectivity.Tarjan(f.graph))
}

func getFunctionCoeffs(function string) ([]uint8, error) {
	coeffs := make([]uint8, len(function))

	for idx := range len(function) {
		n := function[idx] - 0x30
		if n != 0 && n != 1 {
			return nil, errors.New("function coeffs can be only 0 or 1")
		}
		coeffs[idx] = n
	}
	return coeffs, nil
}

func joinArray(array []uint8) string {
	str := make([]string, len(array))
	for i, v := range array {
		str[i] = strconv.Itoa(int(v))
	}
	return strings.Join(str, "")
}

func New(n int, phi string, psi string) (FSM, error) {
	phis, err := getFunctionCoeffs(phi)
	if err != nil {
		return FSM{}, err
	}
	psis, err := getFunctionCoeffs(psi)
	if err != nil {
		return FSM{}, err
	}

	if n <= 0 {
		return FSM{}, errors.New("length can't be negative")
	}

	if len(psis) != int(math.Pow(2, float64(n+1))) {
		return FSM{}, errors.New("length of psi don't equal 2^n")
	}

	if len(phis) != int(math.Pow(2, float64(n+1))) {
		return FSM{}, errors.New("length of phi don't equal 2^n")
	}

	// init structure
	fsm := FSM{
		n:     n,
		phi:   phis,
		psi:   psis,
		graph: gograph.New[State](gograph.Directed()),
		table: map[State]Pair[[]State, []uint8]{},
	}

	// Init graph
	initFsm(&fsm)

	return fsm, nil
}

func initFsm(fsm *FSM) {
	for state := range generateBinaryCombinations(fsm.n) {
		graphNode := NewState(joinArray(state))

		phis := []State{}
		psis := []uint8{}

		for x := range 2 {
			zpPhi := computeZhegalkinPolynomial(uint8(x), state, fsm.phi)
			zpPsi := computeZhegalkinPolynomial(uint8(x), state, fsm.psi)

			newState := NewState(joinArray(append(state[1:], zpPhi)))

			// init graph
			fsm.graph.AddEdge(gograph.NewVertex(graphNode), gograph.NewVertex(newState))

			// init table
			phis = append(phis, newState)
			psis = append(psis, zpPsi)
		}

		fsm.table[graphNode] = Pair[[]State, []uint8]{phis, psis}
	}
}

func generateBinaryCombinations(n int) <-chan []uint8 {
	result := make(chan []uint8)

	go func() {
		defer close(result)

		var generateCombinationsHelper func([]uint8, int)
		generateCombinationsHelper = func(currentCombination []uint8, index int) {
			if index == n {
				combination := make([]uint8, n)
				copy(combination, currentCombination)
				result <- combination
				return
			}
			currentCombination[index] = 0
			generateCombinationsHelper(currentCombination, index+1)
			currentCombination[index] = 1
			generateCombinationsHelper(currentCombination, index+1)
		}

		initialCombination := make([]uint8, n)
		generateCombinationsHelper(initialCombination, 0)
	}()

	return result
}

func computeZhegalkinPolynomial(inputX uint8, currentState []uint8, coeffs []uint8) uint8 {
	// initial state, why?
	zp := []uint8{1}
	extendedCurrentState := append(currentState, inputX)

	for i := 1; i < len(extendedCurrentState)+1; i++ {
		gen := combin.NewCombinationGenerator(len(extendedCurrentState), i)
		for gen.Next() {
			var product uint8 = 1
			for _, idx := range gen.Combination(nil) {
				product *= extendedCurrentState[idx]
			}
			zp = append(zp, uint8(product))
		}
	}

	if len(zp) != len(coeffs) {
		panic("Wtf happen! Why zp != coeffs")
	}

	var count uint8 = 0
	for i := range len(coeffs) {
		zp[i] *= coeffs[i]
		if zp[i] == 1 {
			count++
		}
	}
	return count % 2
}
