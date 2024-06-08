package pkg

import (
	"fmt"
	"gonum.org/v1/gonum/stat/combin"
	"math"
	"strconv"
	"strings"
)

type FSM struct {
	n                int               // длина регистра
	phi              []int             // функция фи
	psi              []int             // функция пси
	graph            *Graph            // граф автомата
	PhiTable         map[State][]State // таблица состояний, полученных из функции фи
	PsiTable         map[State][]int   // таблица состояний, полученных из функции пси
	Delta            int
	Mu               int
	equivalenceClass []EquivalenceClass
	Q                map[State]map[State][][]int
}

// Создание нового экземпляра класса FSM. На вход методу подается длина регистра, функции фи и пси. Метод взвращает
// экземпляр класса FSM
// Входные данные: n int, phi, psi []int
// Выходные данные: *FSM
func NewFSM(n int, phi, psi []int) *FSM {
	return &FSM{
		n:        n,
		phi:      phi,
		psi:      psi,
		graph:    NewGraph(),
		PhiTable: make(map[State][]State),
		PsiTable: make(map[State][]int),
		Q:        make(map[State]map[State][][]int),
	}
}

// Получение компонентов связности автомата
// Входные данные: null
// Выходные данные: [][]State
func (f *FSM) GetConnectivityComponents() [][]State {
	return f.graph.GetConnectivityComponents()
}

// Получение компонентов сильной связности автомата
// Входные данные: null
// Выходные данные: [][]State
func (f *FSM) GetStrongConnectivityComponents() [][]State {
	return f.graph.GetStrongConnectivityComponents()
}

func (f *FSM) GetMemoryFunction() ([]map[State]map[State][][]int, string) {
	var (
		q []map[State]map[State][][]int
	)
	fsm := f

	if fsm.Mu != int(math.Pow(2, float64(f.n))) {
		fsm.minimization()
	}

	fsm.getFirstQ()
	q_1 := fsm.Q
	q = append(q, q_1)

	/*for key, mapa := range fsm.Q {
		fmt.Printf("q_%d(%s): \n", len(q), key)
		for _, val := range mapa {
			//fmt.Printf("%s: \n", k)
			for _, v := range val {
				fmt.Printf("%d \n", v)
			}
		}
	}*/

	maxSteps := (fsm.Mu * (fsm.Mu - 1)) / 2
	for isEqualEdgesInQ(q[len(q)-1]) && len(q) <= maxSteps {
		nextQ := make(map[State]map[State][][]int)
		for state, edges := range q[len(q)-1] { // итерация по q
			nextQ[state] = make(map[State][][]int)
			for point, edge := range edges { // итерация по точкам откуда исходит ребро
				for key, arrays := range q[0][point] { // итерация по вершине, которая находиться в q в начале
					for _, val := range edge {
						for _, value := range arrays {
							resultArray := make([]int, 0)
							resultArray = append(resultArray, value[0])
							resultArray = append(resultArray, val[:len(val)/2]...)
							resultArray = append(resultArray, value[1])
							resultArray = append(resultArray, val[len(val)/2:]...)
							nextQ[state][key] = append(nextQ[state][key], resultArray)
						}
					}
				}
			}
		}

		q = append(q, nextQ)
	}

	n := 0
	for _, edges := range q[len(q)-1] {
		for _, edge := range edges {
			n = len(edge[0])
			break
		}
		break
	}
	table := generateTruthTable(n + 1)

	result := make([]int, 0)
	for i := 0; i < len(table); i++ {
		if i == 25 {
			fmt.Printf("25")
		}
		str := make([]int, 0)
		str = append(str, table[i][:len(table[i])/2]...)
		str = append(str, table[i][len(table[i])/2+1:]...)
		elem := table[i][(len(table[i])-1)/2]
		result = append(result, f.getElemPsiTable(str, elem, q[len(q)-1]))
	}

	polinom := transformationToPolinom(result)
	str := getMemoryFunctionCoefsStr(polinom)
	fmt.Println(str)

	return q, str
}

// Инициализация класса FSM. В рамках данной функции создаются граф, таблица фи и пси, где находятся состояния
func (f *FSM) Init() {
	var (
		zpPhi int
		zpPsi int
	)
	for state := range generateBinaryCombinations(f.n) {
		node := NewState(state)
		for x := range 2 {
			zpPhi = ComputePolinom(x, state, f.phi)
			zpPsi = ComputePolinom(x, state, f.psi)

			newState := NewState(append(state[1:], zpPhi))
			weight := strconv.Itoa(x) + strconv.Itoa(zpPsi)

			f.PhiTable[node] = append(f.PhiTable[node], newState)
			f.PsiTable[node] = append(f.PsiTable[node], zpPsi)

			f.graph.AddEdge(node, newState, weight)
		}
	}
}

// Вычисление полинома Жегалкина. На выход функции подается значение inputX, которое будет иметь значения {0,1}, состояение
// state и функция psi.
// Входные данные: inputX int, state, psi []int
// Выходные данные: int
func ComputePolinom(inputX int, state, psi []int) int {
	zp := []int{1}
	extendedCurrentState := append(state, inputX)

	for i := 1; i < len(extendedCurrentState)+1; i++ {
		gen := combin.NewCombinationGenerator(len(extendedCurrentState), i)
		for gen.Next() {
			var product = 1
			for _, idx := range gen.Combination(nil) {
				product *= extendedCurrentState[idx]
			}
			zp = append(zp, product)
		}
	}

	if len(zp) != len(psi) {
		panic("Wtf happen! Why zp != coeffs")
	}

	var count = 0
	for i := range len(psi) {
		zp[i] *= psi[i]
		if zp[i] == 1 {
			count++
		}
	}
	return count % 2
}

func (f *FSM) AddEquivalenceClass(class []EquivalenceClass) {
	f.equivalenceClass = class
}

func (f *FSM) minimization() {
	for _, subclass := range f.equivalenceClass {
		fixedState := subclass.Class[0]
		if len(subclass.Class) > 1 {
			for i := 1; i <= len(subclass.Class); i++ {
				for key, state := range f.PhiTable {
					if subclass.Class[i] == key {
						state[0] = fixedState
						state[1] = fixedState
					}
				}
				delete(f.PhiTable, subclass.Class[i])
			}
		}
	}
}

func (f *FSM) getFirstQ() {
	for key, states := range f.PhiTable {
		for idx, state := range states {
			tmp := make([]int, 0)
			if len(f.Q[state]) == 0 {
				f.Q[state] = make(map[State][][]int)
			}
			tmp = append(tmp, idx, f.PsiTable[key][idx])
			if !isArrayInStack(tmp, f.Q[state]) {
				f.Q[state][key] = append(f.Q[state][key], tmp)
			}
		}
	}
}

func isEqualEdgesInQ(q map[State]map[State][][]int) bool {
	seen := make(map[string]State)
	for _, edges := range q {
		for idx, edge := range edges {
			key := fmt.Sprintf("%v", edge)
			// проверяем есть ли сторона в просмотренных
			if _, exists := seen[key]; exists {
				// нашли дупликат
				return true
			}
			seen[key] = idx
		}
	}
	return false
}

func isArrayInStack(array []int, state map[State][][]int) bool {
	for _, arrays := range state {
		for _, value := range arrays {
			if compareArrays(array, value) {
				return true
			} else {
				continue
			}
		}
	}
	return false
}

func compareArrays(a, b []int) bool {
	for i := 0; i < len(a); i++ {
		if a[i] == b[i] {
			continue
		} else {
			return false
		}
	}
	return true
}

// Данная функция генерирует значения {0,1}. Генерация выполняется n раз, где n - длина регистра автомата
// Входные данные: n
// Выходные данные: []int = {0,1}
func generateBinaryCombinations(n int) <-chan []int {
	result := make(chan []int)

	go func() {
		defer close(result)

		var generateCombinationsHelper func([]int, int)
		generateCombinationsHelper = func(currentCombination []int, index int) {
			if index == n {
				combination := make([]int, n)
				copy(combination, currentCombination)
				result <- combination
				return
			}
			currentCombination[index] = 0
			generateCombinationsHelper(currentCombination, index+1)
			currentCombination[index] = 1
			generateCombinationsHelper(currentCombination, index+1)
		}

		initialCombination := make([]int, n)
		generateCombinationsHelper(initialCombination, 0)
	}()

	return result
}

func generateTruthTable(n int) [][]int {
	// Количество строк в таблице истинности равно 2^n.
	numRows := int(math.Pow(2, float64(n)))
	// Инициализация таблицы истинности.
	truthTable := make([][]int, numRows)

	// Заполнение таблицы истинности.
	for i := 0; i < numRows; i++ {
		row := make([]int, n)
		for j := 0; j < n; j++ {
			// Вычисление значения переменной (0 или 1).
			row[j] = (i >> uint(n-1-j)) & 1
		}
		truthTable[i] = row
	}

	return truthTable
}

func (f *FSM) getElemPsiTable(array []int, elem int, q map[State]map[State][][]int) int {
	result := 0
	for key, edges := range q {
		for _, edge := range edges {
			for _, val := range edge {
				if compareArrays(val, array) {
					result = f.PsiTable[key][elem]
					return result
				}
			}
		}
	}
	return result
}

func transformationToPolinom(seq []int) []int {
	var (
		seqLeft  []int
		seqRight []int
		seqOut   []int
		temp1    []int
		temp2    []int
	)

	seqLeft = make([]int, len(seq)/2)
	seqRight = make([]int, len(seq)/2)
	seqOut = make([]int, len(seq))

	for i := 0; i < len(seq)/2; i++ {
		seqLeft[i] = seq[i]
		seqRight[i] = (seq[i] + seq[i+len(seq)/2]) % 2
	}

	if len(seq) == 2 {
		seqOut[0] = seqLeft[0]
		seqOut[1] = seqRight[0]
		return seqOut
	}

	temp1 = transformationToPolinom(seqLeft)
	temp2 = transformationToPolinom(seqRight)

	for i := 0; i < len(seqOut)/2; i++ {
		seqOut[i] = temp1[i]
		seqOut[i+len(seqOut)/2] = temp2[i]
	}

	return seqOut
}

func getMemoryFunctionCoefsStr(vector []int) string {
	vectorString := ""
	if vector[0] == 1 {
		vectorString += "1 + "
	}

	lengthOfVector := int(math.Log2(float64(len(vector))))

	for i := 1; i < len(vector); i++ {
		if vector[i] == 1 {
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
					vectorString += coefStr
				}
			}
			vectorString += " + "
		}
	}

	vectorString = strings.TrimSuffix(vectorString, " + ")
	return vectorString
}
