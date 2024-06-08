package pkg

import (
	"fmt"
	"os"
	"sort"
	"strconv"
)

func WriteInputData(file *os.File, n int, phi, psi string) {
	result := fmt.Sprintf("n: %d\n", n)
	result += fmt.Sprintf("phi: %s\n", phi)
	result += fmt.Sprintf("psi: %s\n\n", psi)
	_, err := file.WriteString(result)
	if err != nil {
		panic(err)
	}
}

func WriteTask1(file *os.File, connectivityComponents [][]State) {
	var (
		err    error
		result string
	)

	_, err = file.WriteString("TASK1\n\n")
	if err != nil {
		panic(err)
	}
	for _, components := range connectivityComponents {
		result = fmt.Sprintf("Connected components: ")
		result = result + "{"
		for idx, component := range components {
			if idx == 0 {
				result = result + fmt.Sprintf("%s", component.Value)
			} else {
				result = result + ", " + fmt.Sprintf("%s", component.Value)
			}
		}
		result = result + "}\n"
		_, err = file.WriteString(result)
		if err != nil {
			panic(err)
		}
		result = ""
	}

	number := "Number of components: " + strconv.Itoa(len(connectivityComponents)) + "\n\n"
	_, err = file.WriteString(number)
	if err != nil {
		panic(err)
	}
}

func WriteTask2(file *os.File, connectivityComponents [][]State) {
	var (
		err    error
		result string
	)

	_, err = file.WriteString("TASK2\n\n")
	if err != nil {
		panic(err)
	}
	for _, components := range connectivityComponents {
		result = fmt.Sprintf("Strong connected components: ")
		result = result + "{"
		for idx, component := range components {
			if idx == 0 {
				result = result + fmt.Sprintf("%s", component.Value)
			} else {
				result = result + ", " + fmt.Sprintf("%s", component.Value)
			}
		}
		result = result + "}\n"
		_, err = file.WriteString(result)
		if err != nil {
			panic(err)
		}
		result = ""
	}

	number := "Number of strong components: " + strconv.Itoa(len(connectivityComponents)) + "\n\n"
	_, err = file.WriteString(number)
	if err != nil {
		panic(err)
	}
}

func WriteTask3(file *os.File, equivalenceClasses map[int][]EquivalenceClass, delta, mu int) {
	var (
		err    error
		result string
		keys   []int
	)
	keys = make([]int, 0)

	_, err = file.WriteString("TASK3\n\n")
	if err != nil {
		panic(err)
	}

	for key := range equivalenceClasses {
		keys = append(keys, key)
	}
	sort.Ints(keys)

	for _, key := range keys {
		result = fmt.Sprintf("Equivalence class %d\n", key)
		for idx, subclass := range equivalenceClasses[key] {
			result = result + fmt.Sprintf("Subclass %d: {", idx)
			for index, state := range subclass.Class {
				if index == 0 {
					result = result + fmt.Sprintf("%s", state.Value)
				} else {
					result = result + ", " + state.Value
				}
			}
			result = result + "}\n"
			_, err = file.WriteString(result)
			if err != nil {
				panic(err)
			}
			result = ""
		}
	}

	numbers := fmt.Sprintf("delta(A): %d\n", delta) + fmt.Sprintf("mu(A): %d\n", mu) + "\n\n"
	_, err = file.WriteString(numbers)
	if err != nil {
		panic(err)
	}
}

func WriteTask4(file *os.File, q []map[State]map[State][][]int, polinom string) {
	var (
		err    error
		result string
	)

	_, err = file.WriteString("TASK4\n\n")
	if err != nil {
		panic(err)
	}
	for idx, nextQ := range q {
		keys := make([]string, 0)
		for key := range nextQ {
			keys = append(keys, key.Value)
			sort.Strings(keys)
		}
		for _, key := range keys {
			state := NewStateFromString(key)
			result = fmt.Sprintf("q_%d(%s): \n", idx+1, key)
			for _, val := range nextQ[state] {
				for _, v := range val {
					result = result + fmt.Sprintf("%d \n", v)
				}
			}
			_, err = file.WriteString(result)
			if err != nil {
				panic(err)
			}
			result = ""
		}
	}

	result = ""
	result = fmt.Sprintf("Память автомата: m(A) = %d\n", len(q))
	result = result + fmt.Sprintf("Функция памяти автомата: %s\n", polinom)
	_, err = file.WriteString(result)
	if err != nil {
		panic(err)
	}
}

func WriteTask5(file *os.File, initialState, polynomial string, length int) {
	var (
		err    error
		result string
	)

	result = fmt.Sprintf("TASK5\n\n")
	result = result + fmt.Sprintf("Initial state: %s\n", initialState)
	result = result + fmt.Sprintf("Minimal Polynomial: %s\n", polynomial)
	result = result + fmt.Sprintf("Linear Complexity: %d\n", length)
	_, err = file.WriteString(result)
	if err != nil {
		panic(err)
	}
}
