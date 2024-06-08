package pkg

type EquivalenceClass struct {
	Class []State
}

// Создаем класс новый экземпляр класса эквивалентности
func NewEquivalenceClass() *EquivalenceClass {
	return &EquivalenceClass{
		Class: make([]State, 0),
	}
}

// Сортировка класса эквивалентности
func (ce *EquivalenceClass) sort() {
	quickSort(ce.Class, 0, len(ce.Class)-1)
}

// Алгоритм быстрой сортировки
func quickSort(arr []State, low, high int) []State {
	if low < high {
		var p int
		arr, p = partition(arr, low, high)
		arr = quickSort(arr, low, p-1)
		arr = quickSort(arr, p+1, high)
	}
	return arr
}

func partition(arr []State, low, high int) ([]State, int) {
	pivot := arr[high]
	i := low
	for j := low; j < high; j++ {
		if arr[j].Value < pivot.Value {
			arr[i], arr[j] = arr[j], arr[i]
			i++
		}
	}
	arr[i], arr[high] = arr[high], arr[i]
	return arr, i
}

// Получаем все классы эквивалентности
func (ce *EquivalenceClass) GetEquivalenceClasses(phiTable map[State][]State, psiTable map[State][]int) map[int][]EquivalenceClass {
	var (
		result map[int][]EquivalenceClass
		key    = 1
	)
	result = make(map[int][]EquivalenceClass)
	result[1] = ce.getFirstClass(psiTable)

	for _, class := range result {
		for _, state := range class {
			state.sort()
		}
	}

	for {
		class := result[key]
		result[key+1] = ce.getNextClass(phiTable, class)
		if len(result[key+1]) == len(result[key]) {
			delete(result, key+1)
			break
		} else {
			key++
			continue
		}
	}

	return result
}

// Получаем первый класс эквивалентность, используя таблицу переходов, полученную из функции "пси"
func (ce *EquivalenceClass) getFirstClass(psiTable map[State][]int) []EquivalenceClass {
	var (
		result      []EquivalenceClass
		class       EquivalenceClass
		dictClasses = [][]int{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1},
		}
	)
	result = make([]EquivalenceClass, 0)
	for _, v := range dictClasses {
		for state := range psiTable {
			if compareArray(psiTable[state], v) {
				class.Class = append(class.Class, state)
			}
		}
		if len(class.Class) > 0 {
			result = append(result, class)
		}
		class.Class = nil
	}

	return result
}

// Получаем следующие классы эквивалентности, используя таблицу переходов, полученную из функции "пси"
func (ce *EquivalenceClass) getNextClass(phiTable map[State][]State, class []EquivalenceClass) []EquivalenceClass {
	var (
		result []EquivalenceClass
	)
	result = make([]EquivalenceClass, 0)

	for _, value := range class {
		temp := value.Class
		compute(temp, class, &result, phiTable)
	}

	return result
}

func compute(states []State, class []EquivalenceClass, result *[]EquivalenceClass, phiTable map[State][]State) {
	for _, state := range states {
		newClass := make([]State, 0)
		splitClass := make([]State, 0)
		for i := 1; i < len(states); i++ {
			if getSubclassNumber(phiTable[state][0], class) == getSubclassNumber(phiTable[states[i]][0], class) &&
				getSubclassNumber(phiTable[state][1], class) == getSubclassNumber(phiTable[states[i]][1], class) {
				newClass = append(newClass, states[i]) //те что совпали
			} else {
				splitClass = append(splitClass, states[i]) // те что не совпали
			}
		}
		if len(newClass) > 0 && len(splitClass) > 0 {
			newClass = append(newClass, state)
			equivalenceClass := EquivalenceClass{newClass}
			equivalenceClass.sort()
			*result = append(*result, equivalenceClass)
			states = splitClass
			compute(states, class, result, phiTable)
		}
		if len(newClass) > 0 && len(splitClass) == 0 {
			splitClass = append(splitClass, states...)
			equivalenceClass := EquivalenceClass{splitClass}
			equivalenceClass.sort()
			*result = append(*result, equivalenceClass)
			break
		}
		if len(newClass) == 0 && len(splitClass) > 0 {
			tempClass := make([]State, 0)
			tempClass = append(tempClass, state)
			*result = append(*result, EquivalenceClass{tempClass})
			equivalenceClass := EquivalenceClass{splitClass}
			equivalenceClass.sort()
			*result = append(*result, equivalenceClass)
		}
		if len(newClass) == 0 && len(splitClass) == 0 {
			*result = append(*result, EquivalenceClass{states})
		}
		break
	}
}

// Получаем номер подкласса эквивалентности
func getSubclassNumber(state State, class []EquivalenceClass) int {
	var (
		condition bool
		result    int
	)
	for idx, value := range class {
		for _, v := range value.Class {
			if v == state {
				condition = true
				break
			}
		}
		if condition {
			result = idx
			break
		}
	}

	return result
}

// Сравнение двух массивов
func compareArray(src, dest []int) bool {
	if src[0] == dest[0] && src[1] == dest[1] {
		return true
	}
	return false
}
