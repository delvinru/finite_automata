package fsm

import (
	"log/slog"
	"slices"
)

// first equivalence classes process special way
func (f *FSM) getFirstClasses() (result [][]State) {
	// Always this constants because binary automate
	dictClasses := map[State][]State{
		NewState([]uint8{0, 0}): {},
		NewState([]uint8{0, 1}): {},
		NewState([]uint8{1, 0}): {},
		NewState([]uint8{1, 1}): {},
	}

	for state, pair := range f.table {
		firstClassKey := NewState([]uint8{pair.Psi[0], pair.Psi[1]})
		dictClasses[firstClassKey] = append(dictClasses[firstClassKey], state)
	}

	for _, value := range dictClasses {
		if len(value) > 0 {
			result = append(result, value)
		}
	}

	return result
}

func (f *FSM) isInOneEqualClass(value1, value2 State, oldClasses [][]State) bool {
	transitions1 := f.table[value1].Phi
	transitions2 := f.table[value2].Phi

	check0 := false
	for _, oldClass := range oldClasses {
		if slices.Contains(oldClass, transitions1[0]) && slices.Contains(oldClass, transitions2[0]) {
			check0 = true
			break
		}
	}

	check1 := false
	for _, oldClass := range oldClasses {
		if slices.Contains(oldClass, transitions1[1]) && slices.Contains(oldClass, transitions2[1]) {
			check1 = true
			break
		}
	}

	return check0 && check1
}

func (f *FSM) stepSpliClass(currentClass []State, oldClasses [][]State) []State {
	newClass := []State{currentClass[0]}

	for i := 1; i < len(currentClass); i++ {
		if f.isInOneEqualClass(currentClass[0], currentClass[i], oldClasses) {
			newClass = append(newClass, currentClass[i])
		}
	}

	return newClass
}

func (f *FSM) getDifference(a, b []State) (diff []State) {
	m := make(map[State]bool)

	for _, item := range b {
		m[item] = true
	}

	for _, item := range a {
		if _, ok := m[item]; !ok {
			diff = append(diff, item)
		}
	}

	return
}

func (f *FSM) splitClass(class []State, oldClasses [][]State) (splitClasses [][]State) {
	currentClass := class

	for len(currentClass) > 0 {
		newClass := f.stepSpliClass(currentClass, oldClasses)
		splitClasses = append(splitClasses, newClass)
		currentClass = f.getDifference(currentClass, newClass)
	}

	return splitClasses
}

func (f *FSM) computeKClasses(oldClasses [][]State) (kClasses [][]State) {
	for _, class := range oldClasses {
		splitClasses := f.splitClass(class, oldClasses)
		kClasses = append(kClasses, splitClasses...)
	}
	return kClasses
}

// Equivalence classess
func (f *FSM) GetEquivalenceClasses() map[int][][]State {
	equivalenceClasses := map[int][][]State{
		1: f.getFirstClasses(),
	}
	slog.Debug("class", "k", 1, "class", equivalenceClasses[1])

	for k := 1; ; k++ {
		newClass := f.computeKClasses(equivalenceClasses[k])
		if slices.EqualFunc(newClass, equivalenceClasses[k], func(a, b []State) bool {
			for i := range len(a) {
				if a[i] != b[i] {
					return false
				}
			}
			return true
		}) {
			break
		}

		slog.Debug("class", "k", k+1, "class", newClass)

		equivalenceClasses[k+1] = newClass
	}

	f.EquivalenceClasses = equivalenceClasses
	f.Delta = len(equivalenceClasses)
	f.Mu = len(equivalenceClasses[len(equivalenceClasses)])

	return equivalenceClasses
}
