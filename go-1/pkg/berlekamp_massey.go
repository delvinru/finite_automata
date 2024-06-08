package pkg

import (
	"log"
	"math"
	"slices"
)

func (f *FSM) GetMinimalPolynomial(state []int) []int {
	u := f.getLinearSequence(state)

	return f.berlekampMassey(u)
}

func (f *FSM) getLinearSequence(state []int) []int {
	var (
		u            []int
		currentState State
	)
	u = make([]int, int(math.Pow(2, float64(f.n))))
	currentState = NewState(state)

	for i := range int(math.Pow(2, float64(f.n))) {
		u[i] = f.PsiTable[currentState][0]
		currentState = f.PhiTable[currentState][0]
	}

	return u
}

func (f *FSM) berlekampMassey(u []int) []int {
	var (
		segments    [][]int // таблица u-шек
		polynomials [][]int // таблица многочленов
		zerosCount  []int   // количество нулей
	)
	segments = make([][]int, 0)
	polynomials = make([][]int, 0)
	zerosCount = make([]int, 0)

	// считаем количество нулей для исходной последовательности u
	zerosCount = append(zerosCount, getCountOfFirstZeros(u))
	// добавляем к многочлену 1, так как на первом шаге всегда 1
	polynomials = append(polynomials, []int{1})
	if zerosCount[len(zerosCount)-1] == len(u) {
		return polynomials[0]
	}
	segments = append(segments, u)

	for i := 1; i < int(math.Pow(2, float64(f.n)))-1; i++ {
		// текущая последовательность u с индексом i
		// вычисляется u умножением x^i, в нашем случаем делается
		// просто сдвиг влево
		currentSegments := [][]int{
			segments[len(segments)-1][1:],
		}
		// текущая функция f с индексом i
		// вычисляется f умножением x^i, в нашем случаем делается
		// добавление 0 в начало массива
		currentPolynomials := [][]int{
			append([]int{0}, polynomials[len(polynomials)-1]...),
		}
		// добавляется количество 0 для текущей последовательности u_i
		currentZerosCount := []int{0}
		if zerosCount[len(zerosCount)-1] != 0 {
			currentZerosCount = []int{getCountOfFirstZeros(currentSegments[len(currentSegments)-1])}
		}

		// данный цикл выполняет нахождение "подпоследовательностей"
		// u^(0)_i, если выполняются следующие условия:
		// 1. Если количество нулей в последовательности u, не равно
		// длине текущей последовательности.
		// 2. Если текущее количество нулей уже находится в "пуле",
		// которые были получены для прошлых последовательностей.
		for getCountOfFirstZeros(currentSegments[len(currentSegments)-1]) != len(currentSegments[len(currentSegments)-1]) && slices.Contains(zerosCount, currentZerosCount[len(currentZerosCount)-1]) {
			// получаем номер шага, а точнее номер последовательности
			// которая уже имеет количество нулей, которое равно
			// количеству нулей для текущей последовательности
			index := slices.Index(zerosCount, currentZerosCount[len(currentZerosCount)-1])

			if segments[index][zerosCount[index]] == 0 {
				log.Fatalf("Для нуля не существует обратного, а значит, что ранне допущена ошибка в вычислениях\n")
			}

			// коэффициент r вычисляется как перемножение элементов
			// 2 последовательностей:
			// 1. Элемент текущей последовательности, который находится
			// по индексу index.
			// 2. Элемент последовательности, с которой произошло совпадение,
			// который находится также по индексу index.
			r := currentSegments[len(currentSegments)-1][zerosCount[index]] * segments[index][zerosCount[index]]

			// создается временная последовательность, в которую добавляется
			// последовательность, получившая при перемножении совпавшей
			// последовательности с коэффициентом r
			tmpSegment := make([]int, len(segments[index]))
			for j := 0; j < len(tmpSegment); j++ {
				tmpSegment[j] = segments[index][j] * r
			}

			// добавляем к текущим последовательностям новую
			currentSegments = append(currentSegments, sumSequences(currentSegments[len(currentSegments)-1], tmpSegment))

			// создается временный многочлен, в который добавляется
			// многочлен, получившийся при перемножении совпавшего
			// многочлен с коэффициентом r
			tmpPolynomial := make([]int, len(polynomials[index]))
			for j := 0; j < len(tmpPolynomial); j++ {
				tmpPolynomial[j] = polynomials[index][j] * r
			}
			currentPolynomials = append(currentPolynomials, sumPolynomials(currentPolynomials[len(currentPolynomials)-1], tmpPolynomial))

			// добавляем к текущим счетчикам нулей новый счетчик
			currentZerosCount = append(currentZerosCount, getCountOfFirstZeros(currentSegments[len(currentSegments)-1]))
		}

		// Если количество нулей для текущей последовательности равняется
		// длине текущей последовательности, значит последовательностей
		// состоит из нулей, следовательно, возвращаем многочлен этой послед.
		if getCountOfFirstZeros(currentSegments[len(currentSegments)-1]) == len(currentSegments[len(currentSegments)-1]) {
			polynomials = append(polynomials, currentPolynomials[len(currentPolynomials)-1])

			return polynomials[len(polynomials)-1]
		}

		// Продолжаем вычисления
		segments = append(segments, currentSegments[len(currentSegments)-1])
		polynomials = append(polynomials, currentPolynomials[len(currentPolynomials)-1])
		zerosCount = append(zerosCount, currentZerosCount[len(currentZerosCount)-1])
	}

	polynomials = append(polynomials, append([]int{0}, polynomials[len(polynomials)-1]...))

	return polynomials[len(polynomials)-1]
}

// Получаем количество нулелй в начале последовательности
func getCountOfFirstZeros(u []int) int {
	var (
		count = 0
	)
	for _, val := range u {
		if val == 0 {
			count++
		} else {
			return count
		}
	}

	return count
}

// Выполняем сложение двух последовательностей
func sumSequences(a, b []int) []int {
	result := make([]int, min(len(a), len(b)))
	for i := range min(len(a), len(b)) {
		result[i] = a[i] ^ b[i]
	}

	return result
}

// Выполняем сложение двух многочленов
func sumPolynomials(a, b []int) []int {
	length := max(len(a), len(b))
	if len(a) != length {
		a = append(a, make([]int, length-len(a))...)
	}
	if len(b) != length {
		b = append(b, make([]int, length-len(b))...)
	}

	newList := make([]int, length)
	for i := range length {
		newList[i] = a[i] ^ b[i]
	}
	return newList
}
