package fsm

import (
	"fmt"
	"log/slog"
	"math"
	"slices"
)

// Compute output of automate
func (f *FSM) computeU(initState []uint8) []uint8 {
	size := int(math.Pow(2.0, float64(f.n)))
	u := make([]uint8, size)
	currentState := NewState(initState)

	for i := 0; i < size; i++ {
		// in input only zeroes by task
		u[i] = f.table[currentState].Psi[0]
		currentState = f.table[currentState].Phi[0]
	}

	return u
}

func countOfLeadingZeros(segment []uint8) uint8 {
	// TODO: possible integer overflow, but ok for now
	var count uint8 = 0
	for _, el := range segment {
		if el != 0 {
			return count
		} else {
			count++
		}
	}
	return count
}

// add two segments with different length
func sumOfTwoLists(firstList, secondList []uint8) []uint8 {
	length := min(len(firstList), len(secondList))
	newList := make([]uint8, length)
	for i := range length {
		newList[i] = firstList[i] ^ secondList[i]
	}
	return newList
}

// add two polynomials with different length
func sumOfTwoPolynomials(firstPolynomial, secondPolynomial []uint8) []uint8 {
	length := max(len(firstPolynomial), len(secondPolynomial))
	if len(firstPolynomial) != length {
		firstPolynomial = append(firstPolynomial, make([]uint8, length-len(firstPolynomial))...)
	}
	if len(secondPolynomial) != length {
		secondPolynomial = append(secondPolynomial, make([]uint8, length-len(secondPolynomial))...)
	}

	newList := make([]uint8, length)
	for i := range length {
		newList[i] = firstPolynomial[i] ^ secondPolynomial[i]
	}
	return newList
}

func (f *FSM) berlekampMassey(u []uint8) []uint8 {
	// list of table segments
	segments := [][]uint8{}
	// list of table polynomials
	polynomials := [][]uint8{}
	// list of zeroes in table
	zerosCount := []uint8{}

	// step zero from algorithm
	zerosCount = append(zerosCount, countOfLeadingZeros(u))
	polynomials = append(polynomials, []uint8{1})
	if zerosCount[len(zerosCount)-1] == uint8(len(u)) {
		return polynomials[0]
	}
	segments = append(segments, u)

	// step 1 <= s <= (l - 1)
	for i := 1; i < int(math.Pow(2, float64(f.n)))-1; i++ {
		slog.Debug(fmt.Sprintf("i: %v", i))

		// list of segments u_i, first of all add (u_i)^0
		currentSegments := [][]uint8{
			segments[len(segments)-1][1:],
		}
		// list of polynomials f_i, first of all add (f_i)^0
		currentPolynomials := [][]uint8{
			append([]uint8{0}, polynomials[len(polynomials)-1]...),
		}
		// list of zeros k_i, first of all add (k_i)^0
		currentZerosCount := []uint8{0}
		if zerosCount[len(zerosCount)-1] != 0 {
			currentZerosCount = []uint8{zerosCount[len(zerosCount)-1] - 1}
		}

		slog.Debug(fmt.Sprintf("currentSegments: %v", currentSegments))
		slog.Debug(fmt.Sprintf("currentPolynomials: %v", currentPolynomials))
		slog.Debug(fmt.Sprintf("currentZerosCount: %v", currentZerosCount))

		// case 3
		a := countOfLeadingZeros(currentSegments[len(currentSegments)-1])
		slog.Debug(fmt.Sprintf("countOfLeadingZeros: %v, length: %v", a, len(currentSegments[len(currentSegments)-1])))
		slog.Debug(fmt.Sprintf("currentZerosCount[-1]: %v, zerosCount: %v", currentZerosCount[len(currentZerosCount)-1], zerosCount))

		for countOfLeadingZeros(currentSegments[len(currentSegments)-1]) != uint8(len(currentSegments[len(currentSegments)-1])) && slices.Contains(zerosCount, currentZerosCount[len(currentZerosCount)-1]) {
			// compute "t" from algorithm
			t := slices.Index(zerosCount, currentZerosCount[len(currentZerosCount)-1])
			// compute r in GF(2)
			if segments[t][zerosCount[t]] == 0 {
				panic("alarm! 0 when computing u_t(k_t)^(-1)")
			}
			r := currentSegments[len(currentSegments)-1][zerosCount[t]] * segments[t][zerosCount[t]]

			tmpSegment := make([]uint8, len(segments[t]))
			for j := 0; j < len(tmpSegment); j++ {
				tmpSegment[j] = uint8(segments[t][j]) * r
			}
			currentSegments = append(currentSegments, sumOfTwoLists(currentSegments[len(currentSegments)-1], tmpSegment))

			tmpPolynomial := make([]uint8, len(polynomials[t]))
			for j := 0; j < len(tmpPolynomial); j++ {
				tmpPolynomial[j] = polynomials[t][j] * r
			}
			currentPolynomials = append(currentPolynomials, sumOfTwoPolynomials(currentPolynomials[len(currentPolynomials)-1], tmpPolynomial))

			currentZerosCount = append(currentZerosCount, countOfLeadingZeros(currentSegments[len(currentSegments)-1]))
		}

		// case 1
		if countOfLeadingZeros(currentSegments[len(currentSegments)-1]) == uint8(len(currentSegments[len(currentSegments)-1])) {
			// zerosCount = append(zerosCount, currentZerosCount[len(currentZerosCount)-1])
			polynomials = append(polynomials, currentPolynomials[len(currentPolynomials)-1])
			// segments = append(segments, currentSegments[len(currentSegments)-1])

			return polynomials[len(polynomials)-1]
		}

		// case 2
		segments = append(segments, currentSegments[len(currentSegments)-1])
		polynomials = append(polynomials, currentPolynomials[len(currentPolynomials)-1])
		zerosCount = append(zerosCount, currentZerosCount[len(currentZerosCount)-1])
	}

	// step l
	// segments = append(segments, make([]uint8, 0))
	polynomials = append(polynomials, append([]uint8{0}, polynomials[len(polynomials)-1]...))

	return polynomials[len(polynomials)-1]
}

func (f *FSM) ComputeMinimalPolynomial(initStateStr string) error {
	initState, err := getFunctionCoeffs(initStateStr)
	if err != nil {
		return err
	}

	f.BerlekmapMassey.U = f.computeU(initState)
	f.BerlekmapMassey.MinimalPolynomial = f.berlekampMassey(f.BerlekmapMassey.U)
	f.BerlekmapMassey.LinearyComplexity = len(f.BerlekmapMassey.MinimalPolynomial)

	return nil
}
