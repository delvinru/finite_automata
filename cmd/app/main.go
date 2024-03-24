package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"slices"

	"gonum.org/v1/gonum/stat/combin"
)

func computeZhegalkinPolynom(x uint8, currentState []uint8, coeffs []uint8) uint8 {
	var ZP []uint8

	extendedCurrentState := currentState
	extendedCurrentState = append(extendedCurrentState, x)

	ZP = append(ZP, 1)

	// TODO: why +2?
	for i := 1; i < len(extendedCurrentState)+2; i++ {
		if i > len(extendedCurrentState) {
			continue
		}
		cs := combin.Combinations(len(extendedCurrentState), i)
		for _, c := range cs {
			product := 1
			for j := range i {
				product *= int(extendedCurrentState[c[j]])
			}
			fmt.Printf("PRODUCT=%v\n", product)
			ZP = append(ZP, uint8(product))
		}
	}

	for i := range int(math.Pow(2, float64(len(extendedCurrentState)))) {
		ZP[i] = ZP[i] * coeffs[i]
	}

	count := 0
	for _, v := range ZP {
		if v == 1 {
			count++
		}
	}

	fmt.Printf("ZP=%v COUNT=%v\n", ZP, count)

	return uint8(count % 2)
}

func product(repeat int, args ...[]uint8) [][]uint8 {
	var pools [][]uint8
	for _, arg := range args {
		for i := 0; i < repeat; i++ {
			pools = append(pools, arg)
		}
	}

	result := [][]uint8{{}}
	for _, pool := range pools {
		var newResult [][]uint8
		for _, x := range result {
			for _, y := range pool {
				temp := make([]uint8, len(x))
				copy(temp, x)
				temp = append(temp, y)
				newResult = append(newResult, temp)
			}
		}
		result = newResult
	}

	return result
}

func computeStateTable(n int, phi []uint8, psi []uint8) {
	for _, state := range product(2, []uint8{0, 1}) {
		for x := range 2 {
			res := computeZhegalkinPolynom(uint8(x), state, phi)
			new_state := slices.Concat(state[1:], []uint8{res})
			// psi_small := computeZhegalkinPolynom(uint8(x), state, psi)
			// fmt.Printf("state=%v x=%v new_state=%v psi_small=%v | ", state, x, new_state, psi_small)
			fmt.Printf("state=%v x=%v new_state=%v\n", state, x, new_state)
		}
		fmt.Println()
	}

}

func main() {
	var n int
	var phiFlag string
	var psiFlag string

	flag.IntVar(&n, "n", 0, "length of register")
	flag.StringVar(&phiFlag, "phi", "", "phi")
	flag.StringVar(&psiFlag, "psi", "", "psi")
	flag.Parse()

	if phiFlag == "" || psiFlag == "" || n == 0 {
		fmt.Println("shit")
		os.Exit(1)
	}

	phi := make([]uint8, n)
	psi := make([]uint8, n)

	for _, value := range phiFlag {
		phi = append(phi, uint8(value)-0x30)
	}

	for _, value := range psiFlag {
		psi = append(psi, uint8(value)-0x30)
	}

	computeStateTable(n, phi, psi)
}
