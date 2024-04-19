package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/delvinru/finite_automata/internal/fsm"
)

func main() {
	// filename := flag.String("file", "", "filename for parsing data from file")
	n := flag.Int("n", 0, "length of zhegalkin polynom")
	phi := flag.String("phi", "", "function for fsm")
	psi := flag.String("psi", "", "function for fsm")
	initState := flag.String("init_state", "", "initial state for berlekamp massey")
	flag.Parse()

	if len(os.Args) == 1 {
		fmt.Printf("Usage: %v [-file] [-n]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	fsm, err := fsm.New(*n, *phi, *psi)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	connectedComponents := fsm.ConnectedComponents()
	fmt.Println("ConnectedComponents:", connectedComponents)
	strongConnectedComponents := fsm.StrongConnectedComponents()
	fmt.Println("StrongConnectedComponents:", strongConnectedComponents)
	fmt.Println()

	fsm.GetEquivalenceClasses()
	fmt.Println("Equivalence Classess:", fsm.EquivalenceClasses)
	fmt.Println("delta(A):", fsm.Delta)
	fmt.Println("mu(A):", fsm.Mu)

	fmt.Println()

	minimalPolynomial, _ := fsm.ComputeMinimalPolynomial(*initState)
	fmt.Println("Minimal Polynomial:", minimalPolynomial)
	fmt.Println("Linear Complexity:", len(minimalPolynomial))
}
