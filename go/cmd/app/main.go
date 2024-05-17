package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/delvinru/finite_automata/internal/fsm"
	"github.com/lmittmann/tint"
)

func main() {
	// filename := flag.String("file", "", "filename for parsing data from file")
	n := flag.Int("n", 0, "length of zhegalkin polynom")
	phi := flag.String("phi", "", "function for fsm")
	psi := flag.String("psi", "", "function for fsm")
	// initState := flag.String("init_state", "", "initial state for berlekamp massey")

	verbose := flag.Bool("v", false, "Verbose logging")

	flag.Parse()

	w := os.Stderr
	if *verbose {
		slog.SetDefault(slog.New(
			tint.NewHandler(w, &tint.Options{
				Level:      slog.LevelDebug,
				TimeFormat: time.DateTime,
			}),
		))
	} else {
		slog.SetDefault(slog.New(
			tint.NewHandler(w, &tint.Options{
				Level:      slog.LevelWarn,
				TimeFormat: time.DateTime,
			}),
		))
	}

	if len(os.Args) == 1 {
		fmt.Printf("Usage: %v [-file] [-n]\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	fsm, err := fsm.NewFSM(*n, *phi, *psi)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("ConnectedComponents: %v\n", len(fsm.ConnectedComponents))
	for i, value := range fsm.ConnectedComponents {
		fmt.Printf("Adj_comp # %v: %v\n", i, value)
	}

	fmt.Println()

	fmt.Printf("StrongConnectedComponents: %v\n", len(fsm.StrongConnectedComponents))
	for i, value := range fsm.StrongConnectedComponents {
		fmt.Printf("Strong_adj_comp # %v: %v\n", i, value)
	}
	fmt.Println()

	fsm.GetEquivalenceClasses()
	fmt.Println("Equivalence Classess:", fsm.EquivalenceClasses)
	fmt.Println("delta(A):", fsm.Delta)
	fmt.Println("mu(A):", fsm.Mu)
	fmt.Println()

	// TEST
	fsm.MemoryFunction()

	// minimalPolynomial, _ := fsm.ComputeMinimalPolynomial(*initState)
	// fmt.Println("Minimal Polynomial:", minimalPolynomial)
	// fmt.Println("Linear Complexity:", len(minimalPolynomial))
}
