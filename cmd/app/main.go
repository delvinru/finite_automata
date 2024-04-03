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

	fsm.ConnectedComponents()
}
