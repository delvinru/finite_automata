package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/delvinru/finite_automata/internal/config"
	"github.com/delvinru/finite_automata/internal/filemanager"
	"github.com/lmittmann/tint"
)

var (
	testNumber = flag.String("t", "", "run specific test if specified")
	filename   = flag.String("o", "", "output filename (need only for one test)")
	verbose    = flag.Bool("v", false, "verbose logging")
)

func getSpecificConfig(configs []map[string]config.Config) (config.Config, error) {
	for _, element := range configs {
		for key, config := range element {
			if key == fmt.Sprintf("test_%v", *testNumber) {
				return config, nil
			}
		}
	}

	return config.Config{}, fmt.Errorf("not existing config")
}

func main() {
	flag.Parse()

	logLevel := slog.LevelWarn
	if *verbose {
		logLevel = slog.LevelDebug
	}

	slog.SetDefault(slog.New(
		tint.NewHandler(os.Stderr, &tint.Options{
			Level:      logLevel,
			TimeFormat: time.DateTime,
		}),
	))

	if len(os.Args) == 1 {
		flag.Usage()
		os.Exit(1)
	}

	configs := config.Get()
	if *testNumber != "" {
		_, err := strconv.Atoi(*testNumber)
		if err != nil {
			slog.Error("incorrect test number, input a number")
			os.Exit(1)
		}

		if *filename != "" {
			filemanager.Init(*filename)
			defer filemanager.Close()
		} else {
			slog.Error("need specify output filename")
			os.Exit(1)
		}

		config, err := getSpecificConfig(configs)
		if err != nil {
			slog.Error(err.Error())
			os.Exit(1)
		}

		execute(config)
	} else {
		slog.Info("running all tests")

		for _, element := range configs {
			for key, config := range element {
				filemanager.Init(fmt.Sprintf("%v.txt", key))

				execute(config)

				filemanager.Close()
			}
		}
	}
}

func execute(config config.Config) {
	fmt.Println(config)
	// fsm, err := fsm.NewFSM(*n, *phi, *psi)
	// if err != nil {
	// 	fmt.Println(err)
	// 	os.Exit(1)
	// }

	// fmt.Printf("ConnectedComponents: %v\n", len(fsm.ConnectedComponents))
	// for i, value := range fsm.ConnectedComponents {
	// 	fmt.Printf("Adj_comp # %v: %v\n", i, value)
	// }

	// fmt.Println()

	// fmt.Printf("StrongConnectedComponents: %v\n", len(fsm.StrongConnectedComponents))
	// for i, value := range fsm.StrongConnectedComponents {
	// 	fmt.Printf("Strong_adj_comp # %v: %v\n", i, value)
	// }
	// fmt.Println()

	// fsm.GetEquivalenceClasses()
	// fmt.Println("Equivalence Classess:", fsm.EquivalenceClasses)
	// fmt.Println("delta(A):", fsm.Delta)
	// fmt.Println("mu(A):", fsm.Mu)
	// fmt.Println()

	// // TEST
	// fsm.MemoryFunction()

	// // minimalPolynomial, _ := fsm.ComputeMinimalPolynomial(*initState)
	// // fmt.Println("Minimal Polynomial:", minimalPolynomial)
	// // fmt.Println("Linear Complexity:", len(minimalPolynomial))
}
