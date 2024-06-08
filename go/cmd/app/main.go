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
	"github.com/delvinru/finite_automata/internal/fsm"
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

	logLevel := slog.LevelInfo
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
	// Write basic info about test
	filemanager.Write(fmt.Sprintf("n: %v\n", config.N))
	filemanager.Write(fmt.Sprintf("phi: %v\n", config.Phi))
	filemanager.Write(fmt.Sprintf("psi: %v\n\n", config.Psi))

	// init FSM
	f, err := fsm.New(config.N, config.Phi, config.Psi)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// TASK 1
	slog.Info("task 1")
	filemanager.Write("Задание 1\n")
	for i, value := range f.ConnectedComponents {
		filemanager.Write(fmt.Sprintf("Компонента связаности %v: %v\n", i+1, value))
	}
	filemanager.Write(fmt.Sprintf("Количество компонент связности: %v\n\n", len(f.ConnectedComponents)))

	// TASK 2
	slog.Info("task 2")
	filemanager.Write("\nЗадание 2\n")
	for i, value := range f.StrongConnectedComponents {
		filemanager.Write(fmt.Sprintf("Компонента сильной связаности %v: %v\n", i+1, value))
	}
	filemanager.Write(fmt.Sprintf("Количество компонент сильной связности: %v\n\n", len(f.ConnectedComponents)))

	// TASK 3
	slog.Info("task 3")
	filemanager.Write("\nЗадание 3")
	f.GetEquivalenceClasses()

	for i := range len(f.EquivalenceClasses) {
		filemanager.Write(fmt.Sprintf("\nКласс %v-эквивалентности:\n", i+1))
		for j, value := range f.EquivalenceClasses[i+1] {
			filemanager.Write(fmt.Sprintf("%v", value))
			if j != len(f.EquivalenceClasses[i+1])-1 {
				filemanager.Write(" |_|\n")
			} else {
				filemanager.Write("\n")
			}
		}
	}

	filemanager.Write(fmt.Sprintf("Степень различимости автомата, delta(A)=%v\n", f.Delta))
	filemanager.Write(fmt.Sprintf("Приведенный вес автомата, mu(A)=%v\n", f.Mu))

	// TASK 4
	slog.Info("task 4")
	filemanager.Write("\nЗадание 4\n")
	// we exit from function by timeout because it's very large and slow function that sometimes just hangs
	exitCode := f.MemoryFunction()

	// TASK 5
	slog.Info("task 5")
	filemanager.Write("\nЗадание 5\n")
	filemanager.Write(fmt.Sprintf("Начальное состояние: %v\n", config.State))

	f, _ = fsm.New(config.N, config.Phi, config.Psi)
	f.ComputeMinimalPolynomial(config.State)

	filemanager.Write("Минимальный многочлен: ")
	for i, element := range f.BerlekmapMassey.MinimalPolynomial {
		if i == 0 && element == 1 {
			filemanager.Write("1")
			goto end
		}

		if element == 1 {
			filemanager.Write(fmt.Sprintf("x^(%v)", i))
		}

	end:
		if i != len(f.BerlekmapMassey.MinimalPolynomial)-1 && element == 1 {
			filemanager.Write(" + ")
		}
	}
	filemanager.Write(fmt.Sprintf("\nЛинейная сложность: %v\n", f.BerlekmapMassey.LinearyComplexity))

	os.Exit(exitCode)
}
