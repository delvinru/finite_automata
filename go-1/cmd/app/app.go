package app

import (
	"errors"
	"fmt"
	"github.com/spf13/viper"
	"lab/pkg"
	"math"
	"os"
	"strconv"
	"strings"
)

type App struct {
}

func NewApp() *App {
	return &App{}
}

func (app *App) Run() {
	var (
		err error
	)

	// считываем данные из конфиг-файла
	err = initConfigs()
	if err != nil {
		panic(err)
	}

	// открываем файл, в который будем записывать результаты работы программы
	file, err := os.Create("test1.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	pkg.WriteInputData(file, viper.GetInt("init.n"), viper.GetString("init.phi"), viper.GetString("init.psi"))

	/*err = verificationInputtedData(viper.GetInt("init.n"), viper.GetString("init.phi"), viper.GetString("init.psi"))
	if err != nil {
		log.Fatalf("%s\n", err)
	}*/

	// закидывем исходные данные в автомат и инициализируем его
	fsm := pkg.NewFSM(viper.GetInt("init.n"), createArray(viper.GetString("init.phi")), createArray(viper.GetString("init.psi")))
	fsm.Init()

	for key, val := range fsm.PsiTable {
		fmt.Printf("%s: %d\n", key.Value, val)
		fmt.Printf("%s: %s\n", key.Value, fsm.PhiTable[key])
	}

	// TASK 1
	// находим компоненты связности
	connectivityComponents := fsm.GetConnectivityComponents()
	pkg.WriteTask1(file, connectivityComponents)
	for _, v := range connectivityComponents {
		fmt.Println("Connected components:", v)
	}
	fmt.Printf("Number of components: %d\n", len(connectivityComponents))

	// TASK 2
	// находим компоненты сильной связности
	strongConnectivityComponents := fsm.GetStrongConnectivityComponents()
	pkg.WriteTask2(file, strongConnectivityComponents)
	for _, v := range strongConnectivityComponents {
		fmt.Println("Strong connected components:", v)
	}
	fmt.Printf("Number of strong components: %d\n", len(strongConnectivityComponents))

	// TASK 3
	// находим классы эквивалентности
	equivalenceClass := pkg.NewEquivalenceClass()
	equivalenceClasses := equivalenceClass.GetEquivalenceClasses(fsm.PhiTable, fsm.PsiTable)
	for i, class := range equivalenceClasses {
		fmt.Printf("Equivalence class %d\n", i)
		for idx, v := range class {
			fmt.Printf("Subclass %d: %s\n", idx, v.Class)
		}
	}
	fsm.AddEquivalenceClass(equivalenceClasses[len(equivalenceClasses)])
	fsm.Delta = len(equivalenceClasses)
	fsm.Mu = len(equivalenceClasses[len(equivalenceClasses)])
	fmt.Printf("delta(A): %d\n", fsm.Delta)
	fmt.Printf("mu(A): %d\n", fsm.Mu)
	pkg.WriteTask3(file, equivalenceClasses, fsm.Delta, fsm.Mu)

	// TASK 4
	// получаем функцию памяти автомата
	q, polinom := fsm.GetMemoryFunction()
	pkg.WriteTask4(file, q, polinom)

	// TASK 5
	// находим минимальный многочлен
	is := viper.GetString("init.initial_state")
	initialState := createArray(is)
	minimalPolynomial := fsm.GetMinimalPolynomial(initialState)
	fmt.Println("Initial state:", is)
	fmt.Println("Minimal Polynomial:", polynomialToString(minimalPolynomial))
	fmt.Println("Linear Complexity:", len(minimalPolynomial))
	pkg.WriteTask5(file, is, polynomialToString(minimalPolynomial), len(minimalPolynomial))
}

// Функция отвечает за считывание исходных данных из файла конфигурации, возращает ошибку, если не может произвести считывание
// Входные данные: null
// Выходная данные error
func initConfigs() error {
	viper.AddConfigPath("configs")
	viper.SetConfigName("config")
	viper.SetConfigType("yml")

	return viper.ReadInConfig()
}

// Проверям входные данные
func verificationInputtedData(n int, a, b, s string) error {
	var err error
	if n == 0 || a == "" || b == "" {
		err = errors.New("n and a and b didn't find")
		return err
	}
	if len(s) != int(math.Pow(2, float64(n))) {
		err = errors.New("length of s != 2^n")
		return err
	}

	return nil
}

// На вход функции подается строка, которая преобразуется в числовой массив
// Входные данные: str string
// Выходные данные: arr []int
func createArray(str string) []int {
	tmp := strings.Split(str, "")
	result := make([]int, len(tmp))

	for i, v := range tmp {
		result[i], _ = strconv.Atoi(v)
	}
	return result
}

// Переводим полином из числового вида в строку
func polynomialToString(polynomial []int) string {
	var result string
	for idx, v := range polynomial {
		if idx == 0 && v == 1 {
			result = result + "1"
			continue
		}
		if idx > 0 && v == 1 {
			result = result + fmt.Sprintf(" + x^%d", idx)
		}
	}

	return result
}
