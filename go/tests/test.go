package main

import (
	"fmt"
	"math"
	"sync"
)

func main() {
	var wg sync.WaitGroup

	numbers := []uint64{10, 20, 30, 40, 50, 60, 70, 80, 90}
	for _, num := range numbers {
		wg.Add(1)

		go func(num uint64) {
			defer wg.Done()
			fmt.Printf("num: %v\n", num)

			var counter uint64 = 0
			for i := 1; i < int(math.Pow10(int(num))); i++ {
				counter += 1
			}
			fmt.Printf("counter: %v\n", counter)
		}(num)
	}

	wg.Wait()
}
