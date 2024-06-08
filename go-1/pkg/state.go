package pkg

import "strconv"

type State struct {
	Value string
}

func NewState(value []int) State {
	return State{Value: convert(value)}
}

func NewStateFromString(value string) State {
	return State{Value: value}
}

func convert(value []int) string {
	var result string
	for _, v := range value {
		result += strconv.Itoa(v)
	}

	return result
}
