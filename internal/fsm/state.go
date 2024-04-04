package fsm

import (
	"strings"
)

type State struct {
	Value string
}

func NewState(value []uint8) State {
	converted := strings.Builder{}
	for _, el := range value {
		converted.WriteString(string(el + 0x30))
	}
	return State{Value: converted.String()}
}

func (s *State) ToArray() []uint8 {
	value := make([]uint8, len(s.Value))
	for idx, el := range s.Value {
		value[idx] = uint8(el - 0x30)
	}
	return value
}

func (s State) String() string {
	return s.Value
}
