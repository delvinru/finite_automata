package fsm

type State struct {
	Key string
}

func NewState(key string) State {
	return State{Key: key}
}

func (s State) String() string {
	return s.Key
}
