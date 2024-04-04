package fsm

type Graph struct {
	graph map[State]map[State]bool
}

func NewGraph() *Graph {
	return &Graph{
		graph: map[State]map[State]bool{},
	}
}

func (g *Graph) AddEdge(source, destination State) {
	// add connection
	g.graph[source] = map[State]bool{
		destination: true,
	}
}

func (g *Graph) dfs(current *State, visited map[State]bool, component *[]State) {
	visited[*current] = true
	*component = append(*component, *current)

	for neighbor := range g.graph[*current] {
		if !visited[neighbor] {
			g.dfs(&neighbor, visited, component)
		}
	}
}

func (g *Graph) ConnectedComponents() [][]State {
	visited := make(map[State]bool, len(g.graph))
	connectedComponents := [][]State{}

	for state := range g.graph {
		if !visited[state] {
			component := []State{}
			g.dfs(&state, visited, &component)
			connectedComponents = append(connectedComponents, component)
		}
	}

	return connectedComponents
}

func (g *Graph) transposeGraph() *Graph {
	transposed := NewGraph()

	for state, neighbors := range g.graph {
		for neighbor := range neighbors {
			transposed.AddEdge(neighbor, state)
		}
	}

	return transposed
}

func (g *Graph) StrongConnectedComponents() [][]State {
	stack := make([]State, 0, len(g.graph))
	visited := make(map[State]bool, len(g.graph))

	for state := range g.graph {
		if !visited[state] {
			g.dfs(&state, visited, &stack)
		}
	}

	transposedGraph := g.transposeGraph()
	for s := range visited {
		delete(visited, s)
	}

	strongComponents := [][]State{}
	for i := len(stack) - 1; i >= 0; i-- {
		state := stack[i]
		if !visited[state] {
			component := []State{}
			transposedGraph.dfs(&state, visited, &component)
			strongComponents = append(strongComponents, component)
		}
	}

	return strongComponents
}
