package fsm

import "log/slog"

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
	if g.graph[source] == nil {
		g.graph[source] = make(map[State]bool)
	}
	g.graph[source][destination] = true
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
	slog.Info("computing connected components")
	visited := make(map[State]bool, len(g.graph))
	connectedComponents := [][]State{}

	for state := range g.graph {
		if !visited[state] {
			component := []State{}
			g.dfs(&state, visited, &component)
			slog.Info(
				"get",
				"component", component,
			)
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
	slog.Info("computing strong connected components")
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
			slog.Info(
				"get",
				"component", component,
			)
			strongComponents = append(strongComponents, component)
		}
	}

	return strongComponents
}
