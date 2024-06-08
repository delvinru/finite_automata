package pkg

type Graph struct {
	graphFull map[State]map[State]string
	graph     map[State][]State
}

// Создается новый экземпляр класс Graph
func NewGraph() *Graph {
	return &Graph{
		graphFull: make(map[State]map[State]string),
		graph:     make(map[State][]State),
	}
}

// Добавляется новая сторона графа
func (g *Graph) AddEdge(source, dest State, weight string) {
	g.graph[source] = append(g.graph[source], dest)
}

// Получение компонентов связности автомата
func (g *Graph) GetConnectivityComponents() [][]State {
	var (
		connectivityComponents [][]State
	)

	visited := make(map[State]bool)

	for state := range g.graph {
		if !visited[state] {
			var component []State
			g.dfs(state, visited, &component)
			connectivityComponents = append(connectivityComponents, component)
		}
	}

	return connectivityComponents
}

// Получение компонентов сильной связности автомата
func (g *Graph) GetStrongConnectivityComponents() [][]State {
	stack := make([]State, 0, len(g.graph))
	visited := make(map[State]bool, len(g.graph))

	for state := range g.graph {
		if !visited[state] {
			g.dfs(state, visited, &stack)
		}
	}

	transposedGraph := g.transposeGraph()
	for s := range visited {
		delete(visited, s)
	}

	var strongComponents [][]State
	for i := len(stack) - 1; i >= 0; i-- {
		state := stack[i]
		if !visited[state] {
			var component []State
			transposedGraph.dfs(state, visited, &component)
			strongComponents = append(strongComponents, component)
		}
	}

	return strongComponents
}

// Алгоритм поиска в глубину. На вход подается текущее состояние currentState, карта уже посещенный вершин visited,
// указатель на массив компонент component
// Входные данные: currentState State, visited map[State]bool, component *[]State
func (g *Graph) dfs(currentState State, visited map[State]bool, component *[]State) {
	visited[currentState] = true
	*component = append(*component, currentState)

	for _, child := range g.graph[currentState] {
		if !visited[child] {
			g.dfs(child, visited, component)
		}
	}
}

func (g *Graph) transposeGraph() *Graph {
	transposed := NewGraph()

	for state, neighbors := range g.graph {
		for _, neighbor := range neighbors {
			transposed.AddEdge(neighbor, state, "")
		}
	}

	return transposed
}
