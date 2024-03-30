package models

type Node struct {
	value []uint8
}

type Graph struct {
	graph map[Node][]Node
}
