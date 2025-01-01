# Graphs and trees

## Summary

A library for working with graph and tree data structures.
A graph is a data structure, consisting of nodes and links between some of them. There are different types of graphs, such as graphs with or without loops, directed/undirected, weighted/unweighted and multigraphs.

## Implementation

### Graphs

Graphs can be directed or undirected. For each of them there are the following possibilities when it comes to weights:

- No weights at all;
- Weights only on the nodes;
- Weights only on the links;
- Weights on both the nodes and the links.

### Trees

Trees, defined in this project, are:

- Binary tree;
- Tree with unlimited descendants;
- Tree with unlimited descendants and node weights.

## Functionalities

- Defining an object of any of the given data types;
- Property getters;
- Safely changing the value of a graph or a tree (for example, when a node is removed, all links it takes part in are also removed);
- Representation of given objects;
- Complex algorithms over graphs and trees, such as interval sort, (weighted) vertex cover, (weighted) dominating set, (weighted) independent set, maximal clique and chromatic nodes/links partition for an undirected graph, topological sort for a directed graph and so on.

## Not supported

- Graphs with loops (a link, both ends of which are the same node);
- Multigraphs (where multiple links can exist from one node to another).

