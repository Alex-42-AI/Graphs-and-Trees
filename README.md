# Graphs and Trees

![g0](https://github.com/user-attachments/assets/a7f6b9a5-33ec-4eb2-bcad-05bdfb0c9969)
![g1](https://github.com/user-attachments/assets/2659a505-6308-4779-b278-a0bcb2e3238f)
![g2](https://github.com/user-attachments/assets/a07deef5-ba3c-424c-b840-2515989d93fc)
![g3](https://github.com/user-attachments/assets/d14f3fbf-f71f-425f-b8e9-8d511e096322)
![g4](https://github.com/user-attachments/assets/75d2a798-0d4f-42fa-8fc8-e91ac760d7cb)

## Summary

A library for working with graph and tree data structures. This implementation focuses on speed over memory.
A graph is a data structure, consisting of nodes and links between some of them. There are different types of graphs, such as graphs with or without loops, directed/undirected, weighted/unweighted and multigraphs.

https://en.wikipedia.org/wiki/Graph_theory

https://en.wikipedia.org/wiki/Graph_(abstract_data_type)

## Getting started

```
git clone https://github.com/yourname/Graphs-and-Trees.git
cd Graphs-and-Trees
python3
```

## Implementation

### Graphs

Graphs can be directed or undirected. For each of them there are the following possibilities when it comes to weights:

- No weights at all;
- Weights only on the nodes;
- Weights only on the links;
- Weights on both the nodes and the links.

Additionally, the project includes specialized classes for interval graphs — undirected graphs where each node represents an interval and edges represent interval overlaps.

### Trees

Trees, defined in this project, are:

- Binary tree;
- Tree with unlimited descendants;
- Tree with unlimited descendants and node weights.

## Features

- Defining an object of any given data type;
- Property getters;
- Safely changing the value of a graph or a tree (for example, when a node is removed, all links it takes part in are also removed);
- Representation of given objects;
- Complex algorithms over graphs and trees, such as interval sort, (weighted) vertex cover, (weighted) dominating set, (weighted) independent set, maximal clique and chromatic nodes/links partition for an undirected graph, topological sort and strongly-connected components partition for a directed graph and so on.

## Restrictions

- Node values can only be of a hashable type.

## Not supported

- Multi-graphs (where multiple links can exist from one node to another);
- Graphs with loops (a link, both ends of which are the same node).
