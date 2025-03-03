"""
Module for implementing helper classes Node and Link, abstract base class Graph and helper functions
"""

from collections import defaultdict

from abc import ABC, abstractmethod

from typing import Iterable, Hashable, Any

from itertools import permutations, product


class Node:
    """
    Helper class Node with a hashable value
    """

    def __init__(self, value: Hashable) -> None:
        if not hasattr(value, "__hash__"):
            raise ValueError(f"Unhashable type: {type(value).__name__}!")
        self.__value = value

    @property
    def value(self) -> Hashable:
        """
        Returns:
            Node value
        """
        return self.__value

    def __bool__(self) -> bool:
        return bool(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: Any) -> bool:
        if type(other) == Node:
            return self.value == other.value
        return False

    def __lt__(self, other: "Node") -> bool:
        if isinstance(other, Node):
            return self.value < other.value
        return self.value < other

    def __le__(self, other: "Node") -> bool:
        if isinstance(other, Node):
            return self.value <= other.value
        return self.value <= other

    def __ge__(self, other: "Node") -> bool:
        if isinstance(other, Node):
            return self.value >= other.value
        return self.value >= other

    def __gt__(self, other: "Node") -> bool:
        if isinstance(other, Node):
            return self.value > other.value
        return self.value > other

    def __str__(self) -> str:
        return "(" + str(self.value) + ")"

    __repr__: str = __str__


class Link:
    """
    Helper class, implementing an undirected link
    """

    def __init__(self, u: Node, v: Node) -> None:
        """
        Args:
            u: A node object
            v: A node object
        """
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        self.__u, self.__v = u, v

    @property
    def u(self) -> Node:
        """
        Returns:
            The first given node
        """
        return self.__u

    @property
    def v(self) -> Node:
        """
        Returns:
            The second given node
        """
        return self.__v

    def other(self, n: Node) -> Node:
        if not isinstance(n, Node):
            n = Node(n)
        if n not in self:
            raise KeyError("Unrecognized node!")
        return [(u := self.u), self.v][n == u]

    def __contains__(self, node: Node) -> bool:
        """
        Args:
            node: a Node object
        Returns:
            Whether given node is in the link
        """
        if not isinstance(node, Node):
            node = Node(node)
        return node in {self.u, self.v}

    def __hash__(self) -> int:
        return hash(frozenset({self.u, self.v}))

    def __eq__(self, other: "Link") -> bool:
        if type(other) == Link:
            return {self.u, self.v} == {other.u, other.v}
        return False

    def __str__(self) -> str:
        return f"{self.u}-{self.v}"

    __repr__: str = __str__


class Graph(ABC):
    """
    Abstract base class for graphs
    """

    @abstractmethod
    def nodes(self) -> set[Node]:
        """
        Returns:
            Graph nodes
        """
        pass

    @abstractmethod
    def links(self) -> set[Link | tuple[Node, Node]]:
        """
        Returns:
            Graph links
        """
        pass

    @abstractmethod
    def degrees(self, u: Node = None) -> dict | int:
        """
        Args:
            u: Node object or None
        Returns:
            Node degree or dictionary with all node degrees
        """
        pass

    @abstractmethod
    def remove(self, n: Node, *rest: Node) -> "Graph":
        """
        Remove a non-empty set of nodes
        Args:
            n: First given node
            rest: Other given nodes
        """
        pass

    @abstractmethod
    def connect_all(self, u: Node, *rest: Node) -> "Graph":
        """
        Args:
            u: Node object
            rest: Node objects
        Connect every given node to every other given node, all present in the graph
        """
        pass

    @abstractmethod
    def disconnect_all(self, u: Node, *rest: Node) -> "Graph":
        """
        Args:
            u: First given node
            rest: Other given nodes
        Disconnect all given nodes, all present in the graph
        """
        pass

    @abstractmethod
    def copy(self) -> "Graph":
        """
        Returns:
            Identical copy of the graph
        """
        pass

    @abstractmethod
    def complementary(self) -> "Graph":
        """
        Returns:
            A graph, where there are links between nodes exactly where there aren't in the original graph
        """
        pass

    @abstractmethod
    def component(self, u: Node) -> "Graph":
        pass

    @abstractmethod
    def subgraph(self, u_or_s: Node | Iterable[Node]) -> "Graph":
        """
        Args:
            u_or_s: Given node or set of nodes
        Returns:
            If a set of nodes is given, return the subgraph, that only contains these nodes and all links between them. If a node is given, return the subgraph of all nodes and links, reachable by it, in a directed graph
        """
        pass

    @abstractmethod
    def connection_components(self) -> "list[Graph]":
        """
        Returns:
            A list of all connection components of the graph
        """
        pass

    @abstractmethod
    def connected(self) -> bool:
        """
        Returns:
            Whether the graph is connected
        """
        pass

    @abstractmethod
    def reachable(self, u: Node, v: Node) -> bool:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            Whether the first given node can reach the second one
        """
        pass

    @abstractmethod
    def full(self) -> bool:
        """
        Returns:
            Whether the graph is fully connected
        """
        pass

    @abstractmethod
    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            One shortest path from u to v, if such path exists, otherwise empty list
        """
        pass

    @abstractmethod
    def euler_tour_exists(self) -> bool:
        """
        Returns:
            Whether an Euler tour exists
        """
        pass

    @abstractmethod
    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            Whether an Euler walk from u to v exists
        """
        pass

    @abstractmethod
    def euler_tour(self) -> list[Node]:
        """
        Returns:
             An Euler tour, if such exists, otherwise empty list
        """
        pass

    @abstractmethod
    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
             An Euler walk, if such exists, otherwise empty list
        """
        pass

    @abstractmethod
    def weighted_nodes_graph(self, weights: dict[Node, float] = None) -> "Graph":
        """
        Args:
            weights: A dictionary, mapping nodes to their weights
        Returns:
            The version of the graph with node weights
        """
        pass

    @abstractmethod
    def weighted_links_graph(self, weights: dict) -> "Graph":
        """
        Args:
            weights: A dictionary, mapping links to their weights
        Returns:
            The version of the graph with link weights
        """
        pass

    @abstractmethod
    def weighted_graph(self, node_weights: dict[Node, float] = None, link_weights: dict = None) -> "Graph":
        """
        Args:
            node_weights: A dictionary, mapping nodes to their weights
            link_weights: A dictionary, mapping links to their weights
        Returns:
            The version of the graph with node and link weights
        """
        pass

    @abstractmethod
    def cycle_with_length(self, length: int) -> list[Node]:
        """
        Args:
            length: Length of wanted cycle
        Returns:
            A cycle with given length, if such exists, otherwise empty list
        """
        pass

    @abstractmethod
    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        """
        Args:
            u: First given node
            v: Second given node
            length: Length of wanted path
        Returns:
            A path from u to v with given length, if such path exists, otherwise empty list
        """
        pass

    @abstractmethod
    def hamilton_tour_exists(self) -> bool:
        """
        Returns:
            Whether a Hamilton tour exists
        """
        pass

    @abstractmethod
    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Args:
            u: first given node
            v: second given node
        Returns:
            Whether a Hamilton walk exists from u to v
        """
        pass

    @abstractmethod
    def hamilton_tour(self) -> list[Node]:
        """
        Returns:
            A Hamilton tour, if such exists, otherwise empty list
        """
        pass

    @abstractmethod
    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        """
        Args:
            u: first given node or None
            v: second given node or None
        Returns:
            A Hamilton walk from u to v, if such exists, otherwise empty list. If a node isn't given, it could be any
        """
        pass

    @abstractmethod
    def isomorphic_bijection(self, other: "Graph") -> dict[Node, Node]:
        """
        Args:
            other: another Graph object
        Returns:
            An isomorphic function (bijection) between the nodes of the graph and those of the given graph, if such exists, otherwise empty dictionary. Let f be such a bijection and u and v be nodes in the graph. f(u) and f(v) are nodes in the other graph and f(u) and f(v) are neighbors (or f(u) points to f(v) for directed graphs) exactly when the same applies for u and v. For weighted graphs, the weights are taken into account
        """
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """
        Returns:
            Whether the graph has nodes
        """
        pass

    @abstractmethod
    def __contains__(self, u: Node) -> bool:
        """
        Args:
            u: A node object
        Returns:
            Whether u is a node in the graph
        """
        pass

    @abstractmethod
    def __add__(self, other: "Graph") -> "Graph":
        pass

    @abstractmethod
    def __eq__(self, other: Any):
        """
        Args:
            other: Another Graph object
        Returns:
            Whether both graphs are equal
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


def combine_undirected(graph0: "UndirectedGraph", graph1: "UndirectedGraph") -> "UndirectedGraph":
    if not hasattr(graph1, "neighbors"):
        raise TypeError(f"Addition not defined between type {type(graph0).__name__} and type {type(graph1).__name__}!")
    if hasattr(graph0, "node_weights") and hasattr(graph0, "link_weights"):
        if hasattr(graph1, "node_weights") and hasattr(graph1, "link_weights"):
            res = graph0.copy()
            for n in graph1.nodes:
                if n in res:
                    res.increase_weight(n, graph1.node_weights(n))
                else:
                    res.add((n, graph1.node_weights(n)))
            for l in graph1.links:
                if l in res.links:
                    res.increase_weight(l, graph1.link_weights(l))
                else:
                    res.connect(l.u, {l.v: graph1.link_weights(l)})
            return res
        return graph0 + graph1.weighted_graph()
    if hasattr(graph0, "node_weights"):
        if hasattr(graph1, "link_weights"):
            return graph0.weighted_graph() + graph1
        if hasattr(graph1, "node_weights"):
            res = graph0.copy()
            for n in graph1.nodes:
                if n in res:
                    res.increase_weight(n, graph1.node_weights(n))
                else:
                    res.add((n, graph1.node_weights(n)))
            for l in graph1.links:
                res.connect(l.u, l.v)
            return res
        return graph0 + graph1.weighted_nodes_graph()
    if hasattr(graph0, "link_weights"):
        if hasattr(graph1, "node_weights"):
            return graph1 + graph0
        if hasattr(graph1, "link_weights"):
            res = graph0.copy()
            for n in graph1.nodes:
                res.add(n)
            for l in graph1.links:
                if l in res.links:
                    res.increase_weight(l, graph1.link_weights(l))
                else:
                    res.connect(l.u, {l.v: graph1.link_weights(l)})
            return res
        return graph0 + graph1.weighted_links_graph()
    if hasattr(graph1, "node_weights") or hasattr(graph1, "link_weights"):
        return graph1 + graph0
    res = graph0.copy()
    for n in graph1.nodes:
        res.add(n)
    for l in graph1.links:
        res.connect(l.u, l.v)
    return res


def isomorphic_bijection_undirected(graph0: "UndirectedGraph", graph1: "UndirectedGraph") -> dict[Node, Node]:
    if not hasattr(graph1, "neighbors"):
        return {}
    node_weights = hasattr(graph0, "node_weights") and hasattr(graph1, "node_weights")
    link_weights = hasattr(graph0, "link_weights") and hasattr(graph1, "link_weights")
    if node_weights:
        this_weights, other_weights = defaultdict(int), defaultdict(int)
        for w in graph0.node_weights().values():
            this_weights[w] += 1
        for w in graph1.node_weights().values():
            other_weights[w] += 1
        if this_weights != other_weights:
            return {}
    elif len(graph0.nodes) != len(graph1.nodes):
        return {}
    if link_weights:
        this_weights, other_weights = defaultdict(int), defaultdict(int)
        for w in graph0.link_weights().values():
            this_weights[w] += 1
        for w in graph1.link_weights().values():
            other_weights[w] += 1
        if this_weights != other_weights:
            return {}
    elif len(graph0.links) != len(graph1.links):
        return {}
    this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
    for n in graph0.nodes:
        this_nodes_degrees[graph0.degrees(n)].add(n)
    for n in graph1.nodes:
        other_nodes_degrees[graph1.degrees(n)].add(n)
    if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
        return {}
    this_nodes_degrees = sorted(map(list, this_nodes_degrees.values()), key=len)
    other_nodes_degrees = sorted(map(list, other_nodes_degrees.values()), key=len)
    for possibility in product(*map(permutations, this_nodes_degrees)):
        flatten_self = sum(map(list, possibility), [])
        flatten_other = sum(other_nodes_degrees, [])
        map_dict = dict(zip(flatten_self, flatten_other))
        possible = True
        for n, u in map_dict.items():
            for m, v in map_dict.items():
                if node_weights and graph0.node_weights(n) != graph1.node_weights(u):
                    possible = False
                    break
                link_matching = (m in graph0.neighbors(n)) == (v in graph1.neighbors(u))
                if link_weights:
                    link_matching = graph0.link_weights().get(Link(n, m)) == graph1.link_weights().get(Link(u, v))
                if not link_matching or node_weights and graph0.node_weights(m) != graph1.node_weights(v):
                    possible = False
                    break
            if not possible:
                break
        if possible:
            return map_dict
    return {}


def combine_directed(graph0: "DirectedGraph", graph1: "DirectedGraph") -> "DirectedGraph":
    if not hasattr(graph1, "transposed"):
        raise TypeError(f"Addition not defined between class DirectedGraph and type {type(graph1).__name__}!")
    if hasattr(graph0, "node_weights") and hasattr(graph0, "link_weights"):
        if hasattr(graph1, "node_weights") and hasattr(graph1, "link_weights"):
            res = graph0.copy()
            for n in graph1.nodes:
                if n in res:
                    res.increase_weight(n, graph1.node_weights(n))
                else:
                    res.add((n, graph1.node_weights(n)))
            for u, v in graph1.links:
                if v in res.next(u):
                    res.increase_weight((u, v), graph1.link_weights(u, v))
                else:
                    res.connect(v, {u: graph1.link_weights(u, v)})
            return res
        return graph0 + graph1.weighted_graph()
    if hasattr(graph0, "node_weights"):
        if hasattr(graph1, "link_weights"):
            return graph0.weighted_graph() + graph1
        if hasattr(graph1, "node_weights"):
            res = graph0.copy()
            for n in graph1.nodes:
                if n in res:
                    res.increase_weight(n, graph1.node_weights(n))
                else:
                    res.add((n, graph1.node_weights(n)))
            for u, v in graph1.links:
                res.connect(v, [u])
            return res
        return graph0 + graph1.weighted_nodes_graph()
    if hasattr(graph0, "link_weights"):
        if hasattr(graph1, "node_weights"):
            return graph1 + graph0
        if hasattr(graph1, "link_weights"):
            res = graph0.copy()
            for n in graph1.nodes:
                res.add(n)
            for u, v in graph1.links:
                if v in res.next(u):
                    res.increase_weight((u, v), graph1.link_weights(u, v))
                else:
                    res.connect(v, {u: graph1.link_weights((u, v))})
            return res
        return graph0 + graph1.weighted_links_graph()
    if hasattr(graph1, "node_weights") or hasattr(graph1, "link_weights"):
        return graph1 + graph0
    res = graph0.copy()
    for n in graph1.nodes:
        res.add(n)
    for u, v in graph1.links:
        res.connect(v, [u])
    return res


def isomorphic_bijection_directed(graph0: "DirectedGraph", graph1: "DirectedGraph") -> dict[Node, Node]:
    if not hasattr(graph1, "transposed"):
        return {}
    node_weights = hasattr(graph0, "node_weights") and hasattr(graph1, "node_weights")
    link_weights = hasattr(graph0, "link_weights") and hasattr(graph1, "link_weights")
    if node_weights:
        this_weights, other_weights = defaultdict(int), defaultdict(int)
        for w in graph0.node_weights().values():
            this_weights[w] += 1
        for w in graph1.node_weights().values():
            other_weights[w] += 1
        if this_weights != other_weights:
            return {}
    elif len(graph0.nodes) != len(graph1.nodes):
        return {}
    if link_weights:
        this_weights, other_weights = defaultdict(int), defaultdict(int)
        for w in graph0.link_weights().values():
            this_weights[w] += 1
        for w in graph1.link_weights().values():
            other_weights[w] += 1
        if this_weights != other_weights:
            return {}
    elif len(graph0.links) != len(graph1.links):
        return {}
    this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
    for n in graph0.nodes:
        this_nodes_degrees[graph0.degrees(n)].add(n)
    for n in graph1.nodes:
        other_nodes_degrees[graph1.degrees(n)].add(n)
    if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
        return {}
    this_nodes_degrees = sorted(map(list, this_nodes_degrees.values()), key=len)
    other_nodes_degrees = sorted(map(list, other_nodes_degrees.values()), key=len)
    for possibility in product(*map(permutations, this_nodes_degrees)):
        flatten_self = sum(map(list, possibility), [])
        flatten_other = sum(other_nodes_degrees, [])
        map_dict = dict(zip(flatten_self, flatten_other))
        possible = True
        for n, u in map_dict.items():
            for m, v in map_dict.items():
                if node_weights and graph0.node_weights(n) != graph1.node_weights(u):
                    possible = False
                    break
                link_matching = (m in graph0.next(n)) == (v in graph1.next(u))
                if link_weights:
                    link_matching = graph0.link_weights().get((n, m)) == graph1.link_weights().get((u, v))
                if not link_matching or node_weights and graph0.node_weights(m) != graph1.node_weights(v):
                    possible = False
                    break
            if not possible:
                break
        if possible:
            return map_dict
    return {}


def compare(graph0: "Graph", graph1: "Graph") -> bool:
    if type(graph0).__name__ != type(graph1).__name__:
        return False
    if hasattr(graph0, "node_weights"):
        if graph0.node_weights() != graph1.node_weights():
            return False
    elif graph0.nodes != graph1.nodes:
        return False
    if hasattr(graph0, "link_weights"):
        if graph0.link_weights() != graph1.link_weights():
            return False
    return graph0.links == graph1.links


def string(graph: "Graph") -> str:
    nodes = graph.nodes
    if hasattr(graph, "node_weights"):
        nodes = "{" + ", ".join(f"{n} -> {graph.node_weights(n)}" for n in nodes) + "}"
    links = graph.links
    if hasattr(graph, "neighbors"):
        if hasattr(graph, "link_weights"):
            links = "{" + ", ".join(f"{l} -> {graph.link_weights(l)}" for l in links) + "}"
    else:
        links = "{" + ", ".join(
            f"<{l[0]}, {l[1]}>" + (f" -> {graph.link_weights(l)}" if hasattr(graph, "link_weights") else "") for l in
            graph.links) + "}"
    return f"<{nodes}, {links}>"
