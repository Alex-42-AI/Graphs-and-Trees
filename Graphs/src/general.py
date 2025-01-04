from typing import Iterable

from abc import ABC, abstractmethod


class Node:
    """
    Helper class Node with a hashable value.
    """

    def __init__(self, value) -> None:
        self.__value = value

    @property
    def value(self):
        """
        Returns:
            Node value.
        """
        return self.__value

    def __bool__(self) -> bool:
        return bool(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: "Node") -> bool:
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


class Graph(ABC):
    """
    Abstract base class for graphs.
    """

    @abstractmethod
    def nodes(self) -> set[Node]:
        """
        Returns:
            Graph nodes.
        """
        pass

    @abstractmethod
    def links(self) -> set:
        """
        Returns:
            Graph links.
        """
        pass

    @abstractmethod
    def degrees(self, u: Node = None) -> dict | int:
        """
        Args:
            u: Node object or None.
        Returns:
            Node degree or dictionary with all node degrees.
        """
        pass

    @abstractmethod
    def remove(self, n: Node, *rest: Node) -> "Graph":
        """
        Remove a non-empty set of nodes.
        Args:
            n: First given node.
            rest: Other given nodes.
        """
        pass

    @abstractmethod
    def connect_all(self, u: Node, *rest: Node) -> "Graph":
        """
        Args:
            u: Node object.
            rest: Node objects.
        Connect every given node to every other given node, all present in the graph.
        """
        pass

    @abstractmethod
    def disconnect_all(self, u: Node, *rest: Node) -> "Graph":
        """
        Args:
            u: First given node.
            rest: Other given nodes.
        Disconnect all given nodes, all present in the graph.
        """
        pass

    @abstractmethod
    def copy(self) -> "Graph":
        """
        Returns:
            Identical copy of the graph.
        """
        pass

    @abstractmethod
    def complementary(self) -> "Graph":
        """
        Returns:
            A graph, where there are links between nodes exactly where there aren't in the original graph.
        """
        pass

    @abstractmethod
    def component(self, u: Node) -> "Graph":
        """
        Args:
            u: Given node.
        Returns:
            The weakly connected component, to which a given node belongs.
        """
        pass

    @abstractmethod
    def subgraph(self, u_or_s: Node | Iterable[Node]) -> "Graph":
        """
        Args:
            u_or_s: Given node or set of nodes.
        Returns:
            If a set of nodes is given, return the subgraph, that only contains these nodes and all links between them.
            If a node is given, return the subgraph of all nodes and links, reachable by it, in a directed graph.
        """
        pass

    @abstractmethod
    def connection_components(self) -> "list[Graph]":
        """
        Returns:
            A list of all connection components of the graph.
        """
        pass

    @abstractmethod
    def connected(self) -> bool:
        """
        Returns:
            Whether the graph is connected.
        """
        pass

    @abstractmethod
    def reachable(self, u: Node, v: Node) -> bool:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
            Whether the first given node can reach the second one.
        """
        pass

    @abstractmethod
    def full(self) -> bool:
        """
        Returns:
            Whether the graph is fully connected.
        """
        pass

    @abstractmethod
    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
            One shortest path from u to v, if such path exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def euler_tour_exists(self) -> bool:
        """
        Returns:
            Whether an Euler tour exists.
        """
        pass

    @abstractmethod
    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
            Whether an Euler walk from u to v exists.
        """
        pass

    @abstractmethod
    def euler_tour(self) -> list[Node]:
        """
        Returns:
             An Euler tour, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
             An Euler walk, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def weighted_nodes_graph(self, weights: dict[Node, float] = None) -> "Graph":
        """
        Args:
            weights: A dictionary, mapping nodes to their weights.
        Returns:
            The version of the graph with node weights.
        """
        pass

    @abstractmethod
    def weighted_links_graph(self, weights: dict) -> "Graph":
        """
        Args:
            weights: A dictionary, mapping links to their weights.
        Returns:
            The version of the graph with link weights.
        """
        pass

    @abstractmethod
    def weighted_graph(self, node_weights: dict[Node, float] = None, link_weights: dict = None) -> "Graph":
        """
        Args:
            node_weights: A dictionary, mapping nodes to their weights.
            link_weights: A dictionary, mapping links to their weights.
        Returns:
            The version of the graph with node and link weights.
        """
        pass

    @abstractmethod
    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        """
        Args:
            u: First given node.
            v: Second given node.
            length: Length of the wanted path.
        Returns:
            A path from u to v with given length, if such path exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def cycle_with_length(self, length: int) -> list[Node]:
        """
        Args:
            length: Length of the wanted cycle.
        Returns:
            A cycle with given length, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def hamilton_tour_exists(self) -> bool:
        """
        Returns:
            Whether a Hamilton tour exists.
        """
        pass

    @abstractmethod
    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Args:
            u: first given node
            v: second given node
        Returns:
            Whether a Hamilton walk exists from u to v.
        """
        pass

    @abstractmethod
    def hamilton_tour(self) -> list[Node]:
        """
        Returns:
            A Hamilton tour, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        """
        Args:
            u: first given node or None.
            v: second given node or None.
        Returns:
            A Hamilton walk from u to v, if such exists, otherwise empty list.
            If a node isn't given, it could be any.
        """
        pass

    @abstractmethod
    def isomorphic_bijection(self, other: "Graph") -> dict[Node, Node]:
        """
        Args:
            other: another Graph object
        Returns:
            An isomorphic function (bijection) between the nodes of the graph and those of the
            given graph, if such exists, otherwise empty dictionary. Let f be such a bijection
            and u and v be nodes in the graph. f(u) and f(v) are nodes in the other graph and
            f(u) and f(v) are neighbors (or f(u) points to f(v) for directed graphs) exactly when
            the same applies for u and v. For weighted graphs, the weights are taken into account.
        """
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """
        Returns:
            Whether the graph has nodes.
        """
        pass

    @abstractmethod
    def __reversed__(self) -> "Graph":
        """
        Returns:
            Complementary graph.
        """
        pass

    @abstractmethod
    def __contains__(self, u: Node) -> bool:
        """
        Args:
            u: item.
        Returns:
            Whether u is a node in the graph.
        """
        pass

    @abstractmethod
    def __add__(self, other: "Graph") -> "Graph":
        pass

    @abstractmethod
    def __eq__(self, other: "Graph"):
        """
        Args:
            other: another Graph object
        Returns:
            Whether both graphs are equal.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
