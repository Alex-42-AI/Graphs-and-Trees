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
        Get value.
        """
        return self.__value

    def __bool__(self) -> bool:
        return bool(self.value)

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.value == other.value
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, Node):
            return self.value < other.value
        return self.value < other

    def __le__(self, other) -> bool:
        if isinstance(other, Node):
            return self.value <= other.value
        return self.value <= other

    def __ge__(self, other) -> bool:
        if isinstance(other, Node):
            return self.value >= other.value
        return self.value >= other

    def __gt__(self, other) -> bool:
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
        Get nodes.
        """
        pass

    @abstractmethod
    def links(self) -> set:
        """
        Get links.
        """
        pass

    @abstractmethod
    def degrees(self, n: Node = None) -> dict | int:
        """
        Get degree of a given node or a dictionary with all node degrees.
        """
        pass

    @abstractmethod
    def remove(self, n: Node, *rest: Node) -> "Graph":
        """
        Remove a non-empty set of nodes.
        """
        pass

    @abstractmethod
    def connect_all(self, u: Node, *rest: Node) -> "Graph":
        """
        Add a link between all given nodes.
        """
        pass

    @abstractmethod
    def disconnect_all(self, u: Node, *rest: Node) -> "Graph":
        """
        Disconnect all given nodes.
        """
        pass

    @abstractmethod
    def copy(self) -> "Graph":
        """
        Return identical copy of the graph.
        """
        pass

    @abstractmethod
    def complementary(self) -> "Graph":
        """
        Return a graph, where there are links between nodes exactly
        where there are no links between nodes in the original graph.
        """
        pass

    @abstractmethod
    def component(self, u: Node) -> "Graph":
        """
        Return the weakly connected component, to which a given node belongs.
        """
        pass

    @abstractmethod
    def subgraph(self, u_or_s: Node | Iterable[Node]) -> "Graph":
        """
        If a set of nodes is given, return the subgraph, that only contains these nodes and all links between them.
        If a node is given, return the subgraph of all nodes and links, reachable by it, in a directed graph.
        """
        pass

    @abstractmethod
    def connection_components(self) -> "list[Graph]":
        """
        List out all connected components of the graph.
        """
        pass

    @abstractmethod
    def connected(self) -> bool:
        """
        Check whether the graph is connected.
        """
        pass

    @abstractmethod
    def reachable(self, u: Node, v: Node) -> bool:
        """
        Check whether the first given node can reach the second one.
        """
        pass

    @abstractmethod
    def full(self) -> bool:
        """
        Check whether the graph is fully connected.
        """
        pass

    @abstractmethod
    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        """
        Return one shortest path from the first given node to the second one, if such path exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def euler_tour_exists(self) -> bool:
        """
        Check whether an Euler tour exists.
        """
        pass

    @abstractmethod
    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Check whether an Euler walk exists.
        """
        pass

    @abstractmethod
    def euler_tour(self) -> list[Node]:
        """
        Return an Euler tour, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        """
        Return an Euler walk, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        """
        Return the path from the first given node to the second one, if such path exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def cycle_with_length(self, length: int) -> list[Node]:
        """
        Return a cycle with given length, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def hamilton_tour_exists(self) -> bool:
        """
        Check whether a Hamilton tour exists.
        """
        pass

    @abstractmethod
    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        """
        u: first given node
        v: second given node
        Check whether a Hamilton walk exists from the first given node to the second one.
        """
        pass

    @abstractmethod
    def hamilton_tour(self) -> list[Node]:
        """
        Return a Hamilton tour, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def hamilton_walk(self, u: Node, v: Node) -> list[Node]:
        """
        u: first given node
        v: second given node
        Return a Hamilton walk from node u to node v, if such exists, otherwise empty list.
        """
        pass

    @abstractmethod
    def isomorphic_bijection(self, other: "Graph") -> dict[Node, Node]:
        """
        other: another Graph object
        Return an isomorphic function between the nodes of the graph and those of the
        given graph, if such exists, otherwise empty dictionary. Let f be such a bijection
        and u and v be nodes in the graph. f(u) and f(v) are nodes in the other graph and
        f(u) and f(v) are neighbors (or f(u) points to f(v) for directed graphs) exactly when
        the same applies for u and v. For weighted graphs, the weights are taken into account.
        """
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """
        Check whether the graph has nodes.
        """
        pass

    @abstractmethod
    def __reversed__(self) -> "Graph":
        """
        Complementary graph.
        """
        pass

    @abstractmethod
    def __contains__(self, u: Node) -> bool:
        """
        Check whether a given node is in the graph.
        """
        pass

    @abstractmethod
    def __add__(self, other: "Graph") -> "Graph":
        """
        Combine two graphs.
        """
        pass

    @abstractmethod
    def __eq__(self, other: "Graph"):
        """
        Check whether two graphs are equal.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
