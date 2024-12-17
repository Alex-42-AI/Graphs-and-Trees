from typing import Iterable

from abc import ABC, abstractmethod


class Node:
    def __init__(self, value):
        self.__value = value

    @property
    def value(self):
        return self.__value

    def copy(self):
        return Node(self.value)

    def __bool__(self):
        return bool(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.value == other.value
        return False

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, Node):
            return self.value <= other.value
        return self.value <= other

    def __ge__(self, other):
        if isinstance(other, Node):
            return self.value >= other.value
        return self.value >= other

    def __gt__(self, other):
        if isinstance(other, Node):
            return self.value > other.value
        return self.value > other

    def __str__(self):
        return "(" + str(self.value) + ")"

    __repr__ = __str__


class Graph(ABC):
    @abstractmethod
    def nodes(self) -> set[Node]:
        pass

    @abstractmethod
    def links(self) -> set:
        pass

    @abstractmethod
    def degrees(self, n: Node = None) -> dict | int:
        pass

    @abstractmethod
    def remove(self, n: Node, *rest: Node) -> "Graph":
        pass

    @abstractmethod
    def connect_all(self, u: Node, *rest: Node) -> "Graph":
        pass

    @abstractmethod
    def disconnect_all(self, u: Node, *rest: Node) -> "Graph":
        pass

    @abstractmethod
    def copy(self) -> "Graph":
        pass

    @abstractmethod
    def complementary(self) -> "Graph":
        pass

    @abstractmethod
    def component(self, u_or_s: Node | Iterable[Node]) -> "Graph":
        pass

    @abstractmethod
    def connection_components(self) -> "list[Graph]":
        pass

    @abstractmethod
    def connected(self) -> bool:
        pass

    @abstractmethod
    def reachable(self, u: Node, v: Node) -> bool:
        pass

    @abstractmethod
    def full(self) -> bool:
        pass

    @abstractmethod
    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        pass

    @abstractmethod
    def euler_tour_exists(self) -> bool:
        pass

    @abstractmethod
    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        pass

    @abstractmethod
    def euler_tour(self) -> list[Node]:
        pass

    @abstractmethod
    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        pass

    @abstractmethod
    def pathWithLength(self, u: Node, v: Node, length: int) -> list[Node]:
        pass

    @abstractmethod
    def loopWithLength(self, length: int) -> list[Node]:
        pass

    @abstractmethod
    def hamiltonTourExists(self) -> bool:
        pass

    @abstractmethod
    def hamiltonWalkExists(self, u: Node, v: Node) -> bool:
        pass

    @abstractmethod
    def hamiltonTour(self) -> list[Node]:
        pass

    @abstractmethod
    def hamiltonWalk(self, u: Node, v: Node) -> list[Node]:
        pass

    @abstractmethod
    def isomorphicFunction(self, other: "Graph") -> dict[Node, Node]:
        pass

    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __reversed__(self) -> "Graph":
        pass

    @abstractmethod
    def __contains__(self):
        pass

    @abstractmethod
    def __add__(self, other: "Graph") -> "Graph":
        pass

    @abstractmethod
    def __eq__(self, other: "Graph"):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
