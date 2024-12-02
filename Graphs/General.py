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
    def __init__(self):
        pass

    @abstractmethod
    def nodes(self) -> set[Node]:
        pass

    @abstractmethod
    def links(self) -> set:
        pass

    @abstractmethod
    def degrees(self) -> dict | int:
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def remove(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def complementary(self):
        pass

    @abstractmethod
    def component(self):
        pass

    @abstractmethod
    def connection_components(self):
        pass

    @abstractmethod
    def connected(self) -> bool:
        pass

    @abstractmethod
    def reachable(self) -> bool:
        pass

    @abstractmethod
    def full(self) -> bool:
        pass

    @abstractmethod
    def get_shortest_path(self) -> list[Node]:
        pass

    @abstractmethod
    def euler_tour_exists(self) -> bool:
        pass

    @abstractmethod
    def euler_walk_exists(self) -> bool:
        pass

    @abstractmethod
    def euler_tour(self) -> list[Node]:
        pass

    @abstractmethod
    def euler_walk(self) -> list[Node]:
        pass

    @abstractmethod
    def pathWithLength(self) -> list[Node]:
        pass

    @abstractmethod
    def loopWithLength(self) -> list[Node]:
        pass

    @abstractmethod
    def hamiltonTourExists(self) -> bool:
        pass

    @abstractmethod
    def hamiltonWalkExists(self) -> bool:
        pass

    @abstractmethod
    def hamiltonTour(self) -> list[Node]:
        pass

    @abstractmethod
    def hamiltonWalk(self) -> list[Node]:
        pass

    @abstractmethod
    def isomorphicFunction(self) -> dict[Node, Node]:
        pass

    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __reversed__(self):
        pass

    @abstractmethod
    def __contains__(self):
        pass

    @abstractmethod
    def __add__(self):
        pass

    @abstractmethod
    def __eq__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass
