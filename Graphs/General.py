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


class Link:
    def __init__(self, u: Node, v: Node):
        self.__u, self.__v = u, v

    @property
    def u(self):
        return self.__u

    @property
    def v(self):
        return self.__v

    def __contains__(self, item: Node):
        return item in {self.u, self.v}

    def __hash__(self):
        return hash(frozenset({self.u, self.v}))

    def __eq__(self, other):
        if isinstance(other, Link):
            return {self.u, self.v} == {other.u, other.v}
        return False

    def __str__(self):
        return f"{self.u}-{self.v}"

    __repr__ = __str__
