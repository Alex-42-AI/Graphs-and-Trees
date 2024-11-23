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


def heapify(ll: list[int], l: int, h: int, i: int, f=max):
    left, right = 2 * i - l, 2 * i - l + 1
    res = i
    if left <= h and (el := ll[i - 1]) != f(ll[left - 1], el):
        res = left
    if right <= h and (el := ll[res - 1]) != f(ll[right - 1], el):
        res = right
    if res != i:
        ll[i - 1], ll[res - 1] = ll[res - 1], ll[i - 1]
        heapify(ll, res - l - 1, h, res, f)


def build_heap(ll: list[int], h: int = 0):
    if not h:
        h = len(ll)
    for i in range(h // 2, 0, -1):
        heapify(ll, 0, h, i)
