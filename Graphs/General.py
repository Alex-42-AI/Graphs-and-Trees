class SortedList:
    def __init__(self, *args, f=lambda x: x):
        self.__f, self.__value = f, []
        for arg in args:
            self.insert(arg)

    @property
    def value(self):
        return self.__value

    def f(self, x=None):
        return self.__f if x is None else self.__f(x)

    def pop(self, index: int = -1):
        return self.__value.pop(index)

    def copy(self):
        res = SortedList(f=self.f())
        res.__value = self.value.copy()
        return res

    def insert(self, x):
        try:
            low, high, f_x = 0, len(self), self.f(x)
            while low < high:
                mid = (low + high) // 2
                if f_x == self.f(self[mid]):
                    self.__value.insert(mid, x)
                    return self
                if f_x < self.f(self[mid]):
                    high = mid
                else:
                    if low == mid + 1:
                        return self
                    low = mid + 1
            self.__value.insert(high, x)
        except (ValueError, TypeError):
            pass
        return self

    def remove(self, x):
        try:
            low, high, f_x = 0, len(self), self.f(x)
            while low < high:
                mid = (low + high) // 2
                if f_x == self.f(self[mid]):
                    if x == self[mid]:
                        self.pop(mid)
                        return self
                    i, j, still = mid - 1, mid + 1, True
                    while still:
                        still = False
                        if i >= 0 and self.f(self[i]) == f_x:
                            if x == self[i]:
                                self.pop(i)
                                return self
                            i -= 1
                            still = True
                        if j < len(self) and self.f(self[j]) == f_x:
                            if x == self[j]:
                                self.pop(j)
                                return self
                            j += 1
                            still = True
                    return self
                if f_x < self.f(self[mid]):
                    high = mid
                else:
                    if low == mid:
                        return self
                    low = mid
        except (ValueError, TypeError):
            pass
        return self

    def merge(self, other):
        res = self.copy()
        if isinstance(other, SortedList):
            if any([self.f(el) != other.f(el) for el in self.value + other.value]):
                raise ValueError("Sorting functions of both lists are different!")
            for el in other.value:
                res.insert(el)
        else:
            for x in other:
                res.insert(x)
        return res

    def __call__(self, x):
        return self.f(x)

    def __len__(self):
        return len(self.value)

    def __contains__(self, item):
        try:
            low, high, f_item = 0, len(self), self.f(item)
            while low < high:
                mid = (low + high) // 2
                if f_item == self.f(self[mid]):
                    if item == self[mid]:
                        return True
                    i, j = mid - 1, mid + 1
                    while True:
                        if i >= 0 and self.f(self[i]) == f_item:
                            if item == self[i]:
                                return True
                            i -= 1
                            continue
                        if j < len(self) and self.f(self[j]) == f_item:
                            if item == self[j]:
                                return True
                            j += 1
                            continue
                        break
                    return False
                if f_item < self.f(self[mid]):
                    high = mid
                else:
                    if low == mid:
                        return False
                    low = mid
        except TypeError:
            return False

    def __bool__(self):
        return bool(self.value)

    def __getitem__(self, item: int | slice):
        if isinstance(item, slice):
            res = SortedList(f=self.f())
            res.__value = self.value[item]
            return res
        return self.value[item]

    def __setitem__(self, i: int, value):
        self.remove(self[i]), self.insert(value)

    def __add__(self, other):
        return self.merge(other)

    def __eq__(self, other):
        if isinstance(other, SortedList):
            try:
                if any(self.f(x) != other.f(x) for x in self) or any(self.f(x) != other.f(x) for x in other):
                    return self.value == SortedList(*other.value, f=self.f()).value
                return self.value == other.value
            except (ValueError, TypeError):
                return False
        return self.__value == other

    def __str__(self):
        return "S" + str(self.value)

    __repr__ = __str__


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

    def __repr__(self):
        return "Node" + str(self)


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
        return item in (self.u, self.v)

    def __hash__(self):
        return hash(frozenset((self.u, self.v)))

    def __eq__(self, other):
        if isinstance(other, Link):
            return (self.u, self.v) in [(other.u, other.v), (other.v, other.u)]
        return False

    def __str__(self):
        return f"{self.u}-{self.v}"

    def __repr__(self):
        return f"Link({self.u}, {self.v})"
