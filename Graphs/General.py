from Lists import SortedList


class Dict:
    def __init__(self, *args: (object, object)):
        self.__items = []
        for arg in args:
            if arg not in self.__items:
                if len(arg) != 2: raise ValueError('Pairs of keys and values expected!')
                self.__items.append(arg)
        for i, item1 in enumerate(self.__items):
            for item2 in self.__items[i + 1:]:
                if item1[0] == item2[0]: raise KeyError('No similar keys in a dictionary allowed!')

    def keys(self):
        return [p[0] for p in self.items()]

    def values(self):
        return [p[1] for p in self.items()]

    def items(self):
        return self.__items

    def pop(self, item):
        for p in self.__items:
            if item == p[0]:
                self.__items.remove(p)
                return p[1]

    def popitem(self):
        if self.items():
            res = self.__items[-1]
            self.__items.pop()
            return res

    def copy(self):
        return Dict(*self.items())

    def __len__(self):
        return len(self.items())

    def __contains__(self, item):
        return item in self.keys()

    def __delitem__(self, key):
        self.pop(key)

    def __getitem__(self, item):
        for (k, v) in self.items():
            if k == item:
                return v

    def __setitem__(self, key, value):
        for i in range(len(self.items())):
            if self.items()[i][0] == key:
                self.__items[i] = (key, value)
                return
        self.__items.append((key, value))

    def __add__(self, other):
        if isinstance(other, (dict, Dict)):
            res = self.copy()
            for (k, v) in other.items(): res[k] = v
            return res
        raise TypeError(f'Addition not defined between type Dict and type {type(other)}!')

    def __eq__(self, other):
        if isinstance(other, Dict):
            for i in self.items():
                if i not in other.items():
                    return False
            return len(self.items()) == len(other.items())
        if isinstance(other, dict):
            for k, v in self.items():
                if k not in other.keys() or other.get(k) != v:
                    return False
            return len(self.items()) == len(other.items())
        return False

    def __str__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in self.items()) + '}'

    def __repr__(self):
        return str(self)


class SortedKeysDict(Dict):
    def __init__(self, *args: (object, object), f=lambda x: x):
        self.__items, self.__f = SortedList(lambda x: f(x[0])), f
        for arg in args:
            if arg not in self.__items:
                if len(arg) != 2:
                    raise ValueError('Pairs of keys and values expected!')
                self.__items.insert(arg)
        for i in range(len(self.__items) - 1):
            if self.__items[i][0] == self.__items[i + 1][0]:
                raise KeyError('No similar keys in a dictionary allowed!')

    def pop(self, item):
        low, high = 0, len(self)
        while low < high:
            mid = (low + high) // 2
            if self.__f(item) == self.__f(self.items()[mid][0]):
                return self.__items.pop(mid)[1]
            if self.__f(item) < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    return
                low = mid

    def copy(self):
        return SortedKeysDict(*self.items().value(), self.__f)

    def items(self):
        return self.__items

    def __len__(self):
        return len(self.items())

    def __contains__(self, item):
        low, high = 0, len(self)
        while low < high:
            mid = (low + high) // 2
            if self.__f(item) == self.__f(self.items()[mid][0]):
                return True
            if self.__f(item) < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    return False
                low = mid

    def __getitem__(self, item):
        low, high = 0, len(self)
        while low < high:
            mid = (low + high) // 2
            if self.__f(item) == self.__f(self.items()[mid][0]):
                return self.items()[mid][1]
            if self.__f(item) < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    return
                low = mid

    def __setitem__(self, key, value):
        low, high = 0, len(self)
        while low < high:
            mid = (low + high) // 2
            if self.__f(key) == self.__f(self.items()[mid][0]):
                self.__items[mid] = (key, value)
                return
            if self.__f(key) < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    break
                low = mid
        self.__items.insert((key, value))

    def __add__(self, other):
        if isinstance(other, SortedKeysDict):
            res = self.copy()
            for (k, v) in other.items():
                res[k] = v
            return res
        raise TypeError(f'Addition not defined between type SortedKeysDict and type {type(other)}!')

    def __eq__(self, other):
        if isinstance(other, SortedKeysDict):
            for p in self.items().value():
                if p not in other.items():
                    return False
            return len(self) == len(other)
        return False

    def __str__(self):
        return '{' + ', '.join(f'{k}: {v}' for k, v in self.items().value()) + '}'


class Node:
    def __init__(self, value):
        self.__value = value

    def value(self):
        return self.__value

    def copy(self):
        return Node(self.__value)

    def __bool__(self):
        return bool(self.__value)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__value == other.__value
        return False

    def __lt__(self, other):
        if isinstance(other, Node):
            return self.__value < other.__value
        return False

    def __le__(self, other):
        if isinstance(other, Node):
            return self.__value <= other.__value
        return False

    def __ge__(self, other):
        if isinstance(other, Node):
            return self.__value >= other.__value
        return False

    def __gt__(self, other):
        if isinstance(other, Node):
            return self.__value > other.__value
        return False

    def __str__(self):
        return '(' + str(self.__value) + ')'

    def __repr__(self):
        return str(self)


class BinNode(Node):
    def __init__(self, value=None, left=None, right=None):
        super().__init__(value)
        self.left, self.right = left, right

    def leaf(self):
        return self.left == self.right is None

    def __bool__(self):
        return self.value() is not None

    def __eq__(self, other):
        if isinstance(other, BinNode):
            return (self.value(), self.left, self.right) == (other.value(), other.left, other.right)
        return False


class Link:
    def __init__(self, node1: Node, node2: Node):
        self.__node1, self.__node2 = node1, node2

    def index(self, node: Node):
        if node in (self.__node1, self.__node2):
            return int(node == self.__node2)
        raise Exception('Node not present!')

    def other(self, node: Node):
        if node in [self.__node1, self.__node2]:
            return [self.__node1, self.__node2][node == self.__node1]
        raise ValueError('Unrecognized node!')

    def __contains__(self, item: Node):
        return item in (self.__node1, self.__node2)

    def __len__(self):
        return 1 + (self.__node1 != self.__node2)

    def __getitem__(self, i: int):
        return [self.__node1, self.__node2][i % 2]

    def __eq__(self, other):
        if isinstance(other, Link):
            return (self.__node1, self.__node2) in [(other.__node1, other.__node2), (other.__node2, other.__node1)]
        return False

    def __str__(self):
        return f'{self.__node1}-{self.__node2}'

    def __repr__(self):
        return str(self)
