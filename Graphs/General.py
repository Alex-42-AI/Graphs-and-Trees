from Lists import SortedList


class Dict:
    def __init__(self, *args):
        self.__items = []
        for arg in args:
            if arg not in self.__items:
                if len(arg) != 2:
                    raise ValueError("Pairs of keys and values expected!")
                if arg[0] in self:
                    raise KeyError("No similar keys in a dictionary allowed!")
                self.__items.append(arg)
                
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
            for (k, v) in other.items():
                res[k] = v
            return res
        raise TypeError(f"Addition not defined between type Dict and type {type(other).__name__}!")
        
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
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.items()) + "}"
        
    def __repr__(self):
        return str(self)


class SortedKeysDict(Dict):
    def __init__(self, *args, f=lambda x: x):
        self.__items, self.__f = SortedList(f=lambda x: f(x[0])), f
        for arg in args:
            if arg not in self.__items:
                if len(arg) != 2:
                    raise ValueError("Pairs of keys and values expected!")
                if arg[0] in self:
                    raise KeyError("No similar keys in a dictionary allowed!")
                self[arg[0]] = arg[1]
                
    def f(self, x=None):
        return self.__f if x is None else self.__f(x)
        
    def pop(self, item):
        low, high, f_item = 0, len(self), self.f(item)
        while low < high:
            mid = (low + high) // 2
            if f_item == self.__f(self.items()[mid][0]):
                return self.__items.pop(mid)[1]
            if f_item < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    return
                low = mid
                
    def copy(self):
        return SortedKeysDict(*self.items(), f=self.f())
        
    def items(self):
        return self.__items
        
    def __len__(self):
        return len(self.items())
        
    def __contains__(self, item):
        low, high, f_item = 0, len(self), self.f(item)
        while low < high:
            mid = (low + high) // 2
            if f_item == self.__f(self.items()[mid][0]):
                return True
            if f_item < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    return False
                low = mid
                
    def __getitem__(self, item):
        low, high, f_item = 0, len(self), self.f(item)
        while low < high:
            mid = (low + high) // 2
            if f_item == self.__f(self.items()[mid][0]):
                return self.items()[mid][1]
            if f_item < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    return
                low = mid
                
    def __setitem__(self, key, value):
        low, high, f_key = 0, len(self), self.f(key)
        while low < high:
            mid = (low + high) // 2
            if f_key == self.__f(self.items()[mid][0]):
                self.__items[mid] = (key, value)
                return
            if f_key < self.__f(self.items()[mid][0]):
                high = mid
            else:
                if low == mid:
                    break
                low = mid
        self.__items.insert((key, value))
        
    def __call__(self, x):
        return self.f(x)
        
    def __add__(self, other):
        if isinstance(other, SortedKeysDict):
            res = self.copy()
            for (k, v) in other.items():
                res[k] = v
            return res
        raise TypeError(f"Addition not defined between type SortedKeysDict and type {type(other).__name__}!")
        
    def __eq__(self, other):
        if isinstance(other, SortedKeysDict):
            for p in self.items():
                if p not in other.items():
                    return False
            return len(self) == len(other)
        return False
        
    def __str__(self):
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.items()) + "}"


class Node:
    def __init__(self, value):
        self.__value = value
        
    def value(self):
        return self.__value
        
    def copy(self):
        return Node(self.value())
        
    def __bool__(self):
        return bool(self.value())
        
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.value() == other.value()
        return False
        
    def __lt__(self, other):
        if isinstance(other, Node):
            return self.value() < other.value()
        return self.value() < other
        
    def __le__(self, other):
        if isinstance(other, Node):
            return self.value() <= other.value()
        return self.value() <= other
        
    def __ge__(self, other):
        if isinstance(other, Node):
            return self.value() >= other.value()
        return self.value() >= other
        
    def __gt__(self, other):
        if isinstance(other, Node):
            return self.value() > other.value()
        return self.value() > other
        
    def __str__(self):
        return "(" + str(self.value()) + ")"
        
    def __repr__(self):
        return str(self)


class Link:
    def __init__(self, u: Node, v: Node):
        self.__u, self.__v = u, v
        
    def u(self):
        return self.__u
        
    def v(self):
        return self.__v
        
    def __contains__(self, item: Node):
        return item in (self.u(), self.v())
        
    def __len__(self):
        return 1 + (self.u() != self.v())
        
    def __eq__(self, other):
        if isinstance(other, Link):
            return (self.u(), self.v()) in [(other.u(), other.v()), (other.v(), other.u())]
        return False
        
    def __str__(self):
        return f"{self.u()}-{self.v()}"
        
    def __repr__(self):
        return str(self)
