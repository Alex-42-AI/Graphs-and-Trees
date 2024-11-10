from functools import reduce

from itertools import permutations, product

from Graphs.General import Node, SortedList


class DirectedGraph:
    def __init__(self, neighborhood: dict = {}, f=lambda x: x):
        self.__nodes, self.__f, self.__links = SortedList(f=f), f, []
        self.__prev, self.__next, self.__degrees = {}, {}, {}
        for u, (prev_nodes, next_nodes) in neighborhood.items():
            self.add(u)
            for v in prev_nodes:
                if v in self:
                    self.connect(u, [v])
                else:
                    self.add(v, points_to=[u])
            for v in next_nodes:
                if v in self:
                    self.connect(v, [u])
                else:
                    self.add(v, [u])

    @property
    def nodes(self):
        return self.__nodes

    @property
    def links(self):
        return self.__links

    def degrees(self, u: Node = None):
        return self.__degrees if u is None else self.__degrees[u]

    def next(self, u: Node = None):
        return self.__next if u is None else self.__next[u]

    def prev(self, u: Node = None):
        return self.__prev if u is None else self.__prev[u]

    @property
    def sources(self):
        return [u for u in self.nodes if not self.degrees(u)[0]]

    @property
    def sinks(self):
        return [v for v in self.nodes if not self.degrees(v)[1]]

    @property
    def f(self):
        return self.__f

    def add(self, u: Node, pointed_by: [Node] = (), points_to: [Node] = ()):
        if u not in self:
            self.__nodes.insert(u)
            self.__degrees[u], self.__next[u], self.__prev[u] = [0, 0], SortedList(f=self.f), SortedList(f=self.f)
            DirectedGraph.connect(self, u, pointed_by, points_to)
        return self

    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest:
            if n in self:
                DirectedGraph.disconnect(self, n, self.prev(n).copy(), self.next(n).copy())
                self.__nodes.remove(n), self.__degrees.pop(n), self.__prev.pop(n), self.__next.pop(n)
        return self

    def connect(self, u: Node, pointed_by: [Node] = (), points_to: [Node] = ()):
        if u in self:
            for v in pointed_by:
                if u != v and v not in self.prev(u) and v in self:
                    self.__links.append((v, u)), self.__prev[u].insert(v), self.__next[v].insert(u)
                    self.__degrees[u][0] += 1
                    self.__degrees[v][1] += 1
            for v in points_to:
                if u != v and v not in self.next(u) and v in self:
                    self.__links.append((u, v)), self.__prev[v].insert(u), self.__next[u].insert(v)
                    self.__degrees[u][1] += 1
                    self.__degrees[v][0] += 1
        return self

    def disconnect(self, u: Node, pointed_by: [Node] = (), points_to: [Node] = ()):
        if u in self:
            for v in pointed_by:
                if v in self.prev(u):
                    self.__degrees[u][0] -= 1
                    self.__degrees[v][1] -= 1
                    self.__links.remove((v, u)), self.__next[v].remove(u), self.__prev[u].remove(v)
            for v in points_to:
                if v in self.next(u):
                    self.__degrees[u][1] -= 1
                    self.__degrees[v][0] -= 1
                    self.__links.remove((u, v)), self.__next[u].remove(v), self.__prev[v].remove(u)
        return self

    def copy(self):
        return DirectedGraph({n: (self.prev(n), self.next(n)) for n in self.nodes}, self.f)

    def complementary(self):
        res = DirectedGraph({n: ([], []) for n in self.nodes}, self.f)
        for i, n in enumerate(self.nodes):
            for m in self.nodes[i + 1:]:
                if m not in self.next(n):
                    res.connect(m, [n])
        return res

    def transposed(self):
        return DirectedGraph({u: (self.next(u), self.prev(u)) for u in self.nodes}, self.f)

    def connection_components(self):
        components, rest = [], self.nodes.copy()
        while rest:
            curr = self.component(rest[0])
            components.append(curr)
            for n in curr.nodes:
                rest.remove(n)
        return components

    def connected(self):
        if (m := len(self.links)) + 1 < (n := len(self.nodes)):
            return False
        if m > (n - 1) * (n - 2) or n < 2:
            return True
        return len(self.connection_components()) == 1

    def reachable(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if u == v:
            return True
        return v in self.subgraph(u)

    def component(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], DirectedGraph({u: ([], [])}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, [v])
                else:
                    res.add(n, [v]), queue.append(n)
            for n in self.prev(v):
                if n in res:
                    res.connect(v, [n])
                else:
                    res.add(n, [], [v]), queue.append(n)
        return res

    def subgraph(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], DirectedGraph({u: ([], [])}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, [v])
                else:
                    res.add(n, [v]), queue.append(n)
        return res

    def has_loop(self):
        sources, total, stack = self.sources, SortedList(f=self.f), SortedList(f=self.f)
        if not sources or not self.sinks:
            return True

        def dfs(u):
            for v in self.next(u):
                if v in total:
                    continue
                if v in stack:
                    return True
                stack.insert(v)
                if dfs(v):
                    return True
                stack.remove(v)
            total.insert(u)
            return False

        for n in sources:
            stack.insert(n)
            if dfs(n):
                return True
            stack.remove(n)
        return False

    def dag(self):
        return not self.has_loop()

    def toposort(self):
        if not self.dag():
            return []
        layer, total = self.sources, set()
        res = layer.copy()
        while layer:
            new = SortedList(f=self.f)
            for u in layer:
                total.add(u)
                for v in self.next(u):
                    if v not in new:
                        new.insert(v)
            for u in new.copy():
                if any(v not in total for v in self.prev(u)):
                    new.remove(u)
            res, layer = res + new.value, new.copy()
        return res

    def get_shortest_path(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        previous = {n: None for n in self.nodes}
        queue, total = [u], SortedList(u, f=self.f)
        previous.pop(u)
        while queue:
            if (n := queue.pop(0)) == v:
                res, curr_node = [n], n
                while curr_node != u:
                    res.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]
                return res
            for y in filter(lambda _x: _x not in total, self.next(n)):
                queue.append(y), total.insert(y)
                previous[y] = n

    def euler_tour_exists(self):
        for d in self.degrees().values():
            if d[0] != d[1]:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if self.euler_tour_exists():
            return u == v
        for n in self.nodes:
            if self.degrees(n)[1] + (n == v) != self.degrees(n)[0] + (n == u):
                return False
        return self.connected()

    def euler_tour(self):
        if self.euler_tour_exists():
            tmp = DirectedGraph.copy(self)
            v, u = self.links[0]
            return tmp.disconnect(u, [v]).euler_walk(u, v)
        return []

    def euler_walk(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = DirectedGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i + 1], [path[i]])
            for i, u in enumerate(path):
                while tmp.next(u):
                    curr = tmp.disconnect(v := tmp.next(u)[0], [u]).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j + 1], [curr[j]])
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def strongly_connected_components(self):
        def dfs(x, stack):
            def helper(_x, _stack):
                for _y in filter(lambda _z: _z not in helper_safe, self.next(_x)):
                    if _y not in curr:
                        helper(_y, _stack + [_y])
                    elif _x not in curr:
                        while _stack:
                            total.add(current_node := _stack.pop()), curr.insert(current_node)
                        return
                helper_safe.add(_x)

            for y in filter(lambda z: z not in dfs_safe, self.next(x)):
                if y not in curr and y not in stack:
                    dfs(y, stack + [y])
                if y == n:
                    curr_node = stack.pop()
                    while stack and curr_node != n:
                        curr.insert(curr_node), total.add(curr_node)
                        curr_node = stack.pop()
                    for curr_node in curr:
                        helper_safe = set()
                        helper(curr_node, [])
                    return
            dfs_safe.add(x)

        if self.dag():
            return list(map(lambda x: [x], self.nodes))
        if not self.connected():
            return reduce(lambda x, y: x + y, map(lambda z: z.strongly_connected_components(), self.connection_components()))
        total, res = set(), []
        for n in self.nodes:
            if n not in total:
                dfs_safe, curr = set(), SortedList(n, f=self.f)
                total.add(n), dfs(n, [n]), res.append(curr.value)
        return res

    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack):
            if not l:
                return ([], list(map(lambda link: link[0], stack)) + [v])[x == v]
            for y in filter(lambda _x: (x, _x) not in stack, self.next(x)):
                res = dfs(y, l - 1, stack + [(x, y)])
                if res:
                    return res
            return []

        tmp = self.get_shortest_path(u, v)
        if not tmp or (k := len(tmp)) > length + 1:
            return []
        if length + 1 == k:
            return tmp
        return dfs(u, length, [])

    def loopWithLength(self, length: int):
        if abs(length) < 2:
            return []
        tmp = DirectedGraph.copy(self)
        for l in tmp.links:
            u, v = l
            res = tmp.disconnect(v, [u]).pathWithLength(v, u, length - 1)
            tmp.connect(v, [u])
            if res:
                return res
        return []

    def hamiltonTourExists(self):
        def dfs(x):
            if tmp.nodes == [x]:
                return x in can_end_in
            if all(y not in tmp for y in can_end_in):
                return False
            tmp0, tmp1 = tmp.prev(x).copy(), tmp.next(x).copy()
            tmp.remove(x)
            for y in tmp1:
                if dfs(y):
                    tmp.add(x, tmp0, tmp1)
                    return True
            tmp.add(x, tmp0, tmp1)
            return False

        if len(self.links) > (len(self.nodes) - 1) ** 2 + 1:
            return True
        tmp = DirectedGraph.copy(self)
        can_end_in = tmp.prev(u := self.nodes[0]).copy()
        return dfs(u)

    def hamiltonWalkExists(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if u in self.next(v):
            return True if all(n in {u, v} for n in self.nodes) else self.hamiltonTourExists()
        tmp = DirectedGraph.copy(self)
        return tmp.connect(u, [v]).hamiltonTourExists()

    def hamiltonTour(self):
        if self.sources or self.sinks:
            return []
        for v in self.prev(u := self.nodes[0]):
            if result := self.hamiltonWalk(u, v):
                return result
        return []

    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            too_many = v is not None
            for n in tmp.nodes:
                if not tmp.degrees(n)[0] and n != x:
                    return []
                if not tmp.degrees(n)[1] and n != v:
                    if too_many:
                        return []
                    too_many = True
            tmp0, tmp1 = tmp.prev(x).copy(), tmp.next(x).copy()
            tmp.remove(x)
            if not tmp.nodes:
                tmp.add(x, tmp0, tmp1)
                return stack
            for y in tmp1:
                if y == v:
                    if tmp.nodes == [v]:
                        tmp.add(x, tmp0, tmp1)
                        return stack + [v]
                    continue
                if res := dfs(y, stack + [y]):
                    tmp.add(x, tmp0, tmp1)
                    return res
            tmp.add(x, tmp0, tmp1)
            return []

        tmp = DirectedGraph.copy(self)
        if u is None:
            if v is not None and v not in self:
                raise Exception("Unrecognized node.")
            for _u in sorted(self.nodes, key=lambda _x: self.degrees(_x)[1]):
                if result := dfs(_u, [_u]):
                    return result
            return []
        if u not in self or v is not None and v not in self:
            raise Exception("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphicFunction(self, other):
        if isinstance(other, DirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = {}, {}
            for d in map(tuple, self.degrees().values()):
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in map(tuple, other.degrees().values()):
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            if this_degrees != other_degrees:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = reduce(lambda x, y: x + list(y), possibility, [])
                flatten_other = reduce(lambda x, y: x + y, other_nodes_degrees, [])
                map_dict = dict(zip(flatten_self, flatten_other))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.next(n)) ^ (v in other.next(u)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return {}

    def __bool__(self):
        return bool(self.nodes)

    def __call__(self, x):
        return self.f(x)

    def __reversed__(self):
        return self.complementary()

    def __contains__(self, u: Node):
        return u in self.nodes

    def __add__(self, other):
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if any(self(n) != other(n) for n in self.nodes.value + other.nodes.value):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, (WeightedNodesDirectedGraph, WeightedLinksDirectedGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            if n not in res:
                res.add(n)
        for (u, v) in other.links:
            if v not in res.next(u):
                res.connect(v, [u])
        return res

    def __eq__(self, other):
        if isinstance(other, DirectedGraph):
            if len(self.links) != len(other.links) or self.nodes != other.nodes:
                return False
            for l in self.links:
                if l not in other.links:
                    return False
            return True
        return False

    def __str__(self):
        return "<{" + ", ".join(str(n) for n in self.nodes) + "}, {" + ", ".join(f"<{l[0]}, {l[1]}>" for l in self.links) + "}>"

    __repr__ = __str__


class WeightedNodesDirectedGraph(DirectedGraph):
    def __init__(self, neighborhood: dict = {}, f=lambda x: x):
        super().__init__({}, f=f)
        self.__node_weights = {}
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v in p[1][0]:
                if v in self:
                    self.connect(u, [v])
                else:
                    self.add((v, 0), points_to=[u])
            for v in p[1][1]:
                if v in self:
                    self.connect(v, [u])
                else:
                    self.add((v, 0), [u])

    def node_weights(self, n: Node = None):
        return self.__node_weights if n is None else self.__node_weights.get(n)

    @property
    def total_nodes_weight(self):
        return sum(self.node_weights().values())

    def copy(self):
        return WeightedNodesDirectedGraph({n: (self.node_weights(n), (self.prev(n), self.next(n))) for n in self.nodes}, self.f)

    def add(self, n_w: (Node, float), pointed_by: [Node] = (), points_to: [Node] = ()):
        super().add(n_w[0], pointed_by, points_to)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest:
            self.__node_weights.pop(n)
        return DirectedGraph.remove(self, u, *rest)

    def set_weight(self, u: Node, w: float):
        if u in self:
            self.__node_weights[u] = w
        return self

    def transposed(self):
        return WeightedNodesDirectedGraph({u: (self.node_weights(u), (self.next(u), self.prev(u))) for u in self.nodes}, self.f)

    def component(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedNodesDirectedGraph({u: (self.node_weights(u), ([], []))}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, [v])
                else:
                    res.add((n, self.node_weights(n)), [v]), queue.append(n)
            for n in self.prev(v):
                if n in res:
                    res.connect(v, [n])
                else:
                    res.add((n, self.node_weights(n)), [], [v]), queue.append(n)
        return res

    def subgraph(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedNodesDirectedGraph({u: (self.node_weights(u), ([], []))}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, [v])
                else:
                    res.add((n, self.node_weights(n)), [v]), queue.append(n)
        return res

    def minimalPathNodes(self, u: Node, v: Node):
        neighborhood = {n: (self.node_weights(n), ({}, {m: 0 for m in self.next(n)})) for n in self.nodes}
        return WeightedDirectedGraph(neighborhood, self.f).minimalPath(u, v)

    def isomorphicFunction(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees, this_weights, other_weights = {}, {}, {}, {}
            for d in map(tuple, self.degrees().values()):
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in map(tuple, other.degrees().values()):
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.node_weights().values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.node_weights().values():
                if w in other_weights:
                    other_weights[w] += 1
                else:
                    other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = reduce(lambda x, y: x + list(y), possibility, [])
                flatten_other = reduce(lambda x, y: x + y, other_nodes_degrees, [])
                map_dict = dict(zip(flatten_self, flatten_other))
                possible = True
                for n, u in map_dict.items():
                    if self.node_weights(n) != other.node_weights(u):
                        possible = False
                        break
                    for m, v in map_dict.items():
                        if (m in self.next(n)) ^ (v in other.next(u)) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphicFunction(other)

    def __add__(self, other):
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if any(self(n) != other(n) for n in self.nodes.value + other.nodes.value):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, WeightedDirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksDirectedGraph):
            return WeightedDirectedGraph({}, self.f) + self + other
        res = self.copy()
        if isinstance(other, WeightedNodesDirectedGraph):
            for n in other.nodes:
                if n in res:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u, v in other.links:
                if v not in res.next(u):
                    res.connect(v, [u])
            return res
        for n in other.nodes:
            if n not in res:
                res.add((n, 0))
        for u, v in other.links:
            if v not in res.next(u):
                res.connect(v, [u])
        return res

    def __eq__(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links) != len(other.links) or self.node_weights() != other.node_weights():
                return False
            for l in self.links:
                if l not in other.links:
                    return False
            return True
        return False

    def __str__(self):
        return "<{" + ", ".join(f"{str(n)} -> {self.node_weights(n)}" for n in self.nodes) + "}, {" + ", ".join(f"<{l[0]}, {l[1]}>" for l in self.links) + "}>"


class WeightedLinksDirectedGraph(DirectedGraph):
    def __init__(self, neighborhood: dict = {}, f=lambda x: x):
        super().__init__({}, f=f)
        self.__link_weights = {}
        for u, (prev_pairs, next_pairs) in neighborhood.items():
            if u not in self:
                self.add(u)
            for v, w in prev_pairs.items():
                if v not in self:
                    self.add(v, points_to_weights={u: w})
                elif v not in self.prev(u):
                    self.connect(u, {v: w})
            for v, w in next_pairs.items():
                if v not in self:
                    self.add(v, {u: w})
                elif v not in self.next(u):
                    self.connect(v, {u: w})

    def link_weights(self, u_or_l: Node | tuple = None, v: Node = None):
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return {n: self.__link_weights[(u_or_l, n)] for n in self.next(u_or_l)}
            return self.__link_weights.get((u_or_l, v))
        elif isinstance(u_or_l, tuple):
            return self.__link_weights.get(u_or_l)

    @property
    def total_links_weight(self):
        return sum(self.link_weights().values())

    def add(self, u: Node, pointed_by_weights: dict = {}, points_to_weights: dict = {}):
        if u not in self:
            super().add(u), self.connect(u, pointed_by_weights, points_to_weights)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            for v in self.next(u):
                self.__link_weights.pop((u, v))
            for v in self.prev(u):
                self.__link_weights.pop((v, u))
        return super().remove(n, *rest)

    def connect(self, u: Node, pointed_by_weights: dict = {}, points_to_weights: dict = {}):
        if u in self:
            super().connect(u, pointed_by_weights.keys(), points_to_weights.keys())
            for v, w in pointed_by_weights.items():
                if (v, u) not in self.link_weights():
                    self.set_weight((v, u), w)
            for v, w in points_to_weights.items():
                if (u, v) not in self.link_weights():
                    self.set_weight((u, v), w)
        return self

    def disconnect(self, u: Node, pointed_by: [Node] = (), points_to: [Node] = ()):
        if u in self:
            for v in pointed_by:
                self.__link_weights.pop((v, u))
            for v in points_to:
                self.__link_weights.pop((u, v))
            super().disconnect(u, pointed_by, points_to)
        return self

    def set_weight(self, l: tuple, w: float):
        if l in self.links:
            self.__link_weights[l] = w
        return self

    def transposed(self):
        neighborhood = {u: (self.link_weights(u), {v: self.link_weights(v, u) for v in self.prev(u)}) for u in self.nodes}
        return WeightedLinksDirectedGraph(neighborhood, self.f)

    def copy(self):
        neighborhood = {u: ({v: self.link_weights(v, u) for v in self.prev(u)}, self.link_weights(u)) for u in self.nodes}
        return WeightedLinksDirectedGraph(neighborhood, self.f)

    def component(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksDirectedGraph({u: ({}, {})}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, {v: self.link_weights((v, n))})
                else:
                    res.add(n, {v: self.link_weights((v, n))}), queue.append(n)
            for n in self.prev(v):
                if n in res:
                    res.connect(v, {n: self.link_weights((n, v))})
                else:
                    res.add(n, points_to_weights={v: self.link_weights((n, v))}), queue.append(n)
        return res

    def subgraph(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksDirectedGraph({u: ({}, {})}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})
                else:
                    res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def minimalPathLinks(self, u: Node, v: Node):
        return WeightedDirectedGraph({n: (0, ({}, self.link_weights(n))) for n in self.nodes}, self.f).minimalPath(u, v)

    def isomorphicFunction(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees, this_weights, other_weights = {}, {}, {}, {}
            for d in map(tuple, self.degrees().values()):
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in map(tuple, other.degrees().values()):
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.link_weights().values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.link_weights().values():
                if w in other_weights:
                    other_weights[w] += 1
                else:
                    other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = reduce(lambda x, y: x + list(y), possibility, [])
                flatten_other = reduce(lambda x, y: x + y, other_nodes_degrees, [])
                map_dict = dict(zip(flatten_self, flatten_other))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if self.link_weights((n, m)) != other.link_weights((u, v)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphicFunction(other)

    def __add__(self, other):
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if any(self(n) != other(n) for n in self.nodes.value + other.nodes.value):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, WeightedNodesDirectedGraph):
            return other + self
        res = self.copy()
        if isinstance(other, WeightedLinksDirectedGraph):
            for n in other.nodes:
                if n not in res:
                    res.add(n)
            for u, v in other.links:
                if v in res.next(u):
                    res.set_weight((u, v), res.link_weights(u, v) + other.link_weights(u, v))
                else:
                    res.connect(v, {u: other.link_weights((u, v))})
            return res
        for n in other.nodes:
            if n not in res:
                res.add(n)
        for u, v in other.links:
            res.connect(v, {u: self.link_weights((u, v))})
        return res

    def __eq__(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            return self.nodes == other.nodes and self.link_weights() == other.link_weights()
        return False

    def __str__(self):
        return "<{" + ", ".join(str(n) for n in self.nodes) + "}, " + ", ".join(f"<{l[0]}, {l[1]}>: {self.link_weights(l)}" for l in self.links)


class WeightedDirectedGraph(WeightedNodesDirectedGraph, WeightedLinksDirectedGraph):
    def __init__(self, neighborhood: dict = {}, f=lambda x: x):
        WeightedNodesDirectedGraph.__init__(self, {}, f)
        WeightedLinksDirectedGraph.__init__(self, {}, f)
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v, w in p[1][0].items():
                if v in self:
                    self.connect(u, {v: w})
                else:
                    self.add((v, 0), points_to_weights={u: w})
        for u, p in neighborhood.items():
            for v, w in p[1][1].items():
                if v in self:
                    self.connect(v, {u: w})
                else:
                    self.add((v, 0), {u: w})

    @property
    def total_weight(self):
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: (Node, float), pointed_by_weights: dict = {}, points_to_weights: dict = {}):
        WeightedLinksDirectedGraph.add(self, n_w[0], pointed_by_weights, points_to_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest:
            WeightedLinksDirectedGraph.disconnect(self, n, self.prev(n), self.next(n)), super().remove(n)
        return self

    def connect(self, u: Node, pointed_by_weights: dict = {}, points_to_weights: dict = {}):
        return WeightedLinksDirectedGraph.connect(self, u, pointed_by_weights, points_to_weights)

    def disconnect(self, u: Node, pointed_by: [Node] = (), points_to: [Node] = ()):
        return WeightedLinksDirectedGraph.disconnect(self, u, pointed_by, points_to)

    def set_weight(self, el: Node | tuple, w: float):
        if el in self:
            super().set_weight(el, w)
        elif el in self.links:
            WeightedLinksDirectedGraph.set_weight(self, el, w)
        return self

    def transposed(self):
        neighborhood = {u: (self.node_weights(u), (self.link_weights(u), {v: self.link_weights(v, u) for v in self.prev(u)})) for u in self.nodes}
        return WeightedDirectedGraph(neighborhood, self.f)

    def copy(self):
        neighborhood = {u: (self.node_weights(u), ({v: self.link_weights(v, u) for v in self.prev(u)}, self.link_weights(u))) for u in self.nodes}
        return WeightedDirectedGraph(neighborhood, self.f)

    def component(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedDirectedGraph({u: (self.node_weights(u), ({}, {}))}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})
                else:
                    res.add((n, self.node_weights(n)), {v: self.link_weights((v, n))}), queue.append(n)
            for n in self.prev(v):
                if n in res:
                    res.connect(v, {n: self.link_weights(n, v)})
                else:
                    res.add((n, self.node_weights(n)), points_to_weights={v: self.link_weights((n, v))}), queue.append(n)
        return res

    def subgraph(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedDirectedGraph({u: (self.node_weights(u), ({}, {}))}, self.f)
        while queue:
            for n in self.next(v := queue.pop(0)):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})
                else:
                    res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None:
                res_path = []
            for y in filter(lambda _y: (x, _y) not in curr_path, self.next(x)):
                new_curr_w = curr_w + self.link_weights(x, y) + self.node_weights(y)
                new_total_negative = total_negative
                if (n_w := self.node_weights(y)) < 0:
                    new_total_negative -= n_w
                if (l_w := self.link_weights(x, y)) < 0:
                    new_total_negative -= l_w
                if new_curr_w + new_total_negative >= res_w and res_path:
                    continue
                if y == v and (new_curr_w < res_w or not res_path):
                    res_path, res_w = curr_path + [(x, y)], curr_w + self.link_weights(x, y) + self.node_weights(y)
                curr = dfs(y, curr_path + [(x, y)], new_curr_w, new_total_negative, res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w

        if u in self and v in self:
            if self.reachable(u, v):
                nodes_negative_weights = sum(self.node_weights(n) for n in self.nodes if self.node_weights(n) < 0)
                links_negative_weights = sum(self.link_weights(l) for l in self.links if self.link_weights(l) < 0)
                res = dfs(u, [], self.node_weights(u), nodes_negative_weights + links_negative_weights)
                return [l[0] for l in res[0]] + [res[0][-1][1]], res[1]
            return [], 0
        raise ValueError('Unrecognized node(s)!')

    def isomorphicFunction(self, other):
        if isinstance(other, WeightedDirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = {}, {}
            this_node_weights, other_node_weights = {}, {}
            this_link_weights, other_link_weights = {}, {}
            for d in map(tuple, self.degrees().values()):
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in map(tuple, other.degrees().values()):
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.link_weights().values():
                if w in this_link_weights:
                    this_link_weights[w] += 1
                else:
                    this_link_weights[w] = 1
            for w in other.link_weights().values():
                if w in other_link_weights:
                    other_link_weights[w] += 1
                else:
                    other_link_weights[w] = 1
            for w in self.node_weights().values():
                if w in this_node_weights:
                    this_node_weights[w] += 1
                else:
                    this_node_weights[w] = 1
            for w in other.node_weights().values():
                if w in other_node_weights:
                    other_node_weights[w] += 1
                else:
                    other_node_weights[w] = 1
            if this_degrees != other_degrees or this_node_weights != other_node_weights or this_link_weights != other_link_weights:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = reduce(lambda x, y: x + list(y), possibility, [])
                flatten_other = reduce(lambda x, y: x + y, other_nodes_degrees, [])
                map_dict = dict(zip(flatten_self, flatten_other))
                possible = True
                for n, u in map_dict.items():
                    if self.node_weights(n) != other.node_weights(u):
                        possible = False
                        break
                    for m, v in map_dict.items():
                        if self.link_weights(n, m) != other.link_weights(u, v) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        if isinstance(other, (WeightedNodesDirectedGraph, WeightedLinksDirectedGraph)):
            return type(other).isomorphicFunction(other, self)
        return DirectedGraph.isomorphicFunction(self, other)

    def __add__(self, other):
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if any(self(n) != other(n) for n in self.nodes.value + other.nodes.value):
            raise ValueError("Node sorting functions don't match!")
        res = self.copy()
        if isinstance(other, WeightedDirectedGraph):
            for n in other.nodes:
                if n in res:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u, v in other.links:
                if v in res.next(u):
                    res.set_weight((u, v), res.link_weights(u, v) + other.link_weights(u, v))
                else:
                    res.connect(v, {u: other.link_weights(u, v)})
        elif isinstance(other, WeightedNodesDirectedGraph):
            for n in other.nodes:
                if n in res:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for (u, v) in other.links:
                if v not in res.next(u):
                    res.connect(v, {u: 0})
        elif isinstance(other, WeightedLinksDirectedGraph):
            for n in other.nodes:
                if n not in res:
                    res.add((n, 0))
            for u, v in other.links:
                if v in res.next(u):
                    res.set_weight((u, v), res.link_weights(u, v) + other.link_weights(u, v))
                else:
                    res.connect(v, {u: other.link_weights(u, v)})
        else:
            for n in other.nodes:
                if n not in res:
                    res.add((n, 0))
            for (u, v) in other.links:
                if v not in res.next(u):
                    res.connect(v, {u: 0})
        return res

    def __eq__(self, other):
        if isinstance(other, WeightedDirectedGraph):
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False

    def __str__(self):
        return f"<{self.node_weights()}, {", ".join(f"<{l[0]}, {l[1]}>: {self.link_weights(l)}" for l in self.links)}>"
