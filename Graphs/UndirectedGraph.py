from functools import reduce

from itertools import permutations, product

from Graphs.General import Node, Link, SortedList


class UndirectedGraph:
    def __init__(self, neighborhood: dict, f=lambda x: x):
        self.__nodes, self.__f, self.__links = SortedList(f=f), f, []
        self.__neighboring, self.__degrees = {}, {}
        for u, neighbors in neighborhood.items():
            if u not in self.nodes:
                self.add(u)
            for v in neighbors:
                if v in self.nodes:
                    self.connect(u, v)
                else:
                    self.add(v, u)

    @property
    def nodes(self):
        return self.__nodes

    @property
    def links(self):
        return self.__links

    def neighboring(self, u: Node = None):
        return self.__neighboring if u is None else self.__neighboring[u]

    def degrees(self, u: Node = None):
        return self.__degrees if u is None else self.__degrees[u]

    @property
    def degrees_sum(self):
        return 2 * len(self.links)

    @property
    def f(self, x=None):
        return self.__f if x is None else self.__f(x)

    def add(self, u: Node, *current_nodes: Node):
        if u not in self.nodes:
            self.__nodes.insert(u)
            self.__degrees[u], self.__neighboring[u] = 0, SortedList(f=self.f)
            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            if u in self.nodes:
                if tmp := self.neighboring(u):
                    self.disconnect(u, *tmp)
                self.__nodes.remove(u), self.__degrees.pop(u), self.__neighboring.pop(u)
        return self

    def connect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if u != n and n not in self.neighboring(u) and n in self.nodes:
                self.__degrees[u] += 1
                self.__degrees[n] += 1
                self.__neighboring[u].insert(n), self.__neighboring[n].insert(u), self.__links.append(Link(u, n))
        return self

    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if n in self.neighboring(u):
                self.__degrees[u] -= 1
                self.__degrees[n] -= 1
                self.__neighboring[u].remove(n), self.__neighboring[n].remove(u), self.__links.remove(Link(u, n))
        return self

    def copy(self):
        return UndirectedGraph(self.neighboring(), self.f)

    def width(self):
        result = 0
        for n in self.nodes:
            res, total, queue = 0, SortedList(f=self.f), [n]
            total.insert(n)
            while queue:
                new = []
                while queue:
                    for v in filter(lambda x: x not in total, self.neighboring(_ := queue.pop(0))):
                        total.insert(v), new.append(v)
                queue = new.copy()
                res += bool(new)
            if res > result:
                result = res
        return result

    def complementary(self):
        res = UndirectedGraph({n: [] for n in self.nodes}, self.f)
        for i, n in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[j] not in self.neighboring(n):
                    res.connect(n, self.nodes[j])
        return res

    def connection_components(self):
        if not self:
            return [[]]
        components, queue = [[self.nodes[0]]], [self.nodes[0]]
        total, k = SortedList(self.nodes[0], f=self.f), 1
        while k < (n := len(self.nodes)):
            while queue:
                for v in filter(lambda x: x not in total, self.neighboring(_ := queue.pop(0))):
                    components[-1].append(v), queue.append(v), total.insert(v)
                    k += 1
                    if k == n:
                        return components
            if k < n:
                new = [[n for n in self.nodes if n not in total][0]]
                components.append(new), total.insert(new[0])
                k, queue = k + 1, [new[0]]
                if k == n:
                    return components
        return components

    def connected(self):
        if (m := len(self.links)) + 1 < (n := len(self.nodes)):
            return False
        if 2 * m > (n - 1) * (n - 2) or n < 2:
            return True
        return self.component(self.nodes[0]) == self

    def tree(self):
        if not self.connected():
            for c in self.connection_components():
                if len((comp := self.component(c[0])).nodes) + (not comp.nodes) != len(comp.links) + 1:
                    return False
            return True
        return len(self.nodes) + (not self.nodes) == len(self.links) + 1

    def reachable(self, u: Node, v: Node):
        if u not in self.nodes or v not in self.nodes:
            raise Exception("Unrecognized node(s)!")
        if u == v:
            return True
        return v in self.component(u)

    def component(self, u: Node):
        if u not in self.nodes:
            raise ValueError("Unrecognized node!")
        queue, res = [u], UndirectedGraph({u: []}, self.f)
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res.nodes:
                    res.connect(v, n)
                else:
                    res.add(n, v), queue.append(n)
        return res

    def cut_nodes(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False
            for v in self.neighboring(u):
                if not colors[v]:
                    count += 1
                    if (b := dfs(v, l + 1)) >= l and not is_root:
                        is_cut = True
                    min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            if is_cut or is_root and count > 1:
                res.append(u)
            colors[u] = 2
            return min_back

        levels = {n: 0 for n in self.nodes}
        colors, res = levels.copy(), []
        for n in self.nodes:
            if not colors[n]:
                dfs(n, 0)
        return res

    def bridge_links(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back = 1, l, l
            for v in self.neighboring(u):
                if not colors[v]:
                    if (b := dfs(v, l + 1)) > l:
                        res.append(Link(u, v))
                    else:
                        min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            colors[u] = 2
            return min_back

        levels = {n: 0 for n in self.nodes}
        colors, res = levels.copy(), []
        for n in self.nodes:
            if not colors[n]:
                dfs(n, 0)
        return res

    def full(self):
        return self.degrees_sum == (n := len(self.nodes)) * (n - 1)

    def get_shortest_path(self, u: Node, v: Node):
        if u not in self.nodes or v not in self.nodes:
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
            for m in filter(lambda x: x not in total, self.neighboring(n)):
                queue.append(m), total.insert(m)
                previous[m] = n

    def euler_tour_exists(self):
        for n in self.nodes:
            if self.degrees(n) % 2:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError("Unrecognized node(s)!")
        if u == v:
            return self.euler_tour_exists()
        for n in self.nodes:
            if self.degrees(n) % 2 - (n in {u, v}):
                return False
        return self.connected()

    def euler_tour(self):
        if self.euler_tour_exists():
            tmp = UndirectedGraph.copy(self)
            u, v = tmp.links[0].u, tmp.links[0].v
            return tmp.disconnect(u, v).euler_walk(u, v)
        return []

    def euler_walk(self, u: Node, v: Node):
        def dfs(x, stack):
            for y in self.neighboring(x):
                if Link(x, y) not in result + stack:
                    if y == n:
                        result.insert(i + 1, Link(x, y))
                        while stack:
                            result.insert(i + 1, stack.pop())
                        return True
                    if dfs(y, stack + [Link(x, y)]):
                        return True

        if u not in self.nodes or v not in self.nodes:
            raise Exception("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            tmp = self.get_shortest_path(u, v)
            result = [Link(tmp[i], tmp[i + 1]) for i in range(len(tmp) - 1)]
            while len(result) < len(self.links):
                i = -1
                dfs(n := result[0].u, [])
                for i, l in enumerate(result):
                    dfs(n := l.v, [])
            return list(map(lambda _l: _l.u, result)) + [v]
        return []

    def clique(self, n: Node, *nodes: Node):
        res = SortedList(f=self.f)
        for u in nodes:
            if u not in res and u in self.nodes:
                res.insert(u)
        if not res:
            return True
        if any(u not in self.neighboring(n) for u in res):
            return False
        return self.clique(*res)

    def interval_sort(self):
        if not self.links:
            return self.nodes.value
        if not self.connected():
            res = []
            for c in self.connection_components():
                if not (r := self.component(c[0]).interval_sort()):
                    return []
                res += r
            return res
        if len(self.nodes) < 3 or self.full():
            return self.nodes.value
        res = []
        for n in self.nodes:
            if self.clique(n, *self.neighboring(n)):
                queue, total, res, continuer = [n], SortedList(n, f=self.f), [n], False
                while queue:
                    can_be = True
                    for v in filter(lambda x: x not in total, self.neighboring(_ := queue.pop(0))):
                        can_be = False
                        if self.clique(v, *filter(lambda x: x not in total, self.neighboring(v))):
                            can_be = True
                            queue.append(v), total.insert(v), res.append(v)
                            break
                    if not can_be:
                        res, continuer = [], True
                        break
                if continuer:
                    continue
        return res

    def cliques(self, k: int):
        from itertools import combinations
        result = []
        for p in combinations(self.nodes, abs(k)):
            if self.clique(*p):
                result.append(list(p))
        return result

    def chromaticNumberNodes(self):
        def helper(curr):
            if not tmp.nodes:
                return curr
            if tmp.full():
                return curr + list(map(lambda x: [x], tmp.nodes))
            _result = max_nodes
            for anti_clique in tmp.independentSet():
                neighbors = {n: tmp.neighboring(n).copy() for n in anti_clique}
                for n in anti_clique:
                    tmp.remove(n)
                res = helper(curr + [anti_clique])
                for n in anti_clique:
                    tmp.add(n, *neighbors[n])
                if len(res) < len(_result):
                    _result = res
            return _result

        if not self.connected():
            r = []
            for c in self.connection_components():
                r.append(self.component(c[0]).chromaticNumberNodes())
            final = r[0]
            for c in r[1:]:
                for i in range(min(len(c), len(final))):
                    final[i] += c[i]
                if len(c) > len(final):
                    for i in range(len(final), len(c)):
                        final.append(c[i])
            return final
        if self.tree():
            if not self:
                return [[]]
            queue, c0, c1, total = [self.nodes[0]], self.nodes.copy(), [], SortedList(f=self.f)
            while queue:
                flag = (u := queue.pop(0)) in c0
                for v in filter(lambda x: x not in total, self.neighboring(u)):
                    if flag:
                        c1.append(v), c0.remove(v)
                    queue.append(v), total.insert(v)
            return [c0.value, c1]
        if s := self.interval_sort():
            result = [[s[0]]]
            for u in s[1:]:
                found = False
                for r in range(len(result)):
                    if all(v not in self.neighboring(u) for v in result[r]):
                        result[r].append(u)
                        found = True
                        break
                if not found:
                    result.append([u])
            return result
        max_nodes, tmp = self.nodes.value.copy(), UndirectedGraph.copy(self)
        return helper([])

    def chromaticNumberLinks(self):
        return [list(map(lambda x: x.value, s))
                for s in UndirectedGraph({Node(l0): [Node(l1)
                                                     for l1 in self.links if (l1.u in l0) ^ (l1.v in l0)]
                                          for l0 in self.links}, hash).chromaticNumberNodes()]

    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack: [Link]):
            if not l:
                return list(map(lambda link: link.u, stack)) + [v] if [x == v] else []
            for y in filter(lambda _x: Link(x, _x) not in stack, self.neighboring(x)):
                if res := dfs(y, l - 1, stack + [Link(x, y)]):
                    return res
            return []

        if not (tmp := self.get_shortest_path(u, v)) or (k := len(tmp)) > length + 1:
            return []
        if length + 1 == k:
            return tmp
        return dfs(u, length, [])

    def loopWithLength(self, length: int):
        if abs(length) < 3:
            return []
        tmp = UndirectedGraph.copy(self)
        for l in tmp.links:
            u, v = l.u, l.v
            res = tmp.disconnect(u, v).pathWithLength(v, u, abs(length) - 1)
            tmp.connect(u, v)
            if res:
                return res
        return []

    def vertexCover(self):
        nodes, tmp = self.nodes.copy(), UndirectedGraph.copy(self)

        def helper(curr, i=0):
            if not tmp.links:
                return [curr.copy()]
            result = [nodes]
            for j, u in enumerate(nodes[i:]):
                if neighbors := tmp.neighboring(u).copy():
                    tmp.remove(u)
                    if len((res := helper(curr + [u], i + j + 1))[0]) == len(result[0]):
                        result += res
                    elif len(res[0]) < len(result[0]):
                        result = res
                    tmp.add(u, *neighbors)
            return result

        return helper([])

    def dominatingSet(self):
        nodes = self.nodes.copy()

        def helper(curr, total, i=0):
            if total == self.nodes:
                return [curr.copy()]
            result = [self.nodes.value]
            for j, u in enumerate(nodes[i:]):
                new = SortedList(f=self.f)
                if u not in total:
                    new.insert(u)
                for v in self.neighboring(u):
                    if v not in total:
                        new.insert(v)
                if len((res := helper(curr + [u], total + new, i + j + 1))[0]) == len(result[0]):
                    result += res
                elif len(res[0]) < len(result[0]):
                    result = res
            return result

        isolated = SortedList(f=self.f)
        for n in nodes:
            if not self.degrees(n):
                isolated.insert(n), nodes.remove(n)
        return helper(isolated.value, isolated)

    def independentSet(self):
        return [list(filter(lambda x: x not in res, self.nodes)) for res in self.vertexCover()]

    def hamiltonTourExists(self):
        def dfs(x):
            if tmp.nodes == [x]:
                return x in can_end_in
            if all(y not in tmp.nodes for y in can_end_in):
                return False
            neighbors = tmp.neighboring(x).copy()
            tmp.remove(x)
            for y in neighbors:
                if dfs(y):
                    tmp.add(x, *neighbors)
                    return True
            tmp.add(x, *neighbors)
            return False

        if 2 * (l := len(self.links)) > ((k := len(self.nodes)) - 2) * (k - 1) + 3:
            return True
        if any(self.degrees(n) <= 1 for n in self.nodes) or not self.connected():
            return False
        if all(2 * self.degrees(n) >= k for n in self.nodes) or 2 * l > (k - 1) * (k - 2) + 2:
            return True
        tmp = UndirectedGraph.copy(self)
        can_end_in = tmp.neighboring(u := self.nodes[0]).copy()
        return dfs(u)

    def hamiltonWalkExists(self, u: Node, v: Node):
        if u not in self.nodes or v not in self.nodes:
            raise Exception("Unrecognized node(s).")
        if v in self.neighboring(u):
            return True if all(n in {u, v} for n in self.nodes) else self.hamiltonTourExists()
        return UndirectedGraph.copy(self).connect(u, v).hamiltonTourExists()

    def hamiltonTour(self):
        if any(self.degrees(n) < 2 for n in self.nodes) or not self.connected():
            return []
        for v in self.neighboring(u := self.nodes[0]):
            if res := self.hamiltonWalk(u, v):
                return res
        return []

    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            if not tmp.degrees(x) or v is not None and not tmp.degrees(v):
                return []
            too_many = v is not None
            for n in tmp.nodes:
                if n not in {x, v} and tmp.degrees(n) < 2:
                    if too_many:
                        return []
                    too_many = True
            if not tmp.nodes:
                return stack
            neighbors = tmp.neighboring(x).copy()
            tmp.remove(x)
            if v is None and len(tmp.nodes) == 1 and tmp.nodes == neighbors:
                tmp.add(x, neighbors[0])
                return stack + [neighbors[0]]
            for y in neighbors:
                if y == v:
                    if tmp.nodes == [v]:
                        tmp.add(x, *neighbors)
                        return stack + [v]
                    continue
                if res := dfs(y, stack + [y]):
                    tmp.add(x, *neighbors)
                    return res
            tmp.add(x, *neighbors)
            return []

        tmp = UndirectedGraph.copy(self)
        if u is None:
            if v is not None and v not in self.nodes:
                raise Exception("Unrecognized node.")
            for _u in sorted(self.nodes, key=lambda _x: self.degrees(_x)):
                if result := dfs(_u, [_u]):
                    return result
                if self.degrees(_u) == 1:
                    return []
            return []
        if u not in self.nodes or v is not None and v not in self.nodes:
            raise Exception("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphicFunction(self, other):
        if isinstance(other, UndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            if this_degrees != other_degrees:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes:
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                for n in other.nodes:
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                map_dict = dict(zip(reduce(lambda x, y: x + list(y), possibility, []), reduce(lambda x, y: x + y, other_nodes_degrees, [])))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)):
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

    def __contains__(self, item):
        return item in self.nodes or item in self.links

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class UndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes + other.nodes):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            if n not in res.nodes:
                res.add(n)
        for l in other.links:
            if l.u not in res.neighboring(l.v):
                res.connect(l.u, l.v)
        return res

    def __eq__(self, other):
        if isinstance(other, UndirectedGraph):
            if len(self.links) != len(other.links) or self.nodes != other.nodes:
                return False
            for l in self.links:
                if l.u not in other.neighboring(l.v):
                    return False
            return True
        return False

    def __str__(self):
        return "({" + ", ".join(str(n) for n in self.nodes) + "}, {" + ", ".join(str(l) for l in self.links) + "})"

    __repr__ = __str__


class WeightedNodesUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: dict, f=lambda x: x):
        super().__init__({}, f)
        self.__node_weights = {}
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v in p[1]:
                if v in self.nodes:
                    self.connect(u, v)
                else:
                    self.add((v, 0), u)

    def node_weights(self, u: Node = None):
        return self.__node_weights if u is None else self.__node_weights.get(u)

    @property
    def total_nodes_weight(self):
        return sum(self.node_weights().values())

    def add(self, n_w: (Node, float), *current_nodes: Node):
        super().add(n_w[0], *current_nodes)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            self.__node_weights.pop(u)
        return super().remove(n, *rest)

    def set_weight(self, u: Node, w: float):
        if u in self.nodes:
            self.__node_weights[u] = w
        return self

    def copy(self):
        return WeightedNodesUndirectedGraph({n: (self.node_weights(n), self.neighboring(n)) for n in self.nodes}, self.f)

    def component(self, u: Node):
        if u not in self.nodes:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedNodesUndirectedGraph({u: (self.node_weights(u), [])}, self.f)
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res.nodes:
                    res.connect(v, n)
                else:
                    res.add((n, self.node_weights(n)), v), queue.append(n)
        return res

    def minimalPathNodes(self, u: Node, v: Node):
        return WeightedUndirectedGraph({n: (self.node_weights(n), {m: 0 for m in self.neighboring(n)}) for n in self.nodes}, self.f).minimalPath(u, v)

    def weightedVertexCover(self):
        nodes, weights, tmp = self.nodes.copy(), self.total_nodes_weight, WeightedNodesUndirectedGraph.copy(self)

        def helper(curr, res_sum=0, i=0):
            if not tmp.links:
                return [curr.copy()], res_sum
            result, result_sum = [nodes.copy()], weights
            for j, u in enumerate(nodes[i:]):
                neighbors, w = tmp.neighboring(u).copy(), tmp.node_weights(u)
                if neighbors:
                    tmp.remove(u)
                    cover, weight = helper(curr + [u], res_sum + w, i + j + 1)
                    tmp.add((u, w), *neighbors)
                    if weight == result_sum:
                        result += cover
                    elif weight < result_sum:
                        result, result_sum = cover, weight
            return result, result_sum

        for n in self.nodes:
            if not self.degrees(n):
                nodes.remove(n)
                weights -= self.node_weights(n)
        return helper([])[0]

    def weightedDominatingSet(self):
        nodes = self.nodes.copy()

        def helper(curr, total, total_weight, i=0):
            if total == self.nodes:
                return [curr.copy()], total_weight
            result, result_sum = [self.nodes], self.total_nodes_weight
            for j, u in enumerate(nodes[i:]):
                new = SortedList(f=self.f)
                if u not in total:
                    new.insert(u)
                for v in self.neighboring(u):
                    if v not in total:
                        new.insert(v)
                cover, weight = helper(curr + [u], total + new, total_weight + self.node_weights(u), i + j + 1)
                if weight == result_sum:
                    result += cover
                elif weight < result_sum:
                    result, result_sum = cover, weight
            return result, result_sum

        isolated, weights = SortedList(f=self.f), 0
        for n in nodes:
            if not self.degrees(n):
                isolated.insert(n), nodes.remove(n)
                weights += self.node_weights(n)
        return helper(isolated.value, isolated, weights)[0]

    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedNodesUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees, this_weights, other_weights = {}, {}, {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
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
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes:
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                for n in other.nodes:
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                map_dict = dict(zip(reduce(lambda x, y: x + list(y), possibility, []), reduce(lambda x, y: x + y, other_nodes_degrees, [])))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphicFunction(other)

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedNOdesUndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes + other.nodes):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, WeightedUndirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksUndirectedGraph):
            return WeightedUndirectedGraph({}, self.f) + self + other
        res = self.copy()
        if isinstance(other, WeightedNodesUndirectedGraph):
            for n in other.nodes:
                if n in res.nodes:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links:
                res.connect(l.u, l.v)
            return res
        for n in other.nodes:
            if n not in res.nodes:
                res.add((n, 0))
        for l in other.links:
            res.connect(l.u, l.v)
        return res

    def __eq__(self, other):
        if isinstance(other, WeightedNodesUndirectedGraph):
            if self.node_weights() != other.node_weights() or len(self.links) != len(other.links):
                return False
            for l in self.links:
                if l not in other.links:
                    return False
            return True
        return False

    def __str__(self):
        return "({" + ", ".join(f"{str(n)} -> {self.node_weights(n)}"
        for n in self.nodes) + "}, {" + ", ".join(str(l) for l in self.links) + "})"


class WeightedLinksUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: dict, f=lambda x: x):
        super().__init__({}, f)
        self.__link_weights = {}
        for u, neighbors in neighborhood.items():
            if u not in self.nodes:
                self.add(u)
            for v, w in neighbors.items():
                if v not in self.nodes:
                    self.add(v, {u: w})
                elif v not in self.neighboring(u):
                    self.connect(u, {v: w})

    def link_weights(self, u_or_l: Node | Link = None, v: Node = None):
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return {n: self.__link_weights[Link(n, u_or_l)] for n in self.neighboring(u_or_l)}
            return self.__link_weights.get(Link(u_or_l, v))
        elif isinstance(u_or_l, Link):
            return self.__link_weights.get(u_or_l)

    @property
    def total_links_weight(self):
        return sum(self.link_weights().values())

    def add(self, u: Node, nodes_weights: dict = {}):
        if u not in self.nodes:
            for w in nodes_weights.values():
                if not isinstance(w, (int, float)):
                    raise TypeError("Real numerical values expected!")
            super().add(u, *nodes_weights.keys())
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            for v in self.neighboring(u):
                self.__link_weights.pop(Link(u, v))
        return super().remove(n, *rest)

    def connect(self, u: Node, nodes_weights: dict = {}):
        if nodes_weights:
            super().connect(u, *nodes_weights.keys())
        if u in self.nodes:
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def disconnect(self, u: Node, v: Node, *rest: Node):
        super().disconnect(u, v, *rest)
        for n in (v,) + rest:
            if Link(u, n) in self.link_weights():
                self.__link_weights.pop(Link(u, n))
        return self

    def set_weight(self, l: Link, w: float):
        if l.u in self.neighboring(l.v):
            self.__link_weights[l] = w
        return self

    def copy(self):
        return WeightedLinksUndirectedGraph({n: self.link_weights(n) for n in self.nodes}, self.f)

    def component(self, u: Node):
        if u not in self.nodes:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksUndirectedGraph({u: {}}, self.f)
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res.nodes:
                    res.connect(v, {n: self.link_weights(v, n)})
                else:
                    res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def minimal_spanning_tree(self):
        if self.tree():
            return self.links, self.total_links_weight
        if not self.connected():
            res = []
            for comp in self.connection_components():
                res.append(self.component(comp[0]).minimal_spanning_tree())
            return res
        if not self.links:
            return [], 0
        res_links, total = [], SortedList(self.nodes[0], f=self.f)
        bridge_links = SortedList(*[Link(self.nodes[0], u) for u in self.neighboring(self.nodes[0])], f=lambda x: self.link_weights(x))
        for _ in range(1, len(self.nodes)):
            u, v = bridge_links[0].u, bridge_links[0].v
            res_links.append(bridge_links.pop(0))
            if u in total:
                total.insert(v)
                for _v in self.neighboring(v):
                    if _v in total:
                        bridge_links.remove(Link(v, _v))
                    else:
                        bridge_links.insert(Link(v, _v))
            else:
                total.insert(u)
                for _u in self.neighboring(u):
                    if _u in total:
                        bridge_links.remove(Link(u, _u))
                    else:
                        bridge_links.insert(Link(u, _u))
        return res_links, sum(map(lambda x: self.link_weights(x), res_links))

    def minimalPathLinks(self, u: Node, v: Node):
        return WeightedUndirectedGraph({n: (0, self.link_weights(n)) for n in self.nodes}, self.f).minimalPath(u, v)

    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedLinksUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees, this_weights, other_weights = {}, {}, {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
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
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes:
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                for n in other.nodes:
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                map_dict = dict(zip(reduce(lambda x, y: x + list(y), possibility, []), reduce(lambda x, y: x + y, other_nodes_degrees, [])))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphicFunction(other)

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedLinksUndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes + other.nodes):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, WeightedNodesUndirectedGraph):
            return other + self
        res = self.copy()
        if isinstance(other, WeightedLinksUndirectedGraph):
            for n in other.nodes:
                if n not in res.nodes:
                    res.add(n)
            for l in other.links:
                u, v = l.u, l.v
                if v in res.neighboring(u):
                    res.set_weight(l, res.link_weights(l) + other.link_weights(l))
                else:
                    res.connect(u, {v: other.link_weights(l)})
            return res
        for n in other.nodes:
            if n not in res.nodes:
                res.add(n)
        for l in other.links:
            if l not in res.links:
                res.connect(l.u, {l.v: 0})
        return res

    def __eq__(self, other):
        if isinstance(other, WeightedLinksUndirectedGraph):
            return self.nodes == other.nodes and self.link_weights() == other.link_weights()
        return False

    def __str__(self):
        return "({" + ", ".join(str(n) for n in self.nodes) + "}, " + str(self.link_weights()) + ")"


class WeightedUndirectedGraph(WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph):
    def __init__(self, neighborhood: dict, f=lambda x: x):
        WeightedNodesUndirectedGraph.__init__(self, {}, f), WeightedLinksUndirectedGraph.__init__(self, {}, f)
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v, w in p[1].items():
                if v in self.nodes:
                    self.connect(u, {v: w})
                else:
                    self.add((v, 0), {u: w})

    @property
    def total_weight(self):
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: (Node, float), nodes_weights: dict = {}):
        WeightedLinksUndirectedGraph.add(self, n_w[0], nodes_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest:
            if tmp := self.neighboring(u):
                WeightedLinksUndirectedGraph.disconnect(self, u, *tmp)
            super().remove(n)
        return self

    def connect(self, u: Node, nodes_weights: dict = {}):
        return WeightedLinksUndirectedGraph.connect(self, u, nodes_weights)

    def disconnect(self, u: Node, v: Node, *rest: Node):
        return WeightedLinksUndirectedGraph.disconnect(self, u, v, *rest)

    def set_weight(self, el: Node | Link, w: float):
        if el in self.nodes:
            WeightedNodesUndirectedGraph.set_weight(self, el, w)
        elif el in self.links:
            WeightedLinksUndirectedGraph.set_weight(self, el, w)
        return self

    def copy(self):
        return WeightedUndirectedGraph({n: (self.node_weights(n), self.link_weights(n)) for n in self.nodes}, self.f)

    def component(self, u: Node):
        if u not in self.nodes:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedUndirectedGraph({u: (self.node_weights(u), {})}, self.f)
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res.nodes:
                    res.connect(v, {n: self.link_weights(v, n)})
                else:
                    res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None:
                res_path = []
            for y in filter(lambda _y: Link(x, _y) not in curr_path, self.neighboring(x)):
                if curr_w + self.link_weights(x, y) + total_negative >= res_w and res_path:
                    continue
                if y == v and (curr_w + self.link_weights(x, y) + self.node_weights(y) < res_w or not res_path):
                    res_path, res_w = curr_path + [Link(x, y)], curr_w + self.link_weights(x, y) + self.node_weights(y)
                curr = dfs(y, curr_path + [Link(x, y)], curr_w + self.link_weights(x, y) + self.node_weights(y),
                           total_negative - self.link_weights(x, y) * (self.link_weights(x, y) < 0)
                           - self.node_weights(y) * (self.node_weights(y) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w

        if u in self.nodes and v in self.nodes:
            if self.reachable(u, v):
                res = dfs(u, [], self.node_weights(u), sum(self.link_weights(l) for l in self.links
       if self.link_weights(l) < 0) + sum(self.node_weights(n) for n in self.nodes if self.node_weights(n) < 0))
                return [l.u for l in res[0]] + [res[0][-1].v], res[1]
            return [], 0
        raise ValueError('Unrecognized node(s)!')

    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = {}, {}
            this_node_weights, other_node_weights = {}, {}
            this_link_weights, other_link_weights = {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
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
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes:
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                for n in other.nodes:
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                map_dict = dict(zip(reduce(lambda x, y: x + list(y), possibility, []), reduce(lambda x, y: x + y, other_nodes_degrees, [])))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return type(other).isomorphicFunction(other, self)
        return UndirectedGraph.isomorphicFunction(self, other)

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedUndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes + other.nodes):
            raise ValueError("Node sorting functions don't match!")
        res = self.copy()
        if isinstance(other, WeightedUndirectedGraph):
            for n in other.nodes:
                if n in res.nodes:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links:
                u, v = l.u, l.v
                if v in res.neighboring(u):
                    res.set_weight(l, res.link_weights(l) + other.link_weights(l))
                else:
                    res.connect(u, {v: other.link_weights(l)})
        elif isinstance(other, WeightedNodesUndirectedGraph):
            for n in other.nodes:
                if n in res.nodes:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u in other.nodes:
                for v in other.neighboring(u):
                    if v not in res.neighboring(u):
                        res.connect(u, {v: 0})
        elif isinstance(other, WeightedLinksUndirectedGraph):
            for n in other.nodes:
                if n not in res.nodes:
                    res.add((n, 0))
            for l in other.links:
                u, v = l.u, l.v
                if v in res.neighboring(u):
                    res.set_weight(l, res.link_weights(l) + other.link_weights(l))
                else:
                    res.connect(u, {v: other.link_weights(l)})
        else:
            for n in other.nodes:
                if n not in res.nodes:
                    res.add((n, 0))
            for u in other.nodes:
                for v in other.neighboring(u):
                    if v not in res.neighboring(u):
                        res.connect(u, {v: 0})
        return res

    def __eq__(self, other):
        if isinstance(other, WeightedUndirectedGraph):
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False

    def __str__(self):
        return f"({self.node_weights()}, {self.link_weights()})"
