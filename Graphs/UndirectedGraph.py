from Graphs.General import Node, Link, Dict, SortedKeysDict, SortedList


class UndirectedGraph:
    def __init__(self, neighborhood: Dict = Dict(), f=lambda x: x):
        self.__nodes = SortedList(f=f)
        self.__f = f
        self.__links = []
        self.__neighboring, self.__degrees = SortedKeysDict(f=f), SortedKeysDict(f=f)
        for u, neighbors in neighborhood.items():
            if u not in self.nodes():
                self.add(u)
            for v in neighbors:
                if v in self.nodes():
                    self.connect(u, v)
                else:
                    self.add(v, u)
                    
    def nodes(self):
        return self.__nodes
        
    def links(self):
        return self.__links
        
    def neighboring(self, u: Node = None):
        return self.__neighboring if u is None else self.__neighboring[u]
        
    def degrees(self, u: Node = None):
        return self.__degrees if u is None else self.__degrees[u]
        
    def degrees_sum(self):
        return 2 * len(self.links())
        
    def f(self, x=None):
        return self.__f if x is None else self.__f(x)
        
    def add(self, u: Node, *current_nodes: Node):
        if u not in self.nodes():
            self.__nodes.insert(u)
            self.__degrees[u], self.__neighboring[u] = 0, SortedList(f=self.f())
            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)
        return self
        
    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            if u in self.nodes():
                tmp = self.neighboring(u)
                if tmp: self.disconnect(u, *tmp)
                self.__nodes.remove(u), self.__degrees.pop(u), self.__neighboring.pop(u)
        return self
        
    def connect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if u != n and n not in self.neighboring(u) and n in self.nodes():
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
        return UndirectedGraph(self.neighboring(), self.f())
        
    def width(self):
        res = 0
        for u in self.nodes():
            _res, total, queue = 0, SortedList(f=self.f()), [u]
            total.insert(u)
            while queue:
                new = []
                while queue:
                    u = queue.pop(0)
                    for v in filter(lambda x: x not in total, self.neighboring(u)):
                        total.insert(v), new.append(v)
                queue = new.copy()
                _res += bool(new)
            if _res > res:
                res = _res
        return res
        
    def complementary(self):
        res = UndirectedGraph(Dict(*[(n, []) for n in self.nodes()]), self.f())
        for i, n in enumerate(self.nodes()):
            for j in range(i + 1, len(self.nodes())):
                if self.nodes()[j] not in self.neighboring(n):
                    res.connect(n, self.nodes()[j])
        return res
        
    def connection_components(self):
        if not self:
            return [[]]
        components = [[self.nodes()[0]]]
        queue = [self.nodes()[0]]
        total = SortedList(self.nodes()[0], f=self.f())
        k, n = 1, len(self.nodes())
        while k < n:
            while queue:
                u = queue.pop(0)
                for v in filter(lambda x: x not in total, self.neighboring(u)):
                    components[-1].append(v), queue.append(v), total.insert(v)
                    k += 1
                    if k == n:
                        return components
            if k < n:
                new = [[n for n in self.nodes() if n not in total][0]]
                components.append(new), total.insert(new[0])
                k, queue = k + 1, [new[0]]
                if k == n:
                    return components
        return components
        
    def connected(self):
        m, n = len(self.links()), len(self.nodes())
        if m + 1 < n:
            return False
        if 2 * m > (n - 1) * (n - 2) or n < 2:
            return True
        return self.component(self.nodes()[0]) == self
        
    def tree(self):
        if not self.connected():
            for c in self.connection_components():
                comp = self.component(c[0])
                if len(comp.nodes()) + (not comp.nodes()) != len(comp.links()) + 1:
                    return False
            return True
        return len(self.nodes()) + (not self.nodes()) == len(self.links()) + 1
        
    def reachable(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes():
            raise Exception("Unrecognized node(s)!")
        if u == v:
            return True
        total, queue = SortedList(u, f=self.f()), [u]
        while queue:
            n = queue.pop(0)
            for m in filter(lambda x: x not in total, self.neighboring(n)):
                if m == v:
                    return True
                total.insert(m), queue.append(m)
        return False
        
    def component(self, u: Node):
        if u not in self.nodes():
            raise ValueError("Unrecognized node!")
        queue, res = [u], UndirectedGraph(Dict((u, [])), self.f())
        while queue:
            v = queue.pop(0)
            for n in self.neighboring(v):
                if n in res.nodes():
                    res.connect(v, n)
                else:
                    res.add(n, v), queue.append(n)
        return res
        
    def cut_nodes(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back = 1, l, l
            is_root, count, is_cut = not l, 0, False
            for v in self.neighboring(u):
                if not colors[v]:
                    count += 1
                    b = dfs(v, l + 1)
                    if b >= l and not is_root:
                        is_cut = True
                    min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            if is_cut or is_root and count > 1:
                res.append(u)
            colors[u] = 2
            return min_back
            
        levels = SortedKeysDict(*[(n, 0) for n in self.nodes()], f=self.f())
        colors, res = levels.copy(), []
        for n in self.nodes():
            if not colors[n]:
                dfs(n, 0)
        return res
        
    def bridge_links(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back = 1, l, l
            for v in self.neighboring(u):
                if not colors[v]:
                    b = dfs(v, l + 1)
                    if b > l:
                        res.append(Link(u, v))
                    else:
                        min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            colors[u] = 2
            return min_back
            
        levels, res = SortedKeysDict(*[(n, 0) for n in self.nodes()], f=self.f()), []
        colors = levels.copy()
        for n in self.nodes():
            if not colors[n]:
                dfs(n, 0)
        return res
        
    def full(self):
        return self.degrees_sum() == len(self.nodes()) * (len(self.nodes()) - 1)
        
    def get_shortest_path(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes():
            raise Exception("Unrecognized node(s)!")
        previous, queue, total = SortedKeysDict(*[(n, None) for n in self.nodes()], f=self.f()), [u], SortedList(u, f=self.f())
        previous.pop(u)
        while queue:
            n = queue.pop(0)
            if n == v:
                res, curr_node = [n], n
                while curr_node != u:
                    res.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]
                return res
            for m in filter(lambda x: x not in total, self.neighboring(n)):
                queue.append(m), total.insert(m)
                previous[m] = n
                
    def euler_tour_exists(self):
        for n in self.nodes():
            if self.degrees(n) % 2:
                return False
        return self.connected()
        
    def euler_walk_exists(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes():
            raise ValueError("Unrecognized node(s)!")
        if u == v:
            return self.euler_tour_exists()
        for n in self.nodes():
            if n not in (u, v) and self.degrees(n) % 2:
                return False
        return self.degrees(u) % 2 == self.degrees(v) % 2 == 1 and self.connected()
        
    def euler_tour(self):
        if self.euler_tour_exists():
            tmp = UndirectedGraph.copy(self)
            u, v = tmp.links()[0].u(), tmp.links()[0].v()
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
                        
        if u not in self.nodes() or v not in self.nodes():
            raise Exception("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            tmp = self.get_shortest_path(u, v)
            result = [Link(tmp[i], tmp[i + 1]) for i in range(len(tmp) - 1)]
            while len(result) < len(self.links()):
                i, n = -1, result[0].u()
                dfs(n, [])
                for i, l in enumerate(result):
                    n = l.v()
                    dfs(n, [])
            return list(map(lambda _l: _l.u(), result)) + [v]
        return []
        
    def clique(self, n: Node, *nodes: Node):
        res = SortedList(f=self.f())
        for u in nodes:
            if u not in res and u in self.nodes():
                res.insert(u)
        if not res:
            return True
        if any(u not in self.neighboring(n) for u in res):
            return False
        return self.clique(*res)
        
    def interval_sort(self):
        if not self.links():
            return self.nodes().value()
        if not self.connected():
            res = []
            for c in self.connection_components():
                r = self.component(c[0]).interval_sort()
                if not r:
                    return []
                res += r
            return res
        if len(self.nodes()) < 3 or self.full():
            return self.nodes().value()
        res = []
        for n in self.nodes():
            if self.clique(n, *self.neighboring(n)):
                queue, total = [n], SortedList(n, f=self.f())
                res, continuer = [n], False
                while queue:
                    u, can_be = queue.pop(0), True
                    for v in filter(lambda x: x not in total, self.neighboring(u)):
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
        for p in combinations(self.nodes(), abs(k)):
            if self.clique(*p):
                result.append(list(p))
        return result
        
    def chromaticNumberNodes(self):
        def helper(curr):
            if not tmp.nodes():
                return curr
            if tmp.full():
                return curr + list(map(lambda x: [x], tmp.nodes()))
            _result = max_nodes
            for anti_clique in tmp.independentSet():
                neighbors = SortedKeysDict(*[(n, tmp.neighboring(n).copy()) for n in anti_clique], f=tmp.f())
                for n in anti_clique: tmp.remove(n)
                res = helper(curr + [anti_clique])
                for n in anti_clique: tmp.add(n, *neighbors[n])
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
            queue, total = [self.nodes()[0]], SortedList(f=self.f())
            c0, c1 = self.nodes().value().copy(), []
            while queue:
                u = queue.pop(0)
                flag = u in c0
                for v in filter(lambda x: x not in total, self.neighboring(u)):
                    if flag: c1.append(v), c0.remove(v)
                    queue.append(v), total.insert(v)
            return [c0, c1]
        s = self.interval_sort()
        if s:
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
        max_nodes, tmp = self.nodes().value().copy(), UndirectedGraph.copy(self)
        return helper([])
        
    def chromaticNumberLinks(self):
        res_graph = UndirectedGraph(Dict(*[(Node(l0), [Node(l1) for l1 in self.links() if (l1.u() in l0) ^ (l1.v() in l0)]) for l0 in self.links()]), lambda x: (x.value().u(), x.value().v()))
        for u in res_graph.nodes():
            for v in res_graph.nodes():
                if v != u and (u.value().u() in v.value() or u.value().v() in v.value()):
                    res_graph.connect(u, v)
        return [list(map(lambda x: x.value(), s)) for s in res_graph.chromaticNumberNodes()]
        
    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack: [Link]):
            if not l:
                return ([], list(map(lambda link: link.u(), stack)) + [v])[x == v]
            for y in filter(lambda _x: Link(x, _x) not in stack, self.neighboring(x)):
                res = dfs(y, l - 1, stack + [Link(x, y)])
                if res:
                    return res
            return []
            
        tmp = self.get_shortest_path(u, v)
        if not tmp or len(tmp) > length + 1:
            return []
        if length + 1 == len(tmp):
            return tmp
        return dfs(u, length, [])
        
    def loopWithLength(self, length: int):
        if abs(length) < 3:
            return []
        tmp = UndirectedGraph.copy(self)
        for l in tmp.links():
            u, v = l.u(), l.v()
            res = tmp.disconnect(u, v).pathWithLength(v, u, abs(length) - 1)
            tmp.connect(u, v)
            if res:
                return res
        return []
        
    def vertexCover(self):
        nodes, tmp = self.nodes().copy(), UndirectedGraph.copy(self)
        
        def helper(curr, i=0):
            if not tmp.links():
                return [curr.copy()]
            result = [nodes]
            for j, u in enumerate(nodes[i:]):
                neighbors = tmp.neighboring(u).copy()
                if neighbors:
                    tmp.remove(u)
                    res = helper(curr + [u], i + j + 1)
                    tmp.add(u, *neighbors)
                    if len(res[0]) == len(result[0]):
                        result += res
                    elif len(res[0]) < len(result[0]):
                        result = res
            return result
            
        return helper([])
        
    def dominatingSet(self):
        nodes = self.nodes().copy()
        
        def helper(curr, total, i=0):
            if total == self.nodes():
                return [curr.copy()]
            result = [self.nodes().value()]
            for j, u in enumerate(nodes[i:]):
                new = SortedList(f=self.f())
                if u not in total:
                    new.insert(u)
                for v in self.neighboring(u):
                    if v not in total:
                        new.insert(v)
                res = helper(curr + [u], total + new, i + j + 1)
                if len(res[0]) == len(result[0]):
                    result += res
                elif len(res[0]) < len(result[0]):
                    result = res
            return result
            
        isolated = SortedList(f=self.f())
        for n in nodes:
            if not self.degrees(n):
                isolated.insert(n), nodes.remove(n)
        return helper(isolated.value(), isolated)
        
    def independentSet(self):
        result = []
        for res in self.vertexCover():
            result.append(list(filter(lambda x: x not in res, self.nodes())))
        return result
        
    def hamiltonTourExists(self):
        def dfs(x):
            if tmp.nodes() == [x]:
                return x in can_end_in
            if all(y not in tmp.nodes() for y in can_end_in):
                return False
            neighbors = tmp.neighboring(x).copy()
            tmp.remove(x)
            for y in neighbors:
                res = dfs(y)
                if res:
                    tmp.add(x, *neighbors)
                    return True
            tmp.add(x, *neighbors)
            return False
            
        if 2 * len(self.links()) > (len(self.nodes()) - 2) * (len(self.nodes()) - 1) + 3:
            return True
        if any(self.degrees(n) <= 1 for n in self.nodes()) or not self.connected():
            return False
        if all(2 * self.degrees(n) >= len(self.nodes()) for n in self.nodes()) or 2 * len(self.links()) > (len(self.nodes()) - 1) * (len(self.nodes()) - 2) + 2:
            return True
        u, tmp = self.nodes()[0], UndirectedGraph.copy(self)
        can_end_in = tmp.neighboring(u).copy()
        return dfs(u)
        
    def hamiltonWalkExists(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes():
            raise Exception("Unrecognized node(s).")
        if v in self.neighboring(u):
            return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        return UndirectedGraph.copy(self).connect(u, v).hamiltonTourExists()
        
    def hamiltonTour(self):
        if any(self.degrees(n) < 2 for n in self.nodes()) or not self.connected():
            return []
        u = self.nodes()[0]
        for v in self.neighboring(u):
            res = self.hamiltonWalk(u, v)
            if res:
                return res
        return []
        
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            if not tmp.degrees(x) or v is not None and not tmp.degrees(v):
                return []
            too_many = v is not None
            for n in tmp.nodes():
                if n not in (x, v) and tmp.degrees(n) < 2:
                    if too_many:
                        return []
                    too_many = True
            if not tmp.nodes():
                return stack
            neighbors = tmp.neighboring(x).copy()
            tmp.remove(x)
            if v is None and len(tmp.nodes()) == 1 and tmp.nodes() == neighbors:
                tmp.add(x, neighbors[0])
                return stack + [neighbors[0]]
            for y in neighbors:
                if y == v:
                    if tmp.nodes() == [v]:
                        tmp.add(x, *neighbors)
                        return stack + [v]
                    continue
                res = dfs(y, stack + [y])
                if res:
                    tmp.add(x, *neighbors)
                    return res
            tmp.add(x, *neighbors)
            return []
        
        tmp = UndirectedGraph.copy(self)
        if u is None:
            if v is not None and v not in self.nodes():
                raise Exception("Unrecognized node.")
            for _u in sorted(self.nodes(), key=lambda _x: self.degrees(_x)):
                result = dfs(_u, [_u])
                if result:
                    return result
                if self.degrees(_u) == 1:
                    return []
            return []
        if u not in self.nodes() or v is not None and v not in self.nodes():
            raise Exception("Unrecognized node(s).")
        return dfs(u, [u])
        
    def isomorphicFunction(self, other):
        if isinstance(other, UndirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()):
                return SortedKeysDict()
            this_degrees, other_degrees = dict(), dict()
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
                return SortedKeysDict()
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes():
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            
            from itertools import permutations, product
            
            possibilities = product(*[list(permutations(this_nodes)) for this_nodes in this_nodes_degrees])
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility):
                    map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return SortedKeysDict(*map_dict, f=self.f())
            return SortedKeysDict()
        return SortedKeysDict()
        
    def __bool__(self):
        return bool(self.nodes())
        
    def __call__(self, x):
        return self.f(x)
        
    def __reversed__(self):
        return self.complementary()
        
    def __contains__(self, item):
        return item in self.nodes() or item in self.links()
        
    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class UndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes() + other.nodes()):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes():
            if n not in res.nodes():
                res.add(n)
        for l in other.links():
            if l not in res.links():
                res.connect(l.u(), l.v())
        return res
        
    def __eq__(self, other):
        if isinstance(other, UndirectedGraph):
            if len(self.links()) != len(other.links()) or self.nodes() != other.nodes():
                return False
            for l in self.links():
                if l not in other.links():
                    return False
            return True
        return False
        
    def __str__(self):
        return "({" + ", ".join(str(n) for n in self.nodes()) + "}, {" + ", ".join(str(l) for l in self.links()) + "})"
        
    def __repr__(self):
        return str(self)


class WeightedNodesUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: Dict = Dict(), f=lambda x: x):
        super().__init__(Dict(), f)
        self.__node_weights = SortedKeysDict(f=f)
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v in p[1]:
                if v in self.nodes():
                    self.connect(u, v)
                else:
                    self.add((v, 0), u)
                    
    def node_weights(self, u: Node = None):
        return self.__node_weights[u] if u is not None else self.__node_weights
        
    def total_nodes_weight(self):
        return sum(self.node_weights().values())
        
    def add(self, n_w: (Node, float), *current_nodes: Node):
        if n_w[0] not in self.nodes():
            self.__node_weights[n_w[0]] = n_w[1]
        return super().add(n_w[0], *current_nodes)
        
    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            self.__node_weights.pop(u)
        return super().remove(n, *rest)
        
    def set_weight(self, u: Node, w: float):
        if u in self.nodes():
            self.__node_weights[u] = w
        return self
        
    def copy(self):
        return WeightedNodesUndirectedGraph(Dict(*[(n, (self.node_weights(n), self.neighboring(n))) for n in self.nodes()]), self.f())
        
    def component(self, u: Node):
        if u not in self.nodes():
            raise ValueError("Unrecognized node!")
        res = WeightedNodesUndirectedGraph(Dict((u, (self.node_weights(u), []))), self.f())
        queue = [u]
        while queue:
            v = queue.pop(0)
            for n in self.neighboring(v):
                if n in res.nodes():
                    res.connect(v, n)
                else:
                    res.add((n, self.node_weights(n)), v), queue.append(n)
        return res
        
    def minimalPathNodes(self, u: Node, v: Node):
        from Graphs.DirectedGraph import WeightedLinksDirectedGraph
        
        res = WeightedLinksDirectedGraph(Dict(*[(n, (Dict(), Dict(*[(m, self.node_weights(n)) for m in self.neighboring(n)]))) for n in self.nodes()]), self.f()).minimalPathLinks(u, v)
        return res[0], res[1] + self.node_weights(v) * bool(res[0])
        
    def weightedVertexCover(self):
        nodes, weights = self.nodes().copy(), self.total_nodes_weight()
        tmp = WeightedNodesUndirectedGraph.copy(self)
        
        def helper(curr, res_sum=0, i=0):
            if not tmp.links():
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
            
        for n in self.nodes():
            if not self.degrees(n):
                nodes.remove(n)
                weights -= self.node_weights(n)
        return helper([])[0]
        
    def weightedDominatingSet(self):
        nodes = self.nodes().copy()
        
        def helper(curr, total, total_weight, i=0):
            if total == self.nodes():
                return [curr.copy()], total_weight
            result, result_sum = [self.nodes()], self.total_nodes_weight()
            for j, u in enumerate(nodes[i:]):
                new = SortedList(f=self.f())
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
            
        isolated, weights = SortedList(f=self.f()), 0
        for n in nodes:
            if not self.degrees(n):
                isolated.insert(n), nodes.remove(n)
                weights += self.node_weights(n)
        return helper(isolated.value(), isolated, weights)[0]
        
    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedNodesUndirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()):
                return SortedKeysDict()
            this_degrees, other_degrees, this_weights, other_weights = dict(), dict(), dict(), dict()
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
                return SortedKeysDict()
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes():
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            
            from itertools import permutations, product
            
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility):
                    map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return SortedKeysDict(*map_dict, f=self.f())
            return SortedKeysDict()
        return super().isomorphicFunction(other)
        
    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedNOdesUndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes() + other.nodes()):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, WeightedUndirectedGraph):
            return other + self
        res = self.copy()
        if isinstance(other, WeightedNodesUndirectedGraph):
            for n in other.nodes():
                if n in res.nodes():
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links():
                res.connect(l.u(), l.v())
            return res
        if isinstance(other, WeightedLinksUndirectedGraph):
            return WeightedUndirectedGraph(f=self.f()) + self + other
        for n in other.nodes():
            if n not in res.nodes():
                res.add((n, 0))
        for l in other.links():
            res.connect(l.u(), l.v())
        return res
        
    def __eq__(self, other):
        if isinstance(other, WeightedNodesUndirectedGraph):
            if self.node_weights() != other.node_weights() or len(self.links()) != len(other.links()):
                return False
            for l in self.links():
                if l not in other.links():
                    return False
            return True
        return False
        
    def __str__(self):
        return "({" + ", ".join(f"{str(n)} -> {self.node_weights(n)}" for n in self.nodes()) + "}, {" + ", ".join(str(l) for l in self.links()) + "})"


class WeightedLinksUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: Dict = Dict(), f=lambda x: x):
        super().__init__(Dict(), f)
        self.__link_weights = Dict()
        for u, neighbors in neighborhood.items():
            if u not in self.nodes():
                self.add(u)
            for v, w in neighbors.items():
                if v not in self.nodes():
                    self.add(v, Dict((u, w)))
                elif v not in self.neighboring(u):
                    self.connect(u, Dict((v, w)))
                    
    def link_weights(self, u_or_l: Node | Link = None, v: Node = None):
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return SortedKeysDict(*[(n, self.__link_weights[Link(n, u_or_l)]) for n in self.neighboring(u_or_l)], f=self.f())
            return self.__link_weights[Link(u_or_l, v)]
        elif isinstance(u_or_l, Link):
            return self.__link_weights[u_or_l]
            
    def total_links_weight(self):
        return sum(self.link_weights().values())
        
    def add(self, u: Node, nodes_weights: Dict = Dict()):
        if u not in self.nodes():
            for w in nodes_weights.values():
                if not isinstance(w, (int, float)):
                    raise TypeError("Real numerical values expected!")
            super().add(u, *nodes_weights.keys())
            for v, w in nodes_weights.items():
                self.__link_weights[Link(u, v)] = w
        return self
        
    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            for v in self.neighboring(u):
                self.__link_weights.pop(Link(u, v))
        return super().remove(n, *rest)
        
    def connect(self, u: Node, nodes_weights: Dict = Dict()):
        if u in self.nodes():
            for v, w in nodes_weights.items():
                if v not in self.neighboring(u):
                    self.__link_weights[Link(u, v)] = w
        return super().connect(u, *nodes_weights.keys()) if nodes_weights else self
        
    def disconnect(self, u: Node, v: Node, *rest: Node):
        super().disconnect(u, v, *rest)
        for n in (v,) + rest:
            if Link(u, n) in [l for l in self.link_weights().keys()]:
                self.__link_weights.pop(Link(u, n))
        return self
        
    def set_weight(self, l: Link, w: float):
        if l in self.links():
            self.__link_weights[l] = w
        return self
        
    def copy(self):
        return WeightedLinksUndirectedGraph(Dict(*[(n, self.link_weights(n)) for n in self.nodes()]), self.f())
        
    def component(self, u: Node):
        if u not in self.nodes():
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksUndirectedGraph(Dict((u, Dict())), self.f())
        while queue:
            v = queue.pop(0)
            for n in self.neighboring(v):
                if n in res.nodes():
                    res.connect(v, Dict((n, self.link_weights(v, n))))
                else:
                    res.add(n, Dict((v, self.link_weights(v, n)))), queue.append(n)
        return res
        
    def minimal_spanning_tree(self):
        if self.tree():
            return self.links(), self.total_links_weight()
        if not self.connected():
            res = []
            for comp in self.connection_components():
                curr = self.component(comp[0])
                res.append(curr.minimal_spanning_tree())
            return res
        if not self.links():
            return [], 0
        res_links, total = [], SortedList(self.nodes()[0], f=self.f())
        bridge_links = SortedList(*[Link(self.nodes()[0], u) for u in self.neighboring(self.nodes()[0])], f=lambda x: self.link_weights(x))
        for _ in range(1, len(self.nodes())):
            u, v = bridge_links[0].u(), bridge_links[0].v()
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
        def dfs(x, curr_path, curr_w, total_negative, res_path, res_w=0):
            for y in filter(lambda _y: Link(x, _y) not in curr_path, self.neighboring(x)):
                if curr_w + self.link_weights(Link(y, x)) + total_negative >= res_w and res_path:
                    continue
                if y == v and (curr_w + self.link_weights(Link(x, y)) < res_w or not res_path):
                    res_path, res_w = curr_path + [Link(x, y)], curr_w + self.link_weights(Link(x, y))
                curr = dfs(y, curr_path + [Link(x, y)], curr_w + self.link_weights(Link(x, y)), total_negative - self.link_weights(Link(x, y)) * (self.link_weights(Link(x, y)) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w
            
        if u not in self.nodes() or v not in self.nodes():
            raise ValueError("Unrecognized node(s)!")
        if self.reachable(u, v):
            result = dfs(u, [], 0, sum(self.link_weights(l) for l in self.links() if self.link_weights(l) < 0), [])
            return list(map(lambda l: l.u(), result[0])) + [result[0][-1].v()], result[1]
        return [], 0
        
    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedLinksUndirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()):
                return SortedKeysDict()
            this_degrees, other_degrees, this_weights, other_weights = dict(), dict(), dict(), dict()
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
                return SortedKeysDict()
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes():
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            
            from itertools import permutations, product
            
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility): map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return SortedKeysDict(*map_dict, f=self.f())
            return SortedKeysDict()
        return super().isomorphicFunction(other)
        
    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedLinksUndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes() + other.nodes()):
            raise ValueError("Node sorting functions don't match!")
        if isinstance(other, (WeightedUndirectedGraph, WeightedNodesUndirectedGraph)):
            return other + self
        res = self.copy()
        if isinstance(other, WeightedLinksUndirectedGraph):
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n)
            for l in other.links():
                if l in res.links():
                    res.set_weight(l, res.link_weights(l) + other.link_weights(l))
                else:
                    res.connect(l.u(), Dict((l.v(), other.link_weights(l))))
            return res
        for n in other.nodes():
            if n not in res.nodes():
                res.add(n)
        for l in other.links():
            if l not in res.links():
                res.connect(l.u(), Dict((l.v(), 0)))
        return res
        
    def __eq__(self, other):
        if isinstance(other, WeightedLinksUndirectedGraph):
            return self.nodes() == other.nodes() and self.link_weights() == other.link_weights()
        return False
        
    def __str__(self):
        return "({" + ", ".join(str(n) for n in self.nodes()) + "}, " + str(self.link_weights()) + ")"


class WeightedUndirectedGraph(WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph):
    def __init__(self, neighborhood: Dict = Dict(), f=lambda x: x):
        WeightedNodesUndirectedGraph.__init__(self, Dict(), f), WeightedLinksUndirectedGraph.__init__(self, Dict(), f)
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v, w in p[1].items():
                if v in self.nodes():
                    self.connect(u, Dict((v, w)))
                else:
                    self.add((v, 0), Dict((u, w)))
                    
    def total_weight(self):
        return self.total_nodes_weight() + self.total_links_weight()
        
    def add(self, n_w: (Node, float), nodes_weights: Dict = Dict()):
        WeightedLinksUndirectedGraph.add(self, n_w[0], nodes_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self
        
    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest:
            tmp = self.neighboring(u)
            if tmp:
                WeightedLinksUndirectedGraph.disconnect(self, u, *tmp)
            super().remove(n)
        return self
        
    def connect(self, u: Node, nodes_weights: Dict = Dict()):
        return WeightedLinksUndirectedGraph.connect(self, u, nodes_weights)
        
    def disconnect(self, u: Node, v: Node, *rest: Node):
        return WeightedLinksUndirectedGraph.disconnect(self, u, v, *rest)
        
    def set_weight(self, el: Node | Link, w: float):
        if el in self.nodes():
            WeightedNodesUndirectedGraph.set_weight(self, el, w)
        elif el in self.links():
            WeightedLinksUndirectedGraph.set_weight(self, el, w)
        return self
        
    def copy(self):
        return WeightedUndirectedGraph(Dict(*[(n, (self.node_weights(n), self.link_weights(n))) for n in self.nodes()]), self.f())
        
    def component(self, u: Node):
        if u not in self.nodes():
            raise ValueError("Unrecognized node!")
        res = WeightedUndirectedGraph(Dict((u, (self.node_weights(u), Dict()))), self.f())
        queue = [u]
        while queue:
            v = queue.pop(0)
            for n in self.neighboring(v):
                if n in res.nodes():
                    res.connect(v, Dict((n, self.link_weights(v, n))))
                else:
                    res.add((n, self.node_weights(n)), Dict((v, self.link_weights(v, n)))), queue.append(n)
        return res
        
    def minimalPath(self, u: Node, v: Node):
        from Graphs.DirectedGraph import WeightedLinksDirectedGraph
        res = WeightedLinksDirectedGraph(Dict(*[(n, (Dict(), Dict(*[(m, self.node_weights(n) + self.link_weights(n, m)) for m in self.neighboring(n)]))) for n in self.nodes()]), self.f()).minimalPathLinks(u, v)
        return res[0], res[1] + self.node_weights(v) * bool(res[0])
        
    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedUndirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()):
                return SortedKeysDict()
            this_degrees, other_degrees, this_node_weights, other_node_weights, this_link_weights, other_link_weights = dict(), dict(), dict(), dict(), dict(), dict()
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
                return SortedKeysDict()
            this_nodes_degrees = {d: [] for d in this_degrees.keys()}
            other_nodes_degrees = {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes():
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            
            from itertools import permutations, product
            
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility):
                    map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return SortedKeysDict(*map_dict, f=self.f())
            return SortedKeysDict()
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return type(other).isomorphicFunction(other, self)
        return UndirectedGraph.isomorphicFunction(self, other)
        
    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedUndirectedGraph and type {type(other).__name__}!")
        if any(self(x) != other(x) for x in self.nodes() + other.nodes()):
            raise ValueError("Node sorting functions don't match!")
        res = self.copy()
        if isinstance(other, WeightedUndirectedGraph):
            for n in other.nodes():
                if n in res.nodes():
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u in other.nodes():
                for v in other.neighboring(u):
                    if v in res.neighboring(u):
                        res.set_weight(Link(u, v), res.link_weights(u, v) + other.link_weights(u, v))
                    else:
                        res.connect(u, Dict((v, other.link_weights(u, v))))
        elif isinstance(other, WeightedNodesUndirectedGraph):
            for n in other.nodes():
                if n in res.nodes():
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u in other.nodes():
                for v in other.neighboring(u):
                    if v not in res.neighboring(u):
                        res.connect(u, Dict((v, 0)))
        elif isinstance(other, WeightedLinksUndirectedGraph):
            for n in other.nodes():
                if n not in res.nodes():
                    res.add((n, 0))
            for u in other.nodes():
                for v in other.neighboring(u):
                    if v in res.neighboring(u):
                        res.set_weight(Link(u, v), res.link_weights(u, v) + other.link_weights(u, v))
                    else:
                        res.connect(u, Dict((v, other.link_weights(u, v))))
        for n in other.nodes():
            if n not in res.nodes():
                res.add((n, 0))
        for u in other.nodes():
            for v in other.neighboring(u):
                if v not in res.neighboring(u):
                    res.connect(u, Dict((v, 0)))
        return res
        
    def __eq__(self, other):
        if isinstance(other, WeightedUndirectedGraph):
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False
        
    def __str__(self):
        return f"({self.node_weights()}, {self.link_weights()})"
