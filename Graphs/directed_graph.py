from Graphs.undirected_graph import *


class DirectedGraph(Graph):
    def __init__(self, neighborhood: dict[Node, tuple[Iterable[Node], Iterable[Node]]] = {}):
        self.__nodes, self.__links = set(), set()
        self.__prev, self.__next, self.__degrees = {}, {}, {}
        for u, (prev_nodes, next_nodes) in neighborhood.items():
            self.add(u)
            for v in prev_nodes:
                self.add(v, points_to=[u]), self.connect(u, [v])
            for v in next_nodes:
                self.add(v, [u]), self.connect(v, [u])

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes.copy()

    @property
    def links(self) -> set[tuple[Node, Node]]:
        return self.__links.copy()

    def degrees(self, u: Node = None) -> dict[Node, list[int, int]] | list[int, int]:
        return (self.__degrees if u is None else self.__degrees[u]).copy()

    def next(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        return (self.__next if u is None else self.__next[u]).copy()

    def prev(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        return (self.__prev if u is None else self.__prev[u]).copy()

    @property
    def sources(self) -> set[Node]:
        return {u for u in self.nodes if self.source(u)}

    @property
    def sinks(self) -> set[Node]:
        return {v for v in self.nodes if self.sink(v)}

    def source(self, n: Node) -> bool:
        return not self.degrees(n)[0]

    def sink(self, n: Node) -> bool:
        return not self.degrees(n)[1]

    def add(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "DirectedGraph":
        if u not in self:
            self.__nodes.add(u)
            self.__degrees[u], self.__next[u], self.__prev[u] = [0, 0], set(), set()
            DirectedGraph.connect(self, u, pointed_by, points_to)
        return self

    def remove(self, u: Node, *rest: Node) -> "DirectedGraph":
        for n in (u,) + rest:
            if n in self:
                DirectedGraph.disconnect(self, n, self.prev(n), self.next(n))
                self.__nodes.remove(n), self.__degrees.pop(n), self.__prev.pop(n), self.__next.pop(n)
        return self

    def connect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "DirectedGraph":
        if u in self:
            for v in pointed_by:
                if u != v and v not in self.prev(u) and v in self:
                    self.__links.add((v, u)), self.__prev[u].add(v), self.__next[v].add(u)
                    self.__degrees[u][0] += 1
                    self.__degrees[v][1] += 1
            for v in points_to:
                if u != v and v not in self.next(u) and v in self:
                    self.__links.add((u, v)), self.__prev[v].add(u), self.__next[u].add(v)
                    self.__degrees[u][1] += 1
                    self.__degrees[v][0] += 1
        return self

    def connect_all(self, u: Node, *rest: Node) -> "DirectedGraph":
        if not rest:
            return self
        self.connect(u, rest, rest)
        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "DirectedGraph":
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

    def disconnect_all(self, u: Node, *rest: Node) -> "DirectedGraph":
        if not rest:
            return self
        self.disconnect(u, rest, rest)
        return self.disconnect_all(*rest)

    def copy(self) -> "DirectedGraph":
        return DirectedGraph({n: ([], self.next(n)) for n in self.nodes})

    def complementary(self) -> "DirectedGraph":
        res = DirectedGraph({n: ([], self.nodes) for n in self.nodes})
        for l in self.links:
            res.disconnect(l[0], [l[1]])
        return res

    def transposed(self) -> "DirectedGraph":
        return DirectedGraph({u: (self.next(u), []) for u in self.nodes})

    def undirected(self) -> UndirectedGraph:
        res = UndirectedGraph()
        for u in self.nodes:
            if u not in res:
                res.add(u)
            for v in self.next(u):
                if v not in res:
                    res.add(v)
                if v not in res.neighboring(u):
                    res.connect(u, v)
            for v in self.prev(u):
                if v not in res:
                    res.add(v)
                if v not in res.neighboring(u):
                    res.connect(u, v)
        return res

    def connection_components(self) -> list["DirectedGraph"]:
        components, rest = [], self.nodes
        while rest:
            components.append(curr := self.component(rest.copy().pop()))
            rest -= curr.nodes
        return components

    def connected(self) -> bool:
        if (m := len(self.links)) + 1 < (n := len(self.nodes)):
            return False
        if m > (n - 1) * (n - 2) or n < 2:
            return True
        queue, total = [u := self.nodes.pop()], {u}
        while queue:
            for v in filter(lambda x: x not in total, self.next(u := queue.pop(0)).union(self.prev(u))):
                total.add(v), queue.append(v)
        return len(total) == len(self.nodes)

    def reachable(self, u: Node, v: Node) -> bool:
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if v in {u, *self.next(u), *self.prev(u)}:
            return True
        return v in self.subgraph(u)

    def component(self, u: Node) -> "DirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], DirectedGraph({u: ([], [])})
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

    def full(self) -> bool:
        return len(self.links) == (n := len(self.nodes)) * (n - 1)

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "DirectedGraph":
        if isinstance(u_or_nodes, Node):
            if u_or_nodes not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_nodes], DirectedGraph({u_or_nodes: ([], [])})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, [v])
                    else:
                        res.add(n, [v]), queue.append(n)
            return res
        return DirectedGraph({u: ([], self.next(u).intersection(u_or_nodes)) for u in self.nodes.intersection(u_or_nodes)})

    def has_loop(self) -> bool:
        def dfs(u):
            for v in self.next(u):
                if v in total:
                    continue
                if v in stack:
                    return True
                stack.add(v)
                if dfs(v):
                    return True
                stack.remove(v)
            total.add(u)
            return False

        sources, total, stack = self.sources, set(), set()
        if not sources or not self.sinks:
            return True
        for n in sources:
            stack.add(n)
            if dfs(n):
                return True
            stack.remove(n)
        return total != self.nodes

    def dag(self) -> bool:
        return not self.has_loop()

    def toposort(self) -> list[Node]:
        if not self.dag():
            return []
        layer, total = self.sources, set()
        res = list(layer)
        while layer:
            new = set()
            for u in layer:
                total.add(u), new.update(self.next(u))
            for u in new.copy():
                if any(v not in total for v in self.prev(u)):
                    new.remove(u)
            res, layer = res + list(new), new.copy()
        return res

    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        previous, queue, total = {}, [u], {u}
        while queue:
            if (n := queue.pop(0)) == v:
                result, curr_node = [n], n
                while curr_node != u:
                    result.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]
                return result
            for y in filter(lambda _x: _x not in total, self.next(n)):
                queue.append(y), total.add(y)
                previous[y] = n

    def euler_tour_exists(self) -> bool:
        for d in self.degrees().values():
            if d[0] != d[1]:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if self.euler_tour_exists():
            return u == v
        for n in self.nodes:
            if self.degrees(n)[1] + (n == v) != self.degrees(n)[0] + (n == u):
                return False
        return self.connected()

    def euler_tour(self) -> list[Node]:
        if self.euler_tour_exists():
            tmp = DirectedGraph.copy(self)
            return tmp.disconnect(u := (l := tmp.links.pop())[1], [v := l[0]]).euler_walk(u, v)
        return []

    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = DirectedGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i + 1], [path[i]])
            for i, u in enumerate(path):
                while tmp.next(u):
                    curr = tmp.disconnect(v := tmp.next(u).pop(), [u]).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j + 1], [curr[j]])
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def strongly_connected_component(self, n: Node) -> set[Node]:
        def helper(x):
            def bfs(s):
                previous, queue, so_far = {}, [s], {s}
                while queue:
                    for t in filter(lambda _t: _t not in so_far, self.next(_s := queue.pop(0))):
                        previous[t] = _s
                        if t in path:
                            node = t
                            while node != s:
                                path.append(previous[node])
                                node = previous[node]
                            return
                        queue.append(t), so_far.add(t)

            for y in filter(lambda _y: _y not in total, self.prev(x)):
                if path := self.get_shortest_path(x, y):
                    for u in path:
                        res.add(u), total.add(u)
                        for v in self.next(u):
                            if v not in total and v not in path:
                                bfs(v)
                    return
            res.add(x)

        res, total = set(), {n}
        helper(n)
        return res

    def strongly_connected_components(self) -> list[set[Node]]:
        if self.dag():
            return list(map(lambda x: {x}, self.nodes))
        if not self.connected():
            return sum(map(lambda x: x.strongly_connected_components(), self.connection_components()), [])
        if not self.sources and not self.sinks:
            return [self.nodes]
        rest, res = self.nodes, []
        while rest:
            res.append(curr := self.strongly_connected_component(rest.copy().pop()))
            rest -= curr
        return res

    def scc_dag(self) -> "DirectedGraph":
        result = DirectedGraph()
        scc = self.strongly_connected_components()
        for s in scc:
            result.add(Node(frozenset(s)))
        for u in result.nodes:
            for v in result.nodes:
                if u != v:
                    for x in u.value:
                        if any(y in v.value for y in self.next(x)):
                            result.connect(v, [u])
        return result

    def loop_with_length(self, length: int) -> list[Node]:
        if abs(length) < 2:
            return []
        tmp = DirectedGraph.copy(self)
        for l in tmp.links:
            u, v = l
            res = tmp.disconnect(v, [u]).path_with_length(v, u, length - 1)
            if res:
                return res
            tmp.connect(v, [u])
        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
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

    def hamilton_tour_exists(self) -> bool:
        def dfs(x):
            if tmp.nodes == {x}:
                return x in can_end_in
            if all(y not in tmp for y in can_end_in):
                return False
            tmp0, tmp1 = tmp.prev(x), tmp.next(x)
            tmp.remove(x)
            for y in tmp1:
                if dfs(y):
                    tmp.add(x, tmp0, tmp1)
                    return True
            tmp.add(x, tmp0, tmp1)
            return False

        if (n := len(self.nodes)) == 1 or len(self.links) > (n - 1) ** 2 or all(sum(self.degrees(u)) >= n for u in self.nodes):
            return True
        if self.sources or self.sinks:
            return False
        tmp = DirectedGraph.copy(self)
        can_end_in = tmp.prev(u := self.nodes.pop())
        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if u in self.next(v):
            return True if all(n in {u, v} for n in self.nodes) else self.hamilton_tour_exists()
        return DirectedGraph.copy(self).connect(u, [v]).hamilton_tour_exists()

    def hamilton_tour(self) -> list[Node]:
        if self.sources or self.sinks or not self:
            return []
        for v in self.prev(u := self.nodes.pop()):
            if result := self.hamilton_walk(u, v):
                return result
        return []

    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        def dfs(x, stack):
            too_many = v is not None
            for n in tmp.nodes:
                if not tmp.degrees(n)[0] and n != x:
                    return []
                if not tmp.degrees(n)[1] and n != v:
                    if too_many:
                        return []
                    too_many = True
            prev_x, next_x = tmp.prev(x), tmp.next(x)
            tmp.remove(x)
            if not tmp:
                tmp.add(x, prev_x, next_x)
                return stack
            for y in next_x:
                if y == v:
                    if tmp.nodes == {v}:
                        tmp.add(x, prev_x, next_x)
                        return stack + [v]
                    continue
                if res := dfs(y, stack + [y]):
                    tmp.add(x, prev_x, next_x)
                    return res
            tmp.add(x, prev_x, next_x)
            return []

        tmp = DirectedGraph.copy(self)
        if u is None:
            if v is not None and v not in self:
                raise Exception("Unrecognized node.")
            if self.dag() and (v is None or self.sink(v)):
                if any(self.degrees(n)[0] > 1 or self.degrees(n)[1] > 1 for n in self.nodes):
                    return []
                return self.toposort()
            for _u in self.nodes:
                if result := dfs(_u, [_u]):
                    return result
            return []
        if u not in self or v is not None and v not in self:
            raise Exception("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphic_bijection(self, other: "DirectedGraph") -> dict[Node, Node]:
        if isinstance(other, DirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = defaultdict(int), defaultdict(int)
            for d in map(tuple, self.degrees().values()):
                this_degrees[d] += 1
            for d in map(tuple, other.degrees().values()):
                other_degrees[d] += 1
            if this_degrees != other_degrees:
                return {}
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_degrees, [])
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

    def __reversed__(self) -> "DirectedGraph":
        return self.complementary()

    def __contains__(self, u: Node):
        return u in self.nodes

    def __add__(self, other: "DirectedGraph") -> "DirectedGraph":
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if isinstance(other, (WeightedNodesDirectedGraph, WeightedLinksDirectedGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            res.add(n)
        for u, v in other.links:
            res.connect(v, [u])
        return res

    def __eq__(self, other: "DirectedGraph"):
        if type(other) == DirectedGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self):
        return "<" + str(self.nodes) + ", {" + ", ".join(f"<{l[0]}, {l[1]}>" for l in self.links) + "}>"

    __repr__ = __str__


class WeightedNodesDirectedGraph(DirectedGraph):
    def __init__(self, neighborhood: dict[Node, tuple[float, tuple[Iterable[Node], Iterable[Node]]]] = {}):
        super().__init__()
        self.__node_weights = {}
        for n, (w, _) in neighborhood.items():
            self.add((n, w))
        for u, (_, (prev_u, next_u)) in neighborhood.items():
            for v in prev_u:
                self.add((v, 0), points_to=[u]), self.connect(u, [v])
            for v in next_u:
                self.add((v, 0), [u]), self.connect(v, [u])

    def node_weights(self, n: Node = None) -> dict[Node, float] | float:
        return self.__node_weights.copy() if n is None else self.__node_weights.get(n)

    @property
    def total_nodes_weight(self) -> float:
        return sum(self.node_weights().values())

    def add(self, n_w: tuple[Node, float], pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "WeightedNodesDirectedGraph":
        super().add(n_w[0], pointed_by, points_to)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedNodesDirectedGraph":
        for n in (u,) + rest:
            self.__node_weights.pop(n)
        DirectedGraph.remove(self, u, *rest)
        return self

    def set_weight(self, u: Node, w: float) -> "WeightedNodesDirectedGraph":
        if u in self:
            self.__node_weights[u] = w
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedNodesDirectedGraph":
        if u in self.node_weights:
            self.set_weight(u, self.node_weights(u) + w)
        return self

    def copy(self) -> "WeightedNodesDirectedGraph":
        return WeightedNodesDirectedGraph({n: (self.node_weights(n), ([], self.next(n))) for n in self.nodes})

    def complementary(self) -> "WeightedNodesDirectedGraph":
        res = WeightedNodesDirectedGraph({n: (self.node_weights(n), ([], self.nodes)) for n in self.nodes})
        for l in self.links:
            res.disconnect(l[0], [l[1]])
        return res

    def transposed(self) -> "WeightedNodesDirectedGraph":
        return WeightedNodesDirectedGraph({u: (self.node_weights(u), (self.next(u), [])) for u in self.nodes})

    def undirected(self) -> WeightedNodesUndirectedGraph:
        res = WeightedNodesUndirectedGraph()
        for u in self.nodes:
            if u not in res:
                res.add((u, self.node_weights(u)))
            for v in self.next(u):
                if v not in res:
                    res.add((v, self.node_weights(v)))
                if v not in res.neighboring(u):
                    res.connect(u, v)
            for v in self.prev(u):
                if v not in res:
                    res.add((v, self.node_weights(v)))
                if v not in res.neighboring(u):
                    res.connect(u, v)
        return res

    def component(self, u: Node) -> "WeightedNodesDirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedNodesDirectedGraph({u: (self.node_weights(u), ([], []))})
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

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "WeightedNodesDirectedGraph":
        if isinstance(u_or_nodes, Node):
            if u_or_nodes not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_nodes], WeightedNodesDirectedGraph({u_or_nodes: (self.node_weights(u_or_nodes), ([], []))})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, [v])
                    else:
                        res.add((n, self.node_weights(n)), [v]), queue.append(n)
            return res
        return WeightedNodesDirectedGraph({u: (self.node_weights(u), ([], self.next(u).intersection(u_or_nodes))) for u in self.nodes.intersection(u_or_nodes)})

    def minimal_path_nodes(self, u: Node, v: Node) -> list[Node]:
        neighborhood = {n: (self.node_weights(n), ({}, {m: 0 for m in self.next(n)})) for n in self.nodes}
        return WeightedDirectedGraph(neighborhood).minimal_path(u, v)

    def isomorphic_bijection(self, other: DirectedGraph) -> dict[Node, Node]:
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_weights, other_weights = defaultdict(int), defaultdict(int)
            for w in self.node_weights().values():
                this_weights[w] += 1
            for w in other.node_weights().values():
                other_weights[w] += 1
            if this_weights != other_weights:
                return {}
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_degrees, [])
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
        return super().isomorphic_bijection(other)

    def __add__(self, other: DirectedGraph) -> "WeightedNodesDirectedGraph":
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedDirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksDirectedGraph):
            return WeightedDirectedGraph() + self + other
        if isinstance(other, WeightedNodesDirectedGraph):
            res = self.copy()
            for n in other.nodes:
                if n in res:
                    res.increase_weight(n, other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u, v in other.links:
                res.connect(v, [u])
            return res
        return self + WeightedNodesDirectedGraph({n: (0, ([], other.next(n))) for n in other.nodes})

    def __eq__(self, other: "WeightedNodesDirectedGraph"):
        if type(other) == WeightedNodesDirectedGraph:
            return (self.node_weights, self.links) == (other.node_weights, other.links)
        return False

    def __str__(self):
        return "<{" + ", ".join(f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, {" + ", ".join(f"<{l[0]}, {l[1]}>" for l in self.links) + "}>"


class WeightedLinksDirectedGraph(DirectedGraph):
    def __init__(self, neighborhood: dict[Node, tuple[dict[Node, float], dict[Node, float]]] = {}):
        super().__init__()
        self.__link_weights = {}
        for u, (prev_pairs, next_pairs) in neighborhood.items():
            if u not in self:
                self.add(u)
            for v, w in prev_pairs.items():
                self.add(v, points_to_weights={u: w}), self.connect(u, {v: w})
            for v, w in next_pairs.items():
                self.add(v, {u: w}), self.connect(v, {u: w})

    def link_weights(self, u_or_l: Node | tuple = None, v: Node = None) -> dict[Node, float] | dict[tuple[Node, Node], float] | float:
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return {n: self.__link_weights[(u_or_l, n)] for n in self.next(u_or_l)}
            return self.__link_weights.get((u_or_l, v))
        elif isinstance(u_or_l, tuple):
            return self.__link_weights.get(u_or_l)

    @property
    def total_links_weight(self) -> float:
        return sum(self.link_weights().values())

    def add(self, u: Node, pointed_by_weights: dict[Node, float] = {}, points_to_weights: dict[Node, float] = {}) -> "WeightedLinksDirectedGraph":
        if u not in self:
            super().add(u), self.connect(u, pointed_by_weights, points_to_weights)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedLinksDirectedGraph":
        for u in (n,) + rest:
            for v in self.next(u):
                self.__link_weights.pop((u, v))
            for v in self.prev(u):
                self.__link_weights.pop((v, u))
        return super().remove(n, *rest)

    def connect(self, u: Node, pointed_by_weights: dict[Node, float] = {}, points_to_weights: dict[Node, float] = {}) -> "WeightedLinksDirectedGraph":
        if u in self:
            super().connect(u, pointed_by_weights.keys(), points_to_weights.keys())
            for v, w in pointed_by_weights.items():
                if (v, u) not in self.link_weights():
                    self.set_weight((v, u), w)
            for v, w in points_to_weights.items():
                if (u, v) not in self.link_weights():
                    self.set_weight((u, v), w)
        return self

    def connect_all(self, u: Node, *rest: Node) -> "WeightedLinksDirectedGraph":
        if not rest:
            return self
        self.connect(u, (d := {v: 0 for v in rest}), d)
        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "WeightedLinksDirectedGraph":
        if u in self:
            for v in pointed_by:
                self.__link_weights.pop((v, u))
            for v in points_to:
                self.__link_weights.pop((u, v))
            super().disconnect(u, pointed_by, points_to)
        return self

    def set_weight(self, l: tuple, w: float) -> "WeightedLinksDirectedGraph":
        if l in self.links:
            self.__link_weights[l] = w
        return self

    def increase_weight(self, l: tuple[Node, Node], w: float) -> "WeightedLinksDirectedGraph":
        if l in self.link_weights:
            self.set_weight(l, self.link_weights(l) + w)
        return self

    def copy(self) -> "WeightedLinksDirectedGraph":
        return WeightedLinksDirectedGraph({u: ({}, self.link_weights(u)) for u in self.nodes})

    def transposed(self) -> "WeightedLinksDirectedGraph":
        return WeightedLinksDirectedGraph({u: (self.link_weights(u), {}) for u in self.nodes})

    def undirected(self) -> WeightedLinksUndirectedGraph:
        res = WeightedLinksUndirectedGraph()
        for u in self.nodes:
            if u not in res:
                res.add(u)
            for v in self.next(u):
                if v not in res:
                    res.add(v)
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + self.link_weights(u, v))
                else:
                    res.connect(u, {v: self.link_weights(u, v)})
            for v in self.prev(u):
                if v not in res:
                    res.add(v)
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + self.link_weights(v, u))
                else:
                    res.connect(u, {v: self.link_weights(v, u)})
        return res

    def component(self, u: Node) -> "WeightedLinksDirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksDirectedGraph({u: ({}, {})})
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

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "WeightedLinksDirectedGraph":
        if isinstance(u_or_nodes, Node):
            if u_or_nodes not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_nodes], WeightedLinksDirectedGraph({u_or_nodes: ({}, {})})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, {v: self.link_weights(v, n)})
                    else:
                        res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
            return res
        return WeightedLinksDirectedGraph({u: ({}, {k: v for k, v in self.link_weights(u).items() if k in u_or_nodes}) for u in self.nodes.intersection(u_or_nodes)})

    def minimal_path_links(self, u: Node, v: Node) -> list[Node]:
        return WeightedDirectedGraph({n: (0, ({}, self.link_weights(n))) for n in self.nodes}).minimal_path(u, v)

    def isomorphic_bijection(self, other: DirectedGraph) -> dict[Node, Node]:
        if isinstance(other, WeightedLinksDirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_weights, other_weights = defaultdict(int), defaultdict(int)
            for w in self.link_weights().values():
                this_weights[w] += 1
            for w in other.link_weights().values():
                other_weights[w] += 1
            if this_weights != other_weights:
                return {}
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_degrees, [])
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
        return super().isomorphic_bijection(other)

    def __add__(self, other: DirectedGraph) -> "WeightedLinksDirectedGraph":
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedNodesDirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksDirectedGraph):
            res = self.copy()
            for n in other.nodes:
                res.add(n)
            for u, v in other.links:
                if v in res.next(u):
                    res.increase_weight((u, v), other.link_weights(u, v))
                else:
                    res.connect(v, {u: other.link_weights((u, v))})
            return res
        return self + WeightedLinksDirectedGraph({u: ({}, {v: 0 for v in other.next(u)}) for u in other.nodes})

    def __eq__(self, other: "WeightedLinksDirectedGraph"):
        if type(other) == WeightedLinksDirectedGraph:
            return (self.nodes, self.link_weights()) == (other.nodes, other.link_weights())
        return False

    def __str__(self):
        return "<" + str(self.nodes) + ", {" + ", ".join(f"<{l[0]}, {l[1]}> -> {self.link_weights(l)}" for l in self.links) + "}>"


class WeightedDirectedGraph(WeightedNodesDirectedGraph, WeightedLinksDirectedGraph):
    def __init__(self, neighborhood: dict[Node, tuple[float, tuple[dict[Node, float], dict[Node, float]]]] = {}):
        WeightedNodesDirectedGraph.__init__(self)
        WeightedLinksDirectedGraph.__init__(self)
        for n, (w, _) in neighborhood.items():
            self.add((n, w))
        for u, (_, (prev_u, next_u)) in neighborhood.items():
            for v, w in prev_u.items():
                self.add((v, 0), points_to_weights={u: w}), self.connect(u, {v: w})
            for v, w in next_u.items():
                self.add((v, 0), {u: w}), self.connect(v, {u: w})

    @property
    def total_weight(self) -> float:
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: tuple[Node, float], pointed_by_weights: dict[Node, float] = {}, points_to_weights: dict[Node, float] = {}) -> "WeightedDirectedGraph":
        WeightedLinksDirectedGraph.add(self, n_w[0], pointed_by_weights, points_to_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedDirectedGraph":
        for n in (u,) + rest:
            WeightedLinksDirectedGraph.disconnect(self, n, self.prev(n), self.next(n)), super().remove(n)
        return self

    def connect(self, u: Node, pointed_by_weights: dict[Node, float] = {}, points_to_weights: dict[Node, float] = {}) -> "WeightedDirectedGraph":
        return WeightedLinksDirectedGraph.connect(self, u, pointed_by_weights, points_to_weights)

    def disconnect(self, u: Node, pointed_by: list[Node] = (), points_to: list[Node] = ()) -> "WeightedDirectedGraph":
        return WeightedLinksDirectedGraph.disconnect(self, u, pointed_by, points_to)

    def set_weight(self, el: Node | tuple, w: float) -> "WeightedDirectedGraph":
        if el in self:
            super().set_weight(el, w)
        elif el in self.links:
            WeightedLinksDirectedGraph.set_weight(self, el, w)
        return self

    def increase_weight(self, el: Node | tuple[Node, Node], w: float) -> "WeightedDirectedGraph":
        if el in self.node_weights:
            return self.set_weight(el, self.node_weights(el) + w)
        if el in self.link_weights:
            self.set_weight(el, self.link_weights(el) + w)
        return self

    def copy(self) -> "WeightedDirectedGraph":
        neighborhood = {u: (self.node_weights(u), ({}, self.link_weights(u))) for u in self.nodes}
        return WeightedDirectedGraph(neighborhood)

    def transposed(self) -> "WeightedDirectedGraph":
        neighborhood = {u: (self.node_weights(u), (self.link_weights(u), {})) for u in self.nodes}
        return WeightedDirectedGraph(neighborhood)

    def undirected(self) -> WeightedUndirectedGraph:
        res = WeightedUndirectedGraph()
        for u in self.nodes:
            if u not in res:
                res.add((u, self.node_weights(u)))
            for v in self.next(u):
                if v not in res:
                    res.add((v, self.node_weights(v)))
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + self.link_weights(u, v))
                else:
                    res.connect(u, {v: self.link_weights(u, v)})
            for v in self.prev(u):
                if v not in res:
                    res.add((v, self.node_weights(v)))
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + self.link_weights(v, u))
                else:
                    res.connect(u, {v: self.link_weights(v, u)})
        return res

    def component(self, u: Node) -> "WeightedDirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedDirectedGraph({u: (self.node_weights(u), ({}, {}))})
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

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "WeightedDirectedGraph":
        if isinstance(u_or_nodes, Node):
            if u_or_nodes not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_nodes], WeightedDirectedGraph({u_or_nodes: (self.node_weights(u_or_nodes), ({}, {}))})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, {v: self.link_weights(v, n)})
                    else:
                        res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), queue.append(n)
            return res
        return WeightedDirectedGraph({u: (self.node_weights(u), ({}, {k: v for k, v in self.link_weights(u).items() if k in u_or_nodes})) for u in self.nodes.intersection(u_or_nodes)})

    def minimal_path(self, u: Node, v: Node) -> list[Node]:
        def dfs(x, current_path, current_weight, total_negative, res_path=None, res_weight=0):
            def dijkstra(s, curr_path, curr_weight):
                curr_tmp = tmp.copy()
                for l in curr_path:
                    curr_tmp.disconnect(l[1], [l[0]])
                paths = {n: {m: [] for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                weights_from_to = {n: {m: curr_tmp.total_weight for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                for n in curr_tmp.nodes:
                    weights_from_to[n][n] = 0
                    for m in curr_tmp.next(n):
                        weights_from_to[n][m] = curr_tmp.link_weights(n, m) + curr_tmp.node_weights(m)
                        paths[n][m] = [(n, m)]
                for x1 in curr_tmp.nodes:
                    for x2 in curr_tmp.nodes:
                        for x3 in curr_tmp.nodes:
                            if (new_weight := weights_from_to[x1][x2] + weights_from_to[x2][x3]) < weights_from_to[x1][x3]:
                                weights_from_to[x1][x3] = new_weight
                                paths[x1][x3] = paths[x1][x2] + paths[x2][x3]
                return curr_path + paths[s][v], curr_weight + weights_from_to[s][v]

            if res_path is None:
                res_path = []
            if total_negative:
                for y in filter(lambda _y: (x, _y) not in current_path, tmp.next(x)):
                    new_curr_w = current_weight + (l_w := tmp.link_weights(x, y)) + (n_w := tmp.node_weights(y))
                    new_total_negative = total_negative
                    if n_w < 0:
                        new_total_negative -= n_w
                    if l_w < 0:
                        new_total_negative -= l_w
                    if new_curr_w + new_total_negative >= res_weight and res_path:
                        continue
                    if y == v and (new_curr_w < res_weight or not res_path):
                        res_path, res_weight = current_path + [(x, y)], current_weight + l_w + n_w
                    curr = dfs(y, current_path + [(x, y)], new_curr_w, new_total_negative, res_path, res_weight)
                    if curr[1] < res_weight or not res_path:
                        res_path, res_weight = curr
            else:
                curr = dijkstra(x, current_path, current_weight)
                if curr[1] < res_weight or not res_path:
                    res_path, res_weight = curr
            return res_path, res_weight

        if v in self:
            if v in (tmp := self.subgraph(u)):
                nodes_negative_weights = sum(tmp.node_weights(n) for n in tmp.nodes if tmp.node_weights(n) < 0)
                links_negative_weights = sum(tmp.link_weights(l) for l in tmp.links if tmp.link_weights(l) < 0)
                res = dfs(u, [], tmp.node_weights(u), nodes_negative_weights + links_negative_weights)
                return [l[0] for l in res[0]] + [res[0][-1][1]]
            return []
        raise ValueError('Unrecognized node(s)!')

    def isomorphic_bijection(self, other: DirectedGraph) -> dict[Node, Node]:
        if isinstance(other, WeightedDirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_node_weights, other_node_weights = defaultdict(int), defaultdict(int)
            this_link_weights, other_link_weights = defaultdict(int), defaultdict(int)
            for w in self.link_weights().values():
                this_link_weights[w] += 1
            for w in other.link_weights().values():
                other_link_weights[w] += 1
            for w in self.node_weights().values():
                this_node_weights[w] += 1
            for w in other.node_weights().values():
                other_node_weights[w] += 1
            if this_node_weights != other_node_weights or this_link_weights != other_link_weights:
                return {}
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[tuple(self.degrees(n))].append(n)
            for n in other.nodes:
                other_nodes_degrees[tuple(other.degrees(n))].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_degrees, [])
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
            return type(other).isomorphic_bijection(other, self)
        return DirectedGraph.isomorphic_bijection(self, other)

    def __add__(self, other: DirectedGraph) -> "WeightedDirectedGraph":
        if not isinstance(other, DirectedGraph):
            raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedDirectedGraph):
            res = self.copy()
            for n in other.nodes:
                if n in res:
                    res.increase_weight(n, other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for u, v in other.links:
                if v in res.next(u):
                    res.increase_weight((u, v), other.link_weights(u, v))
                else:
                    res.connect(v, {u: other.link_weights(u, v)})
            return res
        if isinstance(other, WeightedNodesDirectedGraph):
            neighborhood = {u: (other.node_weights(u), ({}, {v: 0 for v in other.next(u)})) for u in other.nodes}
            return self + WeightedDirectedGraph(neighborhood)
        if isinstance(other, WeightedLinksDirectedGraph):
            return self + WeightedDirectedGraph({u: (0, ({}, other.link_weights(u))) for u in other.nodes})
        return self + WeightedDirectedGraph({u: (0, ({}, {v: 0 for v in other.next(u)})) for u in other.nodes})

    def __eq__(self, other: "WeightedDirectedGraph"):
        if type(other) == WeightedDirectedGraph:
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False

    def __str__(self):
        return "<{" + ", ".join(f"{n} -> {self.node_weights(n)}" for n in self.nodes) + ", " + ", ".join(f"<{l[0]}, {l[1]}> -> {self.link_weights(l)}" for l in self.links) + "}>"
