"""
Module for implementing directed multi-graphs.
"""

from collections import defaultdict

from .directed_graph import Node, DirectedGraph, Iterable, product, permutations

from .undirected_multigraph import UndirectedMultiGraph


class DirectedMultiGraph:
    """
    Class for implementing an unweighted directed multi-graph.
    """

    def __init__(self, neighborhood: dict[Node, tuple[dict[Node, int], dict[Node, int]]] = {}) \
            -> None:
        self.__nodes, self.__links = set(), defaultdict(int)
        self.__prev, self.__next = {}, {}
        for u, (prev_nodes, next_nodes) in neighborhood.items():
            self.add(u)
            for v in prev_nodes:
                self.add(v)
            for v in next_nodes:
                self.add(v)
        for u, (prev_nodes, next_nodes) in neighborhood.items():
            self.connect(u, prev_nodes, next_nodes)

    @property
    def nodes(self) -> set[Node]:
        """
        Get nodes.
        """
        return self.__nodes.copy()

    @property
    def links(self) -> dict[tuple[Node, Node], int]:
        """
        Get links.
        """
        return self.__links.copy()

    def degrees(self, u: Node = None) -> dict[Node, tuple[int, int]] | tuple[int, int]:
        """
        Get in-degree and out-degree of a given node or the same for all nodes.
        """
        if u is None:
            return {n: self.degrees(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return sum(self.__prev[u].values()), sum(self.__next[u].values())

    def next(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None.
        Returns:
            A set of all nodes, that node u points to, if it's given,
            otherwise the same for all nodes.
        """
        if u is None:
            return self.__next.copy()
        if not isinstance(u, Node):
            u = Node(u)
        return self.__next[u].copy()

    def prev(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None.
        Returns:
            A set of all nodes, that point to node u, if it's given,
            otherwise the same for all nodes.
        """
        if u is None:
            return self.__prev.copy()
        if not isinstance(u, Node):
            u = Node(u)
        return self.__prev[u].copy()

    @property
    def sources(self) -> set[Node]:
        """
        Returns:
            All sources.
        """
        return {u for u in self.nodes if self.source(u)}

    @property
    def sinks(self) -> set[Node]:
        """
        Returns:
            All sinks.
        """
        return {v for v in self.nodes if self.sink(v)}

    def source(self, n: Node) -> bool:
        """
        Args:
            n: A present node.
        Returns:
            Whether node n is a source.
        """
        if not isinstance(n, Node):
            n = Node(n)
        return not self.degrees(n)[0]

    def sink(self, n: Node) -> bool:
        """
        Args:
            n: A present node.
        Returns:
            Whether node n is a sink.
        """
        if not isinstance(n, Node):
            n = Node(n)
        return not self.degrees(n)[1]

    def add(self, u: Node, pointed_by: dict[Node, int] = {}, points_to: dict[Node, int] = {}) \
            -> "DirectedMultiGraph":
        """
        Add a new node to the graph.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            self.__nodes.add(u)
            self.__next[u], self.__prev[u] = set(), set()
            DirectedMultiGraph.connect(self, u, pointed_by, points_to)
        return self

    def remove(self, u: Node, *rest: Node) -> "DirectedMultiGraph":
        for n in (u, *rest):
            if not isinstance(n, Node):
                n = Node(n)
            if n in self:
                DirectedMultiGraph.disconnect(self, n,
                                              {m: self.links[(m, n)] for m in self.prev(n)},
                                              {m: self.links[(n, m)] for m in self.next(n)})
                self.__nodes.remove(n), self.__prev.pop(n), self.__next.pop(n)
        return self

    def connect(self, u: Node, pointed_by: dict[Node, int], points_to: dict[Node, int]) \
            -> "DirectedMultiGraph":
        """
        Args:
            u: A present node.
            pointed_by: A set of present nodes.
            points_to: A set of present nodes.
        Connect node u such, that it's pointed by nodes, listed in pointed_by,
        and points to nodes, listed in points_to.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            for v, num in pointed_by.items():
                if not isinstance(v, Node):
                    v = Node(v)
                if u != v and v in self:
                    num = max(0, num)
                    self.__links[(v, u)] += num
                    self.__prev[u][v] += num
                    self.__next[v][u] += num
            for v, num in points_to.items():
                if not isinstance(v, Node):
                    v = Node(v)
                if u != v and v in self:
                    num = max(0, num)
                    self.__links[(u, v)] += num
                    self.__prev[v][u] += num
                    self.__next[u][v] += num
        return self

    def connect_all(self, u: Node, *rest: Node) -> "DirectedMultiGraph":
        if not rest:
            return self
        self.connect(u, {v: 1 for v in rest}, {v: 1 for v in rest})
        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: dict[Node, int] = {},
                   points_to: dict[Node, int] = {}) -> "DirectedMultiGraph":
        """
        Args:
            u: A present node.
            pointed_by: A set of present nodes.
            points_to: A set of present nodes.
        Remove all links from nodes in pointed_by to node u and from node u to nodes in points_to.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            for v, num in pointed_by.items():
                if not isinstance(v, Node):
                    v = Node(v)
                num = max(0, num)
                if v in self.prev(u):
                    self.__links[(v, u)] -= num
                    self.__next[v][u] -= num
                    self.__prev[u][v] -= num
                    if self.links[(v, u)] <= 0:
                        self.__prev[u].pop(v)
                        self.__next[v].pop(u)
                        self.__links.pop((v, u))
            for v, num in points_to.items():
                if not isinstance(v, Node):
                    v = Node(v)
                num = max(0, num)
                if v in self.next(u):
                    self.__links[(u, v)] -= num
                    self.__next[u][v] -= num
                    self.__prev[v][u] -= num
                    if self.links[(u, v)] <= 0:
                        self.__prev[v].pop(u)
                        self.__next[u].pop(v)
                        self.__links.pop((u, v))
        return self

    def disconnect_all(self, u: Node, *rest: Node) -> "DirectedMultiGraph":
        if not rest:
            return self
        self.disconnect(u, {v: self.links[(v, u)] for v in rest},
                        {v: self.links[(u, v)] for v in rest})
        return self.disconnect_all(*rest)

    def copy(self) -> "DirectedMultiGraph":
        return DirectedMultiGraph(
            {n: ({}, {m: self.links[(n, m)] for m in self.next(n)}) for n in self.nodes})

    def non_multi_graph(self) -> DirectedGraph:
        return DirectedGraph({u: ([], self.next(u)) for u in self.nodes})

    def complementary(self) -> DirectedGraph:
        return self.non_multi_graph().complementary()

    def transposed(self) -> "DirectedMultiGraph":
        """
        Returns:
            A graph, where each link points to the opposite direction.
        """
        return DirectedMultiGraph(
            {n: ({m: self.links[(n, m)] for m in self.next(n)}, {}) for n in self.nodes})

    def undirected(self) -> "UndirectedMultiGraph":
        """
        Returns:
            The undirected version of the graph.
        """
        neighborhood = {n: {m: self.links[(m, n)] + self.links[(n, m)] for m in
                            set(self.prev(n)).union(self.next(n))} for n in self.nodes}
        return UndirectedMultiGraph(neighborhood)

    def connection_components(self) -> list["DirectedMultiGraph"]:
        components, rest = [], self.nodes
        while rest:
            components.append(curr := self.component(rest.copy().pop()))
            rest -= curr.nodes
        return components

    def connected(self) -> bool:
        return DirectedMultiGraph.undirected(self).connected()

    def reachable(self, u: Node, v: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s).")
        if v in {u, *self.next(u)}:
            return True
        return v in self.subgraph(u)

    def component(self, u: Node) -> "DirectedMultiGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        queue, total = [u], {u}
        while queue:
            queue += list((next_nodes := self.prev(v := queue.pop(0)).union(self.next(v)) - total))
            total.update(next_nodes)
        return self.subgraph(total)

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "DirectedMultiGraph":
        try:
            u_or_nodes = self.nodes.intersection(u_or_nodes)
            return DirectedMultiGraph({u: ({}, {v: self.links[(u, v)] for v in
                                                self.next(u).intersection(u_or_nodes)})
                                       for u in u_or_nodes})
        except TypeError:
            if not isinstance(u_or_nodes, Node):
                u_or_nodes = Node(u_or_nodes)
            if u_or_nodes not in self:
                raise KeyError("Unrecognized node!")
            queue, res = [u_or_nodes], DirectedMultiGraph({u_or_nodes: ({}, {})})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, [v])
                    else:
                        res.add(n, [v]), queue.append(n)
            return res

    def has_cycle(self) -> bool:
        """
        Returns:
            Whether the graph has a cycle.
        """

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
        """
        Returns:
            Whether the graph is a DAG (directed acyclic graph).
        """
        return not self.has_cycle()

    def toposort(self) -> list[Node]:
        """
        Returns:
            A topological sort of the nodes if the graph is a DAG, otherwise an empty list.
        A topological sort has the following property: Let u and v be nodes in the graph and let u
        come before v. Then there's no path from v to u in the graph.
        (That's also the reason a graph with cycles has no topological sort.)
        """
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
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)!")
        previous, queue, total = {}, [u], {u}
        while queue:
            if (n := queue.pop(0)) == v:
                result, curr_node = [n], n
                while curr_node != u:
                    result.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]
                return result
            for y in self.next(n) - total:
                queue.append(y), total.add(y)
                previous[y] = n

    def euler_tour_exists(self) -> bool:
        for d in self.degrees().values():
            if d[0] != d[1]:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)!")
        if self.euler_tour_exists():
            return u == v
        for n in self.nodes:
            if self.degrees(n)[1] + (n == v) != self.degrees(n)[0] + (n == u):
                return False
        return self.connected()

    def euler_tour(self) -> list[Node]:
        if self.euler_tour_exists():
            tmp = DirectedMultiGraph.copy(self)
            return tmp.disconnect(u := (l := set(tmp.links.keys()).pop())[1],
                                  {(v := l[0]): 1}).euler_walk(u, v) + [u]
        return []

    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = DirectedMultiGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i + 1], {path[i]: 1})
            for i, u in enumerate(path):
                while tmp.next(u):
                    curr = tmp.disconnect(v := tmp.next(u).pop(),
                                          {u: 1}).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j + 1], {curr[j]: 1})
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def strongly_connected_component(self, n: Node) -> set[Node]:
        """
        Args:
            n: A present node.
        Returns:
            The maximal by inclusion strongly-connected component, to which a given node belongs.
        A strongly-connected component is a set of nodes, where
        there exists a path from every node to every other node.
        """

        def helper(x):
            def bfs(s):
                previous, queue, so_far = {}, [s], {s}
                while queue:
                    for t in self.next(_s := queue.pop(0)) - so_far:
                        previous[t] = _s
                        if t in path:
                            node = t
                            while node != s:
                                path.append(previous[node])
                                node = previous[node]
                            return
                        queue.append(t), so_far.add(t)

            for y in self.prev(x) - total:
                if path := self.get_shortest_path(x, y):
                    for u in path:
                        res.add(u), total.add(u)
                        for v in self.next(u):
                            if v not in total and v not in path:
                                bfs(v)
                    return
            res.add(x)

        if not isinstance(n, Node):
            n = Node(n)
        res, total = set(), {n}
        helper(n)
        return res

    def strongly_connected_components(self) -> list[set[Node]]:
        """
        Returns:
            A list of all strongly-connected components of hte graph.
        """
        if not self.connected():
            return sum(map(lambda x: x.strongly_connected_components(), self.connection_components()), [])
        if self.dag():
            return list(map(lambda x: {x}, self.nodes))
        if not self.sources and not self.sinks:
            return [self.nodes]
        rest, res = self.nodes, []
        while rest:
            res.append(curr := self.strongly_connected_component(rest.copy().pop()))
            rest -= curr
        return res

    def scc_dag(self) -> "DirectedMultiGraph":
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected
            components, and the links of which are according to whether any
            node of one SCC points to any node of another SCC.
        """
        result = DirectedMultiGraph()
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

    def cycle_with_length(self, length: int) -> list[Node]:
        try:
            length = int(length)
        except TypeError:
            raise TypeError("Integer expected!")
        if length < 2:
            return []
        tmp = DirectedMultiGraph.copy(self)
        for l in tmp.links:
            u, v = l
            res = tmp.disconnect(v, [u]).path_with_length(v, u, length - 1)
            if res:
                return res + [u]
            tmp.connect(v, [u])
        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        def dfs(x, l, stack):
            if not l:
                return (list(map(lambda link: link[0], stack)) + [v]) if x == v else []
            for y in filter(lambda _x: (x, _x) not in stack, self.next(x)):
                res = dfs(y, l - 1, stack + [(x, y)])
                if res:
                    return res
            return []

        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        try:
            length = int(length)
        except TypeError:
            raise TypeError("Integer expected!")
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

        if ((n := len(self.nodes)) == 1 or len(self.links) > (n - 1) ** 2
                or all(sum(self.degrees(u)) >= n for u in self.nodes)):
            return True
        if self.sources or self.sinks:
            return False
        tmp = DirectedMultiGraph.copy(self)
        can_end_in = tmp.prev(u := self.nodes.pop())
        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s).")
        if u in self.next(v):
            return True if all(n in {u, v} for n in self.nodes) else self.hamilton_tour_exists()
        return DirectedMultiGraph.copy(self).connect(u, [v]).hamilton_tour_exists()

    def hamilton_tour(self) -> list[Node]:
        if self.sources or self.sinks or not self:
            return []
        u = self.nodes.pop()
        for v in self.prev(u):
            if result := self.hamilton_walk(u, v):
                return result + [u]
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

        tmp = DirectedMultiGraph.copy(self)
        if u is None:
            if v is not None and v not in self:
                raise KeyError("Unrecognized node.")
            if self.dag() and (v is None or self.sink(v)):
                if any(self.degrees(n)[0] > 1 or self.degrees(n)[1] > 1 for n in self.nodes):
                    return []
                return self.toposort()
            for _u in self.nodes:
                if result := dfs(_u, [_u]):
                    return result
            return []
        if not isinstance(u, Node):
            u = Node(u)
        if v is not None and not isinstance(v, Node):
            v = Node(v)
        if u not in self or v is not None and v not in self:
            raise KeyError("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphic_bijection(self, other: "DirectedMultiGraph") -> dict[Node, Node]:
        if isinstance(other, DirectedMultiGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = defaultdict(int), defaultdict(int)
            for d in self.degrees().values():
                this_degrees[d] += 1
            for d in other.degrees().values():
                other_degrees[d] += 1
            if this_degrees != other_degrees:
                return {}
            this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].add(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].add(n)
            this_nodes_degrees = list(sorted(map(list, this_nodes_degrees.values()), key=len))
            other_nodes_degrees = list(sorted(map(list, other_nodes_degrees.values()), key=len))
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

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __reversed__(self) -> "DirectedMultiGraph":
        return self.complementary()

    def __contains__(self, u: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        return u in self.nodes

    def __add__(self, other: "DirectedMultiGraph") -> "DirectedMultiGraph":
        """
        Args:
            other: another DirectedMultiGraph object.
        Returns:
            Combination of two directed graphs.
        """
        if isinstance(other, DirectedGraph):
            other = DirectedMultiGraph(
                {u: ({}, {v: 1 for v in other.next(u)}) for u in other.nodes})
        if not isinstance(other, DirectedMultiGraph):
            raise TypeError(f"Addition not defined between class "
                            f"DirectedGraph and type {type(other).__name__}!")
        if isinstance(other, (WeightedNodesDirectedMultiGraph, WeightedLinksDirectedMultiGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            res.add(n)
        for u, v in other.links:
            res.connect(v, [u])
        return res

    def __eq__(self, other: "DirectedMultiGraph") -> bool:
        if type(other) == DirectedMultiGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self) -> str:
        return ("<" + str(self.nodes) + ", {" +
                ", ".join(f"<{l[0]}, {l[1]}>: {self.links[l]}" for l in self.links) + "}>")

    __repr__: str = __str__


class WeightedNodesDirectedMultiGraph(DirectedMultiGraph):
    ...


class WeightedLinksDirectedMultiGraph(DirectedMultiGraph):
    ...


class WeightedDirectedMultiGraph(WeightedLinksDirectedMultiGraph, WeightedNodesDirectedMultiGraph):
    ...
