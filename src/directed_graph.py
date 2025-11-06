"""
Module for implementing directed graphs
"""

from __future__ import annotations

from math import inf

from base import combine_directed, isomorphic_bijection_directed, compare, string, Any, Path, DLink

from undirected_graph import *

__all__ = ["DirectedGraph", "WeightedNodesDirectedGraph", "WeightedLinksDirectedGraph", "WeightedDirectedGraph"]


def scc_dag(graph: DirectedGraph) -> DirectedGraph:
    node_weights = isinstance(graph, WeightedNodesDirectedGraph)
    link_weights = isinstance(graph, WeightedLinksDirectedGraph)

    result = type(graph)()
    scc = graph.strongly_connected_components()

    for s in scc:
        result.add((Node(frozenset(s)), sum(map(graph.node_weights, s))) if node_weights else Node(frozenset(s)))

    for u in result.nodes:
        for v in result.nodes:
            if u != v:
                for x in u.value:
                    if any(y in v.value for y in graph.next(x)):
                        if link_weights:
                            result.connect(v, {u: 0})

                            for y in graph.next(x):
                                if y in v.value:
                                    result.increase_weight((u, v), graph.link_weights(x, y))

                        else:
                            result.connect(v, {u})

    return result


def transposed(graph: DirectedGraph) -> DirectedGraph:
    neighborhood = {u: (graph.next(u), {}) for u in graph.nodes}

    if isinstance(graph, WeightedLinksDirectedGraph):
        neighborhood = {u: (graph.link_weights(u), {}) for u in neighborhood}

    if isinstance(graph, WeightedNodesDirectedGraph):
        neighborhood = {u: (graph.node_weights(u), pair) for u, pair in neighborhood.items()}

    return type(graph)(neighborhood)


def complementary(graph: DirectedGraph) -> DirectedGraph:
    node_weights = isinstance(graph, WeightedNodesDirectedGraph)

    if node_weights:
        res = WeightedNodesDirectedGraph({u: (graph.node_weights(u), ([], graph.nodes)) for u in graph.nodes})

    else:
        res = DirectedGraph({u: ([], graph.nodes) for u in graph.nodes})

    for l in graph.links:
        res.disconnect(l[1], {l[0]})

    return res


class DirectedGraph(Graph):
    """
    Class for implementing an unweighted directed graph
    """

    def __init__(self, neighborhood: dict[Node, tuple[Iterable[Node], Iterable[Node]]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple of 2 sets of nodes. The first one is the nodes, which point to it, and the second one is the nodes it points to
        """

        if neighborhood is None:
            neighborhood = {}

        self.__nodes, self.__links = set(), set()
        self.__prev, self.__next = {}, {}

        for u, (prev_nodes, next_nodes) in neighborhood.items():
            self.add(u)

            for v in prev_nodes:
                self.add(v, points_to={u}), self.connect(u, {v})

            for v in next_nodes:
                self.add(v, {u}), self.connect(v, {u})

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes.copy()

    @property
    def links(self) -> set[DLink]:
        return self.__links.copy()

    def degree(self, u: Node = None) -> dict[Node, tuple[int, int]] | tuple[int, int]:
        """
        Get in-degree and out-degree of a given node or the same for all nodes.
        """

        if u is None:
            return {n: self.degree(n) for n in self.nodes}

        return len(self.prev(u)), len(self.next(u))

    def prev(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None
        Returns:
            A set of all nodes, that point to node u, if it's given, otherwise the same for all nodes
        """

        if u is None:
            return self.__prev.copy()

        if u not in self:
            raise KeyError("Unrecognized node")

        return self.__prev[Node(u)].copy()

    def next(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None
        Returns:
            A set of all nodes, that node u points to, if it's given, otherwise the same for all nodes
        """

        if u is None:
            return self.__next.copy()

        if u not in self:
            raise KeyError("Unrecognized node")

        return self.__next[Node(u)].copy()

    @property
    def sources(self) -> set[Node]:
        """
        Returns:
            All sources
        """

        return {u for u in self.nodes if self.source(u)}

    @property
    def sinks(self) -> set[Node]:
        """
        Returns:
            All sinks
        """

        return {v for v in self.nodes if self.sink(v)}

    def source(self, n: Node) -> bool:
        """
        Args:
            n: A present node
        Returns:
            Whether node n is a source (has a 0 in-degree)
        """

        return not self.degree(n)[0]

    def sink(self, n: Node) -> bool:
        """
        Args:
            n: A present node
        Returns:
            Whether node n is a sink (has a 0 out-degree)
        """

        return not self.degree(n)[1]

    def add(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> DirectedGraph:
        """
        Add a new node to the graph
        """

        u = Node(u)

        if u not in self:
            self.__nodes.add(u)
            self.__prev[u], self.__next[u] = set(), set()
            DirectedGraph.connect(self, u, pointed_by, points_to)

        return self

    def remove(self, u: Node, *rest: Node) -> DirectedGraph:
        for n in {u, *rest}:
            n = Node(n)

            if n in self:
                DirectedGraph.disconnect(self, n, self.prev(n), self.next(n))
                self.__nodes.remove(n), self.__prev.pop(n), self.__next.pop(n)

        return self

    def connect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> DirectedGraph:
        """
        Args:
            u: A present node
            pointed_by: A set of present nodes
            points_to: A set of present nodes
        Connect node u such, that it's pointed by nodes, listed in pointed_by, and points to nodes, listed in points_to
        """

        u = Node(u)

        if u in self:
            for v in pointed_by:
                v = Node(v)

                if u != v and (v, u) not in self.links and v in self:
                    self.__links.add((v, u))
                    self.__prev[u].add(v)
                    self.__next[v].add(u)

            for v in points_to:
                v = Node(v)

                if u != v and (u, v) not in self.links and v in self:
                    self.__links.add((u, v))
                    self.__prev[v].add(u)
                    self.__next[u].add(v)

        return self

    def connect_all(self, u: Node, *rest: Node) -> DirectedGraph:
        if not rest:
            return self

        rest = set(rest) - {u}
        self.connect(u, rest, rest)

        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> DirectedGraph:
        """
        Args:
            u: A present node
            pointed_by: A set of present nodes
            points_to: A set of present nodes
        Remove all links from nodes in pointed_by to node u and from node u to nodes in points_to
        """

        u = Node(u)

        if u in self:
            for v in pointed_by:
                v = Node(v)

                if (v, u) in self.links:
                    self.__links.remove((v, u))
                    self.__prev[u].discard(v)
                    self.__next[v].discard(u)

            for v in points_to:
                v = Node(v)

                if (u, v) in self.links:
                    self.__links.remove((u, v))
                    self.__prev[v].discard(u)
                    self.__next[u].discard(v)

        return self

    def disconnect_all(self, u: Node, *rest: Node) -> DirectedGraph:
        if not rest:
            return self

        rest = set(rest) - {u}
        self.disconnect(u, rest, rest)

        return self.disconnect_all(*rest)

    def copy(self) -> DirectedGraph:
        return DirectedGraph({n: ([], self.next(n)) for n in self.nodes})

    def complementary(self) -> DirectedGraph:
        return complementary(self)

    def transposed(self) -> DirectedGraph:
        """
        Returns:
            A graph, where each link points to the opposite direction
        """

        return transposed(self)

    def weighted_nodes_graph(self, weights: dict[Node, float] = None) -> WeightedNodesDirectedGraph:
        if weights is None:
            weights = {n: 0 for n in self.nodes}

        for n in self.nodes - set(weights):
            weights[n] = 0

        return WeightedNodesDirectedGraph({n: (weights[n], ([], self.next(n))) for n in self.nodes})

    def weighted_links_graph(self, weights: dict[DLink, float] = None) -> WeightedLinksDirectedGraph:
        if weights is None:
            weights = {l: 0 for l in self.links}

        for l in self.links - set(weights):
            weights[l] = 0

        return WeightedLinksDirectedGraph({u: ({}, {v: weights[(u, v)] for v in self.next(u)}) for u in self.nodes})

    def weighted_graph(self, node_weights: dict[Node, float] = None,
                       link_weights: dict[DLink, float] = None) -> WeightedDirectedGraph:
        return self.weighted_links_graph(link_weights).weighted_graph(node_weights)

    def undirected(self) -> UndirectedGraph:
        """
        Returns:
            The undirected version of the graph
        """

        return UndirectedGraph({n: self.prev(n).union(self.next(n)) for n in self.nodes})

    def connection_components(self) -> list[DirectedGraph]:
        components, rest = [], self.nodes

        while rest:
            components.append(curr := self.component(rest.pop()))
            rest -= curr.nodes

        return components

    def connected(self) -> bool:
        return DirectedGraph.undirected(self).connected()

    def reachable(self, u: Node, v: Node) -> bool:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        queue, total = deque([u]), {u}

        while queue:
            if (n := queue.popleft()) == v:
                return True

            queue += (new := self.next(n) - total)
            total.update(new)

        return False

    def component(self, u: Node) -> DirectedGraph:
        """
        Args:
            u: Given node
        Returns:
            The weakly connected component, to which a given node belongs
        """

        u = Node(u)

        if u not in self:
            return DirectedGraph()

        queue, total = deque([u]), {u}

        while queue:
            queue += list(next_nodes := self.prev(v := queue.popleft()).union(self.next(v)) - total)
            total.update(next_nodes)

        return self.subgraph_nodes(total)

    def full(self) -> bool:
        return len(self.links) == (n := len(self.nodes)) * (n - 1)

    def subgraph_nodes(self, nodes: Iterable[Node]):
        """
        Args:
            nodes: A set of nodes
        Returns:
            The subgraph that only contains these nodes and all links between them
        """

        nodes = self.nodes.intersection(nodes)

        return DirectedGraph({u: ([], self.next(u).intersection(nodes)) for u in nodes})

    def subgraph(self, u: Node) -> DirectedGraph:
        """
        Args:
            u: Given node
        Returns:
            The subgraph of all nodes and links, reachable by u, in a directed graph
        """

        if u not in self:
            return DirectedGraph()

        rest, res = {u}, DirectedGraph({u: ([], [])})

        while rest:
            for n in self.next(v := rest.pop()):
                if n in res:
                    res.connect(n, {v})

                else:
                    res.add(n, {v}), rest.add(n)

        return res

    def supergraph(self, u: Node) -> DirectedGraph:
        """
        Args:
            u: Given node
        Returns:
            The graph of nodes which can reach u
        """

        if u not in self:
            return DirectedGraph()

        rest, res = {u}, DirectedGraph({u: ([], [])})

        while rest:
            for n in self.prev(v := rest.pop()):
                if n in res:
                    res.connect(n, {v})

                else:
                    res.add(n, {v}), rest.add(n)

        return res

    def dag(self) -> bool:
        """
        Returns:
            Whether the graph is a DAG (directed acyclic graph)
        """

        def dfs(u):
            for v in self.next(u):
                if v in total:
                    continue

                if v in stack:
                    return False

                stack.add(v)

                if not dfs(v):
                    return False

                stack.remove(v)

            total.add(u)

            return True

        sources, total, stack = self.sources, set(), set()

        if not sources or not self.sinks:
            return False

        for n in sources:
            stack.add(n)

            if not dfs(n):
                return False

            stack.remove(n)

        return total == self.nodes

    def toposort(self) -> Path:
        """
        Returns:
            A topological sort of the nodes if the graph is a DAG, otherwise an empty list
        A topological sort has the following property: Let u and v be nodes in the graph and let u come before v. Then there's no path from v to u in the graph. (That's also the reason a graph with cycles has no topological sort)
        """

        tmp = DirectedGraph.copy(self)
        total, layer, res = set(), tmp.sources, []

        while layer:
            res.append(u := layer.pop()), total.add(u)

            for v in tmp.next(u):
                if v in total:
                    return []

                tmp.disconnect(v, {u})

                if tmp.source(v):
                    layer.add(v)

        return res if total == tmp.nodes else []

    def get_shortest_path(self, u: Node, v: Node) -> Path:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        previous, queue, total = {}, deque([u]), {u}

        while queue:
            if (n := queue.popleft()) == v:
                result, curr_node = [n], n

                while curr_node != u:
                    result.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]

                return result

            for m in self.next(n) - total:
                queue.append(m), total.add(m)
                previous[m] = n

        return []

    def euler_tour_exists(self) -> bool:
        for d in self.degree().values():
            if d[0] != d[1]:
                return False

        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        if self.euler_tour_exists():
            return u == v

        for n in self.nodes:
            if self.degree(n)[1] + (n == v) != self.degree(n)[0] + (n == u):
                return False

        return self.connected()

    def euler_tour(self) -> Path:
        if self.euler_tour_exists():
            tmp = DirectedGraph.copy(self)

            return tmp.disconnect(u := (l := tmp.links.pop())[1], {v := l[0]}).euler_walk(u, v) + [u]

        return []

    def euler_walk(self, u: Node, v: Node) -> Path:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = DirectedGraph.copy(self)

            for i in range(len(path) - 1):
                tmp.disconnect(path[i + 1], {path[i]})

            for i, u in enumerate(path):
                neighbors = tmp.next(u)

                while neighbors:
                    curr = tmp.disconnect(v := neighbors.pop(), {u}).get_shortest_path(v, u)

                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j + 1], {curr[j]})

                    while curr:
                        path.insert(i + 1, curr.pop())

            return path

        return []

    def strongly_connected_component(self, n: Node) -> set[Node]:
        """
        Args:
            n: A present node
        Returns:
            The maximal by inclusion strongly-connected component, to which a given node belongs
        A strongly-connected component is a set of nodes, where there exists a path from every node to every other node
        """

        def helper(x):
            def bfs(s):
                previous, queue, so_far = {}, deque([s]), {s}

                while queue:
                    for t in self.next(w := queue.popleft()) - so_far:
                        previous[t] = w

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

        n = Node(n)

        if not self.sources and not self.sinks:
            return self.nodes

        if self.dag():
            return {n}

        res, total = set(), {n}
        helper(n)

        return res

    def strongly_connected_components(self) -> list[set[Node]]:
        """
        Returns:
            A list of all strongly-connected components of the graph
        """

        if not self.connected():
            return sum(
                map(lambda x: x.strongly_connected_components(), self.connection_components()), [])

        if not self.sources and not self.sinks:
            return [self.nodes]

        if self.dag():
            return list(map(lambda x: {x}, self.nodes))

        rest, res = self.nodes, []

        while rest:
            res.append(curr := self.strongly_connected_component(rest.pop()))
            rest -= curr

        return res

    def scc_dag(self) -> DirectedGraph:
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected components, and the links of which are according to whether any node of one SCC points to any node of another SCC
        """

        return scc_dag(self)

    def cycle_with_length(self, length: int) -> Path:
        try:
            length = int(length)

        except TypeError:
            raise TypeError("Integer expected")

        if length < 2:
            return []

        tmp = DirectedGraph.copy(self)

        for u, v in tmp.links:
            res = tmp.disconnect(v, {u}).path_with_length(v, u, length - 1)

            if res:
                return res + [u]

            tmp.connect(v, {u})

        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> Path:
        def dfs(x, l, stack):
            if not l:
                return (list(map(lambda link: link[0], stack)) + [v]) if x == v else []

            for y in filter(lambda _x: (x, _x) not in stack, self.next(x)):
                res = dfs(y, l - 1, stack + [(x, y)])

                if res:
                    return res

            return []

        u, v = Node(u), Node(v)

        try:
            length = int(length)

        except ValueError:
            raise TypeError("Integer expected")

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

            prev_x, next_x = tmp.prev(x), tmp.next(x)
            tmp.remove(x)

            for y in next_x:
                if dfs(y):
                    return True

            tmp.add(x, prev_x, next_x)

            return False

        if (n := len(self.nodes)) == 1 or len(self.links) > (n - 1) ** 2 or all(
                sum(self.degree(u)) >= n for u in self.nodes):
            return True

        if self.sources or self.sinks:
            return False

        tmp = DirectedGraph.copy(self)
        can_end_in = tmp.prev(u := self.nodes.pop())

        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        if (v, u) in self.links:
            return self.nodes == {u, v} or self.hamilton_tour_exists()

        return DirectedGraph.copy(self).connect(u, {v}).hamilton_tour_exists()

    def hamilton_tour(self) -> Path:
        if self.sources or self.sinks or not self:
            return []

        u = self.nodes.pop()

        for v in self.prev(u):
            if res := self.hamilton_walk(u, v):
                return res + [u]

        return []

    def hamilton_walk(self, u: Node = None, v: Node = None) -> Path:
        def dfs(x, stack):
            if not tmp.connected:
                return []

            too_many = v is not None

            for n in tmp.nodes:
                if not tmp.degree(n)[0] and n != x:
                    return []

                if not tmp.degree(n)[1] and n != v:
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
                        return stack + [v]

                    continue

                if res := dfs(y, stack + [y]):
                    return res

            tmp.add(x, prev_x, next_x)

            return []

        tmp = DirectedGraph.copy(self)

        if u is None:
            if v is not None and v not in self:
                raise KeyError("Unrecognized node")

            del u

            for u in self.nodes:
                if result := dfs(u, [u]):
                    return result

            return []

        u = Node(u)

        if v is not None:
            v = Node(v)

        if u not in self or v is not None and v not in self:
            raise KeyError("Unrecognized node(s)")

        return dfs(u, [u])

    def isomorphic_bijection(self, other: DirectedGraph) -> dict[Node, Node]:
        return isomorphic_bijection_directed(self, other)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __reversed__(self) -> DirectedGraph:
        return self.transposed()

    def __contains__(self, u: Node) -> bool:
        return Node(u) in self.nodes

    def __add__(self, other: DirectedGraph) -> DirectedGraph:
        """
        Args:
            other: another DirectedGraph object
        Returns:
            Combination of two directed graphs
        """

        return combine_directed(self, other)

    def __eq__(self, other: Any) -> bool:
        return compare(self, other)

    def __str__(self) -> str:
        return string(self)

    __repr__: str = __str__


class WeightedNodesDirectedGraph(DirectedGraph):
    """
    Class for implementing a directed graph with node weights
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, tuple[Iterable[Node], Iterable[Node]]]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple with 2 elements. The first one is the node's weight. The second one is a tuple with 2 sets of nodes. The first one is the nodes, that point to it, and the second one are the nodes it points to
        """

        if neighborhood is None:
            neighborhood = {}

        super().__init__()
        self.__node_weights = {}

        for n, (w, _) in neighborhood.items():
            self.add((n, w))

        for u, (_, (prev_u, next_u)) in neighborhood.items():
            for v in prev_u:
                self.add((v, 0), points_to={u}), self.connect(u, {v})

            for v in next_u:
                self.add((v, 0), {u}), self.connect(v, {u})

    def node_weights(self, n: Node = None) -> dict[Node, float] | float:
        """
        Args:
            n: A present node or None
        Returns:
            The weight of node n or the dictionary with all node weights
        """

        return {u: self.node_weights(u) for u in self.nodes} if n is None else self.__node_weights[Node(n)]

    @property
    def total_nodes_weight(self) -> float:
        """
        Returns:
            The sum of all node weights
        """

        return sum(self.node_weights().values())

    def add(self, n_w: tuple[Node, float], pointed_by: Iterable[Node] = (),
            points_to: Iterable[Node] = ()) -> WeightedNodesDirectedGraph:
        n = n_w[0]

        if n not in self:
            DirectedGraph.add(self, n, pointed_by, points_to)
            self.set_weight(*n_w)

        return self

    def remove(self, u: Node, *rest: Node) -> WeightedNodesDirectedGraph:
        for n in {u, *rest}:
            if n in self:
                self.__node_weights.pop(Node(n))

        return super().remove(u, *rest)

    def set_weight(self, u: Node, w: float) -> WeightedNodesDirectedGraph:
        """
        Args:
            u: A present node
            w: The new weight of node u
        Set the weight of node u to w
        """

        if u in self:
            try:
                self.__node_weights[Node(u)] = float(w)

            except ValueError:
                raise TypeError("Real value expected")

        return self

    def increase_weight(self, u: Node, w: float) -> WeightedNodesDirectedGraph:
        """
        Args:
            u: A present node
            w: A real value
        Increase the weight of node u by w
        """

        if Node(u) in self.node_weights():
            try:
                self.set_weight(u, self.node_weights(u) + float(w))

            except ValueError:
                raise TypeError("Real value expected")

        return self

    def copy(self) -> WeightedNodesDirectedGraph:
        return WeightedNodesDirectedGraph({n: (self.node_weights(n), ([], self.next(n))) for n in self.nodes})

    def weighted_graph(self, weights: dict[DLink, float] = None) -> WeightedDirectedGraph:
        if weights is None:
            weights = {l: 0 for l in self.links}

        for l in self.links - set(weights):
            weights[l] = 0

        return super().weighted_graph(self.node_weights(), weights)

    def undirected(self) -> WeightedNodesUndirectedGraph:
        neighborhood = {n: (self.node_weights(n), self.prev(n).union(self.next(n))) for n in self.nodes}

        return WeightedNodesUndirectedGraph(neighborhood)

    def subgraph_nodes(self, nodes: Iterable[Node]) -> WeightedNodesDirectedGraph:
        nodes = self.nodes.intersection(nodes)
        neighborhood = {u: (self.node_weights(u), ([], self.next(u).intersection(nodes))) for u in nodes}

        return WeightedNodesDirectedGraph(neighborhood)

    def subgraph(self, u: Node) -> WeightedNodesDirectedGraph:
        if u not in self:
            return WeightedNodesDirectedGraph()

        rest = {u}
        res = WeightedNodesDirectedGraph({u: (self.node_weights(u), ([], []))})

        while rest:
            for n in self.next(v := rest.pop()):
                if n in res:
                    res.connect(n, {v})

                else:
                    res.add((n, self.node_weights(n)), {v}), rest.add(n)

        return res

    def supergraph(self, u: Node) -> WeightedNodesDirectedGraph:
        if u not in self:
            return WeightedNodesDirectedGraph()

        rest, res = {u}, WeightedNodesDirectedGraph({u: (self.node_weights(u), ([], []))})

        while rest:
            for n in self.prev(v := rest.pop()):
                if n in res:
                    res.connect(n, {v})

                else:
                    res.add((n, self.node_weights(u)), {v}), rest.add(n)

        return res

    def minimal_path_nodes(self, u: Node, v: Node) -> Path:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path between u and v with the least possible sum of node weights
        """

        return self.weighted_graph().minimal_path(u, v)


class WeightedLinksDirectedGraph(DirectedGraph):
    """
    Class for implementing directed graphs with link weights
    """

    def __init__(self, neighborhood: dict[Node, tuple[dict[Node, float], dict[Node, float]]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple with 2 dictionaries. The first one contains the nodes, which point to it, and the second one contains the nodes it points to. The values in these dictionaries are the link weights
        """

        if neighborhood is None:
            neighborhood = {}

        super().__init__()
        self.__link_weights = {}

        for u, (prev_pairs, next_pairs) in neighborhood.items():
            if u not in self:
                self.add(u)

            for v, w in prev_pairs.items():
                self.add(v, points_to_weights={u: w}), self.connect(u, {v: w})

            for v, w in next_pairs.items():
                self.add(v, {u: w}), self.connect(v, {u: w})

    def link_weights(self, u_l: Node | DLink = None, v: Node = None) -> dict[Node, float] | dict[DLink, float] | float:
        """
        Args:
            u_l: Given first node, a link or None
            v: Given second node or None
        Returns:
            Information about link weights the following way:
            If no argument is passed, return the weights of all links;
            if a link or two nodes are passed, return the weight of the given link between them;
            If one node is passed, return a dictionary with all nodes it points to and the weight of the link from that node to each of them
        """

        if u_l is None:
            return self.__link_weights.copy()

        if isinstance(u_l, tuple):
            return self.__link_weights[u_l]

        u_l = Node(u_l)

        return {n: self.link_weights((u_l, n)) for n in self.next(u_l)} if v is None else self.link_weights(
            (u_l, Node(v)))

    @property
    def total_links_weight(self) -> float:
        """
        Returns:
            The sum of all link weights
        """

        return sum(self.link_weights().values())

    def add(self, u: Node, pointed_by_weights: dict[Node, float] = None,
            points_to_weights: dict[Node, float] = None) -> WeightedLinksDirectedGraph:
        u = Node(u)

        if pointed_by_weights is None:
            pointed_by_weights = {}

        if points_to_weights is None:
            points_to_weights = {}

        if u not in self:
            DirectedGraph.add(self, u, pointed_by_weights.keys(), points_to_weights.keys())

            for v, w in pointed_by_weights.items():
                self.set_weight((v, u), w)

            for v, w in points_to_weights.items():
                self.set_weight((u, v), w)

        return self

    def remove(self, u: Node, *rest: Node) -> WeightedLinksDirectedGraph:
        for n in {u, *rest}:
            n = Node(n)

            if n in self:
                for v in self.next(n):
                    if (n, v) in self.link_weights():
                        self.__link_weights.pop((n, v))

                for v in self.prev(n):
                    if (v, n) in self.link_weights():
                        self.__link_weights.pop((v, n))

        return super().remove(u, *rest)

    def connect(self, u: Node, pointed_by_weights: dict[Node, float] = None,
                points_to_weights: dict[Node, float] = None) -> WeightedLinksDirectedGraph:
        u = Node(u)

        if pointed_by_weights is None:
            pointed_by_weights = {}

        if points_to_weights is None:
            points_to_weights = {}

        if u in self:
            pointed_by_weights = {Node(k): v for k, v in pointed_by_weights.items()}
            pointed_by_weights = {v: w for v, w in pointed_by_weights.items() if v not in self.prev(u)}
            points_to_weights = {Node(k): v for k, v in points_to_weights.items()}
            points_to_weights = {v: w for v, w in points_to_weights.items() if v not in self.next(u)}
            super().connect(u, pointed_by_weights.keys(), points_to_weights.keys())

            for v, w in pointed_by_weights.items():
                v = Node(v)

                if (v, u) not in self.link_weights():
                    self.set_weight((v, u), w)

            for v, w in points_to_weights.items():
                v = Node(v)

                if (u, v) not in self.link_weights():
                    self.set_weight((u, v), w)

        return self

    def connect_all(self, u: Node, *rest: Node) -> WeightedLinksDirectedGraph:
        if not rest:
            return self

        rest = set(rest) - {u}
        self.connect(u, (d := {v: 0 for v in rest}), d)

        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: Iterable[Node] = (),
                   points_to: Iterable[Node] = ()) -> WeightedLinksDirectedGraph:
        u = Node(u)

        if u in self:
            for v in pointed_by:
                v = Node(v)

                if (v, u) in self.links:
                    self.__link_weights.pop((v, u))

            for v in points_to:
                v = Node(v)

                if (u, v) in self.links:
                    self.__link_weights.pop((u, v))

            super().disconnect(u, pointed_by, points_to)

        return self

    def set_weight(self, l: DLink, w: float) -> WeightedLinksDirectedGraph:
        """
        Args:
            l: A present link
            w: The new weight of link l
        Set the weight of link l to w
        """

        try:
            l = tuple(map(Node, l))

            if l in self.links:
                try:
                    self.__link_weights[l] = float(w)

                except TypeError:
                    raise TypeError("Real value expected")

            return self

        except ValueError:
            raise TypeError("Directed link is of type tuple[Node, Node]")

    def increase_weight(self, l: DLink, w: float) -> WeightedLinksDirectedGraph:
        """
        Args:
            l: A present link
            w: A real value
        Increase the weight of link l with w
        """

        try:
            l = tuple(l)

            if len(l) != 2:
                raise ValueError("Directed link expected")

            l = (Node(l[0]), Node(l[1]))

            if l in self.link_weights():
                try:
                    self.set_weight(l, self.link_weights(l) + float(w))

                except TypeError:
                    raise TypeError("Real value expected")

            return self

        except ValueError as t:
            if "Real value expected" in t.args:
                raise t

            raise TypeError("Directed link is of type tuple[Node, Node]")

    def copy(self) -> WeightedLinksDirectedGraph:
        return WeightedLinksDirectedGraph({u: ({}, self.link_weights(u)) for u in self.nodes})

    def weighted_graph(self, weights: dict[Node, float] = None) -> WeightedDirectedGraph:
        if weights is None:
            weights = {n: 0 for n in self.nodes}

        for n in self.nodes - set(weights):
            weights[n] = 0

        return WeightedDirectedGraph(
            {u: (weights[u], ({}, {v: self.link_weights(u, v) for v in self.next(u)})) for u in self.nodes})

    def undirected(self) -> WeightedLinksUndirectedGraph:
        res = WeightedLinksUndirectedGraph({n: {} for n in self.nodes})

        for u, v in self.links:
            if v in res.neighbors(u):
                res.increase_weight(Link(u, v), self.link_weights(u, v))

            else:
                res.connect(u, {v: self.link_weights(u, v)})

        return res

    def subgraph_nodes(self, nodes: Iterable[Node]) -> WeightedLinksDirectedGraph:
        nodes = self.nodes.intersection(nodes)
        neighborhood = {u: ({}, {k: v for k, v in self.link_weights(u).items() if k in nodes}) for u in nodes}

        return WeightedLinksDirectedGraph(neighborhood)

    def subgraph(self, u: Node) -> WeightedLinksDirectedGraph:
        if u not in self:
            return WeightedLinksDirectedGraph()

        rest = {u}
        res = WeightedLinksDirectedGraph({u: ({}, {})})

        while rest:
            for n in self.next(v := rest.pop()):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})

                else:
                    res.add(n, {v: self.link_weights(v, n)}), rest.add(n)

        return res

    def supergraph(self, u: Node) -> WeightedLinksDirectedGraph:
        if u not in self:
            return WeightedLinksDirectedGraph()

        rest = {u}
        res = WeightedLinksDirectedGraph({u: ({}, {})})

        while rest:
            for n in self.prev(v := rest.pop()):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})

                else:
                    res.add(n, {v: self.link_weights(v, n)}), rest.add(n)

        return res

    def minimal_path_links(self, u: Node, v: Node) -> Path:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path from u to v with the least possible sum of link weights
        """

        return self.weighted_graph().minimal_path(u, v)


class WeightedDirectedGraph(WeightedLinksDirectedGraph, WeightedNodesDirectedGraph):
    """
    Class for implementing directed graph with weights on the nodes and the links
    """

    def __init__(self,
                 neighborhood: dict[Node, tuple[float, tuple[dict[Node, float], dict[Node, float]]]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple with its weight and another tuple with 2 dictionaries. The first one contains the nodes, which point to it, and the second one contains the nodes it points to. The values in these dictionaries are the link weights
        """

        if neighborhood is None:
            neighborhood = {}

        WeightedNodesDirectedGraph.__init__(self), WeightedLinksDirectedGraph.__init__(self)

        for n, (w, _) in neighborhood.items():
            self.add((n, w))

        for u, (_, (prev_u, next_u)) in neighborhood.items():
            for v, w in prev_u.items():
                self.add((v, 0), points_to_weights={u: w})
                self.connect(u, {v: w})

            for v, w in next_u.items():
                self.add((v, 0), {u: w})
                self.connect(v, {u: w})

    @property
    def total_weight(self) -> float:
        """
        Returns:
            The sum of all weights in the graph
        """

        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: tuple[Node, float], pointed_by_weights: dict[Node, float] = None,
            points_to_weights: dict[Node, float] = None) -> WeightedDirectedGraph:
        n = Node(n_w[0])

        if pointed_by_weights is None:
            pointed_by_weights = {}

        if points_to_weights is None:
            points_to_weights = {}

        if n not in self:
            super().add(n, pointed_by_weights, points_to_weights)
            self.set_weight(*n_w)

        return self

    def remove(self, u: Node, *rest: Node) -> WeightedDirectedGraph:
        for n in {u, *rest}:
            if n in self:
                super().disconnect(n, self.prev(n), self.next(n))

        return WeightedNodesDirectedGraph.remove(self, u, *rest)

    def set_weight(self, el: Node | DLink, w: float) -> WeightedDirectedGraph:
        """
        Args:
            el: A present node or link
            w: The new weight of object el
        Set the weight of object el to w
        """

        if isinstance(el, tuple):
            el = tuple(map(Node, el))

        if isinstance(el, tuple) and len(el) == 2 and (Node(el[0]), Node(el[1])) in self.links:
            super().set_weight(el, float(w))

        elif el in self:
            WeightedNodesDirectedGraph.set_weight(self, el, float(w))

        return self

    def increase_weight(self, el: Node | DLink, w: float) -> WeightedDirectedGraph:
        """
        Args:
            el: A present node or link
            w: A real value
        Increase the weight of object el with w
        """

        try:
            if el in self.link_weights():
                self.set_weight(el, self.link_weights(el) + float(w))

            elif Node(el) in self.node_weights():
                return self.set_weight(el, self.node_weights(el) + float(w))

            return self

        except ValueError:
            raise TypeError("Real value expected")

    def copy(self) -> WeightedDirectedGraph:
        neighborhood = {u: (self.node_weights(u), ({}, self.link_weights(u))) for u in self.nodes}

        return WeightedDirectedGraph(neighborhood)

    def undirected(self) -> WeightedUndirectedGraph:
        res = WeightedUndirectedGraph({n: (self.node_weights(n), {}) for n in self.nodes})

        for u, v in self.links:
            if v in res.neighbors(u):
                res.increase_weight(Link(u, v), self.link_weights(u, v))

            else:
                res.connect(u, {v: self.link_weights(u, v)})

        return res

    def subgraph_nodes(self, nodes: Iterable[Node]) -> WeightedDirectedGraph:
        nodes = self.nodes.intersection(nodes)
        neighborhood = {u: (self.node_weights(u), ({}, {k: v for k, v in self.link_weights(u).items() if k in nodes}))
                        for u in nodes}

        return WeightedDirectedGraph(neighborhood)

    def subgraph(self, u: Node) -> WeightedDirectedGraph:
        if u not in self:
            return WeightedDirectedGraph()

        rest = {u}
        res = WeightedDirectedGraph({u: (self.node_weights(u), ({}, {}))})

        while rest:
            for n in self.next(v := rest.pop()):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})

                else:
                    res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), rest.add(n)

        return res

    def supergraph(self, u: Node) -> WeightedDirectedGraph:
        if u not in self:
            return WeightedDirectedGraph()

        rest = {u}
        res = WeightedDirectedGraph({u: (self.node_weights(u), ({}, {}))})

        while rest:
            for n in self.prev(v := rest.pop()):
                if n in res:
                    res.connect(n, {v: self.link_weights(v, n)})

                else:
                    res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), rest.add(n)

        return res

    def minimal_path(self, u: Node, v: Node) -> Path:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path between u and v with the least possible sum of node and link weights
        """

        def dfs(x, tmp):
            nonlocal curr_path, curr_weight, res_path, res_weight

            def dag_path(s):
                nonlocal res_path, res_weight

                prev_dist = {x: (None, inf) for x in tmp.nodes}
                prev_dist[s] = (None, 0)

                for x in sort:
                    for y, w in tmp.link_weights(x).items():
                        if (new_w := tmp.node_weights(y) + w + prev_dist[x][1]) < prev_dist[y][1]:
                            prev_dist[y] = (x, new_w)

                res, curr = [], v

                while (prev := prev_dist[curr][0]) is not None:
                    res.insert(0, curr)
                    curr = prev

                result = (curr_path + res, curr_weight + prev_dist[v][1])

                if result[1] < res_weight:
                    res_path, res_weight = result

            def dijkstra(s):
                nonlocal res_path, res_weight

                pq = [(0, s)]
                prev_weight = {n: (None, inf) for n in tmp.nodes}
                prev_weight[s] = (None, 0)

                while pq:
                    s_weight, s_ = heappop(pq)

                    if s_ == v:
                        break

                    for t_ in tmp.next(s_):
                        if (new_w := s_weight + tmp.link_weights(s_, t_) + tmp.node_weights(t_)) < prev_weight[t_][1]:
                            prev_weight[t_] = (s_, new_w)
                            heappush(pq, (new_w, t_))

                else:
                    return

                result, curr_node = [], v

                while curr_node != s:
                    result.insert(0, curr_node)
                    curr_node = prev_weight[curr_node][0]

                result = (curr_path + result, curr_weight + prev_weight[v][1])

                if result[1] < res_weight:
                    res_path, res_weight = result

            def SPFA(s):
                nonlocal res_path, res_weight

                prev_weight = {n: (None, inf) for n in tmp.nodes}
                prev_weight[s] = (None, 0)
                queue, in_queue = deque([s]), {s}

                while queue:
                    x = queue.popleft()
                    in_queue.remove(x)

                    for y, w in tmp.link_weights(x).items():
                        if prev_weight[x][1] + w < prev_weight[y][1]:
                            prev_weight[y] = (x, prev_weight[x][1] + w)

                            if y not in in_queue:
                                queue.append(y), in_queue.add(y)

                for _u, _v in tmp.links:
                    w = tmp.link_weights(_u, _v) + tmp.node_weights(_v)

                    if prev_weight[_u][1] + w < prev_weight[_v][1]:
                        raise ValueError

                path, curr_node = [], v

                while curr_node != s:
                    path.insert(0, curr_node)
                    curr_node = prev_weight[curr_node][0]

                result = (curr_path + path, curr_weight + prev_weight[v][1])

                if result[1] < res_weight:
                    res_path, res_weight = result

            if not tmp or x == v and tmp.source(x):
                return

            if sort := tmp.toposort():
                dag_path(x)

                return

            nodes_negative_weights = sum(tmp.node_weights(n) for n in tmp.nodes if tmp.node_weights(n) < 0)
            links_negative_weights = sum(tmp.link_weights(l) for l in tmp.links if tmp.link_weights(l) < 0)
            total_negative = nodes_negative_weights + links_negative_weights

            if total_negative:
                try:
                    SPFA(x)

                except ValueError:
                    for y in tmp.next(x):
                        n_w, l_w = tmp.node_weights(y), tmp.link_weights(x, y)

                        if curr_weight + max(n_w, 0) + max(l_w, 0) + total_negative >= res_weight:
                            continue

                        if n_w < 0:
                            total_negative -= n_w

                        if l_w < 0:
                            total_negative -= l_w

                        tmp.disconnect(y, {x})
                        curr_weight += n_w + l_w

                        if y == v and curr_weight < res_weight:
                            res_path = curr_path + [v]
                            res_weight = curr_weight

                        curr_path.append(y)
                        dfs(y, tmp.subgraph(y).supergraph(v))
                        curr_path.pop()
                        curr_weight -= n_w + l_w
                        tmp.connect(y, {x: l_w})

                        if n_w < 0:
                            total_negative += n_w

                        if l_w < 0:
                            total_negative += l_w

            else:
                dijkstra(x)

        u, v = Node(u), Node(v)

        if v in self:
            if v in (g := self.subgraph(u)):
                res_path, res_weight = [], inf
                curr_path, curr_weight = [u], g.node_weights(u)
                dfs(u, g)

                return res_path

            return []

        raise KeyError("Unrecognized node(s)")
