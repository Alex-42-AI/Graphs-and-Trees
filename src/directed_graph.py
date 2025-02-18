"""
Module for implementing directed graphs
"""

__all__ = ["DirectedGraph", "WeightedNodesDirectedGraph", "WeightedLinksDirectedGraph", "WeightedDirectedGraph"]

from .base import combine_directed, isomorphic_bijection_directed

from .undirected_graph import *


class DirectedGraph(Graph):
    """
    Class for implementing an unweighted directed graph
    """

    def __init__(self, neighborhood: dict[Node, tuple[Iterable[Node], Iterable[Node]]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple of 2 sets of nodes. The first one is the nodes, which point to it, and the second one is the nodes it points to
        """
        self.__nodes, self.__links = set(), set()
        self.__prev, self.__next = {}, {}
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

    def degrees(self, u: Node = None) -> dict[Node, tuple[int, int]] | tuple[int, int]:
        """
        Get in-degree and out-degree of a given node or the same for all nodes.
        """
        if u is None:
            return {n: self.degrees(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return len(self.prev(u)), len(self.next(u))

    def next(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None
        Returns:
            A set of all nodes, that node u points to, if it's given, otherwise the same for all nodes
        """
        if u is None:
            return {n: self.next(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return self.__next[u].copy()

    def prev(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None
        Returns:
            A set of all nodes, that point to node u, if it's given, otherwise the same for all nodes
        """
        if u is None:
            return {n: self.prev(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return self.__prev[u].copy()

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
            Whether node n is a source
        """
        if not isinstance(n, Node):
            n = Node(n)
        return not self.degrees(n)[0]

    def sink(self, n: Node) -> bool:
        """
        Args:
            n: A present node
        Returns:
            Whether node n is a sink
        """
        if not isinstance(n, Node):
            n = Node(n)
        return not self.degrees(n)[1]

    def add(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "DirectedGraph":
        """
        Add a new node to the graph
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            self.__nodes.add(u)
            self.__next[u], self.__prev[u] = set(), set()
            DirectedGraph.connect(self, u, pointed_by, points_to)
        return self

    def remove(self, u: Node, *rest: Node) -> "DirectedGraph":
        for n in {u, *rest}:
            if not isinstance(n, Node):
                n = Node(n)
            if n in self:
                DirectedGraph.disconnect(self, n, self.prev(n), self.next(n))
                self.__nodes.remove(n), self.__prev.pop(n), self.__next.pop(n)
        return self

    def connect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "DirectedGraph":
        """
        Args:
            u: A present node
            pointed_by: A set of present nodes
            points_to: A set of present nodes
        Connect node u such, that it's pointed by nodes, listed in pointed_by, and points to nodes, listed in points_to
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            for v in pointed_by:
                if not isinstance(v, Node):
                    v = Node(v)
                if u != v and v not in self.prev(u) and v in self:
                    self.__links.add((v, u))
                    self.__prev[u].add(v)
                    self.__next[v].add(u)
            for v in points_to:
                if not isinstance(v, Node):
                    v = Node(v)
                if u != v and v not in self.next(u) and v in self:
                    self.__links.add((u, v))
                    self.__prev[v].add(u)
                    self.__next[u].add(v)
        return self

    def connect_all(self, u: Node, *rest: Node) -> "DirectedGraph":
        if not rest:
            return self
        rest = set(rest) - {u}
        self.connect(u, rest, rest)
        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: Iterable[Node] = (), points_to: Iterable[Node] = ()) -> "DirectedGraph":
        """
        Args:
            u: A present node
            pointed_by: A set of present nodes
            points_to: A set of present nodes
        Remove all links from .nodes in pointed_by to node u and from .node u to nodes in points_to
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            for v in pointed_by:
                if not isinstance(v, Node):
                    v = Node(v)
                if v in self.prev(u):
                    self.__links.remove((v, u))
                    self.__next[v].remove(u)
                    self.__prev[u].remove(v)
            for v in points_to:
                if not isinstance(v, Node):
                    v = Node(v)
                if v in self.next(u):
                    self.__links.remove((u, v))
                    self.__next[u].remove(v)
                    self.__prev[v].remove(u)
        return self

    def disconnect_all(self, u: Node, *rest: Node) -> "DirectedGraph":
        if not rest:
            return self
        rest = set(rest) - {u}
        self.disconnect(u, rest, rest)
        return self.disconnect_all(*rest)

    def copy(self) -> "DirectedGraph":
        return DirectedGraph({n: ([], self.next(n)) for n in self.nodes})

    def complementary(self) -> "DirectedGraph":
        res = DirectedGraph({n: ([], self.nodes) for n in self.nodes})
        for l in self.links:
            res.disconnect(l[1], {l[0]})
        return res

    def transposed(self) -> "DirectedGraph":
        """
        Returns:
            A graph, where each link points to the opposite direction
        """
        return DirectedGraph({u: (self.next(u), []) for u in self.nodes})

    def weighted_nodes_graph(self, weights: dict[Node, float] = None) -> "WeightedNodesDirectedGraph":
        if weights is None:
            weights = {n: 0 for n in self.nodes}
        for n in self.nodes - set(weights):
            weights[n] = 0
        return WeightedNodesDirectedGraph({n: (weights[n], ([], self.next(n))) for n in self.nodes})

    def weighted_links_graph(self, weights: dict[tuple[Node, Node], float] = None) -> "WeightedLinksDirectedGraph":
        if weights is None:
            weights = {l: 0 for l in self.links}
        for l in self.links - set(weights):
            weights[l] = 0
        return WeightedLinksDirectedGraph({u: ({}, {v: weights[(u, v)] for v in self.next(u)}) for u in self.nodes})

    def weighted_graph(self, node_weights: dict[Node, float] = None,
                       link_weights: dict[tuple[Node, Node], float] = None) -> "WeightedDirectedGraph":
        return self.weighted_links_graph(link_weights).weighted_graph(node_weights)

    def undirected(self) -> "UndirectedGraph":
        """
        Returns:
            The undirected version of the graph
        """
        return UndirectedGraph({n: self.prev(n).union(self.next(n)) for n in self.nodes})

    def connection_components(self) -> list["DirectedGraph"]:
        components, rest = [], self.nodes
        while rest:
            components.append(curr := self.component(rest.copy().pop()))
            rest -= curr.nodes
        return components

    def connected(self) -> bool:
        return DirectedGraph.undirected(self).connected()

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

    def component(self, u: Node) -> "DirectedGraph":
        """
        Args:
            u: Given node
        Returns:
            The weakly connected component, to which a given node belongs
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        queue, total = [u], {u}
        while queue:
            queue += list(next_nodes := self.prev(v := queue.pop(0)).union(self.next(v)) - total)
            total.update(next_nodes)
        return self.subgraph(total)

    def full(self) -> bool:
        return len(self.links) == (n := len(self.nodes)) * (n - 1)

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "DirectedGraph":
        try:
            u_or_nodes = self.nodes.intersection(u_or_nodes)
            return DirectedGraph({u: ([], self.next(u).intersection(u_or_nodes)) for u in u_or_nodes})
        except TypeError:
            if not isinstance(u_or_nodes, Node):
                u_or_nodes = Node(u_or_nodes)
            if u_or_nodes not in self:
                raise KeyError("Unrecognized node!")
            queue, res = [u_or_nodes], DirectedGraph({u_or_nodes: ([], [])})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, [v])
                    else:
                        res.add(n, [v]), queue.append(n)
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

    def toposort(self) -> list[Node]:
        """
        Returns:
            A topological sort of the nodes if the graph is a DAG, otherwise an empty list
        A topological sort has the following property: Let u and v be nodes in the graph and let u come before v. Then there's no path from .v to u in the graph. (That's also the reason a graph with cycles has no topological sort)
        """
        if not self.dag():
            return []
        layer, total = self.sources, set()
        res = list(layer)
        while layer:
            new = set()
            total.update(layer)
            for u in layer:
                new.update(self.next(u))
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
        return []

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
            tmp = DirectedGraph.copy(self)
            return tmp.disconnect(u := (l := tmp.links.pop())[1], [v := l[0]]).euler_walk(u, v) + [u]
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
        """
        Args:
            n: A present node
        Returns:
            The maximal by inclusion strongly-connected component, to which a given node belongs
        A strongly-connected component is a set of nodes, where there exists a path from .every node to every other node
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
            A list of all strongly-connected components of the graph
        """
        if not self.connected():
            return sum(
                map(lambda x: x.strongly_connected_components(), self.connection_components()), [])
        if self.dag():
            return list(map(lambda x: {x}, self.nodes))
        if not self.sources and not self.sinks:
            return [self.nodes]
        rest, res = self.nodes, []
        while rest:
            res.append(curr := self.strongly_connected_component(rest.copy().pop()))
            rest -= curr
        return res

    def scc_dag(self) -> "DirectedGraph":
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected components, and the links of which are according to whether any node of one SCC points to any node of another SCC
        """
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

    def cycle_with_length(self, length: int) -> list[Node]:
        try:
            length = int(length)
        except TypeError:
            raise TypeError("Integer expected!")
        if length < 2:
            return []
        tmp = DirectedGraph.copy(self)
        for u, v in tmp.links:
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

        if (n := len(self.nodes)) == 1 or len(self.links) > (n - 1) ** 2 or all(
                sum(self.degrees(u)) >= n for u in self.nodes):
            return True
        if self.sources or self.sinks:
            return False
        tmp = DirectedGraph.copy(self)
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
        return DirectedGraph.copy(self).connect(u, [v]).hamilton_tour_exists()

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

        tmp = DirectedGraph.copy(self)
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

    def isomorphic_bijection(self, other: "DirectedGraph") -> dict[Node, Node]:
        return isomorphic_bijection_directed(self, other)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __reversed__(self) -> "DirectedGraph":
        return self.transposed()

    def __contains__(self, u: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        return u in self.nodes

    def __add__(self, other: "DirectedGraph") -> "DirectedGraph":
        """
        Args:
            other: another DirectedGraph object
        Returns:
            Combination of two directed graphs
        """
        return combine_directed(self, other)

    def __eq__(self, other: "DirectedGraph") -> bool:
        return compare(self, other)

    def __str__(self) -> str:
        return string(self)

    def __repr__(self) -> str:
        return str(self)


class WeightedNodesDirectedGraph(DirectedGraph):
    """
    Class for implementing a directed graph with node weights
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, tuple[Iterable[Node], Iterable[Node]]]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple with 2 elements. The first one is the node's weight. The second one is a tuple with 2 sets of nodes. The first one is the nodes, that point to it, and the second one are the nodes it points to
        """
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
        """
        Args:
            n: A present node or None
        Returns:
            The weight of node n or the dictionary with all node weights
        """
        if n is not None and not isinstance(n, Node):
            n = Node(n)
        return self.__node_weights.copy() if n is None else self.__node_weights[n]

    @property
    def total_nodes_weight(self) -> float:
        """
        Returns:
            The sum of all node weights
        """
        return sum(self.node_weights().values())

    def add(self, n_w: tuple[Node, float], pointed_by: Iterable[Node] = (),
            points_to: Iterable[Node] = ()) -> "WeightedNodesDirectedGraph":
        DirectedGraph.add(self, n_w[0], pointed_by, points_to)
        n = n_w[0] if isinstance(n_w[0], Node) else Node(n_w[0])
        if n not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedNodesDirectedGraph":
        for n in {u, *rest}:
            if not isinstance(n, Node):
                n = Node(n)
            if n in self.node_weights():
                self.__node_weights.pop(n)
        DirectedGraph.remove(self, u, *rest)
        return self

    def set_weight(self, u: Node, w: float) -> "WeightedNodesDirectedGraph":
        """
        Args:
            u: A present node
            w: The new weight of node u
        Set the weight of node u to w
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            try:
                self.__node_weights[u] = float(w)
            except ValueError:
                raise TypeError("Real value expected!")
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedNodesDirectedGraph":
        """
        Args:
            u: A present node
            w: A real value
        Increase the weight of node u by w
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self.node_weights():
            try:
                self.set_weight(u, self.node_weights(u) + float(w))
            except ValueError:
                raise TypeError("Real value expected!")
        return self

    def copy(self) -> "WeightedNodesDirectedGraph":
        return WeightedNodesDirectedGraph({n: (self.node_weights(n), ([], self.next(n))) for n in self.nodes})

    def complementary(self) -> "WeightedNodesDirectedGraph":
        res = WeightedNodesDirectedGraph({n: (self.node_weights(n), ([], self.nodes)) for n in self.nodes})
        for l in self.links:
            res.disconnect(l[1], [l[0]])
        return res

    def transposed(self) -> "WeightedNodesDirectedGraph":
        return WeightedNodesDirectedGraph({u: (self.node_weights(u), (self.next(u), [])) for u in self.nodes})

    def weighted_graph(self, weights: dict[tuple[Node, Node], float] = None) -> "WeightedDirectedGraph":
        if weights is None:
            weights = {l: 0 for l in self.links}
        for l in self.links - set(weights):
            weights[l] = 0
        return super().weighted_graph(self.node_weights(), weights)

    def undirected(self) -> "WeightedNodesUndirectedGraph":
        neighborhood = {n: (self.node_weights(n), self.prev(n).union(self.next(n))) for n in self.nodes}
        return WeightedNodesUndirectedGraph(neighborhood)

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "WeightedNodesDirectedGraph":
        try:
            u_or_nodes = self.nodes.intersection(u_or_nodes)
            neighborhood = {u: (self.node_weights(u), ([], self.next(u).intersection(u_or_nodes))) for u in u_or_nodes}
            return WeightedNodesDirectedGraph(neighborhood)
        except TypeError:
            if not isinstance(u_or_nodes, Node):
                u_or_nodes = Node(u_or_nodes)
            if u_or_nodes not in self:
                raise KeyError("Unrecognized node!")
            queue = [u_or_nodes]
            res = WeightedNodesDirectedGraph({u_or_nodes: (self.node_weights(u_or_nodes), ([], []))})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, [v])
                    else:
                        res.add((n, self.node_weights(n)), [v]), queue.append(n)
            return res

    def scc_dag(self) -> "WeightedNodesDirectedGraph":
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected components, and the links of which are according to whether any node of one SCC points to any node of another SCC
        """
        result = WeightedNodesDirectedGraph()
        scc = self.strongly_connected_components()
        for s in scc:
            result.add((Node(frozenset(s)), sum(map(self.node_weights, s))))
        for u in result.nodes:
            for v in result.nodes:
                if u != v:
                    for x in u.value:
                        if any(y in v.value for y in self.next(x)):
                            result.connect(v, [u])
        return result

    def minimal_path_nodes(self, u: Node, v: Node) -> list[Node]:
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

    def __init__(self, neighborhood: dict[Node, tuple[dict[Node, float], dict[Node, float]]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple with 2 dictionaries. The first one contains the nodes, which point to it, and the second one contains the nodes it points to. The values in these dictionaries are the link weights
        """
        super().__init__()
        self.__link_weights = {}
        for u, (prev_pairs, next_pairs) in neighborhood.items():
            if u not in self:
                self.add(u)
            for v, w in prev_pairs.items():
                self.add(v, points_to_weights={u: w}), self.connect(u, {v: w})
            for v, w in next_pairs.items():
                self.add(v, {u: w}), self.connect(v, {u: w})

    def link_weights(self, u_or_l: Node | tuple = None, v: Node = None) -> dict[Node, float] | dict[
        tuple[Node, Node], float] | float:
        """
        Args:
            u_or_l: Given first node, a link or None
            v: Given second node or None
        Returns:
            Information about link weights the following way:
            If no argument is passed, return the weights of all links;
            if a link or two nodes are passed, return the weight of the given link between them;
            If one node is passed, return a dictionary with all nodes it points to and the weight of the link from .that node to each of them
        """
        if u_or_l is None:
            return self.__link_weights.copy()
        elif isinstance(u_or_l, tuple):
            return self.__link_weights.get(u_or_l)
        else:
            if not isinstance(u_or_l, Node):
                u_or_l = Node(u_or_l)
            if v is None:
                return {n: self.link_weights((u_or_l, n)) for n in self.next(u_or_l)}
            if not isinstance(v, Node):
                v = Node(v)
            return self.__link_weights[(u_or_l, v)]

    @property
    def total_links_weight(self) -> float:
        """
        Returns:
            The sum of all link weights
        """
        return sum(self.link_weights().values())

    def add(self, u: Node, pointed_by_weights: dict[Node, float] = {},
            points_to_weights: dict[Node, float] = {}) -> "WeightedLinksDirectedGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            DirectedGraph.add(self, u), self.connect(u, pointed_by_weights, points_to_weights)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedLinksDirectedGraph":
        for n in {u, *rest}:
            if not isinstance(n, Node):
                n = Node(n)
            if n in self:
                for v in self.next(n):
                    if (n, v) in self.link_weights():
                        self.__link_weights.pop((n, v))
                for v in self.prev(n):
                    if (v, n) in self.link_weights():
                        self.__link_weights.pop((v, n))
        return super().remove(u, *rest)

    def connect(self, u: Node, pointed_by_weights: dict[Node, float] = {},
                points_to_weights: dict[Node, float] = {}) -> "WeightedLinksDirectedGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            super().connect(u, pointed_by_weights.keys(), points_to_weights.keys())
            for v, w in pointed_by_weights.items():
                if not isinstance(v, Node):
                    v = Node(v)
                if (v, u) not in self.link_weights():
                    self.set_weight((v, u), w)
            for v, w in points_to_weights.items():
                if not isinstance(v, Node):
                    v = Node(v)
                if (u, v) not in self.link_weights():
                    self.set_weight((u, v), w)
        return self

    def connect_all(self, u: Node, *rest: Node) -> "WeightedLinksDirectedGraph":
        if not rest:
            return self
        self.connect(u, (d := {v: 0 for v in rest}), d)
        return self.connect_all(*rest)

    def disconnect(self, u: Node, pointed_by: Iterable[Node] = (),
                   points_to: Iterable[Node] = ()) -> "WeightedLinksDirectedGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            for v in pointed_by:
                if not isinstance(v, Node):
                    v = Node(v)
                if v in self.prev(u):
                    self.__link_weights.pop((v, u))
            for v in points_to:
                if not isinstance(v, Node):
                    v = Node(v)
                if v in self.next(u):
                    self.__link_weights.pop((u, v))
            super().disconnect(u, pointed_by, points_to)
        return self

    def set_weight(self, l: tuple, w: float) -> "WeightedLinksDirectedGraph":
        """
        Args:
            l: A present link
            w: The new weight of link l
        Set the weight of link l to w
        """
        try:
            l = tuple(l)
            if len(l) != 2:
                raise ValueError("Directed link expected!")
            l = (l[0] if isinstance(l[0], Node) else Node(l[0]), l[1] if isinstance(l[1], Node) else Node(l[1]))
            if l in self.links:
                try:
                    self.__link_weights[l] = float(w)
                except TypeError:
                    raise TypeError("Real value expected!")
            return self
        except ValueError:
            raise TypeError("Directed link is of type tuple[Node, Node]!")

    def increase_weight(self, l: tuple[Node, Node], w: float) -> "WeightedLinksDirectedGraph":
        """
        Args:
            l: A present link
            w: A real value
        Increase the weight of link l with w
        """
        try:
            l = tuple(l)
            if len(l) != 2:
                raise ValueError("Directed link expected!")
            l = (l[0] if isinstance(l[0], Node) else Node(l[0]), l[1] if isinstance(l[1], Node) else Node(l[1]))
            if l in self.link_weights():
                try:
                    self.set_weight(l, self.link_weights(l) + float(w))
                except TypeError:
                    raise TypeError("Real value expected!")
            return self
        except ValueError as t:
            if "Real value expected!" in t.args:
                raise t
            raise TypeError("Directed link is of type tuple[Node, Node]!")

    def copy(self) -> "WeightedLinksDirectedGraph":
        return WeightedLinksDirectedGraph({u: ({}, self.link_weights(u)) for u in self.nodes})

    def transposed(self) -> "WeightedLinksDirectedGraph":
        return WeightedLinksDirectedGraph({u: (self.link_weights(u), {}) for u in self.nodes})

    def weighted_graph(self, weights: dict[Node, float] = None) -> "WeightedDirectedGraph":
        if weights is None:
            weights = {n: 0 for n in self.nodes}
        for n in self.nodes - set(weights):
            weights[n] = 0
        return WeightedDirectedGraph(
            {u: (weights[u], ({}, {v: self.link_weights(u, v) for v in self.next(u)})) for u in self.nodes})

    def undirected(self) -> "WeightedLinksUndirectedGraph":
        res = WeightedLinksUndirectedGraph({n: {} for n in self.nodes})
        for u, v in self.links:
            if v in res.neighbors(u):
                res.increase_weight(Link(u, v), self.link_weights(u, v))
            else:
                res.connect(u, {v: self.link_weights(u, v)})
        return res

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "WeightedLinksDirectedGraph":
        try:
            u_or_nodes = self.nodes.intersection(u_or_nodes)
            neighborhood = {u: ({}, {k: v for k, v in self.link_weights(u).items() if k in u_or_nodes}) for u in
                            u_or_nodes}
            return WeightedLinksDirectedGraph(neighborhood)
        except TypeError:
            if not isinstance(u_or_nodes, Node):
                u_or_nodes = Node(u_or_nodes)
            if u_or_nodes not in self:
                raise KeyError("Unrecognized node!")
            queue, res = [u_or_nodes], WeightedLinksDirectedGraph({u_or_nodes: ({}, {})})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, {v: self.link_weights(v, n)})
                    else:
                        res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
            return res

    def scc_dag(self) -> "WeightedLinksDirectedGraph":
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected components, and the links of which are according to whether any node of one SCC points to any node of another SCC
        """
        result = WeightedLinksDirectedGraph()
        scc = self.strongly_connected_components()
        for s in scc:
            result.add(Node(frozenset(s)))
        for u in result.nodes:
            for v in result.nodes:
                if u != v:
                    for x in u.value:
                        if any(y in v.value for y in self.next(x)):
                            result.connect(v, {u: 0})
                            for y in self.next(x):
                                if y in v.value:
                                    result.increase_weight((u, v), self.link_weights(x, y))
        return result

    def minimal_path_links(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path from .u to v with the least possible sum of link weights
        """
        return self.weighted_graph().minimal_path(u, v)


class WeightedDirectedGraph(WeightedLinksDirectedGraph, WeightedNodesDirectedGraph):
    """
    Class for implementing directed graph with weights on the nodes and the links
    """

    def __init__(self,
                 neighborhood: dict[Node, tuple[float, tuple[dict[Node, float], dict[Node, float]]]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary with nodes for keys. The value of each node is a tuple with its weight and another tuple with 2 dictionaries. The first one contains the nodes, which point to it, and the second one contains the nodes it points to. The values in these dictionaries are the link weights
        """
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

    def add(self, n_w: tuple[Node, float], pointed_by_weights: dict[Node, float] = {},
            points_to_weights: dict[Node, float] = {}) -> "WeightedDirectedGraph":
        super().add(n_w[0], pointed_by_weights, points_to_weights)
        n = n_w[0] if isinstance(n_w[0], Node) else Node(n_w[0])
        if n not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedDirectedGraph":
        for n in {u, *rest}:
            if n in self:
                super().disconnect(n, self.prev(n), self.next(n))
        return WeightedNodesDirectedGraph.remove(self, u, *rest)

    def set_weight(self, el: Node | tuple, w: float) -> "WeightedDirectedGraph":
        """
        Args:
            el: A present node or link
            w: The new weight of object el
        Set the weight of object el to w
        """
        if el in self.links:
            super().set_weight(el, w)
        else:
            if not isinstance(el, Node):
                el = Node(el)
            if el in self:
                WeightedNodesDirectedGraph.set_weight(self, el, w)
        return self

    def increase_weight(self, el: Node | tuple[Node, Node], w: float) -> "WeightedDirectedGraph":
        """
        Args:
            el: A present node or link
            w: A real value
        Increase the weight of object el with w
        """
        if el in self.link_weights():
            try:
                self.set_weight(el, self.link_weights(el) + float(w))
            except TypeError:
                raise TypeError("Real value expected!")
            return self
        if not isinstance(el, Node):
            el = Node(el)
        if el in self.node_weights():
            try:
                return self.set_weight(el, self.node_weights(el) + float(w))
            except ValueError:
                raise TypeError("Real value expected!")
        return self

    def copy(self) -> "WeightedDirectedGraph":
        neighborhood = {u: (self.node_weights(u), ({}, self.link_weights(u))) for u in self.nodes}
        return WeightedDirectedGraph(neighborhood)

    def transposed(self) -> "WeightedDirectedGraph":
        neighborhood = {u: (self.node_weights(u), (self.link_weights(u), {})) for u in self.nodes}
        return WeightedDirectedGraph(neighborhood)

    def undirected(self) -> "WeightedUndirectedGraph":
        res = WeightedUndirectedGraph({n: (self.node_weights(n), {}) for n in self.nodes})
        for u, v in self.links:
            if v in res.neighbors(u):
                res.increase_weight(Link(u, v), self.link_weights(u, v))
            else:
                res.connect(u, {v: self.link_weights(u, v)})
        return res

    def subgraph(self, u_or_nodes: Node | Iterable[Node]) -> "WeightedDirectedGraph":
        try:
            u_or_nodes = self.nodes.intersection(u_or_nodes)
            neighborhood = {
                u: (self.node_weights(u), ({}, {k: v for k, v in self.link_weights(u).items() if k in u_or_nodes})) for
                u in u_or_nodes}
            return WeightedDirectedGraph(neighborhood)
        except TypeError:
            if not isinstance(u_or_nodes, Node):
                u_or_nodes = Node(u_or_nodes)
            if u_or_nodes not in self:
                raise KeyError("Unrecognized node!")
            queue, res = [u_or_nodes], WeightedDirectedGraph({u_or_nodes: (self.node_weights(u_or_nodes), ({}, {}))})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, {v: self.link_weights(v, n)})
                    else:
                        res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), queue.append(n)
            return res

    def scc_dag(self) -> "WeightedDirectedGraph":
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected components, and the links of which are according to whether any node of one SCC points to any node of another SCC
        """
        result = WeightedDirectedGraph()
        scc = self.strongly_connected_components()
        for s in scc:
            result.add((Node(frozenset(s)), sum(map(self.node_weights, s))))
        for u in result.nodes:
            for v in result.nodes:
                if u != v:
                    for x in u.value:
                        if any(y in v.value for y in self.next(x)):
                            result.connect(v, {u: 0})
                            for y in self.next(x):
                                if y in v.value:
                                    result.increase_weight((u, v), self.link_weights(x, y))
        return result

    def minimal_path(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path between u and v with the least possible sum of node and link weights
        """

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
                            if (new_weight := weights_from_to[x1][x2] + weights_from_to[x2][x3]) < weights_from_to[x1][
                                x3]:
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

        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if v in self:
            if v in (tmp := self.subgraph(u)):
                nodes_negative_weights = sum(tmp.node_weights(n) for n in tmp.nodes if tmp.node_weights(n) < 0)
                links_negative_weights = sum(tmp.link_weights(l) for l in tmp.links if tmp.link_weights(l) < 0)
                res = dfs(u, [], tmp.node_weights(u), nodes_negative_weights + links_negative_weights)
                return [l[0] for l in res[0]] + [res[0][-1][1]]
            return []
        raise KeyError("Unrecognized node(s)!")
