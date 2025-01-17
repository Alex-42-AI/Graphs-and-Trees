"""
Module for implementing directed multi-graphs.
"""

from .base import MultiGraph

from .undirected_multigraph import UndirectedMultiGraph

from .directed_graph import Node, DirectedGraph, Iterable, product, permutations, defaultdict


class DirectedMultiGraph(MultiGraph):
    """
    Class for implementing an unweighted directed multi-graph.
    """

    def __init__(self,
                 neighborhood: dict[Node, tuple[dict[Node, int], dict[Node, int]]] = {}) -> None:
        self.__nodes, self.__links = set(), defaultdict(int)
        self.__prev, self.__next = defaultdict(dict[Node, int]), defaultdict(dict[Node, int])
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
        return self.__nodes.copy()

    @property
    def links(self) -> dict[tuple[Node, Node], int]:
        return self.__links.copy()

    def degrees(self, u: Node = None) -> dict[Node, tuple[int, int]] | tuple[int, int]:
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
            return {n: self.next(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return set(self.__next[u].keys())

    def prev(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None.
        Returns:
            A set of all nodes, that point to node u, if it's given,
            otherwise the same for all nodes.
        """
        if u is None:
            return {n: self.prev(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return set(self.__prev[u].keys())

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

    def add(self, u: Node, pointed_by: dict[Node, int] = {},
            points_to: dict[Node, int] = {}) -> "DirectedMultiGraph":
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

    def connect(self, u: Node, pointed_by: dict[Node, int] = {},
                points_to: dict[Node, int] = {}) -> "DirectedMultiGraph":
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
                    if not self.links[(v, u)]:
                        self.__prev[u][v] = 0
                        self.__next[v][u] = 0
                    self.__links[(v, u)] += num
                    self.__prev[u][v] += num
                    self.__next[v][u] += num
            for v, num in points_to.items():
                if not isinstance(v, Node):
                    v = Node(v)
                if u != v and v in self:
                    num = max(0, num)
                    if not self.links[(u, v)]:
                        self.__prev[v][u] = 0
                        self.__next[u][v] = 0
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
            neighborhood = {
                u: ({}, {v: self.links[(u, v)] for v in self.next(u).intersection(u_or_nodes)}) for
                u in u_or_nodes}
            return DirectedMultiGraph(neighborhood)
        except TypeError:
            if not isinstance(u_or_nodes, Node):
                u_or_nodes = Node(u_or_nodes)
            if u_or_nodes not in self:
                raise KeyError("Unrecognized node!")
            queue, res = [u_or_nodes], DirectedMultiGraph({u_or_nodes: ({}, {})})
            while queue:
                for n in self.next(v := queue.pop(0)):
                    if n in res:
                        res.connect(n, {v: self.links[(v, n)]})
                    else:
                        res.add(n, {v: self.links[(v, n)]}), queue.append(n)
            return res

    def has_cycle(self) -> bool:
        """
        Returns:
            Whether the graph has a cycle.
        """
        return self.non_multi_graph().has_cycle()

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
        return self.non_multi_graph().toposort()

    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        return self.non_multi_graph().get_shortest_path(u, v)

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
                    curr = tmp.disconnect(v := tmp.next(u).pop(), {u: 1}).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j + 1], {curr[j]: 1})
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def weighted_nodes_graph(self, weights: dict[
        Node, float] = None) -> "WeightedNodesDirectedMultiGraph":
        ...

    def weighted_links_graph(self, weights: dict[
        tuple[Node, Node], float]) -> "WeightedLinksDirectedMultiGraph":
        ...

    def weighted_graph(self, node_weights: dict[Node, float] = None, link_weights: dict[
        "Link" | tuple[Node, Node], float] = None) -> "MultiGraph":
        ...

    def strongly_connected_component(self, n: Node) -> set[Node]:
        """
        Args:
            n: A present node.
        Returns:
            The maximal by inclusion strongly-connected component, to which a given node belongs.
        A strongly-connected component is a set of nodes, where
        there exists a path from every node to every other node.
        """
        return self.non_multi_graph().strongly_connected_component(n)

    def strongly_connected_components(self) -> list[set[Node]]:
        """
        Returns:
            A list of all strongly-connected components of the graph.
        """
        return self.non_multi_graph().strongly_connected_components()

    def scc_dag(self) -> "DirectedGraph":
        """
        Returns:
            The DAG, the nodes of which are the individual strongly-connected
            components, and the links of which are according to whether any
            node of one SCC points to any node of another SCC.
        """
        return self.non_multi_graph().scc_dag()

    def cycle_with_length(self, length: int) -> list[Node]:
        return self.non_multi_graph().cycle_with_length(length)

    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        return self.non_multi_graph().path_with_length(u, v, length)

    def hamilton_tour_exists(self) -> bool:
        return self.non_multi_graph().hamilton_tour_exists()

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        return self.non_multi_graph().hamilton_walk_exists(u, v)

    def hamilton_tour(self) -> list[Node]:
        return self.non_multi_graph().hamilton_tour()

    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        return self.non_multi_graph().hamilton_walk(u, v)

    def isomorphic_bijection(self, other: "DirectedMultiGraph") -> dict[Node, Node]:
        if isinstance(other, DirectedMultiGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(
                    other.nodes) or sorted(self.links.values()) != sorted(other.links.values()):
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
                        if self.links[(n, m)] != other.links[(u, v)]:
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

    def __reversed__(self) -> "DirectedGraph":
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
            Combination of two directed multi-graphs.
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
        for l, num in other.links.items():
            res.connect(l[1], {l[0]: num})
        return res

    def __eq__(self, other: "DirectedMultiGraph") -> bool:
        if type(other) == DirectedMultiGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self) -> str:
        return "<" + str(self.nodes) + ", {" + ", ".join(
            f"<{l[0]}, {l[1]}>: {self.links[l]}" for l in self.links) + "}>"

    __repr__: str = __str__


class WeightedNodesDirectedMultiGraph(DirectedMultiGraph):
    ...


class WeightedLinksDirectedMultiGraph(DirectedMultiGraph):
    ...


class WeightedDirectedMultiGraph(WeightedLinksDirectedMultiGraph, WeightedNodesDirectedMultiGraph):
    ...
