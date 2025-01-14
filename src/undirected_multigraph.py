"""
Module for implementing undirected graphs.
"""

from .undirected_graph import (Node, Link, UndirectedGraph, defaultdict, product, permutations,
                               Iterable)


class UndirectedMultiGraph:
    """
    Class for implementing an unweighted undirected multi-graph.
    """

    def __init__(self, neighborhood: dict[Node, dict[Node, int]] = {}) -> None:
        self.__nodes, self.__links = set(), defaultdict(int)
        self.__neighbors = {}
        for u, neighbors in neighborhood.items():
            if u not in self:
                self.add(u)
            for v in neighbors:
                if v not in self:
                    self.add(v)
        for u, neighbors in neighborhood.items():
            self.connect(u, neighbors)

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes.copy()

    @property
    def links(self) -> dict[Link, int]:
        return self.__links.copy()

    @property
    def degrees_sum(self) -> int:
        return 2 * sum(self.links.values())

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Graph leaves.
        """
        return {n for n in self.nodes if self.leaf(n)}

    def neighbors(self, u: Node = None) -> set[Node] | dict[Node, set[Node]]:
        if u is None:
            return {n: self.neighbors(n) for n in self.__nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return set(self.__neighbors[u].keys())

    def degrees(self, u: Node = None) -> int | dict[Node, int]:
        if u is None:
            return {n: self.degrees(n) for n in self.__nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return sum(self.__neighbors[u].values())

    def leaf(self, n: Node) -> bool:
        """
        Args:
            n: Node object.
        Returns:
            Whether n is a leaf (if it has a degree of 1).
        """
        if not isinstance(n, Node):
            n = Node(n)
        return self.degrees(n) == 1

    def add(self, u: Node, connections: dict[Node, int] = {}) -> "UndirectedMultiGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            self.__nodes.add(u)
            self.__neighbors[u] = defaultdict(int)
            UndirectedMultiGraph.connect(self, u, connections)
        return self

    def remove(self, n: Node, *rest: Node) -> "UndirectedMultiGraph":
        for u in (n, *rest):
            if not isinstance(u, Node):
                u = Node(u)
            if u in self:
                if tmp := self.neighbors(u):
                    res = {}
                    for v in tmp:
                        res[v] = self.__neighbors[u][v]
                    UndirectedMultiGraph.disconnect(self, u, res)
                self.__nodes.remove(u), self.__neighbors.pop(u)
        return self

    def connect(self, u: Node, rest: dict[Node, int]) -> "UndirectedMultiGraph":
        if not isinstance(u, Node):
            u = Node(u)
        for v, num in rest.items():
            if not isinstance(v, Node):
                v = Node(v)
            num = max(0, num)
            if u != v and v in self:
                self.__neighbors[u][v] += num
                self.__neighbors[v][u] += num
                self.__links[Link(u, v)] += num
        return self

    def connect_all(self, u: Node, *rest: Node) -> "UndirectedMultiGraph":
        if not rest:
            return self
        self.connect(u, {v: 1 for v in rest})
        return self.connect_all(*rest)

    def disconnect(self, u: Node, rest: dict[Node, int]) -> "UndirectedMultiGraph":
        if not isinstance(u, Node):
            u = Node(u)
        for v, num in rest.items():
            if not isinstance(v, Node):
                v = Node(v)
            num = max(0, num)
            if v in self.neighbors(u):
                self.__neighbors[u][v] = self.__neighbors[u][v] - num
                self.__neighbors[v][u] = self.__neighbors[v][u] - num
                self.__links[Link(u, v)] -= num
                if self.links[Link(u, v)] <= 0:
                    self.__neighbors[u].pop(v)
                    self.__neighbors[v].pop(u)
                    self.__links.pop(Link(u, v))
        return self

    def disconnect_all(self, u: Node, *rest: Node) -> "UndirectedMultiGraph":
        if not rest:
            return self
        self.disconnect(u, {v: self.links[Link(u, v)] for v in rest})
        return self.disconnect_all(*rest)

    def copy(self) -> "UndirectedMultiGraph":
        neighborhood = {}
        for u in self.nodes:
            neighborhood[u] = sum([[v] * self.__neighbors[u][v] for v in self.neighbors(u)], [])
        return UndirectedMultiGraph(neighborhood)

    def excentricity(self, u: Node) -> int:
        """
        Args:
            u: Node, present in the graph.
        Returns:
            Excentricity of u (the length of the longest of all shortest paths, starting from it).
        """
        if not isinstance(u, Node):
            u = Node(u)
        res, total, layer = -1, {u}, [u]
        while layer:
            new = []
            while layer:
                for v in self.neighbors(_ := layer.pop(0)) - total:
                    new.append(v), total.add(v)
            layer = new.copy()
            res += 1
        return res

    def diameter(self) -> int:
        """
        Returns:
            The greatest of all excentricity values in the graph.
        """
        return max(self.excentricity(u) for u in self.nodes)

    def non_multi_graph(self) -> UndirectedGraph:
        return UndirectedGraph(self.neighbors())

    def complementary(self) -> UndirectedGraph:
        return self.non_multi_graph().complementary()

    def connected(self) -> bool:
        return self.non_multi_graph().connected()

    def cycle_with_length_3(self) -> list[Node]:
        """
        Returns:
            A cycle with a length of 3 if such exists, otherwise an empty list.
        """
        for l in self.links:
            if intersection := self.neighbors(u := l.u).intersection(self.neighbors(v := l.v)):
                return [u, v, intersection.pop(), u]
        return []

    def planar(self) -> bool:
        """
        Returns:
            Whether the graph is planar (whether it could be drawn on a
            flat surface without intersecting).
        """
        for tmp in self.connection_components():
            if len(tmp.links) > (2 + bool(tmp.cycle_with_length_3())) * (len(tmp.nodes) - 2):
                return False
        return True

    def reachable(self, u: Node, v: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)!")
        if v in {u, *self.neighbors(u)}:
            return True
        return v in self.component(u)

    def subgraph(self, nodes: Iterable[Node]) -> "UndirectedMultiGraph":
        try:
            nodes = list(filter(lambda n: n in self, nodes))
            neighborhood = {}
            for u in nodes:
                for v in self.neighbors(u).intersection(nodes):
                    neighborhood[u] += [v] * self.links[Link(u, v)]
            return UndirectedMultiGraph(neighborhood)
        except TypeError:
            raise TypeError("Iterable of nodes expected!")

    def component(self, u: Node) -> "UndirectedMultiGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        queue, total = [u], {u}
        while queue:
            queue += list((next_nodes := self.neighbors(queue.pop(0)) - total))
            total.update(next_nodes)
        return self.subgraph(total)

    def connection_components(self) -> list["UndirectedMultiGraph"]:
        components, rest = [], self.nodes
        while rest:
            components.append(curr := self.component(rest.pop()))
            rest -= curr.nodes
        return components

    def cut_nodes(self) -> set[Node]:
        """
        A cut node in a graph is such, that if it's removed,
        the graph splits into more connection components.
        Returns:
            All cut nodes.
        """
        return self.non_multi_graph().cut_nodes()

    def bridge_links(self) -> set[Link]:
        """
        A bridge link is such, that if it's removed from the graph,
        it splits into one more connection component.
        Returns:
            All bridge links.
        """
        return set(filter(lambda l: self.links[l] == 1, self.non_multi_graph().bridge_links()))

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
                res, curr_node = [n], n
                while curr_node != u:
                    res.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]
                return res
            for m in self.neighbors(n) - total:
                queue.append(m), total.add(m)
                previous[m] = n
        return []

    def euler_tour_exists(self) -> bool:
        for n in self.nodes:
            if self.degrees(n) % 2:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)!")
        if u == v:
            return self.euler_tour_exists()
        for n in self.nodes:
            if self.degrees(n) % 2 - (n in {u, v}):
                return False
        return self.connected()

    def euler_tour(self) -> list[Node]:
        if self.euler_tour_exists():
            tmp = UndirectedMultiGraph.copy(self)
            return tmp.disconnect(u := (l := set(tmp.links.keys()).pop()).u,
                                  {(v := l.v): 1}).euler_walk(u, v) + [u]
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
            tmp = UndirectedMultiGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i], {path[i + 1]: 1})
            for i, u in enumerate(path):
                while tmp.neighbors(u):
                    curr = tmp.disconnect(u, {(v := tmp.neighbors(u).pop()): 1}).get_shortest_path(
                        v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j], {curr[j + 1]: 1})
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def links_graph(self) -> "UndirectedMultiGraph":
        nodes: set[Node(tuple[Link, int])] = set()
        for l, num in self.links.items():
            for k in range(num):
                nodes.add(Node((l, k)))
        neighborhood: dict[Node, dict[Node, int]] = {}
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if {u.value[0].u, u.value[0].v}.intersection({v.value[0].u, v.value[0].v}):
                    neighborhood[u][v] += 1
        return UndirectedMultiGraph(neighborhood)

    def all_maximal_cliques_node(self, u: Node) -> list[set[Node]]:
        """
        Args:
            u: A present node.
        Returns:
            All maximal by inclusion cliques in the graph, to which node u belongs.
        """
        if not isinstance(u, Node):
            u = Node(u)
        g = self.subgraph(self.neighbors(u)).complementary()
        return list(map({u}.union, g.maximal_independent_sets()))

    def maximal_independent_sets(self) -> list[set[Node]]:
        """
        Returns:
            All maximal by inclusion independent sets in the graph.
        """

        def generator(curr=set(), total=set(), i=0):
            for j, n in enumerate(list(self.nodes)[i:]):
                if curr.isdisjoint(self.neighbors(n)):
                    for res in generator({n, *curr}, {n, *self.neighbors(n), *total}, i + j + 1):
                        yield res
            if total == self.nodes:
                yield curr

        return [i_s for i_s in generator()]

    def chromatic_nodes_partition(self) -> list[set[Node]]:
        """
        Returns:
            A list of independent sets in the graph, that cover all nodes without intersecting.
            This list has as few elements as possible.
        """
        return self.non_multi_graph().chromatic_nodes_partition()

    def chromatic_links_partition(self) -> list[set[Link]]:
        """
        Similar to chromatic nodes partition, except links,
        that share a node, are in different sets.
        """
        return [set(map(lambda x: x.value[0], s)) for s in
                UndirectedMultiGraph.links_graph(self).chromatic_nodes_partition()]

    def vertex_cover(self) -> set[Node]:
        return self.non_multi_graph().vertex_cover()

    def dominating_sets(self) -> set[Node]:
        return self.non_multi_graph().dominating_set()

    def independent_set(self) -> set[Node]:
        return self.non_multi_graph().independent_set()

    def cycle_with_length(self, length: int) -> list[Node]:
        try:
            length = int(length)
        except TypeError:
            raise TypeError("Integer expected!")
        if length < 2:
            return []
        if length == 2:
            for l, num in self.links.items():
                if num > 1:
                    return [l.u, l.v]
            return []
        if length == 3:
            return self.cycle_with_length_3()
        tmp = UndirectedMultiGraph.copy(self)
        for l in tmp.links:
            res = tmp.disconnect(u := l.u, {(v := l.v): 1}).path_with_length(v, u, length - 1)
            if res:
                return res + [u]
            tmp.connect(u, {v: 1})
        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        def dfs(x: Node, l: int, stack: list[Link]):
            if not l:
                return list(map(lambda link: link.u, stack)) + [v] if x == v else []
            for y in filter(lambda _x: Link(x, _x) not in stack, self.neighbors(x)):
                if res := dfs(y, l - 1, stack + [Link(x, y)]):
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
        if not (tmp := self.get_shortest_path(u, v)) or (k := len(tmp)) > length + 1:
            return []
        if length + 1 == k:
            return tmp
        return dfs(u, length, [])

    def hamilton_tour_exists(self) -> bool:
        return self.non_multi_graph().hamilton_tour_exists()

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        return self.non_multi_graph().hamilton_walk_exists(u, v)

    def hamilton_tour(self) -> list[Node]:
        return self.non_multi_graph().hamilton_tour()

    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        return self.non_multi_graph().hamilton_walk(u, v)

    def isomorphic_bijection(self, other) -> dict[Node, Node]:
        if isinstance(other, UndirectedMultiGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].add(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].add(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d])
                   for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(map(list, this_nodes_degrees.values()), key=len)
            other_nodes_degrees = sorted(map(list, other_nodes_degrees.values()), key=len)
            for possibility in product(*map(permutations, this_nodes_degrees)):
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_degrees, [])
                map_dict = dict(zip(flatten_self, flatten_other))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if self.links[Link(n, m)] != other.links[Link(u, v)]:
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

    def __contains__(self, u: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        return u in self.nodes

    def __add__(self, other: "UndirectedMultiGraph") -> "UndirectedMultiGraph":
        """
        Args:
            other: another UndirectedMultiGraph object.
        Returns:
            Combination of two undirected multi-graphs.
        """
        if isinstance(other, UndirectedGraph):
            other = UndirectedMultiGraph(
                {u: {v: 1 for v in other.neighbors(u)} for u in other.nodes})
        if not isinstance(other, UndirectedMultiGraph):
            raise TypeError(f"Addition not defined between class "
                            f"UndirectedMultiGraph and type {type(other).__name__}!")
        if isinstance(other,
                      (WeightedNodesUndirectedMultiGraph, WeightedLinksUndirectedMultiGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            res.add(n)
        for l in other.links:
            res.connect(l.u, {l.v: other.links[l]})
        return res

    def __eq__(self, other: "UndirectedMultiGraph") -> bool:
        if type(other) == UndirectedMultiGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self) -> str:
        return f"<{self.nodes}, {self.links}>"

    __repr__: str = __str__


class WeightedNodesUndirectedMultiGraph(UndirectedMultiGraph):
    ...


class WeightedLinksUndirectedMultiGraph(UndirectedMultiGraph):
    ...


class WeightedMultiGraph(WeightedLinksUndirectedMultiGraph, WeightedNodesUndirectedMultiGraph):
    ...
