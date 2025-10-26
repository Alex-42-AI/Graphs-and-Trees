"""
Module for implementing undirected graphs
"""

from __future__ import annotations

from math import inf

from collections import defaultdict

from functools import reduce

from itertools import combinations

from base import Node, Link, Graph, Iterable, combine_undirected, isomorphic_bijection_undirected, compare, string, \
    Any, Path

__all__ = ["Node", "Link", "Graph", "UndirectedGraph", "WeightedNodesUndirectedGraph", "WeightedLinksUndirectedGraph",
           "WeightedUndirectedGraph", "reduce", "Iterable"]


def links_graph(graph: UndirectedGraph) -> UndirectedGraph:
    links = list(graph.links)

    if isinstance(graph, WeightedUndirectedGraph):
        result = WeightedUndirectedGraph({Node(l): (graph.link_weights(l), {}) for l in links})
        for i, l0 in enumerate(links):
            for l1 in links[i + 1:]:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): graph.node_weights(s.pop())})

        return result

    if isinstance(graph, WeightedNodesUndirectedGraph):
        result = WeightedLinksUndirectedGraph({Node(l): {} for l in links})
        for i, l0 in enumerate(links):
            for l1 in links[i + 1:]:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): graph.node_weights(s.pop())})

        return result

    if isinstance(graph, WeightedLinksUndirectedGraph):
        neighborhood = {
            Node(l0): (graph.link_weights(l0), [Node(l1) for l1 in links[i + 1:] if (l1.u in l0) ^ (l1.v in l0)])
            for i, l0 in enumerate(links)}

        return WeightedNodesUndirectedGraph(neighborhood)

    neighborhood = {Node(l0): [Node(l1) for l1 in links[i + 1:] if (l1.u in l0) or (l1.v in l0)] for i, l0 in
                    enumerate(links)}

    return UndirectedGraph(neighborhood)


def cliques_graph(graph: UndirectedGraph) -> UndirectedGraph:
    node_weights = isinstance(graph, WeightedNodesUndirectedGraph)
    result = WeightedUndirectedGraph() if node_weights else UndirectedGraph()
    cliques = graph.complementary().maximal_independent_sets()

    if node_weights:
        for u in cliques:
            result.add((Node(frozenset(u)), sum(map(graph.node_weights, u))))

        for i, u in enumerate(cliques):
            for v in cliques[i + 1:]:
                if common := u.intersection(v):
                    result.connect(Node(frozenset(u)), {Node(frozenset(v)): sum(map(graph.node_weights, common))})

    else:
        for u in cliques:
            result.add(Node(frozenset(u)))

        for i, u in enumerate(cliques):
            for v in cliques[i + 1:]:
                if u.intersection(v):
                    result.connect(Node(frozenset(u)), Node(frozenset(v)))

    return result


def complementary(graph: UndirectedGraph) -> UndirectedGraph:
    node_weights = isinstance(graph, WeightedNodesUndirectedGraph)
    res = UndirectedGraph({u: graph.nodes for u in graph.nodes})

    if node_weights:
        res = WeightedNodesUndirectedGraph({u: (graph.node_weights(u), graph.nodes) for u in graph.nodes})

    for l in graph.links:
        res.disconnect(l.u, l.v)

    return res


class UndirectedGraph(Graph):
    """
    Class for implementing an unweighted undirected graph
    """

    def __init__(self, neighborhood: dict[Node, Iterable[Node]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary that maps a node to its neighbors in the graph
        """

        if neighborhood is None:
            neighborhood = {}

        self.__nodes, self.__links = set(), set()
        self.__neighbors = {}

        for u, neighbors in neighborhood.items():
            if u not in self:
                self.add(u)

            for v in neighbors:
                self.add(v), self.connect(u, v)

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes.copy()

    @property
    def links(self) -> set[Link]:
        return self.__links.copy()

    @property
    def degrees_sum(self) -> int:
        """
        Returns:
            The total sum of all node degrees
        """

        return 2 * len(self.links)

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Graph leaves
        """

        return {n for n in self.nodes if self.leaf(n)}

    def neighbors(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: Node object or None
        Returns:
            Neighbors of u or dictionary of all nodes and their neighbors
        """

        if u is None:
            return self.__neighbors.copy()

        try:
            return self.__neighbors[Node(u)].copy()

        except KeyError:
            raise KeyError("Unrecognized node")

    def degree(self, u: Node = None) -> dict[Node, int] | int:
        if u is None:
            return {n: self.degree(n) for n in self.nodes}

        return len(self.neighbors(u))

    def leaf(self, n: Node) -> bool:
        """
        Args:
            n: Node object
        Returns:
            Whether n is a leaf (if it has a degree of 1)
        """

        return self.degree(n) == 1

    def simplicial(self, u: Node) -> bool:
        """
        A simplicial node in a graph is one, the neighbors of which are in a clique
        Args:
            u: Node object
        Returns:
            Whether u is simplicial
        """

        return self.clique(*self.neighbors(u))

    def add(self, u: Node, *current_nodes: Node) -> UndirectedGraph:
        """
        Args:
            u: a new node
            current_nodes: nodes already present in the graph
        Add node u to already present nodes
        """

        if (u := Node(u)) not in self:
            self.__nodes.add(u)
            self.__neighbors[u] = set()

            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)

        return self

    def remove(self, n: Node, *rest: Node) -> UndirectedGraph:
        for u in {n, *rest}:
            if (u := Node(u)) in self:
                if tmp := self.neighbors(u):
                    UndirectedGraph.disconnect(self, u, *tmp)

                self.__nodes.remove(u)
                self.__neighbors.pop(u)

        return self

    def connect(self, u: Node, v: Node, *rest: Node) -> UndirectedGraph:
        """
        Args:
            u: Node object
            v: Node object
            rest: Node objects
        Connect u to v and nodes in rest, all present in the graph
        """

        if (u := Node(u)) in self:
            for n in {v, *rest}:
                n = Node(n)

                if u != n and Link(n, u) not in self.links and n in self:
                    self.__links.add(Link(u, n))
                    self.__neighbors[u].add(n)
                    self.__neighbors[n].add(u)

        return self

    def connect_all(self, u: Node, *rest: Node) -> UndirectedGraph:
        if not rest:
            return self

        rest = set(rest) - {u}
        self.connect(u, *rest)

        return self.connect_all(*rest)

    def disconnect(self, u: Node, v: Node, *rest: Node) -> UndirectedGraph:
        """
        Args:
            u: Node object
            v: Node object
            rest: Node objects
        Disconnect a given node from a non-empty set of nodes, all present in the graph
        """

        u = Node(u)

        for n in {v, *rest}:
            n = Node(n)

            if (l := Link(n, u)) in self.links:
                self.__links.remove(l)
                self.__neighbors[u].discard(n)
                self.__neighbors[n].discard(u)

        return self

    def disconnect_all(self, n: Node, *rest: Node) -> UndirectedGraph:
        if not rest:
            return self

        rest = set(rest) - {n}
        self.disconnect(n, *rest)

        return self.disconnect_all(*rest)

    def copy(self) -> UndirectedGraph:
        return UndirectedGraph(self.neighbors())

    def excentricity(self, u: Node) -> int:
        """
        Args:
            u: A present node
        Returns:
            Excentricity of u (the length of the longest of all shortest paths, starting from it)
        """

        if not self:
            raise ValueError("Unrecognized node")

        if self.full():
            return 1 - (len(self.nodes) == 1)

        res, total, layer = -1, {u := Node(u)}, {u}

        while layer:
            new = set()

            while layer:
                new.update(nodes := self.neighbors(layer.pop()) - total)
                total.update(nodes)

            layer = new.copy()
            res += 1

        return res

    def diameter(self) -> int:
        """
        Returns:
            The greatest of all excentricity values in the graph
        """

        return max(map(self.excentricity, self.nodes))

    def complementary(self) -> UndirectedGraph:
        return complementary(self)

    def connected(self) -> bool:
        if len(self.links) + 1 < (n := len(self.nodes)):
            return False

        if self.degrees_sum > (n - 1) * (n - 2) or n < 2:
            return True

        queue, total = [u := next(iter(self.nodes))], {u}

        while queue:
            queue += list(next_nodes := self.neighbors(queue.pop(0)) - total)
            total.update(next_nodes)

        return total == self.nodes

    def is_tree(self, connected: bool = False) -> bool:
        """
        Args:
            connected: Boolean flag about whether the graph is already known to be connected
        Returns:
            Whether the graph could be a tree
        """

        return len(self.nodes) == len(self.links) + 1 and (connected or self.connected())

    def tree(self, root: Node = None, dfs: bool = False) -> "Tree":
        """
        Args:
            root: a present node
            dfs: a boolean flag, indicating whether the search algorithm should use DFS or BFS
        Returns:
             A tree representation of the graph with root n
        """

        from tree import Tree

        if root is None:
            try:
                root = next(iter(self.nodes))

            except KeyError:
                raise ValueError("Can't make an empty tree")

        tree = Tree(root := Node(root))
        rest, total = [root], {root}

        while rest:
            for v in self.neighbors(u := rest.pop(-bool(dfs))) - total:
                tree.add(u, v), rest.append(v), total.add(v)

        return tree

    def reachable(self, u: Node, v: Node) -> bool:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        rest, total = {u}, {u}

        while rest:
            if (n := rest.pop()) == v:
                return True

            rest.update(new := self.neighbors(n) - total)
            total.update(new)

        return False

    def subgraph(self, nodes: Iterable[Node]) -> UndirectedGraph:
        """
        Args:
            nodes: Given set of nodes
        Returns:
            The subgraph that only contains these nodes and all links between them
        """

        try:
            nodes = self.nodes.intersection(nodes)

            return UndirectedGraph({u: self.neighbors(u).intersection(nodes) for u in nodes})

        except TypeError:
            raise TypeError("Iterable of nodes expected")

    def component(self, u: Node) -> UndirectedGraph:
        """
        Args:
            u: Given node
        Returns:
            The connection component of the given node
        """

        if u not in self:
            raise KeyError("Unrecognized node")

        rest, total = {u := Node(u)}, {u}

        while rest:
            rest.update(next_nodes := self.neighbors(rest.pop()) - total)
            total.update(next_nodes)

        return self.subgraph(total)

    def connection_components(self) -> list[UndirectedGraph]:
        components, rest = [], self.nodes

        while rest:
            components.append(curr := self.component(rest.pop()))
            rest -= curr.nodes

        return components

    def cut_nodes(self) -> set[Node]:
        """
        A cut node in a graph is such that, if it's removed, the graph splits into more connection components
        Returns:
            All cut nodes
        """

        def dfs(u, l):
            colors[u], levels[u], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False

            for v in self.neighbors(u):
                if not colors[v]:
                    count += 1
                    is_cut |= (b := dfs(v, l + 1)) >= l and not is_root
                    min_back = min(min_back, b)

                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]

            if is_cut or is_root and count > 1:
                res.add(u)

            colors[u] = 2

            return min_back

        levels = {n: 0 for n in self.nodes}
        colors, res = levels.copy(), set()

        for n in self.nodes:
            if not colors[n]:
                dfs(n, 0)

        return res

    def bridge_links(self) -> set[Link]:
        """
        A bridge link is such that, if it's removed from the graph, it splits into one more connection component
        Returns:
            All bridge links
        """

        def dfs(u, l):
            colors[u], levels[u], min_back = 1, l, l

            for v in self.neighbors(u):
                if not colors[v]:
                    if (b := dfs(v, l + 1)) > l:
                        res.add(Link(u, v))

                    else:
                        min_back = min(min_back, b)

                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]

            colors[u] = 2

            return min_back

        levels = {n: 0 for n in self.nodes}
        colors, res = levels.copy(), set()

        for n in self.nodes:
            if not colors[n]:
                dfs(n, 0)

        return res

    def full(self) -> bool:
        return self.degrees_sum == (n := len(self.nodes)) * (n - 1)

    def get_shortest_path(self, u: Node, v: Node) -> Path:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

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
            if self.degree(n) % 2:
                return False

        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        if u == v:
            return self.euler_tour_exists()

        for n in self.nodes:
            if self.degree(n) % 2 - (n in {u, v}):
                return False

        return self.connected()

    def euler_tour(self) -> Path:
        if self.euler_tour_exists():
            tmp = UndirectedGraph.copy(self)

            return tmp.disconnect(u := (l := next(iter(tmp.links))).u, v := l.v).euler_walk(u, v) + [u]

        return []

    def euler_walk(self, u: Node, v: Node) -> Path:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = UndirectedGraph.copy(self)

            for i in range(len(path) - 1):
                tmp.disconnect(path[i], path[i + 1])

            for i, u in enumerate(path):
                neighbors = tmp.neighbors(u)

                while neighbors:
                    curr = tmp.disconnect(u, v := neighbors.pop()).get_shortest_path(v, u)

                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j], curr[j + 1])

                    while curr:
                        path.insert(i + 1, curr.pop())

            return path

        return []

    def links_graph(self) -> UndirectedGraph:
        """
        Returns:
            A graph, the nodes of which represent the links of the original one. Say two of its nodes represent links, which share a node in the original graph. This is shown by the fact that these nodes are connected in the links graph. If the graph has node weights, they become link weights and if it has link weights, they become node weights
        """

        return links_graph(self)

    def weighted_nodes_graph(self, weights: dict[Node, float] = None) -> WeightedNodesUndirectedGraph:
        res_weights = defaultdict(float)

        if weights is None:
            weights = {}

        for k, v in weights.items():
            res_weights[k] = v

        return WeightedNodesUndirectedGraph({n: (res_weights[n], self.neighbors(n)) for n in self.nodes})

    def weighted_links_graph(self, weights: dict[Link, float] = None) -> WeightedLinksUndirectedGraph:
        res_weights = defaultdict(float)

        if weights is None:
            weights = {}

        for k, v in weights.items():
            res_weights[k] = v

        return WeightedLinksUndirectedGraph(
            {u: {v: res_weights[Link(u, v)] for v in self.neighbors(u)} for u in self.nodes})

    def weighted_graph(self, node_weights: dict[Node, float] = None,
                       link_weights: dict[Link, float] = None) -> WeightedUndirectedGraph:
        return self.weighted_links_graph(link_weights).weighted_graph(node_weights)

    def lex_bfs(self, start: Node = None) -> Path:
        """
        Args:
            start: A present node or None
        Returns:
            A lexicographical order of the graph nodes, starting from a given (or arbitrary) node
        """

        def shift(node):
            new_priority = priority[node]
            j = i

            while priority[remaining[j]] > new_priority:
                j -= 1

            remaining.pop(i)
            remaining.insert(j, node)

        if start is None:
            start = next(iter(self.nodes))

        remaining = self.neighbors(start)
        remaining = list(remaining) + list(self.nodes - {start := Node(start)} - remaining)
        priority = {node: 0 for node in remaining}
        order = [start]

        while remaining:
            order.append(max_node := remaining.pop(0))
            priority.pop(max_node)

            for n in priority:
                priority[n] *= 2

            for i, node in enumerate(remaining):
                if node in self.neighbors(max_node):
                    priority[node] += 1
                    shift(node)

        return order

    def interval_sort(self, start: Node = None) -> Path:
        """
        Assume a set of intervals over the real number line, some of which could intersect. Such a set of intervals can be sorted based on multiple criteria. An undirected graph could be defined to represent the intervals the following way: Nodes represent the intervals and two nodes are connected exactly when the intervals they represent intersect
        Args:
            start: A present node or None
        Returns:
            A sort of the graph nodes, based on how early the interval a particular node could represent begins. If it fails, it returns an empty list. If start is given, it only tries to find a sort with start in the beginning
        """

        def find_start_node(graph, nodes):
            subgraph = graph.subgraph(nodes)
            simplicial = {n for n in nodes if subgraph.simplicial(n)}

            if len(simplicial) == 1:
                return simplicial.pop()

            return max(simplicial, key=graph.excentricity, default=None)

        def component_interval_sort(graph, nodes, priority):
            max_priority = priority[n := nodes[0]]
            max_priority_nodes = {n}

            for u in nodes[1:]:
                if priority[u] < max_priority:
                    break

                max_priority_nodes.add(u)

            start = find_start_node(curr_graph := graph.subgraph(nodes), max_priority_nodes)

            if start is None:
                return []

            new_neighbors = graph.neighbors(start)

            return helper(start, curr_graph, {u: 2 * priority[u] + (u in new_neighbors) for u in set(nodes) - {start}})

        def helper(u, graph, priority):
            def extend_last(ll):
                last = ll[-1]

                return ll + (last,) * (max_length - len(ll))

            if graph.full():
                return [u, *sorted(graph.nodes - {u}, key=priority.get, reverse=True)]

            result, neighbors, comps = [u], graph.neighbors(u), []
            final_neighbors, final, total = set(), [], {u}

            for v in neighbors:
                if v not in total:
                    total.add(v)
                    rest, comp, this_final = {v}, {v}, False

                    while rest:
                        for n in graph.neighbors(_ := rest.pop()):
                            if n not in total:
                                if n in neighbors:
                                    rest.add(n), comp.add(n), total.add(n)

                                else:
                                    if final:
                                        return []

                                    this_final = True

                    if this_final:
                        final_neighbors = comp
                        final = sorted((graph.nodes - neighbors - {u}).union(comp),
                                       key=lambda x: (priority[x], x in neighbors), reverse=True)

                    else:
                        comps.append(sorted(comp, key=priority.get, reverse=True))

            max_length = max(map(len, comps), default=0)
            comps = sorted(comps, key=lambda c: extend_last(tuple(map(priority.get, c))), reverse=True)

            if final:
                if set(final[:len(final_neighbors)]) != final_neighbors:
                    return []

                comps.append(final)

                del final, final_neighbors

            for i in range(len(comps) - 1):
                if priority[comps[i][-1]] < priority[comps[i + 1][0]]:
                    return []

            for comp in comps:
                if not (curr_sort := component_interval_sort(graph, comp, priority)):
                    return []

                result += curr_sort

            return result

        if not self.connected():
            components = self.connection_components()

            if start is None:
                result = []

                for component in components:
                    if not (curr := component.interval_sort()):
                        return []

                    result += curr

                return result

            start = Node(start)

            for c in components:
                if start in c:
                    begin = c
                    break

            else:
                raise KeyError("Unrecognized node")

            components.remove(begin)

            if not (result := begin.interval_sort(start)):
                return []

            for component in components:
                if not (curr := component.interval_sort()):
                    return []

                result += curr

            return result

        if start is None:
            start = find_start_node(self, self.nodes)

            if start is None:
                return []

        if (start := Node(start)) not in self:
            raise KeyError("Unrecognized node")

        return helper(start, self, {u: u in self.neighbors(start) for u in self.nodes - {start}})

    def is_full_k_partite(self, k: int = None) -> bool:
        """
        Args:
            k: number of partitions or None
        Returns:
            Whether the graph has k independent sets and as many links as possible, given this condition. If k is not given, it can be anything
        """

        if k is not None and not isinstance(k, int):
            try:
                k = int(k)

            except ValueError:
                raise TypeError("Integer expected")

        return k in {None, len(comps := self.complementary().connection_components())} and all(c.full() for c in comps)

    def clique(self, *nodes: Node) -> bool:
        """
        Args:
            nodes: A set of present nodes
        Returns:
            Whether these given nodes form a clique
        """

        def helper(rest):
            if not rest:
                return True

            if not rest.issubset(self.neighbors(rest.pop())):
                return False

            return helper(rest)

        nodes = {Node(x) for x in nodes}
        nodes.intersection_update(self.nodes)

        if nodes == self.nodes:
            return self.full()

        return helper(nodes)

    def cliques(self, k: int) -> list[set[Node]]:
        """
        Args:
            k: Wanted size of cliques
        Returns:
            All cliques in the graph of size k
        """

        try:
            k = int(k)

        except ValueError:
            raise TypeError("Integer expected")

        if k < 0:
            return []

        if not self.connected():
            if not k:
                return [set()]

            return reduce(lambda x, y: x + y, map(lambda g: g.cliques(k), self.connection_components()))

        return [set(p) for p in combinations(self.nodes, abs(k)) if self.clique(*p)]

    def maximal_cliques(self) -> list[set[Node]]:
        """
        Returns:
            All maximal by inclusion cliques in the graph
        """

        if sort := self.interval_sort():
            cliques = []
            rest = self.nodes

            for n in reversed(sort):
                curr = self.neighbors(n).intersection(rest).union({n})
                rest.remove(n)

                if all(not curr.issubset(x) for x in cliques):
                    cliques.append(curr)

            return cliques

        return self.complementary().maximal_independent_sets()

    def max_cliques(self) -> list[set[Node]]:
        """
        Returns:
            All maximal by cardinality cliques in the graph
        """

        result, low, high, k = [set()], 1, len(self.nodes), 0

        while low <= high:
            if curr := self.cliques(mid := (low + high) // 2):
                low = mid + 1

                if mid > k:
                    result, k = curr, mid

            else:
                high = mid - 1

        return result

    def max_cliques_node(self, u: Node) -> list[set[Node]]:
        """
        Args:
            u: A present node
        Returns:
            All, maximal by cardinality, cliques in the graph, to which node u belongs
        """

        return list(map({Node(u)}.union, self.subgraph(self.neighbors(u)).max_cliques()))

    def maximal_cliques_node(self, u: Node) -> list[set[Node]]:
        """
        Args:
            u: A present node
        Returns:
            All maximal by inclusion cliques in the graph, to which node u belongs
        """

        return list(map({Node(u)}.union, self.subgraph(self.neighbors(u)).maximal_cliques()))

    def maximal_independent_sets(self) -> list[set[Node]]:
        """
        Returns:
            All maximal by inclusion independent sets in the graph
        """

        def generator(result=set(), total=set(), i=0):
            for j, n in enumerate(nodes[i:]):
                if result.isdisjoint(self.neighbors(n)):
                    for res in generator({n, *result}, {n, *self.neighbors(n), *total}, i + j + 1):
                        yield res

            if total == self.nodes:
                yield result

        nodes = list(self.nodes)

        return [i_s for i_s in generator()]

    def cliques_graph(self) -> UndirectedGraph:
        """
        A cliques graph of a given one is a graph, the nodes of which represent the individual maximal by inclusion cliques in the original graph, and the links of which represent whether the cliques two nodes represent intersect
        Returns:
            The clique graph of the original one
        """

        return cliques_graph(self)

    def halls_marriage_problem(self, nodes: Iterable[Node]) -> set[Link]:
        """
        Args:
            nodes: A set of present nodes
        Returns:
            A set of bridge links between the given set of nodes and the rest, where each link is connected to exactly one node from the partition, which has no more nodes than the other. Each link connects a node from the "lesser" set to a unique node from the other one.
        """

        def helper(nodes, neighborhood, res=set()):
            while True:
                removed = set()

                for k, v in neighborhood.items():
                    if len(v) == 1:
                        res.add(Link(k, (n := v.pop())))
                        removed.add((k, n))

                if not removed:
                    break

                for k, n in removed:
                    nodes.remove(k)
                    neighborhood.pop(k)

                removed = {p[1] for p in removed}

                for k, v in neighborhood.items():
                    neighborhood[k] = v - removed

            if not nodes:
                return res

            rest = neighborhood[u := nodes.pop()]
            neighborhood.pop(u)
            neighborhood_copy = neighborhood.copy()

            for v in rest:
                res.add(l := Link(u, v))

                for k in neighborhood:
                    neighborhood[k].discard(v)

                if curr := helper(nodes, neighborhood, res):
                    return curr

                neighborhood = neighborhood_copy
                res.remove(l)

            return set()

        nodes = {Node(n) for n in nodes}
        nodes.intersection_update(self.nodes)

        if len(nodes) > len(rest := self.nodes - nodes):
            nodes, rest = rest, nodes

        neighborhood = {u: self.neighbors(u).intersection(rest) for u in nodes}

        if any(not neighborhood[u] for u in nodes):
            return set()

        return helper(nodes, neighborhood)

    def chromatic_nodes_partition(self) -> list[set[Node]]:
        """
        Returns:
            A list of independent sets in the graph that covers all nodes without any of its elements intersecting. This list has as few elements as possible
        """

        if not self.connected():
            r = [comp.chromatic_nodes_partition() for comp in self.connection_components()]
            final = max(r, key=len)
            r.remove(final)

            for part in r:
                for i, i_s in enumerate(part):
                    final[i].update(i_s)

            return final

        if self.is_tree(True):
            queue, c0, c1, total = [next(iter(self.nodes))], self.nodes, set(), set()

            while queue:
                flag = (u := queue.pop(0)) in c0

                for v in self.neighbors(u) - total:
                    if flag:
                        c1.add(v), c0.remove(v)

                    queue.append(v), total.add(v)

            return [c0, c1]

        if self.is_full_k_partite():
            return [comp.nodes for comp in self.complementary().connection_components()]

        if sort := self.interval_sort():
            result = []

            for u in sort:
                for i, partition in enumerate(result):
                    if self.neighbors(u).isdisjoint(partition):
                        result[i].add(u)
                        break

                else:
                    result.append({u})

            return result

        result = list(map(lambda x: {x}, self.nodes))

        for i_s in self.maximal_independent_sets():
            curr = [i_s] + self.subgraph(self.nodes - i_s).chromatic_nodes_partition()

            if len(curr) == 2:
                return curr

            if len(curr) < len(result):
                result = curr.copy()

        return result

    def chromatic_links_partition(self) -> list[set[Link]]:
        """
        Similar to chromatic nodes partition, except links that share a node are in different sets
        """

        return [set(map(lambda x: x.value, s)) for s in
                UndirectedGraph.copy(self).links_graph().chromatic_nodes_partition()]

    def vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimal by cardinality set of nodes that cover all links in the graph
        """

        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimal by cardinality set of nodes that cover all other nodes in the graph
        """

        def helper(curr=set(), total=set(), i=0):
            if total == self.nodes:
                return curr.copy()

            result = self.nodes

            for j, u in enumerate(nodes[i:]):
                if len(res := helper({u, *curr}, {u, *self.neighbors(u), *total}, i + j + 1)) < len(result):
                    result = res

            return result

        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.dominating_set() for comp in self.connection_components()])

        if self.is_tree(True):
            return self.tree().dominating_set()

        if self.is_full_k_partite():
            if not self:
                return set()

            res = {u := max(self.nodes, key=self.degree)}

            if (neighbors := self.neighbors(u)) and {u, *neighbors} != self.nodes:
                res.add(neighbors.pop())

            return res

        nodes = list(self.nodes)

        return helper()

    def independent_set(self) -> set[Node]:
        """
        Returns:
            A maximal by cardinality set of nodes, no two of which are neighbors
        """

        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.independent_set() for comp in self.connection_components()])

        if not self:
            return set()

        if self.is_tree(True):
            return self.tree().independent_set()

        if self.is_full_k_partite():
            return max([comp.nodes for comp in self.complementary().connection_components()], key=len)

        if sort := self.interval_sort():
            result = set()

            for u in reversed(sort):
                if self.neighbors(u).isdisjoint(result):
                    result.add(u)

            return result

        return max(self.maximal_independent_sets(), key=len)

    def cycle_with_length(self, length: int) -> Path:
        try:
            length = int(length)

        except ValueError:
            raise TypeError("Integer expected")

        if length < 3:
            return []

        if length == 3:
            for l in self.links:
                if intersection := self.neighbors(u := l.u).intersection(self.neighbors(v := l.v)):
                    return [u, v, intersection.pop(), u]

            return []

        tmp = UndirectedGraph.copy(self)

        for l in tmp.links:
            res = tmp.disconnect(u := l.u, v := l.v).path_with_length(v, u, length - 1)

            if res:
                return [u] + res

            tmp.connect(u, v)

        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> Path:
        def dfs(x: Node, l: int, stack: list[Link]):
            if not l:
                return [link.u for link in stack] + [v] if x == v else []

            for y in {y for y in self.neighbors(x) if Link(x, y) not in stack}:
                if res := dfs(y, l - 1, stack + [Link(x, y)]):
                    return res

            return []

        u, v = Node(u), Node(v)

        try:
            length = int(length)

        except ValueError:
            raise TypeError("Integer expected")

        if not (tmp := self.get_shortest_path(u, v)) or (k := len(tmp)) > length + 1:
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

            neighbors = tmp.neighbors(x)
            tmp.remove(x)

            for y in neighbors:
                if dfs(y):
                    return True

            tmp.add(x, *neighbors)

            return False

        if (n := len(self.nodes)) == 1 or (2 * (m := len(self.links)) > (n - 1) * (n - 2) + 2 or n > 2 and all(
                2 * self.degree(node) >= n for node in self.nodes)):
            return True

        if n > m or self.leaves or not self.connected():
            return False

        tmp = UndirectedGraph.copy(self)
        can_end_in = tmp.neighbors(u := next(iter(self.nodes)))

        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        u, v = Node(u), Node(v)

        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s)")

        if Link(u, v) in self.links:
            return self.nodes == {u, v} or self.hamilton_tour_exists()

        return UndirectedGraph.copy(self).connect(u, v).hamilton_tour_exists()

    def hamilton_tour(self) -> Path:
        if len(self.nodes) == 1:
            return [next(iter(self.nodes))]

        if not self or self.leaves or not self.connected():
            return []

        u = next(iter(self.nodes))

        for v in self.neighbors(u):
            if res := self.hamilton_walk(u, v):
                return res + [u]

        return []

    def hamilton_walk(self, u: Node = None, v: Node = None) -> Path:
        def dfs(x, stack):
            if not tmp.connected():
                return []

            too_many = v is not None

            for n in tmp.nodes - {x, v}:
                if tmp.leaf(n):
                    if too_many:
                        return []

                    too_many = True

            neighbors = tmp.neighbors(x)
            tmp.remove(x)

            if not tmp:
                return stack

            for y in neighbors:
                if y == v:
                    if tmp.nodes == {v}:
                        return stack + [v]

                    continue

                if res := dfs(y, stack + [y]):
                    return res

            tmp.add(x, *neighbors)

            return []

        if u is not None:
            u = Node(u)

        if v is not None:
            v = Node(v)

        tmp = UndirectedGraph.copy(self)

        if u is None:
            u, v = v, u

        if u is None:
            for n in self.nodes:
                if result := dfs(n, [n]):
                    return result

                if self.leaf(n):
                    return []

            return []

        if u not in self or v is not None and v not in self:
            raise KeyError("Unrecognized node(s)")

        return dfs(u, [u])

    def isomorphic_bijection(self, other: UndirectedGraph) -> dict[Node, Node]:
        return isomorphic_bijection_undirected(self, other)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __contains__(self, u: Node) -> bool:
        return Node(u) in self.nodes

    def __add__(self, other: UndirectedGraph) -> UndirectedGraph:
        """
        Args:
            other: another UndirectedGraph object
        Returns:
            Combination of two undirected graphs
        """

        return combine_undirected(self, other)

    def __eq__(self, other: Any) -> bool:
        return compare(self, other)

    def __str__(self) -> str:
        return string(self)

    __repr__: str = __str__


class WeightedNodesUndirectedGraph(UndirectedGraph):
    """
    Class for implementing and undirected graph with weights on the nodes
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, Iterable[Node]]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary that maps a node to a tuple of its weight and its neighbors
        """

        if neighborhood is None:
            neighborhood = {}

        super().__init__()
        self.__node_weights = {}

        for n, (w, _) in neighborhood.items():
            self.add((n, w))

        for u, (_, neighbors) in neighborhood.items():
            for v in neighbors:
                self.add((v, 0), u), self.connect(u, v)

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

    def add(self, n_w: tuple[Node, float], *current_nodes: Node) -> WeightedNodesUndirectedGraph:
        n = n_w[0]

        if n not in self:
            UndirectedGraph.add(self, n, *current_nodes)
            self.set_weight(*n_w)

        return self

    def remove(self, n: Node, *rest: Node) -> WeightedNodesUndirectedGraph:
        for u in {n, *rest}:
            if u in self:
                self.__node_weights.pop(Node(u))

        return super().remove(n, *rest)

    def set_weight(self, u: Node, w: float) -> WeightedNodesUndirectedGraph:
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

    def increase_weight(self, u: Node, w: float) -> WeightedNodesUndirectedGraph:
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

    def copy(self) -> WeightedNodesUndirectedGraph:
        return WeightedNodesUndirectedGraph({n: (self.node_weights(n), self.neighbors(n)) for n in self.nodes})

    def weighted_tree(self, n: Node = None, dfs: bool = False) -> "WeightedTree":
        """
        Args:
            n: A present node
            dfs: a boolean flag, indicating whether the search algorithm should use DFS or BFS
        Returns:
             A weighted tree representation of the graph with root n
        """

        from tree import WeightedTree

        if n is None:
            n = next(iter(self.nodes))

        n = Node(n)

        tree = WeightedTree((n, self.node_weights(n)))
        queue, total = [n], {n}

        while queue:
            for v in self.neighbors(u := queue.pop(-bool(dfs))) - total:
                tree.add(u, {v: self.node_weights(v)}), queue.append(v), total.add(v)

        return tree

    def subgraph(self, nodes: Iterable[Node]) -> WeightedNodesUndirectedGraph:
        try:
            nodes = self.nodes.intersection(nodes)
            neighborhood = {u: (self.node_weights(u), self.neighbors(u).intersection(nodes)) for u in nodes}

            return WeightedNodesUndirectedGraph(neighborhood)

        except TypeError:
            raise TypeError("Iterable of nodes expected")

    def weighted_graph(self, weights: dict[Link, float] = None) -> WeightedUndirectedGraph:
        res_weights = defaultdict(float)

        if weights is None:
            weights = {}

        for k, v in weights.items():
            res_weights[k] = v

        return super().weighted_graph(self.node_weights(), res_weights)

    def minimal_path_nodes(self, u: Node, v: Node) -> Path:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path between u and v with the least possible sum of node weights
        """

        return self.weighted_graph().minimal_path(u, v)

    def weighted_vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimal by sum of the weights set of nodes that covers all links in the graph
        """

        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimal by sum of the weights set of nodes that covers all nodes in the graph
        """

        def helper(curr=set(), total=set(), res_weight=0.0, i=0):
            if total == self.nodes:
                return curr.copy(), res_weight

            result, result_weight = self.nodes, self.total_nodes_weight

            for j, u in enumerate(nodes[i:]):
                cover, weight = helper({u, *curr}, {u, *self.neighbors(u), *total},
                                       res_weight + self.node_weights(u), i + j + 1)

                if weight < result_weight:
                    result, result_weight = cover, weight

            return result, result_weight

        if not self:
            return set()

        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.weighted_dominating_set() for comp in
                                                    self.connection_components()])

        nodes = list(self.nodes)

        if self.is_tree(True):
            return self.weighted_tree().weighted_dominating_set()

        if self.is_full_k_partite():
            if not self:
                return set()

            res = {min({u for u in nodes if self.degree(u) + 1 == len(nodes)}, key=self.node_weights)}
            potential = [{n, m} for l in self.links if
                         self.degree(n := l.u) + 1 < len(nodes) and self.degree(m := l.v) + 1 < len(nodes)]
            potential = min(potential, key=lambda s: sum(map(self.node_weights, s)))

            return min([res, potential], key=lambda s: sum(map(self.node_weights, s)))

        return helper()[0]

    def weighted_independent_set(self) -> set[Node]:
        """
        Returns:
            A set of non-neighboring nodes with a maximal possible sum of the weights
        """

        def helper(curr=set(), total=set(), res_weight=0.0, i=0):
            if total == self.nodes:
                return curr, res_weight

            result, result_weight = set(), 0

            for j, u in enumerate(nodes[i:]):
                if u not in total and self.node_weights(u) > 0:
                    cover, weight = helper({u, *curr}, {u, *total, *self.neighbors(u)},
                                           res_weight + self.node_weights(u), i + j + 1)

                    if weight > result_weight:
                        result, result_weight = cover, weight

            return result, result_weight

        if not self:
            return set()

        if not self.connected():
            return reduce(lambda x, y: x.union(y),
                          [comp.weighted_independent_set() for comp in self.connection_components()])

        if self.is_tree(True):
            return self.weighted_tree().weighted_independent_set()

        nodes = list(self.nodes)

        return helper()[0]


class WeightedLinksUndirectedGraph(UndirectedGraph):
    """
    Class for implementing and undirected graph with weights on the links
    """

    def __init__(self, neighborhood: dict[Node, dict[Node, float]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary, mapping to each node the link weight between it and each of its neighbors
        """

        if neighborhood is None:
            neighborhood = {}

        super().__init__()
        self.__link_weights = {}

        for u, neighbors in neighborhood.items():
            self.add(u)

            for v, w in neighbors.items():
                self.add(v, {u: w}), self.connect(v, {u: w})

    def link_weights(self, u_l: Node | Link = None, v: Node = None) -> dict[Node, float] | dict[Link, float] | float:
        """
        Args:
            u_l: Given first node, a link or None
            v: Given second node or None
        Returns:
            Information about link weights the following way:
            If no argument is passed, return the weights of all links;
            if a link or two nodes are passed, return the weight of the given link between them;
            If one node is passed, return a dictionary with all of its neighbors and the weight of the link it shares with each of them
        """

        if u_l is None:
            return self.__link_weights.copy()

        if isinstance(u_l, Link):
            return self.__link_weights[u_l]

        return {n: self.link_weights(Link(n, u_l)) for n in self.neighbors(u_l)} if v is None else self.link_weights(
            Link(u_l, v))

    @property
    def total_links_weight(self) -> float:
        """
        Returns:
            The sum of all link weights
        """

        return sum(self.link_weights().values())

    def add(self, u: Node, nodes_weights: dict[Node, float] = None) -> WeightedLinksUndirectedGraph:
        if nodes_weights is None:
            nodes_weights = {}

        if u not in self:
            UndirectedGraph.add(self, u, *nodes_weights)

            for v, w in nodes_weights.items():
                self.set_weight(Link(u, v), w)

        return self

    def remove(self, n: Node, *rest: Node) -> WeightedLinksUndirectedGraph:
        for u in {n, *rest}:
            if u in self:
                for v in self.neighbors(u):
                    self.__link_weights.pop(Link(u, v))

        return super().remove(n, *rest)

    def connect(self, u: Node, nodes_weights: dict[Node, float] = None) -> WeightedLinksUndirectedGraph:
        if nodes_weights is None:
            nodes_weights = {}

        if u in self:
            nodes_weights = {Node(k): v for k, v in nodes_weights.items()}
            nodes_weights = {v: w for v, w in nodes_weights.items() if v not in self.neighbors(u)}

            if nodes_weights:
                super().connect(u, *nodes_weights)

                for v, w in nodes_weights.items():
                    self.set_weight(Link(u, v), w)

        return self

    def connect_all(self, u: Node, *rest: Node) -> WeightedLinksUndirectedGraph:
        if not rest:
            return self

        rest = set(rest) - {u}
        self.connect(u, {v: 0 for v in rest})

        return self.connect_all(*rest)

    def disconnect(self, u: Node, v: Node, *rest: Node) -> WeightedLinksUndirectedGraph:
        super().disconnect(u, v, *rest)

        for n in {v, *rest}:
            if (l := Link(u, n)) in self.link_weights():
                self.__link_weights.pop(l)

        return self

    def set_weight(self, l: Link, w: float) -> WeightedLinksUndirectedGraph:
        """
        Args:
            l: A present link
            w: The new weight of link l
        Set the weight of link l to w
        """

        try:
            if l in self.links:
                self.__link_weights[l] = float(w)

            return self

        except TypeError:
            raise TypeError("Real value expected")

    def increase_weight(self, l: Link, w: float) -> WeightedLinksUndirectedGraph:
        """
        Args:
            l: A present link
            w: A real value
        Increase the weight of link l with w
        """

        try:
            if l in self.link_weights():
                self.set_weight(l, self.link_weights(l) + float(w))

            return self

        except ValueError:
            raise TypeError("Real value expected")

    def copy(self) -> WeightedLinksUndirectedGraph:
        return WeightedLinksUndirectedGraph({n: self.link_weights(n) for n in self.nodes})

    def subgraph(self, nodes: Iterable[Node]) -> WeightedLinksUndirectedGraph:
        try:
            nodes = self.nodes.intersection(nodes)
            neighborhood = {u: {k: v for k, v in self.link_weights(u).items() if k in nodes} for u in nodes}

            return WeightedLinksUndirectedGraph(neighborhood)

        except TypeError:
            raise TypeError("Iterable of nodes expected")

    def minimal_spanning_tree(self) -> set[Link]:
        """
        Returns:
            A spanning tree or forrest of trees of the graph with the minimal possible weights sum
        """

        def insert(x):
            low, high, w = 0, len(bridge_links), self.link_weights(x)

            while low < high:
                mid = (low + high) // 2

                if w == (mid_weight := self.link_weights(bridge_links[mid])):
                    bridge_links.insert(mid, x)

                    return

                if w < mid_weight:
                    high = mid

                else:
                    if low == mid:
                        break

                    low = mid + 1

            bridge_links.insert(high, x)

        def remove(x):
            low, high, w = 0, len(bridge_links), self.link_weights(x)

            while low < high:
                mid = (low + high) // 2

                if w == (f_mid := self.link_weights(mid_l := bridge_links[mid])):
                    if x == mid_l:
                        bridge_links.pop(mid)

                        return

                    i, j, still = mid - 1, mid + 1, True

                    while still:
                        still = False

                        if i >= 0 and self.link_weights(bridge_links[i]) == w:
                            if x == bridge_links[i]:
                                bridge_links.pop(i)

                                return

                            i -= 1
                            still = True

                        if j < len(bridge_links) and self.link_weights(bridge_links[j]) == w:
                            if x == bridge_links[j]:
                                bridge_links.pop(j)

                                return

                            j += 1
                            still = True

                    return

                if w < f_mid:
                    high = mid
                else:
                    if low == mid:
                        return

                    low = mid

        if not self.connected():
            return reduce(lambda x, y: x.union(y),
                          [comp.minimal_spanning_tree() for comp in self.connection_components()]

        if self.is_tree(True):
            return self.links

        res_links, total = set(), {u := next(iter(self.nodes))}
        bridge_links = []

        for v in self.neighbors(u):
            insert(Link(u, v))

        for _ in range(1, len(self.nodes)):
            res_links.add(l := bridge_links.pop(0))

            if (u := l.u) in total:
                total.add(v := l.v)

                for _v in self.neighbors(v):
                    l = Link(v, _v)

                    if _v in total:
                        remove(l)

                    else:
                        insert(l)

            else:
                total.add(u)

                for _u in self.neighbors(u):
                    l = Link(u, _u)

                    if _u in total:
                        remove(l)

                    else:
                        insert(l)

        return res_links

    def weighted_graph(self, weights: dict[Node, float] = None) -> WeightedUndirectedGraph:
        res_weights = defaultdict(float)

        if weights is None:
            weights = {}

        for k, v in weights.items():
            res_weights[k] = v

        return WeightedUndirectedGraph({n: (res_weights[n], self.link_weights(n)) for n in self.nodes})

    def minimal_path_links(self, u: Node, v: Node) -> Path:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path between u and v with the least possible sum of link weights
        """

        return self.weighted_graph().minimal_path(u, v)


class WeightedUndirectedGraph(WeightedLinksUndirectedGraph, WeightedNodesUndirectedGraph):
    """
    Class for implementing an undirected graph with weights on the nodes and the links
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, dict[Node, float]]] = None) -> None:
        """
        Args:
            neighborhood: A dictionary, mapping nodes to a tuple with each node's weight and another dictionary, representing the weight of the link between it and each of its neighbors
        """

        if neighborhood is None:
            neighborhood = {}

        WeightedNodesUndirectedGraph.__init__(self), WeightedLinksUndirectedGraph.__init__(self)

        for n, (w, _) in neighborhood.items():
            self.add((n, w))

        for u, (_, neighbors) in neighborhood.items():
            for v, w in neighbors.items():
                self.add((v, 0), {u: w}), self.connect(u, {v: w})

    @property
    def total_weight(self) -> float:
        """
        Returns:
            The sum of all weights in the graph
        """

        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: tuple[Node, float], nodes_weights: dict[Node, float] = None) -> WeightedUndirectedGraph:
        if nodes_weights is None:
            nodes_weights = {}

        if (n := n_w[0]) not in self:
            super().add(n, nodes_weights)
            self.set_weight(*n_w)

        return self

    def remove(self, u: Node, *rest: Node) -> WeightedUndirectedGraph:
        for n in {u, *rest}:
            if n in self:
                if neighbors := self.neighbors(n):
                    super().disconnect(n, *neighbors)

        WeightedNodesUndirectedGraph.remove(self, u, *rest)

        return self

    def set_weight(self, el: Node | Link, w: float) -> WeightedUndirectedGraph:
        """
        Args:
            el: A present node or link
            w: The new weight of object el
        Set the weight of object el to w
        """

        try:
            if el in self.links:
                super().set_weight(el, float(w))

            elif el in self:
                WeightedNodesUndirectedGraph.set_weight(self, el, float(w))

            return self

        except ValueError:
            raise TypeError("Real value expected")

    def increase_weight(self, el: Node | Link, w: float) -> WeightedUndirectedGraph:
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

    def copy(self) -> WeightedUndirectedGraph:
        neighborhood = {n: (self.node_weights(n), self.link_weights(n)) for n in self.nodes}

        return WeightedUndirectedGraph(neighborhood)

    def subgraph(self, nodes: Iterable[Node]) -> WeightedUndirectedGraph:
        try:
            nodes = self.nodes.intersection(nodes)
            neighborhood = {u: (self.node_weights(u), {k: v for k, v in self.link_weights(u).items() if k in nodes})
                            for u in nodes}

            return WeightedUndirectedGraph(neighborhood)

        except TypeError:
            raise TypeError("Iterable of nodes expected")

    def minimal_path(self, u: Node, v: Node) -> Path:
        """
        Args:
            u: First given node
            v: Second given node
        Returns:
            A path between u and v with the least possible sum of node and link weights
        """

        def dfs(x, res_path, res_weight):
            nonlocal total_negative, curr_path, curr_weight

            def dijkstra(s):
                tmp_cpy = tmp.copy()

                for l in curr_path:
                    tmp_cpy.disconnect(l.u, l.v)

                pq = {s}
                prev_weight: dict[Node, tuple[Node, float]] = {s: (None, 0)}

                while pq:
                    s_ = min(pq, key=lambda _s: prev_weight[_s][1])
                    pq.remove(s_)

                    if s_ == v:
                        break

                    s_weight = prev_weight[s_][1]

                    for t_ in tmp_cpy.neighbors(s_):
                        weight = tmp_cpy.link_weights(s_, t_) + tmp_cpy.node_weights(t_)

                        if t_ not in prev_weight or s_weight + weight < prev_weight[t_][1]:
                            prev_weight[t_] = (s_, s_weight + weight)
                            pq.add(t_)

                else:
                    return [], inf

                result, curr_node = [], v

                while curr_node != s:
                    result.insert(0, Link(prev_weight[curr_node][0], curr_node))
                    curr_node = prev_weight[curr_node][0]

                return curr_path + result, curr_weight + prev_weight[v][1]

            if total_negative:
                for y in {y for y in tmp.neighbors(x) if Link(x, y) not in curr_path}:
                    if (n_w := tmp.node_weights(y)) < 0:
                        total_negative -= n_w

                    if (l_w := tmp.link_weights(x, y)) < 0:
                        total_negative -= l_w

                    if curr_weight + n_w + l_w + total_negative >= res_weight:
                        continue

                    curr_weight += n_w + l_w

                    if y == v and curr_weight < res_weight:
                        res_path = curr_path + [Link(x, y)]
                        res_weight = curr_weight

                    curr_path.append(Link(x, y))
                    curr = dfs(y, res_path, res_weight)
                    curr_path.pop()
                    curr_weight -= n_w + l_w

                    if n_w < 0:
                        total_negative += n_w

                    if l_w < 0:
                        total_negative += l_w

                    if curr[1] < res_weight:
                        res_path, res_weight = curr

            else:
                curr = dijkstra(x)

                if curr[1] < res_weight:
                    res_path, res_weight = curr

            return res_path, res_weight

        u, v = Node(u), Node(v)

        if v in self:
            if v in (tmp := self.component(u)):
                nodes_negative_weights = sum(tmp.node_weights(n) for n in tmp.nodes if tmp.node_weights(n) < 0)
                links_negative_weights = sum(tmp.link_weights(l) for l in tmp.links if tmp.link_weights(l) < 0)
                total_negative = nodes_negative_weights + links_negative_weights
                curr_path, curr_weight = [], tmp.node_weights(u)
                res = dfs(u, [], inf)[0]

                return [l.u for l in res] + [res[-1].v]

            return []

        raise KeyError("Unrecognized node(s)")



