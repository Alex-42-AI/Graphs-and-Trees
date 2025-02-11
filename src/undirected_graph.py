"""
Module for implementing undirected graphs and functions for working with them.
"""

from functools import reduce

from collections import defaultdict

from itertools import permutations, combinations, product


from .base import Node, Link, Graph, Iterable


def links_graph(graph: "UndirectedGraph") -> "UndirectedGraph":
    if isinstance(graph, WeightedUndirectedGraph):
        result = WeightedUndirectedGraph(
            {Node(l): (graph.link_weights(l), {}) for l in graph.links})
        for l0 in graph.links:
            for l1 in graph.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): graph.node_weights(s.pop())})
        return result
    if isinstance(graph, WeightedNodesUndirectedGraph):
        result = WeightedLinksUndirectedGraph({Node(l): {} for l in graph.links})
        for l0 in graph.links:
            for l1 in graph.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): graph.node_weights(s.pop())})
        return result
    if isinstance(graph, WeightedLinksUndirectedGraph):
        neighborhood = {Node(l0): (graph.link_weights(l0),
                                   [Node(l1) for l1 in graph.links if (l1.u in l0) ^ (l1.v in l0)])
                        for l0 in graph.links}
        return WeightedNodesUndirectedGraph(neighborhood)
    neighborhood = {Node(l0): [Node(l1) for l1 in graph.links if (l1.u in l0) or (l1.v in l0)]
                    for l0 in graph.links}
    return UndirectedGraph(neighborhood)


def combine_graphs(graph0: "UndirectedGraph", graph1: "UndirectedGraph") -> "UndirectedGraph":
    if not isinstance(graph1, UndirectedGraph):
        raise TypeError("Addition not defined between class "
                        f"{type(graph0).__name__} and type {type(graph1).__name__}!")
    if isinstance(graph0, WeightedUndirectedGraph):
        if isinstance(graph1, WeightedUndirectedGraph):
            res = graph0.copy()
            for n in graph1.nodes:
                if n in res:
                    res.increase_weight(n, graph1.node_weights(n))
                else:
                    res.add((n, graph1.node_weights(n)))
            for l in graph1.links:
                if l in res.links:
                    res.increase_weight(l, graph1.link_weights(l))
                else:
                    res.connect(l.u, {l.v: graph1.link_weights(l)})
            return res
        return graph0 + graph1.weighted_graph()
    if isinstance(graph0, WeightedNodesUndirectedGraph):
        if isinstance(graph1, WeightedLinksUndirectedGraph):
            return graph0.weighted_graph() + graph1
        if isinstance(graph1, WeightedNodesUndirectedGraph):
            res = graph0.copy()
            for n in graph1.nodes:
                if n in res:
                    res.increase_weight(n, graph1.node_weights(n))
                else:
                    res.add((n, graph1.node_weights(n)))
            for l in graph1.links:
                res.connect(l.u, l.v)
            return res
        return graph0 + graph1.weighted_nodes_graph()
    if isinstance(graph0, WeightedLinksUndirectedGraph):
        if isinstance(graph1, WeightedNodesUndirectedGraph):
            return graph1 + graph0
        if isinstance(graph1, WeightedLinksUndirectedGraph):
            res = graph0.copy()
            for n in graph1.nodes:
                res.add(n)
            for l in graph1.links:
                if l in res.links:
                    res.increase_weight(l, graph1.link_weights(l))
                else:
                    res.connect(l.u, {l.v: graph1.link_weights(l)})
            return res
        return graph0 + graph1.weighted_links_graph()
    if isinstance(graph1, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
        return graph1 + graph0
    res = graph0.copy()
    for n in graph1.nodes:
        res.add(n)
    for l in graph1.links:
        res.connect(l.u, l.v)
    return res


class UndirectedGraph(Graph):
    """
    Class for implementing an unweighted undirected graph.
    """

    def __init__(self, neighborhood: dict[Node, Iterable[Node]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary, that associates a node to its neighbors in the graph.
        """
        self.__nodes, self.__links = set(), set()
        self.__neighbors = {}
        for u, neighbors in neighborhood.items():
            if u not in self:
                self.add(u)
            for v in neighbors:
                self.add(v, u), self.connect(u, v)

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
            The total sum of all node degrees.
        """
        return 2 * len(self.links)

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Graph leaves.
        """
        return {n for n in self.nodes if self.leaf(n)}

    def neighbors(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: Node object or None.
        Returns:
            Neighbors of u or dictionary of all nodes and their neighbors.
        """
        if u is None:
            return {n: self.neighbors(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return self.__neighbors[u].copy()

    def degrees(self, u: Node = None) -> dict[Node, int] | int:
        if u is None:
            return {n: len(self.neighbors(n)) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return len(self.__neighbors[u])

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

    def add(self, u: Node, *current_nodes: Node) -> "UndirectedGraph":
        """
        Args:
            u: a new node.
            current_nodes: nodes already present in the graph.
        Add node u to already present nodes.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            self.__nodes.add(u)
            self.__neighbors[u] = set()
            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)
        return self

    def remove(self, n: Node, *rest: Node) -> "UndirectedGraph":
        for u in (n, *rest):
            if not isinstance(u, Node):
                u = Node(u)
            if u in self:
                if tmp := self.neighbors(u):
                    self.disconnect(u, *tmp)
                self.__nodes.remove(u), self.__neighbors.pop(u)
        return self

    def connect(self, u: Node, v: Node, *rest: Node) -> "UndirectedGraph":
        """
        Args:
            u: Node object.
            v: Node object.
            rest: Node objects.
        Connect u to v and nodes in rest, all present in the graph.
        """
        if not isinstance(u, Node):
            u = Node(u)
        for n in (v, *rest):
            if not isinstance(n, Node):
                n = Node(n)
            if u != n and n not in self.neighbors(u) and n in self:
                self.__neighbors[u].add(n)
                self.__neighbors[n].add(u)
                self.__links.add(Link(u, n))
        return self

    def connect_all(self, u: Node, *rest: Node) -> "UndirectedGraph":
        if not rest:
            return self
        self.connect(u, *rest)
        return self.connect_all(*rest)

    def disconnect(self, u: Node, v: Node, *rest: Node) -> "UndirectedGraph":
        """
        Args:
            u: Node object.
            v: Node object.
            rest: Node objects.
        Disconnect a given node from a non-empty set of nodes, all present in the graph.
        """
        if not isinstance(u, Node):
            u = Node(u)
        for n in (v, *rest):
            if not isinstance(n, Node):
                n = Node(n)
            if n in self.neighbors(u):
                self.__neighbors[u].remove(n)
                self.__neighbors[n].remove(u)
                self.__links.remove(Link(u, n))
        return self

    def disconnect_all(self, n: Node, *rest: Node) -> "UndirectedGraph":
        if not rest:
            return self
        self.disconnect(n, *rest)
        return self.disconnect_all(*rest)

    def copy(self) -> "UndirectedGraph":
        return UndirectedGraph(self.neighbors())

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

    def complementary(self) -> "UndirectedGraph":
        res = UndirectedGraph({u: self.nodes for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def connected(self) -> bool:
        if len(self.links) + 1 < (n := len(self.nodes)):
            return False
        if self.degrees_sum > (n - 1) * (n - 2) or n < 2:
            return True
        queue, total = [u := self.nodes.pop()], {u}
        while queue:
            queue += list((next_nodes := self.neighbors(queue.pop(0)) - total))
            total.update(next_nodes)
        return total == self.nodes

    def is_tree(self, connected: bool = False) -> bool:
        """
        Args:
            connected: Boolean flag about whether the graph is already known to be connected.
        Returns:
            Whether the graph could be a tree.
        """
        return not self or len(self.nodes) == len(self.links) + 1 and (
                connected or self.connected())

    def tree(self, root: Node = None, dfs: bool = False) -> "Tree":
        """
        Args:
            root: a present node.
            dfs: a boolean flag, answering to whether the search algorithm should use DFS or BFS.
        Returns:
             A tree representation of the graph with root n.
        """

        from .tree import Tree

        if root is None:
            root = self.nodes.pop()
        if not isinstance(root, Node):
            root = Node(root)
        tree = Tree(root)
        rest, total = [root], {root}
        while rest:
            for v in self.neighbors(u := rest.pop(-bool(dfs))) - total:
                tree.add(u, v), rest.append(v), total.add(v)
        return tree

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

    def subgraph(self, nodes: Iterable[Node]) -> "UndirectedGraph":
        """
        Args:
            nodes: Given set of nodes.
        Returns:
            The subgraph, that only contains these nodes and all links between them.
        """
        try:
            nodes = self.nodes.intersection(nodes)
            return UndirectedGraph({u: self.neighbors(u).intersection(nodes) for u in nodes})
        except TypeError:
            raise TypeError("Iterable of nodes expected!")

    def component(self, u: Node) -> "UndirectedGraph":
        """
        Args:
            u: Given node.
        Returns:
            The connection component, to which a given node belongs.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        stack, total = [u], {u}
        while stack:
            stack += list((next_nodes := self.neighbors(stack.pop()) - total))
            total.update(next_nodes)
        return self.subgraph(total)

    def connection_components(self) -> list["UndirectedGraph"]:
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
        A bridge link is such, that if it's removed from the graph,
        it splits into one more connection component.
        Returns:
            All bridge links.
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
            tmp = UndirectedGraph.copy(self)
            return tmp.disconnect(u := (l := tmp.links.pop()).u, v := l.v).euler_walk(u, v) + [u]
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
            tmp = UndirectedGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i], path[i + 1])
            for i, u in enumerate(path):
                while tmp.neighbors(u):
                    curr = tmp.disconnect(u, v := tmp.neighbors(u).pop()).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j], curr[j + 1])
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def links_graph(self) -> "UndirectedGraph":
        """
        Returns:
            A graph, the nodes of which represent the links of the original one. Say two of its
            nodes represent links, which share a node in the original graph. This is shown by
            the fact, that these nodes are connected in the links graph. If the graph has node
            weights, they become link weights and if it has link weights, they become node weights.
        """
        return links_graph(self)

    def weighted_nodes_graph(self,
                             weights: dict[Node, float] = None) -> "WeightedNodesUndirectedGraph":
        if weights is None:
            weights = {n: 0 for n in self.nodes}
        for n in self.nodes - set(weights):
            weights[n] = 0
        return WeightedNodesUndirectedGraph(
            {n: (weights[n], self.neighbors(n)) for n in self.nodes})

    def weighted_links_graph(self,
                             weights: dict[Link, float] = None) -> "WeightedLinksUndirectedGraph":
        if weights is None:
            weights = {l: 0 for l in self.links}
        for l in self.links - set(weights):
            weights[l] = 0
        return WeightedLinksUndirectedGraph(
            {u: {v: weights[Link(u, v)] for v in self.neighbors(u)} for u in self.nodes})

    def weighted_graph(self, node_weights: dict[Node, float] = None,
                       link_weights: dict[Link, float] = None) -> "WeightedUndirectedGraph":
        if node_weights is None:
            node_weights = {n: 0 for n in self.nodes}
        for n in self.nodes - set(node_weights):
            node_weights[n] = 0
        if link_weights is None:
            link_weights = {l: 0 for l in self.links}
        for l in self.links - set(link_weights):
            link_weights[l] = 0
        return self.weighted_links_graph(link_weights).weighted_graph(node_weights)

    def interval_sort(self, start: Node = None) -> list[Node]:
        """
        Assume a set of intervals over the real number line, some of which could intersect. Such a
        set of intervals can be sorted based on multiple criteria. An undirected graph could be
        defined to represent the intervals the following way: Nodes represent the intervals and two
        nodes are connected exactly when the intervals they represent intersect.
        Args:
            start: A present node or None.
        Returns:
            A sort of the graph nodes, based on how early the interval a particular node could
            represent begins. If it fails, it returns an empty list. If start is given,
            it only tries to find a way to start from it.
        """

        def extend_0s(ll, max_l):
            return ll + (0,) * (max_l - len(ll))

        def bfs(g, res, limit):
            queue, total = [res], {res}
            while queue:
                for v in g.neighbors(queue.pop(0)) - total:
                    queue.append(v), total.add(v)
                    if limit(v):
                        res = v
            return res

        def helper(u, graph, priority):
            if graph.full():
                return [u, *sorted(graph.nodes - {u}, key=priority.get, reverse=True)]
            order, neighbors = [u], graph.neighbors(u)
            comps, total, final = [], {u}, set()
            for v in neighbors:
                if v not in total:
                    total.add(v)
                    rest, comp, this_final = {v}, {v}, False
                    while rest:
                        for _u in graph.neighbors(_ := rest.pop()):
                            if _u not in total:
                                if _u in neighbors:
                                    rest.add(_u), comp.add(_u), total.add(_u)
                                else:
                                    if final:
                                        return []
                                    this_final = True
                    if this_final:
                        final = sorted((graph.nodes - neighbors - {u}).union(comp),
                                       key=lambda x: (priority[x], x in neighbors), reverse=True)
                    else:
                        comps.append(sorted(comp, key=priority.get, reverse=True))
            max_length = max(map(len, comps)) if comps else 0
            comparison_function = lambda c: extend_0s(tuple(map(priority.get, c)), max_length)
            comps = sorted(comps, key=comparison_function, reverse=True)
            for i in range(len(comps) - 1):
                if priority[comps[i][-1]] < priority[comps[i + 1][0]]:
                    return []
            if final and comps and priority[final[0]] > priority[comps[-1][-1]]:
                return []
            final_neighbors = neighbors.intersection(final)
            if final and set(final[:len(final_neighbors)]) != final_neighbors:
                return []
            for comp in comps:
                max_priority = priority[comp[0]]
                curr_graph = graph.subgraph(comp)
                max_nodes = [n for n in comp if priority[n] == max_priority]
                start, d = max_nodes[0], curr_graph.excentricity(max_nodes[0])
                for n in max_nodes[1:]:
                    if (e := curr_graph.excentricity(n)) > d:
                        start, d = n, e
                new_neighbors = graph.neighbors(start)
                if not (curr_sort := helper(start, curr_graph,
                                            {k: 2 * priority[k] + (k in new_neighbors) for k in
                                             comp})):
                    return []
                order += curr_sort
            if set(order) == graph.nodes:
                return order
            max_priority = priority[final[0]]
            starting = bfs(curr_graph := graph.subgraph(final), final[-1],
                           lambda x: priority[x] == max_priority)
            new_neighbors = graph.neighbors(starting)
            if not (curr_sort := helper(starting, curr_graph,
                                        {k: 2 * priority[k] + (k in new_neighbors) for k in
                                         final})):
                return []
            return order + curr_sort

        if not self.connected():
            if start is None:
                result = []
                for component in self.connection_components():
                    if not (curr := component.interval_sort()):
                        return []
                    result += curr
                return result
            if not isinstance(start, Node):
                start = Node(start)
            components = self.connection_components()
            for c in components:
                if start in c:
                    begin = c
                    break
            else:
                raise KeyError("Unrecognized node!")
            components.remove(begin)
            components = [begin, *components]
            if not (result := components[0].interval_sort(start)):
                return []
            for component in components[1:]:
                if not (curr := component.interval_sort()):
                    return []
                result += curr
            return result
        if start is None:
            d = -1
            for u in self.nodes:
                if (e := self.excentricity(u)) > d:
                    start, d = u, e
        if not isinstance(start, Node):
            start = Node(start)
        if start not in self:
            raise KeyError("Unrecognized node!")
        return helper(start, self, {u: u in self.neighbors(start) for u in self.nodes})

    def is_full_k_partite(self, k: int = None) -> bool:
        """
        Args:
            k: number of partitions or None.
        Returns:
            Whether the graph has k independent sets and as many links as possible,
            given this condition. If k is not given, it can be anything.
        """
        if k is not None and not isinstance(k, int):
            try:
                k = int(k)
            except ValueError:
                raise TypeError("Integer expected!")
        return k in {None, len(comps := self.complementary().connection_components())} and all(
            c.full() for c in comps)

    def clique(self, n: Node, *rest: Node) -> bool:
        """
        Args:
            n: A present node.
            rest: A set of present nodes.
        Returns:
            Whether these given nodes form a clique.
        """

        def helper(nodes):
            if not nodes:
                return True
            u = nodes.pop()
            if any(v not in self.neighbors(u) for v in nodes):
                return False
            return helper(nodes)

        if not isinstance(n, Node):
            n = Node(n)
        rest = list(map(lambda x: x if isinstance(x, Node) else Node(x), rest))
        result = {n, *rest}.intersection(self.nodes)
        if result == self.nodes:
            return self.full()
        return helper(result)

    def cliques(self, k: int) -> list[set[Node]]:
        """
        Args:
            k: Wanted size of cliques.
        Returns:
            All cliques in the graph of size k.
        """
        try:
            k = int(k)
        except ValueError:
            raise TypeError("Integer expected!")
        if k < 0:
            return []
        if not k:
            return [set()]
        if not self.connected():
            return reduce(lambda x, y: x + y,
                          map(lambda g: g.cliques(k), self.connection_components()))
        result = []
        for p in combinations(self.nodes, abs(k)):
            if self.clique(*p):
                result.append(set(p))
        return result

    def max_cliques(self) -> list[set[Node]]:
        """
        Returns:
            All maximal by cardinality cliques in the graph.
        """
        result, low, high = [set()], 1, len(self.nodes)
        k = 0
        while low <= high:
            if curr := self.cliques(mid := (low + high) // 2):
                low = mid + 1
                if not result or mid > k:
                    result, k = curr, mid
            else:
                high = mid - 1
                k = 0
        return result

    def max_cliques_node(self, u: Node) -> list[set[Node]]:
        """
        Args:
            u: A present node.
        Returns:
            All, maximal by cardinality, cliques in the graph, to which node u belongs.
        """
        if not isinstance(u, Node):
            u = Node(u)
        return list(map({u}.union, self.subgraph(self.neighbors(u)).max_cliques()))

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

    def cliques_graph(self) -> "UndirectedGraph":
        """
        A cliques graph of a given one is a graph, the nodes of which represent the
        individual maximal by inclusion cliques in the original graph, and the links
        of which represent whether the cliques two nodes represent intersect.
        Returns:
            The clique graph of the original one.
        """
        result, cliques = UndirectedGraph(), self.complementary().maximal_independent_sets()
        for u in cliques:
            result.add(Node(frozenset(u)))
        for i, u in enumerate(cliques):
            for v in cliques[i + 1:]:
                if u.intersection(v):
                    result.connect(Node(frozenset(u)), Node(frozenset(v)))
        return result

    def chromatic_nodes_partition(self) -> list[set[Node]]:
        """
        Returns:
            A list of independent sets in the graph, that cover all nodes without intersecting.
            This list has as few elements as possible.
        """

        def helper(partition, union=set(), i=0):
            if union == self.nodes:
                return partition
            res, entered = list(map(lambda x: {x}, self.nodes)), False
            for j, s in enumerate(independent_sets[i:]):
                if {*s, *union} == self.nodes or s.isdisjoint(union):
                    entered = True
                    if len(curr := helper(partition + [s - union], {*s, *union}, i + j + 1)) == 2:
                        return curr
                    res = min(res, curr, key=len)
            if not entered:
                res = partition + self.subgraph(self.nodes - union).chromatic_nodes_partition()
            return res

        if not self.connected():
            r = [comp.chromatic_nodes_partition() for comp in self.connection_components()]
            final = r[0]
            for c in r[1:]:
                for i in range(min(len(c), len(final))):
                    final[i].update(c[i])
                for i in range(len(final), len(c)):
                    final.append(c[i])
            return final
        if self.is_full_k_partite():
            return [comp.nodes for comp in self.complementary().connection_components()]
        if self.is_tree(True):
            queue, c0, c1, total = [self.nodes.pop()], self.nodes, set(), set()
            while queue:
                flag = (u := queue.pop(0)) in c0
                for v in self.neighbors(u) - total:
                    if flag:
                        c1.add(v), c0.remove(v)
                    queue.append(v), total.add(v)
            return [c0, c1]
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
        independent_sets = self.maximal_independent_sets()
        return helper([])

    def chromatic_links_partition(self) -> list[set[Link]]:
        """
        Similar to chromatic nodes partition, except links,
        that share a node, are in different sets.
        """
        return [set(map(lambda x: x.value, s)) for s in
                UndirectedGraph.copy(self).links_graph().chromatic_nodes_partition()]

    def vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimal by cardinality set of nodes, that cover all links in the graph.
        """
        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimal by cardinality set of nodes, that cover all nodes in the graph.
        """

        def helper(curr, total, i=0):
            if total == self.nodes:
                return curr.copy()
            result = self.nodes
            for j, u in enumerate(list(nodes)[i:]):
                if len((res := helper({u, *curr}, {u, *self.neighbors(u), *total},
                                      i + j + 1))) < len(result):
                    result = res
            return result

        if not self.connected():
            return reduce(lambda x, y: x.union(y),
                          [comp.dominating_set() for comp in self.connection_components()])
        if self.is_tree(True):
            return self.tree().dominating_set()
        if self.is_full_k_partite():
            if not self:
                return set()
            res = {u := max(self.nodes, key=self.degrees)}
            if (neighbors := self.neighbors(u)) and {u, *neighbors} != self.nodes:
                res.add(neighbors.pop())
            return res
        nodes, isolated = self.nodes, set()
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
        return helper(isolated, isolated)

    def independent_set(self) -> set[Node]:
        """
        Returns:
            A maximal by cardinality set of nodes, no two of which are neighbors.
        """
        if not self.connected():
            return reduce(lambda x, y: x.union(y),
                          [comp.independent_set() for comp in self.connection_components()])
        if self.is_tree(True):
            return self.tree().independent_set()
        if self.is_full_k_partite():
            if not self:
                return set()
            return max([comp.nodes for comp in self.complementary().connection_components()],
                       key=len)
        if sort := self.interval_sort():
            result = set()
            for u in reversed(sort):
                if self.neighbors(u).isdisjoint(result):
                    result.add(u)
            return result
        return max(self.maximal_independent_sets(), key=len)

    def cycle_with_length(self, length: int) -> list[Node]:
        try:
            length = int(length)
        except ValueError:
            raise TypeError("Integer expected!")
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
        except ValueError:
            raise TypeError("Integer expected!")
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
                    tmp.add(x, *neighbors)
                    return True
            tmp.add(x, *neighbors)
            return False

        if (n := len(self.nodes)) == 1 or (
                2 * (m := len(self.links)) > (n - 1) * (n - 2) + 2 or n > 2 and all(
            2 * self.degrees(node) >= n for node in self.nodes)):
            return True
        if n > m or self.leaves or not self.connected():
            return False
        tmp = UndirectedGraph.copy(self)
        can_end_in = tmp.neighbors(u := self.nodes.pop())
        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        if not isinstance(u, Node):
            u = Node(u)
        if not isinstance(v, Node):
            v = Node(v)
        if u not in self or v not in self:
            raise KeyError("Unrecognized node(s).")
        if v in self.neighbors(u):
            return True if all(n in {u, v} for n in self.nodes) else self.hamilton_tour_exists()
        return UndirectedGraph.copy(self).connect(u, v).hamilton_tour_exists()

    def hamilton_tour(self) -> list[Node]:
        if len(self.nodes) == 1:
            return [self.nodes.pop()]
        if not self or self.leaves or not self.connected():
            return []
        u = self.nodes.pop()
        for v in self.neighbors(u):
            if res := self.hamilton_walk(u, v):
                return res + [u]
        return []

    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        def dfs(x, stack):
            if not tmp.degrees(x) or v is not None and not tmp.degrees(v) or not tmp.connected():
                return []
            too_many = v is not None
            for n in tmp.nodes:
                if n not in {x, v} and tmp.leaf(n):
                    if too_many:
                        return []
                    too_many = True
            if not tmp.nodes:
                return stack
            neighbors = tmp.neighbors(x)
            tmp.remove(x)
            if v is None and len(tmp.nodes) == 1 and tmp.nodes == neighbors:
                tmp.add(x, (y := neighbors.copy().pop()))
                return stack + [y]
            for y in neighbors:
                if y == v:
                    if tmp.nodes == {v}:
                        tmp.add(x, *neighbors)
                        return stack + [v]
                    continue
                if res := dfs(y, stack + [y]):
                    tmp.add(x, *neighbors)
                    return res
            tmp.add(x, *neighbors)
            return []

        if u is not None and not isinstance(u, Node):
            u = Node(u)
        if v is not None and not isinstance(v, Node):
            v = Node(v)
        tmp = UndirectedGraph.copy(self)
        if u is None:
            u, v = v, u
        if u is None:
            if v is not None and v not in self:
                raise KeyError("Unrecognized node.")
            for _u in self.nodes:
                if result := dfs(_u, [_u]):
                    return result
                if self.leaf(_u):
                    return []
            return []
        if u not in self or v is not None and v not in self:
            raise KeyError("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphic_bijection(self, other: "UndirectedGraph") -> dict[Node, Node]:
        if isinstance(other, UndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].add(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].add(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in
                   this_nodes_degrees):
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
                        if (m in self.neighbors(n)) ^ (v in other.neighbors(u)):
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

    def __add__(self, other: "UndirectedGraph") -> "UndirectedGraph":
        """
        Args:
            other: another UndirectedGraph object.
        Returns:
            Combination of two undirected graphs.
        """
        return combine_graphs(self, other)

    def __eq__(self, other: "UndirectedGraph") -> bool:
        if type(other) == UndirectedGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self) -> str:
        return f"<{self.nodes}, {self.links}>"

    def __repr__(self) -> str:
        return str(self)


class WeightedNodesUndirectedGraph(UndirectedGraph):
    """
    Class for implementing and undirected graph with weights on the nodes.
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, Iterable[Node]]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary, that maps a node
            to a tuple with its weight and its neighbors.
        """
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
            n: A present node or None.
        Returns:
            The weight of node n or the dictionary with all node weights.
        """
        if n is not None and not isinstance(n, Node):
            n = Node(n)
        return self.__node_weights.copy() if n is None else self.__node_weights.get(n)

    @property
    def total_nodes_weight(self) -> float:
        """
        Returns:
            The sum of all node weights.
        """
        return sum(self.node_weights().values())

    def add(self, n_w: tuple[Node, float], *current_nodes: Node) -> "WeightedNodesUndirectedGraph":
        UndirectedGraph.add(self, n_w[0], *current_nodes)
        n = n_w[0] if isinstance(n_w[0], Node) else Node(n_w[0])
        if n not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedNodesUndirectedGraph":
        for u in (n, *rest):
            if not isinstance(u, Node):
                u = Node(u)
            if u in self.node_weights():
                self.__node_weights.pop(u)
        return super().remove(n, *rest)

    def set_weight(self, u: Node, w: float) -> "WeightedNodesUndirectedGraph":
        """
        Args:
            u: A present node.
            w: The new weight of node u.
        Set the weight of node u to w.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            try:
                self.__node_weights[u] = float(w)
            except ValueError:
                raise TypeError("Real value expected!")
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedNodesUndirectedGraph":
        """
        Args:
            u: A present node.
            w: A real value.
        Increase the weight of node u by w.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self.node_weights():
            try:
                self.set_weight(u, self.node_weights(u) + float(w))
            except ValueError:
                raise TypeError("Real value expected!")
        return self

    def copy(self) -> "WeightedNodesUndirectedGraph":
        return WeightedNodesUndirectedGraph(
            {n: (self.node_weights(n), self.neighbors(n)) for n in self.nodes})

    def complementary(self) -> "WeightedNodesUndirectedGraph":
        res = WeightedNodesUndirectedGraph(
            {u: (self.node_weights(u), self.nodes) for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def weighted_tree(self, n: Node = None, dfs: bool = False) -> "WeightedTree":
        """
        Args:
            n: A present node.
            dfs: a boolean flag, answering to whether the search algorithm should use DFS or BFS.
        Returns:
             A weighted tree representation of the graph with root n.
        """

        from .tree import WeightedTree

        if n is None:
            n = self.nodes.pop()
        if not isinstance(n, Node):
            n = Node(n)
        tree = WeightedTree((n, self.node_weights(n)))
        queue, total = [n], {n}
        while queue:
            for v in self.neighbors(u := queue.pop(-bool(dfs))) - total:
                tree.add(u, {v: self.node_weights(v)}), queue.append(v), total.add(v)
        return tree

    def subgraph(self, nodes: Iterable[Node]) -> "WeightedNodesUndirectedGraph":
        try:
            nodes = self.nodes.intersection(nodes)
            neighborhood = {u: (self.node_weights(u), self.neighbors(u).intersection(nodes)) for u
                            in nodes}
            return WeightedNodesUndirectedGraph(neighborhood)
        except TypeError:
            raise TypeError("Iterable of nodes expected!")

    def weighted_graph(self, weights: dict[Link, float] = None) -> "WeightedUndirectedGraph":
        if weights is None:
            weights = {l: 0 for l in self.links}
        for l in self.links - set(weights):
            weights[l] = 0
        return super().weighted_graph(self.node_weights(), weights)

    def minimal_path_nodes(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
            A path between u and v with the least possible sum of node weights.
        """
        return self.weighted_graph().minimal_path(u, v)

    def weighted_vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimal by sum of the weights set of nodes, that cover all links in the graph.
        """
        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimal by sum of the weights set of nodes, that cover all nodes in the graph.
        """

        def helper(curr, total, total_weight, i=0):
            if total == self.nodes:
                return curr.copy(), total_weight
            result, result_sum = self.nodes, self.total_nodes_weight
            for j, u in enumerate(list(nodes)[i:]):
                cover, weight = helper({u, *curr}, {u, *self.neighbors(u), *total},
                                       total_weight + self.node_weights(u), i + j + 1)
                if weight < result_sum:
                    result, result_sum = cover, weight
            return result, result_sum

        if not self:
            return set()
        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.weighted_dominating_set() for comp in
                                                    self.connection_components()])
        if self.is_tree(True):
            return self.weighted_tree().weighted_dominating_set()
        nodes, isolated, weights = self.nodes, set(), 0
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
                weights += self.node_weights(n)
        return helper(isolated, isolated, weights)[0]

    def weighted_independent_set(self) -> set[Node]:
        """
        Returns:
            A set of non-neighboring nodes with a maximal possible sum of the weights.
        """

        def helper(curr, total=set(), res_sum=0.0, i=0):
            if total == self.nodes:
                return curr, res_sum
            result, result_sum = set(), 0
            for j, u in enumerate(list(self.nodes)[i:]):
                if u not in total and self.node_weights(u) > 0:
                    cover, weight = helper({u, *curr}, {u, *total, *self.neighbors(u)},
                                           res_sum + self.node_weights(u), i + j + 1)
                    if weight > result_sum:
                        result, result_sum = cover, weight
            return result, result_sum

        if not self:
            return set()
        if not self.connected():
            return reduce(lambda x, y: x.union(y),
                          [comp.independent_set() for comp in self.connection_components()])
        if self.is_tree(True):
            return self.weighted_tree().weighted_independent_set()
        weights = self.total_nodes_weight
        return helper(set())[0]

    def isomorphic_bijection(self, other: UndirectedGraph) -> dict[Node, Node]:
        if isinstance(other, WeightedNodesUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_weights, other_weights = defaultdict(int), defaultdict(int)
            for w in self.node_weights().values():
                this_weights[w] += 1
            for w in other.node_weights().values():
                other_weights[w] += 1
            if this_weights != other_weights:
                return {}
            this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].add(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].add(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in
                   this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(map(list, this_nodes_degrees.values()), key=len)
            other_nodes_degrees = sorted(map(list, other_nodes_degrees.values()), key=len)
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
                        if (m in self.neighbors(n)) ^ (
                                v in other.neighbors(u)) or self.node_weights(
                            m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphic_bijection(other)

    def __eq__(self, other: "WeightedNodesUndirectedGraph") -> bool:
        if type(other) == WeightedNodesUndirectedGraph:
            return (self.node_weights(), self.links) == (other.node_weights(), other.links)
        return False

    def __str__(self) -> str:
        return "<{" + ", ".join(
            f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, " + str(self.links) + ">"


class WeightedLinksUndirectedGraph(UndirectedGraph):
    """
    Class for implementing and undirected graph with weights on the links.
    """

    def __init__(self, neighborhood: dict[Node, dict[Node, float]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary of nodes and another dictionary, associated
            with each node. Each such dictionary has for keys the neighbors of said
            node and the value of each neighbor is the weight of the link between them.
        """
        super().__init__()
        self.__link_weights = {}
        for u, neighbors in neighborhood.items():
            self.add(u)
            for v, w in neighbors.items():
                self.add(v, {u: w}), self.connect(v, {u: w})

    def link_weights(self, u_or_l: Node | Link = None, v: Node = None) -> dict[Node, float] | dict[
        Link, float] | float:
        """
        Args:
            u_or_l: Given first node, a link or None.
            v: Given second node or None.
        Returns:
            Information about link weights the following way:
            If no argument is passed, return the weights of all links;
            if a link or two nodes are passed, return the weight of the given link between them;
            If one node is passed, return a dictionary with all of its neighbors and the weight
            of the link it shares with each of them.
        """
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Link):
            return self.__link_weights.get(u_or_l)
        else:
            if v is None:
                return {n: self.__link_weights[Link(n, u_or_l)] for n in self.neighbors(u_or_l)}
            return self.__link_weights.get(Link(u_or_l, v))

    @property
    def total_links_weight(self) -> float:
        """
        Returns:
            The sum of all link weights.
        """
        return sum(self.link_weights().values())

    def add(self, u: Node,
            nodes_weights: dict[Node, float] = {}) -> "WeightedLinksUndirectedGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            UndirectedGraph.add(self, u, *nodes_weights.keys())
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedLinksUndirectedGraph":
        for u in (n, *rest):
            if u in self:
                for v in self.neighbors(u):
                    self.__link_weights.pop(Link(u, v))
        return super().remove(n, *rest)

    def connect(self, u: Node,
                nodes_weights: dict[Node, float] = {}) -> "WeightedLinksUndirectedGraph":
        if not isinstance(u, Node):
            u = Node(u)
        if nodes_weights:
            super().connect(u, *nodes_weights.keys())
        if u in self:
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def connect_all(self, u: Node, *rest: Node) -> "WeightedLinksUndirectedGraph":
        if not rest:
            return self
        self.connect(u, {v: 0 for v in rest})
        return self.connect_all(*rest)

    def disconnect(self, u: Node, v: Node, *rest: Node) -> "WeightedLinksUndirectedGraph":
        super().disconnect(u, v, *rest)
        for n in (v, *rest):
            if (l := Link(u, n)) in self.link_weights():
                self.__link_weights.pop(l)
        return self

    def set_weight(self, l: Link, w: float) -> "WeightedLinksUndirectedGraph":
        """
        Args:
            l: A present link.
            w: The new weight of link l.
        Set the weight of link l to w.
        """
        try:
            if l in self.links:
                self.__link_weights[l] = float(w)
            return self
        except TypeError:
            raise TypeError("Real value expected!")

    def increase_weight(self, l: Link, w: float) -> "WeightedLinksUndirectedGraph":
        """
        Args:
            l: A present link.
            w: A real value.
        Increase the weight of link l with w.
        """
        try:
            if l in self.link_weights():
                self.set_weight(l, self.link_weights(l) + float(w))
            return self
        except ValueError:
            raise TypeError("Real value expected!")

    def copy(self) -> "WeightedLinksUndirectedGraph":
        return WeightedLinksUndirectedGraph({n: self.link_weights(n) for n in self.nodes})

    def subgraph(self, nodes: Iterable[Node]) -> "WeightedLinksUndirectedGraph":
        try:
            nodes = self.nodes.intersection(nodes)
            neighborhood = {u: {k: v for k, v in self.link_weights(u).items() if k in nodes} for u
                            in nodes}
            return WeightedLinksUndirectedGraph(neighborhood)
        except TypeError:
            raise TypeError("Iterable of nodes expected!")

    def minimal_spanning_tree(self) -> set[Link]:
        """
        Returns:
            A spanning tree or forrest of trees of the graph with the minimal possible weights sum.
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
            return reduce(lambda x, y: x.union(y.minimal_spanning_tree()),
                          self.connection_components(), set())
        if self.is_tree(True):
            return self.links
        res_links, total = set(), {u := self.nodes.pop()}
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

    def weighted_graph(self, weights: dict[Node, float] = None) -> "WeightedUndirectedGraph":
        if weights is None:
            weights = {n: 0 for n in self.nodes}
        for n in self.nodes - set(weights):
            weights[n] = 0
        return WeightedUndirectedGraph({n: (weights[n], self.link_weights(n)) for n in self.nodes})

    def minimal_path_links(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
            A path between u and v with the least possible sum of link weights.
        """
        return self.weighted_graph().minimal_path(u, v)

    def isomorphic_bijection(self, other: UndirectedGraph) -> dict[Node, Node]:
        if isinstance(other, WeightedLinksUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_weights, other_weights = defaultdict(int), defaultdict(int)
            for w in self.link_weights().values():
                this_weights[w] += 1
            for w in other.link_weights().values():
                other_weights[w] += 1
            if this_weights != other_weights:
                return {}
            this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].add(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].add(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in
                   this_nodes_degrees):
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
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphic_bijection(other)

    def __eq__(self, other: "WeightedLinksUndirectedGraph") -> bool:
        if type(other) == WeightedLinksUndirectedGraph:
            return (self.nodes, self.link_weights()) == (other.nodes, other.link_weights())
        return False

    def __str__(self) -> str:
        return "<" + str(self.nodes) + ", {" + ", ".join(
            f"{l} -> {self.link_weights(l)}" for l in self.links) + "}>"


class WeightedUndirectedGraph(WeightedLinksUndirectedGraph, WeightedNodesUndirectedGraph):
    """
    Class for implementing an undirected graph with weights on the nodes and the links.
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, dict[Node, float]]] = {}) -> None:
        """
        Args:
            neighborhood: A dictionary of nodes and a tuple with each node's weight and another
            dictionary, associated with each node. Each such dictionary has for keys the neighbors
            of said node and the value of each neighbor is the weight of the link between them.
        """
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
            The sum of all weights in the graph.
        """
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: tuple[Node, float],
            nodes_weights: dict[Node, float] = {}) -> "WeightedUndirectedGraph":
        super().add(n_w[0], nodes_weights)
        n = n_w[0] if isinstance(n_w[0], Node) else Node(n_w[0])
        if n not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedUndirectedGraph":
        for n in (u, *rest):
            if not isinstance(n, Node):
                n = Node(n)
            if n in self:
                if tmp := self.neighbors(n):
                    super().disconnect(n, *tmp)
        WeightedNodesUndirectedGraph.remove(self, u, *rest)
        return self

    def set_weight(self, el: Node | Link, w: float) -> "WeightedUndirectedGraph":
        """
        Args:
            el: A present node or link.
            w: The new weight of object el.
        Set the weight of object el to w.
        """
        if not isinstance(el, (Node, Link)):
            el = Node(el)
        try:
            if el in self:
                WeightedNodesUndirectedGraph.set_weight(self, el, float(w))
            elif el in self.links:
                super().set_weight(el, float(w))
            return self
        except ValueError:
            raise TypeError("Real value expected!")

    def increase_weight(self, el: Node | Link, w: float) -> "WeightedUndirectedGraph":
        """
        Args:
            el: A present node or link.
            w: A real value.
        Increase the weight of object el with w.
        """
        try:
            if el in self.link_weights():
                self.set_weight(el, self.link_weights(el) + float(w))
            else:
                if not isinstance(el, Node):
                    el = Node(el)
                if el in self.node_weights():
                    return self.set_weight(el, self.node_weights(el) + float(w))
            return self
        except ValueError:
            raise TypeError("Real value expected!")

    def copy(self) -> "WeightedUndirectedGraph":
        neighborhood = {n: (self.node_weights(n), self.link_weights(n)) for n in self.nodes}
        return WeightedUndirectedGraph(neighborhood)

    def subgraph(self, nodes: Iterable[Node]) -> "WeightedUndirectedGraph":
        try:
            nodes = self.nodes.intersection(nodes)
            neighborhood = {u: (
                self.node_weights(u),
                {k: v for k, v in self.link_weights(u).items() if k in nodes})
                for u in nodes}
            return WeightedUndirectedGraph(neighborhood)
        except TypeError:
            raise TypeError("Iterable of nodes expected!")

    def minimal_path(self, u: Node, v: Node) -> list[Node]:
        """
        Args:
            u: First given node.
            v: Second given node.
        Returns:
            A path between u and v with the least possible sum of node and link weights.
        """

        def dfs(x, current_path, current_weight, total_negative, res_path=None, res_weight=0):
            def dijkstra(s, curr_path, curr_weight):
                curr_tmp = tmp.copy()
                for l in curr_path:
                    curr_tmp.disconnect(l.u, l.v)
                paths = {n: {m: [] for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                weights_from_to = {n: {m: curr_tmp.total_weight for m in curr_tmp.nodes} for n in
                                   curr_tmp.nodes}
                for n in curr_tmp.nodes:
                    weights_from_to[n][n] = 0
                    for m in curr_tmp.neighbors(n):
                        weights_from_to[n][m] = curr_tmp.link_weights(n,
                                                                      m) + curr_tmp.node_weights(m)
                        paths[n][m] = [Link(n, m)]
                for x1 in curr_tmp.nodes:
                    for x2 in curr_tmp.nodes:
                        for x3 in curr_tmp.nodes:
                            if (new_weight := weights_from_to[x1][x2] + weights_from_to[x2][x3]) < \
                                    weights_from_to[x1][x3]:
                                weights_from_to[x1][x3] = new_weight
                                paths[x1][x3] = paths[x1][x2] + paths[x2][x3]
                return curr_path + paths[s][v], curr_weight + weights_from_to[s][v]

            if res_path is None:
                res_path = []
            if total_negative:
                for y in filter(lambda _y: Link(x, _y) not in current_path, tmp.neighbors(x)):
                    new_curr_w = current_weight + (l_w := tmp.link_weights(x, y)) + (
                        n_w := tmp.node_weights(y))
                    new_total_negative = total_negative
                    if n_w < 0:
                        new_total_negative -= n_w
                    if l_w < 0:
                        new_total_negative -= l_w
                    if new_curr_w + new_total_negative >= res_weight and res_path:
                        continue
                    if y == v and (new_curr_w < res_weight or not res_path):
                        res_path = current_path + [Link(x, y)]
                        res_weight = current_weight + l_w + n_w
                    curr = dfs(y, current_path + [Link(x, y)], new_curr_w, new_total_negative,
                               res_path, res_weight)
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
            if v in (tmp := self.component(u)):
                nodes_negative_weights = sum(
                    tmp.node_weights(n) for n in tmp.nodes if tmp.node_weights(n) < 0)
                links_negative_weights = sum(
                    tmp.link_weights(l) for l in tmp.links if tmp.link_weights(l) < 0)
                res = dfs(u, [], tmp.node_weights(u),
                          nodes_negative_weights + links_negative_weights)
                return [l.u for l in res[0]] + [res[0][-1].v]
            return []
        raise KeyError("Unrecognized node(s)!")

    def isomorphic_bijection(self, other: UndirectedGraph) -> dict[Node, Node]:
        if isinstance(other, WeightedUndirectedGraph):
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
            this_nodes_degrees, other_nodes_degrees = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].add(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].add(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in
                   this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(map(list, this_nodes_degrees.values()), key=len)
            other_nodes_degrees = sorted(map(list, other_nodes_degrees.values()), key=len)
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
                        if self.link_weights(Link(n, m)) != other.link_weights(
                                Link(u, v)) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            res = type(other).isomorphic_bijection(other, self)
            return {v: k for k, v in res.items()}
        return UndirectedGraph.isomorphic_bijection(self, other)

    def __eq__(self, other: "WeightedUndirectedGraph") -> bool:
        if type(other) == WeightedUndirectedGraph:
            return (self.node_weights(), self.link_weights()) == (
                other.node_weights(), other.link_weights())
        return False

    def __str__(self) -> str:
        return "<{" + ", ".join(
            f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, {" + ", ".join(
            f"{l} -> {self.link_weights(l)}" for l in self.links) + "}>"
