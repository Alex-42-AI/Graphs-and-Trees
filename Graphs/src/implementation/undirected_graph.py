from functools import reduce

from collections import defaultdict

from itertools import permutations, combinations, product

from Personal.Graphs.src.implementation.general import *


class Link:
    """
    Helper class, implementing an undirected link.
    """

    def __init__(self, u: Node, v: Node) -> None:
        self.__u, self.__v = u, v

    @property
    def u(self) -> Node:
        """
        Get the first given node.
        """
        return self.__u

    @property
    def v(self) -> Node:
        """
        Get the second given node.
        """
        return self.__v

    def __contains__(self, item: Node) -> bool:
        """
        item: a Node object.
        Check whether given node is in the link.
        """
        return item in {self.u, self.v}

    def __hash__(self) -> int:
        return hash(frozenset({self.u, self.v}))

    def __eq__(self, other) -> bool:
        if isinstance(other, Link):
            return {self.u, self.v} == {other.u, other.v}
        return False

    def __str__(self) -> str:
        return f"{self.u}-{self.v}"

    __repr__: str = __str__


class UndirectedGraph(Graph):
    """
    Class for implementing an unweighted undirected graph.
    """

    def __init__(self, neighborhood: dict[Node, Iterable[Node]] = {}) -> None:
        self.__nodes, self.__links = set(), set()
        self.__neighboring, self.__degrees = {}, {}
        for u, neighbors in neighborhood.items():
            if u not in self:
                self.add(u)
            for v in neighbors:
                self.add(v, u), self.connect(u, v)

    @property
    def nodes(self) -> set[Node]:
        """
        Get nodes.
        """
        return self.__nodes.copy()

    @property
    def links(self) -> set[Link]:
        """
        Get links.
        """
        return self.__links.copy()

    @property
    def leaves(self) -> set[Node]:
        """
        Get leaves.
        """
        return {n for n in self.nodes if self.leaf(n)}

    def leaf(self, n: Node) -> bool:
        """
        Check whether a given node is a leaf (if it has a degree of 1).
        """
        return self.degrees(n) == 1

    def neighbors(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Get neighbors of a given node or the neighboring dictionary of all nodes.
        """
        return (self.__neighboring if u is None else self.__neighboring[u]).copy()

    def degrees(self, u: Node = None) -> dict[Node, int] | int:
        """
        Get degree of a given node or the degrees dictionary of all nodes.
        """
        return self.__degrees.copy() if u is None else self.__degrees[u]

    @property
    def degrees_sum(self) -> int:
        """
        Get the total sum of all degrees.
        """
        return 2 * len(self.links)

    def add(self, u: Node, *current_nodes: Node) -> "UndirectedGraph":
        """
        Add a new node to already present nodes.
        """
        if u not in self:
            self.__nodes.add(u)
            self.__degrees[u], self.__neighboring[u] = 0, set()
            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)
        return self

    def remove(self, n: Node, *rest: Node) -> "UndirectedGraph":
        """
        Remove a non-empty set of nodes.
        """
        for u in (n, *rest):
            if u in self:
                if tmp := self.neighbors(u):
                    self.disconnect(u, *tmp)
                self.__nodes.remove(u), self.__degrees.pop(u), self.__neighboring.pop(u)
        return self

    def connect(self, u: Node, v: Node, *rest: Node) -> "UndirectedGraph":
        """
        Connect a given node to a non-empty set of nodes, all present.
        """
        for n in (v, *rest):
            if u != n and n not in self.neighbors(u) and n in self:
                self.__degrees[u] += 1
                self.__degrees[n] += 1
                self.__neighboring[u].add(n), self.__neighboring[n].add(u), self.__links.add(Link(u, n))
        return self

    def connect_all(self, u: Node, *rest: Node) -> "UndirectedGraph":
        """
        Add a link between all given nodes.
        """
        if not rest:
            return self
        self.connect(u, *rest)
        return self.connect_all(*rest)

    def disconnect(self, u: Node, v: Node, *rest: Node) -> "UndirectedGraph":
        """
        Disconnect a given node from a non-empty set of nodes, all present.
        """
        for n in (v, *rest):
            if n in self.neighbors(u):
                self.__degrees[u] -= 1
                self.__degrees[n] -= 1
                self.__neighboring[u].remove(n), self.__neighboring[n].remove(u), self.__links.remove(Link(u, n))
        return self

    def disconnect_all(self, n: Node, *rest: Node) -> "UndirectedGraph":
        """
        Disconnect all given nodes.
        """
        if not rest:
            return self
        self.disconnect(n, *rest)
        return self.disconnect_all(*rest)

    def copy(self) -> "UndirectedGraph":
        """
        Return an identical copy of the graph.
        """
        return UndirectedGraph(self.neighbors())

    def excentricity(self, u: Node) -> int:
        """
        Get the excentricity of a given node (the length of the longest of all shortest paths, that start from it).
        """
        res, total, queue = 0, {u}, [u]
        while queue:
            new = []
            while queue:
                for v in filter(lambda x: x not in total, self.neighbors(_ := queue.pop(0))):
                    new.append(v), total.add(v)
            queue = new.copy()
            res += bool(new)
        return res

    def diameter(self) -> int:
        """
        Return the greatest of all excentricity values of any node.
        """
        return max(self.excentricity(u) for u in self.nodes)

    def complementary(self) -> "UndirectedGraph":
        """
        Return a graph, where there are links between nodes exactly
        where there are no links between nodes in the original graph.
        """
        res = UndirectedGraph({u: self.nodes for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def connection_components(self) -> list["UndirectedGraph"]:
        """
        List out all connected components of the graph.
        """
        components, rest = [], self.nodes
        while rest:
            components.append(curr := self.component(rest.pop()))
            rest -= curr.nodes
        return components

    def connected(self) -> bool:
        """
        Check whether the graph is connected.
        """
        if len(self.links) + 1 < (n := len(self.nodes)):
            return False
        if self.degrees_sum > (n - 1) * (n - 2) or n < 2:
            return True
        queue, total = [u := self.nodes.pop()], {u}
        while queue:
            for v in filter(lambda x: x not in total, self.neighbors(queue.pop(0))):
                total.add(v), queue.append(v)
        return total == self.nodes

    def is_tree(self) -> bool:
        """
        Check whether the graph could be a tree.
        """
        return len(self.nodes) == len(self.links) + bool(self.nodes) and self.connected()

    def tree(self, n: Node, depth: bool = False) -> "Tree":
        """
        Return a tree representation of the graph with a given node as a root.
        The depth flag asks whether the nodes traversing algorithm is DFS or BFS.
        """

        from Personal.Graphs.src.implementation.tree import Tree

        tree = Tree(n)
        rest, total = [n], {n}
        while rest:
            for v in filter(lambda x: x not in total, self.neighbors(u := rest.pop(-depth))):
                tree.add(u, v), rest.append(v), total.add(v)
        return tree

    def cycle_with_length_3(self) -> list[Node]:
        """
        Return a cycle with a length of 3 if such exists, otherwise an empty list.
        """
        for l in self.links:
            if intersection := self.neighbors(u := l.u).intersection(self.neighbors(v := l.v)):
                return [u, v, intersection.pop()]
        return []

    def planar(self) -> bool:
        """
        Check whether the graph is planar (whether it could be drawn on a flat surface without intersecting).
        """
        return all(len(tmp.links) <= (2 + bool(tmp.cycle_with_length_3())) * (len(tmp.nodes) - 2) for tmp in self.connection_components())

    def reachable(self, u: Node, v: Node) -> bool:
        """
        Check whether the given nodes can reach one-another.
        """
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        if v in {u, *self.neighbors(u)}:
            return True
        return v in self.component(u)

    def component(self, u: Node) -> "UndirectedGraph":
        """
        Return the connection component of a given node in the graph.
        """
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], UndirectedGraph({u: []})
        while queue:
            for n in self.neighbors(v := queue.pop(0)):
                if n in res.nodes:
                    res.connect(v, n)
                else:
                    res.add(n, v), queue.append(n)
        return res

    def subgraph(self, nodes: Iterable[Node]) -> "UndirectedGraph":
        """
        Return the subgraph, that only contains the nodes in the given set and all links between them.
        """
        return UndirectedGraph({u: self.neighbors(u).intersection(nodes) for u in self.nodes.intersection(nodes)})

    def cut_nodes(self) -> set[Node]:
        """
        A cut node in a graph is such, that if it's removed, the graph splits into more connection components.
        This method lists out all such nodes in the graph.
        """

        def dfs(u: Node, l: int):
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

    def bridge_links(self) -> set[Node]:
        """
        A bridge link is such, that if it's removed from the graph, it splits into one more connection component.
        This method lists out all such links in the graph.
        """

        def dfs(u: Node, l: int):
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
        """
        Check whether the graph is fully connected.
        """
        return self.degrees_sum == (n := len(self.nodes)) * (n - 1)

    def get_shortest_path(self, u: Node, v: Node) -> list[Node]:
        """
        Return one shortest path between the given nodes, if such path exists, otherwise empty list.
        """

        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        previous, queue, total = {}, [u], {u}
        while queue:
            if (n := queue.pop(0)) == v:
                res, curr_node = [n], n
                while curr_node != u:
                    res.insert(0, previous[curr_node])
                    curr_node = previous[curr_node]
                return res
            for m in filter(lambda x: x not in total, self.neighbors(n)):
                queue.append(m), total.add(m)
                previous[m] = n
        return []

    def euler_tour_exists(self) -> bool:
        """
        Check if the graph is Eulerian.
        """
        for n in self.nodes:
            if self.degrees(n) % 2:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Check if the graph has an Euler walk between two given nodes.
        """
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if u == v:
            return self.euler_tour_exists()
        for n in self.nodes:
            if self.degrees(n) % 2 - (n in {u, v}):
                return False
        return self.connected()

    def euler_tour(self) -> list[Node]:
        """
        Return an Euler tour if such exists, otherwise an empty list.
        """
        if self.euler_tour_exists():
            tmp = UndirectedGraph.copy(self)
            return tmp.disconnect(u := (l := tmp.links.pop()).u, v := l.v).euler_walk(u, v)
        return []

    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        """
        Return an Euler walk between two given nodes if such exists, otherwise an empty list.
        """
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
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
        This is a graph, the nodes of which represent the links of the original one. Say two
        of its nodes represent links, which share a node in the original graph. This is shown
        by the fact, that these nodes are connected in the links graph. If the graph has node
        weights, they become link weights and if it has link weights, they become node weights.
        """
        neighborhood = {Node(l0): [Node(l1) for l1 in self.links if (l1.u in l0) ^ (l1.v in l0)] for l0 in self.links}
        return UndirectedGraph(neighborhood)

    def interval_sort(self, start: Node = None) -> list[Node]:
        """
        Assume a set of intervals on the real numberline, some of which could intersect.
        Such a set of intervals can be sorted based on multiple criteria.
        An undirected graph could be defined to represent the intervals the following way:
        Nodes represent the intervals and two nodes are connected exactly when the intervals
        they represent intersect. This method tries to find a sort of the graph nodes, based on
        how early the interval a particular node could represent begins. If it fails, it returns an empty list.
        If a starting node is given, it necessarily tries to find a way to start from it.
        """

        def extract(nodes, predicate):
            first, second = [], []
            for u in nodes:
                if predicate(u):
                    first.append(u)
                else:
                    second.append(u)
            return first + second

        def bfs(graph, res=None, limit=lambda _: True):
            if res is None:
                res = graph.nodes.pop()
            queue, total = [res], {res}
            while queue:
                for v in graph.neighbors(queue.pop(0)) - total:
                    queue.append(v), total.add(v)
                    if limit(v):
                        res = v
            return res

        def helper(u):
            if self.full():
                return [u, *(self.nodes - {u})]
            order, neighbors = [u], self.neighbors(u)
            comps, total, final = [], {u}, set()
            for v in neighbors:
                if v not in total:
                    total.add(v)
                    rest, comp, this_final = {v}, {v}, False
                    while rest:
                        for _u in self.neighbors(_ := rest.pop()):
                            if _u not in total:
                                if _u in neighbors:
                                    rest.add(_u), comp.add(_u), total.add(_u)
                                else:
                                    if final:
                                        return []
                                    this_final = True
                    if this_final:
                        final = comp.copy()
                    else:
                        comps.append(comp)
            final.update(self.nodes - neighbors.union({u}))
            for comp in comps:
                starting = bfs(curr := self.subgraph(comp))
                if not (curr_sort := curr.interval_sort(starting)):
                    return []
                order += curr_sort
            if set(order) == self.nodes:
                return order
            final_neighbors = neighbors.intersection(final)
            avoid = final - neighbors
            starting = None
            for v in final_neighbors:
                if not (self.neighbors(v) - {u}).issubset(avoid):
                    starting = bfs(self.subgraph(final_neighbors), v, lambda x: self.neighbors(x).isdisjoint(avoid))
                    break
            else:
                starting = final_neighbors.pop()
            if not (curr_sort := self.subgraph(final).interval_sort(starting)):
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
            components = extract(self.connection_components(), lambda g: start in g)
            if not components[0].interval_sort(start):
                return []
            result = []
            for component in components[1:]:
                if not (curr := component.interval_sort()):
                    return []
                result += curr
            return result
        if start is None:
            for n in self.nodes:
                if result := helper(n):
                    return result
            return []
        if start not in self:
            raise ValueError("Unrecognized node!")
        return helper(start)

    def is_full_k_partite(self, k: int = None) -> bool:
        """
        Check whether the graph has k independent sets and as many links as possible,
        given this condition. k doesn't have to be given.
        """
        return k in {None, len(comps := self.complementary().connection_components())} and all(c.full() for c in comps)

    def clique(self, n: Node, *nodes: Node) -> bool:
        """
        Check whether given nodes from the graph
        are all connected to one-another.
        """
        if {n, *nodes} == self.nodes:
            return self.full()
        if not nodes:
            return True
        if any(u not in self.neighbors(n) for u in nodes):
            return False
        return self.clique(*nodes)

    def cliques(self, k: int) -> list[set[Node]]:
        """
        Return all cliques in the graph of size k.
        """
        return [set(p) for p in combinations(self.nodes, abs(k)) if self.clique(*p)]

    def max_cliques_node(self, u: Node) -> list[set[Node]]:
        """
        Return all, maximum by cardinality, cliques in the graph, to which a given node belongs.
        """
        if (tmp := self.component(u)).full():
            return [tmp.nodes]
        if not (neighbors := tmp.neighbors(u)):
            return [{u}]
        cliques = [{u, v} for v in neighbors]
        while True:
            new = set()
            for i, cl1 in enumerate(cliques):
                for cl2 in cliques[i + 1:]:
                    compatible = True
                    for x in cl1 - {u}:
                        for y in cl2 - cl1:
                            if x != y and y not in tmp.neighbors(x):
                                compatible = False
                                break
                        if not compatible:
                            break
                    if compatible:
                        new.add(frozenset(cl1.union(cl2)))
            if not new:
                max_card = max(map(len, cliques))
                return [set(cl) for cl in cliques if len(cl) == max_card]
            newer = new.copy()
            for cl1 in new:
                for cl2 in new:
                    if len(cl1) < len(cl2):
                        newer.remove(cl1)
                        break
            cliques = list(newer)

    def max_cliques(self) -> list[set[Node]]:
        """
        Return all maximum by cardinality cliques in the graph.
        """
        result, low, high = [set()], 1, len(self.nodes)
        while low <= high:
            if not (curr := self.cliques(mid := (low + high) // 2)):
                high = mid - 1
            else:
                low = mid + 1
                if len(curr[0]) > len(result[0]):
                    result = curr
        return result

    def all_maximal_cliques_node(self, u: Node) -> list[set[Node]]:
        """
        Return all maximal by inclusion cliques in the graph, to which a given node belongs.
        """
        if (tmp := self.component(u)).full():
            return [tmp.nodes]
        if not (neighbors := tmp.neighbors(u)):
            return [{u}]
        cliques = [{u, v} for v in neighbors]
        result = {frozenset((u, v)) for v in neighbors}
        while True:
            changed = False
            for i, cl1 in enumerate(cliques):
                for cl2 in cliques[i + 1:]:
                    compatible = True
                    for x in cl1 - {u}:
                        for y in cl2 - cl1:
                            if x != y and y not in tmp.neighbors(x):
                                compatible = False
                                break
                        if not compatible:
                            break
                    if compatible:
                        changed = True
                        result.add(frozenset(cl1.union(cl2)))
                        result.discard(frozenset(cl1)), result.discard(frozenset(cl2))
            if not changed:
                return list(map(set, result))
            new = result.copy()
            for cl1 in result:
                for cl2 in result:
                    if cl1 != cl2 and cl1.issubset(cl2):
                        new.remove(cl1)
                        break
            cliques, result = list(new), new.copy()

    def maximal_independent_sets(self):
        """
        Return all maximal by inclusion independent sets in the graph.
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
        """
        result, independent_sets = UndirectedGraph(), self.complementary().maximal_independent_sets()
        for i, u in enumerate(independent_sets):
            for v in independent_sets[i + 1:]:
                if u.value.intersection(v.value):
                    result.connect(u, v)
        return result

    def chromatic_nodes_partition(self) -> list[set[Node]]:
        """
        Return a list of independent sets in the graph. This list has as few elements as possible.
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
        if len(self.nodes) == len(self.links) + 1:
            queue, c0, c1, total = [self.nodes.pop()], self.nodes, set(), set()
            while queue:
                flag = (u := queue.pop(0)) in c0
                for v in filter(lambda x: x not in total, self.neighbors(u)):
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

    def chromatic_links_partition(self) -> list[set[Node]]:
        """
        Similar to chromatic nodes partition, except links, that share a node, are in different sets.
        """
        return [set(map(lambda x: x.value, s)) for s in UndirectedGraph.links_graph(self).chromatic_nodes_partition()]

    def vertex_cover(self) -> set[Node]:
        """
        Return a minimum set of nodes, that cover all links in the graph.
        """
        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        """
        Return a minimum set of nodes, that cover all nodes in the graph.
        """

        def helper(curr, total, i=0):
            if total == self.nodes:
                return curr.copy()
            result = self.nodes
            for j, u in enumerate(list(nodes)[i:]):
                if len((res := helper({u, *curr}, {u, *self.neighbors(u), *total}, i + j + 1))) < len(result):
                    result = res
            return result

        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.dominating_set() for comp in self.connection_components()])
        if len(self.nodes) == len(self.links) + 1:
            return self.tree(self.nodes.pop()).dominating_set()
        if self.is_full_k_partite():
            if not self:
                return set()
            res = {u := self.nodes.pop()}
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
        Return a maximum set of nodes, no two of which are neighbors.
        """
        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.independent_set() for comp in self.connection_components()])
        if len(self.nodes) == len(self.links) + 1:
            return self.tree(self.nodes.pop()).independent_set()
        if self.is_full_k_partite():
            if not self:
                return set()
            return max([comp.nodes for comp in self.complementary().connection_components()], key=len)
        if sort := list(reversed(self.interval_sort())):
            result = set()
            for u in sort:
                if self.neighbors(u).isdisjoint(result):
                    result.add(u)
            return result
        return self.complementary().max_cliques()[0]

    def cycle_with_length(self, length: int) -> list[Node]:
        """
        Return a cycle with given length, if such exists, otherwise empty list.
        """
        if (length := abs(length)) < 3:
            return []
        if length == 3:
            return self.cycle_with_length_3()
        for l in (tmp := UndirectedGraph.copy(self)).links:
            res = tmp.disconnect(u := l.u, v := l.v).path_with_length(v, u, length - 1)
            if res:
                return res
            tmp.connect(u, v)
        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        """
        Return a path between given nodes with given length, if such exists, otherwise empty list.
        """

        def dfs(x: Node, l: int, stack: list[Link]):
            if not l:
                return list(map(lambda link: link.u, stack)) + [v] if [x == v] else []
            for y in filter(lambda _x: Link(x, _x) not in stack, self.neighbors(x)):
                if res := dfs(y, l - 1, stack + [Link(x, y)]):
                    return res
            return []

        if not (tmp := self.get_shortest_path(u, v)) or (k := len(tmp)) > length + 1:
            return []
        if length + 1 == k:
            return tmp
        return dfs(u, length, [])

    def hamilton_tour_exists(self) -> bool:
        """
        Check whether the graph is Hamiltonian.
        """

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

        if (n := len(self.nodes)) == 1 or (2 * (l := len(self.links)) > (n - 1) * (n - 2) + 2 or n > 2 and all(2 * self.degrees(n) >= n for n in self.nodes)):
            return True
        if n > l or self.leaves or not self.connected():
            return False
        tmp = UndirectedGraph.copy(self)
        can_end_in = tmp.neighbors(u := self.nodes.pop())
        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        """
        Check whether the graph has a Hamilton walk between two given nodes.
        """
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if v in self.neighbors(u):
            return True if all(n in {u, v} for n in self.nodes) else self.hamilton_tour_exists()
        return UndirectedGraph.copy(self).connect(u, v).hamilton_tour_exists()

    def hamilton_tour(self) -> list[Node]:
        """
        Return a Hamilton tour if such exists, otherwise empty list.
        """
        if len(self.nodes) == 1:
            return [self.nodes.pop()]
        if self.leaves or not self or not self.connected():
            return []
        for v in self.neighbors(u := self.nodes.pop()):
            if res := self.hamilton_walk(u, v):
                return res
        return []

    def hamilton_walk(self, u: Node = None, v: Node = None) -> list[Node]:
        """
        Return a Hamilton walk between two given nodes, if such exists, otherwise empty list.
        """

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

        tmp = UndirectedGraph.copy(self)
        if u is None:
            if v is not None and v not in self:
                raise Exception("Unrecognized node.")
            for _u in self.nodes:
                if result := dfs(_u, [_u]):
                    return result
                if self.leaf(_u):
                    return []
            return []
        if u is None and v is not None:
            u, v = v, u
        if u not in self or v is not None and v not in self:
            raise Exception("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphic_bijection(self, other) -> dict[Node, Node]:
        if isinstance(other, UndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_nodes_degrees, other_nodes_degrees = defaultdict(list), defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))
            other_nodes_degrees = sorted(other_nodes_degrees.values(), key=lambda _p: len(_p))
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
        """
        Check whether the graph has nodes.
        """
        return bool(self.nodes)

    def __reversed__(self) -> "UndirectedGraph":
        """
        Complementary graph.
        """
        return self.complementary()

    def __contains__(self, u: Node) -> bool:
        """
        Check whether a given node is in the graph.
        """
        return u in self.nodes

    def __add__(self, other: "UndirectedGraph") -> "UndirectedGraph":
        """
        Combine two undirected graphs.
        """
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class UndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            res.add(n)
        for l in other.links:
            res.connect(l.u, l.v)
        return res

    def __eq__(self, other: "UndirectedGraph") -> bool:
        """
        Compare two undirected graphs.
        """
        if type(other) == UndirectedGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self) -> str:
        """
        Represent the graph as a string.
        """
        return f"<{self.nodes}, {self.links}>"

    __repr__: str = __str__


class WeightedNodesUndirectedGraph(UndirectedGraph):
    """
    Class for implementing and undirected graph with weights on the nodes.
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, Iterable[Node]]] = {}) -> None:
        super().__init__()
        self.__node_weights = {}
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v in p[1]:
                self.add((v, 0), u), self.connect(u, v)

    def node_weights(self, u: Node = None) -> dict[Node, float] | float:
        """
        Node weights getter.
        """
        return self.__node_weights.copy() if u is None else self.__node_weights.get(u)

    @property
    def total_nodes_weight(self) -> float:
        """
        Return the sum of all node weights.
        """
        return sum(self.node_weights().values())

    def add(self, n_w: tuple[Node, float], *current_nodes: Node) -> "WeightedNodesUndirectedGraph":
        super().add(n_w[0], *current_nodes)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedNodesUndirectedGraph":
        for u in (n, *rest):
            self.__node_weights.pop(u)
        return super().remove(n, *rest)

    def set_weight(self, u: Node, w: float) -> "WeightedNodesUndirectedGraph":
        """
        Set the weight of a given node to a given value.
        """
        if u in self:
            self.__node_weights[u] = w
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedNodesUndirectedGraph":
        """
        Increase the weight of a given node by a given value.
        """
        if u in self.node_weights:
            self.set_weight(u, self.node_weights(u) + w)
        return self

    def copy(self) -> "WeightedNodesUndirectedGraph":
        return WeightedNodesUndirectedGraph({n: (self.node_weights(n), self.neighbors(n)) for n in self.nodes})

    def complementary(self) -> "WeightedNodesUndirectedGraph":
        res = WeightedNodesUndirectedGraph({u: (self.node_weights(u), self.nodes) for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def weighted_tree(self, n: Node, depth: bool = False) -> "WeightedTree":
        """
        Return a tree representation of the graph with a given node as a root.
        The depth flag asks whether the nodes traversing algorithm is DFS or BFS.
        """

        from Personal.Graphs.src.implementation.tree import WeightedTree

        tree = WeightedTree((n, self.node_weights(n)))
        queue, total = [n], {n}
        while queue:
            for v in filter(lambda x: x not in total, self.neighbors(u := queue.pop(-depth))):
                tree.add(u, {v: self.node_weights(v)}), queue.append(v), total.add(v)
        return tree

    def component(self, u: Node) -> "WeightedNodesUndirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedNodesUndirectedGraph({u: (self.node_weights(u), [])})
        while queue:
            for v in self.neighbors(u := queue.pop(0)):
                if v in res:
                    res.connect(u, v)
                else:
                    res.add((v, self.node_weights(v)), u), queue.append(v)
        return res

    def subgraph(self, nodes: Iterable[Node]) -> "WeightedNodesUndirectedGraph":
        neighborhood = {u: (self.node_weights(u), self.neighbors(u).intersection(nodes)) for u in self.nodes.intersection(nodes)}
        return WeightedNodesUndirectedGraph(neighborhood)

    def links_graph(self) -> "WeightedLinksUndirectedGraph":
        result = WeightedLinksUndirectedGraph({Node(l): {} for l in self.links})
        for l0 in self.links:
            for l1 in self.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): self.node_weights(Node(s.pop()))})
        return result

    def minimal_path_nodes(self, u: Node, v: Node) -> list[Node]:
        """
        Return a path between two given nodes with the least possible sum of node weights.
        """
        neighborhood = {n: (self.node_weights(n), {m: 0 for m in self.neighbors(n)}) for n in self.nodes}
        return WeightedUndirectedGraph(neighborhood).minimal_path(u, v)

    def weighted_vertex_cover(self) -> set[Node]:
        """
        Return a minimum by sum of the node weights set of nodes, that cover all links in the graph.
        """
        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        """
        Return a minimum by sum of the node weights set of nodes, that cover all nodes in the graph.
        """

        def helper(curr, total, total_weight, i=0):
            if total == self.nodes:
                return curr.copy(), total_weight
            result, result_sum = self.nodes, self.total_nodes_weight
            for j, u in enumerate(list(nodes)[i:]):
                cover, weight = helper({u, *curr}, {u, *self.neighbors(u), *total}, total_weight + self.node_weights(u), i + j + 1)
                if weight < result_sum:
                    result, result_sum = cover, weight
            return result, result_sum

        if not self:
            return set()
        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.weighted_dominating_set() for comp in self.connection_components()])
        if len(self.nodes) == len(self.links) + 1:
            return self.weighted_tree(self.nodes.pop()).weighted_dominating_set()
        nodes, isolated, weights = self.nodes, set(), 0
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
                weights += self.node_weights(n)
        return helper(isolated, isolated, weights)[0]

    def weighted_independent_set(self) -> set[Node]:
        """
        Return a set of non-neighboring set of nodes with a maximum possible sum of the node weights.
        """

        def helper(curr, total=set(), res_sum=0.0, i=0):
            if total == self.nodes:
                return curr, res_sum
            result, result_sum = nodes.copy(), weights
            for j, u in enumerate(list(nodes)[i:]):
                if u not in total and self.node_weights(u) > 0 and (neighbors := self.neighbors(u) - total):
                    cover, weight = helper({u, *curr}, {u, *total, *neighbors}, res_sum + self.node_weights(u), i + j + 1)
                    if weight > result_sum:
                        result, result_sum = cover, weight
            return result, result_sum

        if not self:
            return set()
        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.independent_set() for comp in self.connection_components()])
        if len(self.nodes) == len(self.links) + 1:
            return self.weighted_tree(self.nodes.pop()).weighted_independent_set()
        nodes, weights = self.nodes, self.total_nodes_weight
        for n in self.nodes:
            if not self.degrees(n):
                nodes.remove(n)
                weights -= self.node_weights(n)
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
            this_nodes_degrees, other_nodes_degrees = defaultdict(list), defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))
            other_nodes_degrees = sorted(other_nodes_degrees.values(), key=lambda _p: len(_p))
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
                        if (m in self.neighbors(n)) ^ (v in other.neighbors(u)) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphic_bijection(other)

    def __add__(self, other: UndirectedGraph) -> "WeightedNodesUndirectedGraph":
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedNodesUndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedUndirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksUndirectedGraph):
            return WeightedUndirectedGraph() + self + other
        if isinstance(other, WeightedNodesUndirectedGraph):
            res = self.copy()
            for n in other.nodes:
                if n in res:
                    res.increase_weight(n, other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links:
                res.connect(l.u, l.v)
            return res
        return self + WeightedNodesUndirectedGraph({n: (0, other.neighbors(n)) for n in other.nodes})

    def __eq__(self, other: "WeightedNodesUndirectedGraph") -> bool:
        if type(other) == WeightedNodesUndirectedGraph:
            return (self.node_weights, self.links) == (other.node_weights, other.links)
        return False

    def __str__(self) -> str:
        return "<{" + ", ".join(f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, " + str(self.links) + ">"


class WeightedLinksUndirectedGraph(UndirectedGraph):
    """
    Class for implementing and undirected graph with weights on the links.
    """

    def __init__(self, neighborhood: dict[Node, dict[Node, float]] = {}) -> None:
        super().__init__()
        self.__link_weights = {}
        for u, neighbors in neighborhood.items():
            self.add(u)
            for v, w in neighbors.items():
                self.add(v, {u: w}), self.connect(v, {u: w})

    def link_weights(self, u_or_l: Node | Link = None, v: Node = None) -> dict[Node, float] | dict[Link, float] | float:
        """
        Give information for link weights the following way:
        If no argument is passed, return the weights of all links;
        if a link or two nodes are passed, return the weight of the given link between them;
        If one node is passed, return a dictionary with all of its neighbors and the weight
        of the link it shares with each of them.
        """
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return {n: self.__link_weights[Link(n, u_or_l)] for n in self.neighbors(u_or_l)}
            return self.__link_weights.get(Link(u_or_l, v))
        elif isinstance(u_or_l, Link):
            return self.__link_weights.get(u_or_l)

    @property
    def total_links_weight(self) -> float:
        """
        Return the sum of all link weights.
        """
        return sum(self.link_weights().values())

    def add(self, u: Node, nodes_weights: dict[Node, float] = {}) -> "WeightedLinksUndirectedGraph":
        if u not in self:
            super().add(u, *nodes_weights.keys())
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedLinksUndirectedGraph":
        for u in (n, *rest):
            for v in self.neighbors(u):
                self.__link_weights.pop(Link(u, v))
        return super().remove(n, *rest)

    def connect(self, u: Node, nodes_weights: dict[Node, float] = {}) -> "WeightedLinksUndirectedGraph":
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
            if Link(u, n) in self.link_weights():
                self.__link_weights.pop(Link(u, n))
        return self

    def set_weight(self, l: Link, w: float) -> "WeightedLinksUndirectedGraph":
        """
        Set the weight of a given link to a given value.
        """
        if l in self.links:
            self.__link_weights[l] = w
        return self

    def increase_weight(self, l: Link, w: float) -> "WeightedLinksUndirectedGraph":
        """
        Increase the weight of a given link by a given value.
        """
        if l in self.link_weights:
            self.set_weight(l, self.link_weights(l) + w)
        return self

    def copy(self) -> "WeightedLinksUndirectedGraph":
        return WeightedLinksUndirectedGraph({n: self.link_weights(n) for n in self.nodes})

    def component(self, u: Node) -> "WeightedLinksUndirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksUndirectedGraph({u: {}})
        while queue:
            for n in self.neighbors(v := queue.pop(0)):
                if n in res:
                    res.connect(v, {n: self.link_weights(v, n)})
                else:
                    res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def subgraph(self, nodes: Iterable[Node]) -> "WeightedLinksUndirectedGraph":
        neighborhood = {u: {k: v for k, v in self.link_weights(u).items() if k in nodes} for u in self.nodes.intersection(nodes)}
        return WeightedLinksUndirectedGraph(neighborhood)

    def minimal_spanning_tree(self) -> set[Link]:
        """
        Return a subset of the graph's links, that cover all nodes,
        don't form a cycle and have the minimal possible weights sum.
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
            return reduce(lambda x, y: x.union(y.minimal_spanning_tree()), self.connection_components())
        if len(self.nodes) == len(self.links) + bool(self):
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

    def links_graph(self) -> WeightedNodesUndirectedGraph:
        neighborhood = {Node(l0): (self.link_weights(l0), [Node(l1) for l1 in self.links if (l1.u in l0) ^ (l1.v in l0)]) for l0 in self.links}
        return WeightedNodesUndirectedGraph(neighborhood)

    def minimal_path_links(self, u: Node, v: Node) -> list[Node]:
        """
        Return a path between two given nodes with a minimal possible sum of the link weights.
        """
        return WeightedUndirectedGraph({n: (0, self.link_weights(n)) for n in self.nodes}).minimal_path(u, v)

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
            this_nodes_degrees, other_nodes_degrees = defaultdict(list), defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))
            other_nodes_degrees = sorted(other_nodes_degrees.values(), key=lambda _p: len(_p))
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

    def __add__(self, other: UndirectedGraph) -> "WeightedLinksUndirectedGraph":
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedLinksUndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedNodesUndirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksUndirectedGraph):
            res = self.copy()
            for n in other.nodes:
                res.add(n)
            for l in other.links:
                if l in res.links:
                    res.increase_weight(l, other.link_weights(l))
                else:
                    res.connect(l.u, {l.v: other.link_weights(l)})
            return res
        return self + WeightedLinksUndirectedGraph({u: {v: 0 for v in other.neighbors(u)} for u in other.nodes})

    def __eq__(self, other: "WeightedLinksUndirectedGraph") -> bool:
        if type(other) == WeightedLinksUndirectedGraph:
            return (self.nodes, self.link_weights()) == (other.nodes, other.link_weights())
        return False

    def __str__(self) -> str:
        return "<" + str(self.nodes) + ", {" + ", ".join(f"{l} -> {self.link_weights(l)}" for l in self.links) + "}>"


class WeightedUndirectedGraph(WeightedLinksUndirectedGraph, WeightedNodesUndirectedGraph):
    """
    Class for implementing an undirected graph with weights on the nodes and the links.
    """

    def __init__(self, neighborhood: dict[Node, tuple[float, dict[Node, float]]] = {}) -> None:
        WeightedNodesUndirectedGraph.__init__(self), WeightedLinksUndirectedGraph.__init__(self)
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v, w in p[1].items():
                self.add((v, 0), {u: w}), self.connect(u, {v: w})

    @property
    def total_weight(self) -> float:
        """
        Return the sum of all weights in the graph.
        """
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: tuple[Node, float], nodes_weights: dict[Node, float] = {}) -> "WeightedUndirectedGraph":
        WeightedLinksUndirectedGraph.add(self, n_w[0], nodes_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedUndirectedGraph":
        for n in (u, *rest):
            if tmp := self.neighbors(n):
                super().disconnect(n, *tmp)
        WeightedNodesUndirectedGraph.remove(self, u, *rest)
        return self

    def set_weight(self, el: Node | Link, w: float) -> "WeightedUndirectedGraph":
        """
        Set the weight of a given node or link to a given value.
        """
        if el in self:
            WeightedNodesUndirectedGraph.set_weight(self, el, w)
        elif el in self.links:
            super().set_weight(el, w)
        return self

    def increase_weight(self, el: Node | Link, w: float) -> "WeightedUndirectedGraph":
        """
        Increase the weight of a given node or link by a given value.
        """
        if el in self.node_weights:
            return self.set_weight(el, self.node_weights(el) + w)
        if el in self.link_weights:
            self.set_weight(el, self.link_weights(el) + w)
        return self

    def copy(self) -> "WeightedUndirectedGraph":
        return WeightedUndirectedGraph({n: (self.node_weights(n), self.link_weights(n)) for n in self.nodes})

    def component(self, u: Node) -> "WeightedUndirectedGraph":
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedUndirectedGraph({u: (self.node_weights(u), {})})
        while queue:
            for u in self.neighbors(v := queue.pop(0)):
                if u in res:
                    res.connect(v, {u: self.link_weights(v, u)})
                else:
                    res.add((u, self.node_weights(u)), {v: self.link_weights(v, u)}), queue.append(u)
        return res

    def subgraph(self, nodes: Iterable[Node]) -> "WeightedUndirectedGraph":
        neighborhood = {u: (self.node_weights(u), {k: v for k, v in self.link_weights(u).items() if k in nodes}) for u in self.nodes.intersection(nodes)}
        return WeightedUndirectedGraph(neighborhood)

    def links_graph(self) -> "WeightedUndirectedGraph":
        result = WeightedUndirectedGraph({Node(l): (self.link_weights(l), {}) for l in self.links})
        for l0 in self.links:
            for l1 in self.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): self.node_weights(Node(s.pop()))})
        return result

    def minimal_path(self, u: Node, v: Node) -> list[Node]:
        """
        Return the path between two given nodes with the minimal possible total sum of nodes and weights on it.
        """

        def dfs(x, current_path, current_weight, total_negative, res_path=None, res_weight=0):
            def dijkstra(s, curr_path, curr_weight):
                curr_tmp = tmp.copy()
                for l in curr_path:
                    curr_tmp.disconnect(l.u, l.v)
                paths = {n: {m: [] for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                weights_from_to = {n: {m: curr_tmp.total_weight for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                for n in curr_tmp.nodes:
                    weights_from_to[n][n] = 0
                    for m in curr_tmp.neighbors(n):
                        weights_from_to[n][m] = curr_tmp.link_weights(n, m) + curr_tmp.node_weights(m)
                        paths[n][m] = [Link(n, m)]
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
                for y in filter(lambda _y: Link(x, _y) not in current_path, tmp.neighbors(x)):
                    new_curr_w = current_weight + (l_w := tmp.link_weights(x, y)) + (n_w := tmp.node_weights(y))
                    new_total_negative = total_negative
                    if n_w < 0:
                        new_total_negative -= n_w
                    if l_w < 0:
                        new_total_negative -= l_w
                    if new_curr_w + new_total_negative >= res_weight and res_path:
                        continue
                    if y == v and (new_curr_w < res_weight or not res_path):
                        res_path, res_weight = current_path + [Link(x, y)], current_weight + l_w + n_w
                    curr = dfs(y, current_path + [Link(x, y)], new_curr_w, new_total_negative, res_path, res_weight)
                    if curr[1] < res_weight or not res_path:
                        res_path, res_weight = curr
            else:
                curr = dijkstra(x, current_path, current_weight)
                if curr[1] < res_weight or not res_path:
                    res_path, res_weight = curr
            return res_path, res_weight

        if v in self:
            if v in (tmp := self.component(u)):
                nodes_negative_weights = sum(tmp.node_weights(n) for n in tmp.nodes if tmp.node_weights(n) < 0)
                links_negative_weights = sum(tmp.link_weights(l) for l in tmp.links if tmp.link_weights(l) < 0)
                res = dfs(u, [], tmp.node_weights(u), nodes_negative_weights + links_negative_weights)
                return [l.u for l in res[0]] + [res[0][-1].v]
            return []
        raise ValueError('Unrecognized node(s)!')

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
            this_nodes_degrees, other_nodes_degrees = defaultdict(list), defaultdict(list)
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
            if any(len(this_nodes_degrees[d]) != len(other_nodes_degrees[d]) for d in this_nodes_degrees):
                return {}
            this_nodes_degrees = sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))
            other_nodes_degrees = sorted(other_nodes_degrees.values(), key=lambda _p: len(_p))
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
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return type(other).isomorphic_bijection(other, self)
        return UndirectedGraph.isomorphic_bijection(self, other)

    def __add__(self, other: UndirectedGraph) -> "WeightedUndirectedGraph":
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedUndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedUndirectedGraph):
            res = self.copy()
            for n in other.nodes:
                if n in res:
                    res.increase_weight(n, other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links:
                if l in res.links:
                    res.increase_weight(l, other.link_weights(l))
                else:
                    res.connect(l.u, {l.v: other.link_weights(l)})
            return res
        if isinstance(other, WeightedNodesUndirectedGraph):
            return self + WeightedUndirectedGraph({u: (other.node_weights(u), {v: 0 for v in other.neighbors(u)}) for u in other.nodes})
        if isinstance(other, WeightedLinksUndirectedGraph):
            return self + WeightedUndirectedGraph({n: (0, other.link_weights(n)) for n in other.nodes})
        return self + WeightedUndirectedGraph({u: (0, {v: 0 for v in other.neighbors(u)}) for u in other.nodes})

    def __eq__(self, other: "WeightedUndirectedGraph") -> bool:
        if type(other) == WeightedUndirectedGraph:
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False

    def __str__(self) -> str:
        return "<{" + ", ".join(f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, {" + ", ".join(f"{l} -> {self.link_weights(l)}" for l in self.links) + "}>"


def clique_to_SAT(cnf: list[list[tuple[str, bool]]]) -> list[set[tuple[str, bool]]]:
    def compatible(var1, var2):
        return var1[0] != var2[0] or var1[1] == var2[1]

    def independent_set(x):
        for i_s in independent_sets:
            if x in i_s:
                return i_s

    i = 0
    while i < len(cnf):
        j, clause, contradiction = 0, cnf[i], False
        while j < len(clause):
            for var0 in clause[j + 1:]:
                if not compatible(var := clause[j], var0):
                    contradiction = True
                    break
                if var == var0:
                    clause.pop(j)
                    j -= 1
                    break
                j += 1
            if contradiction:
                break
        if contradiction:
            i -= 1
            cnf.pop(i)
        i += 1
    if not cnf:
        return [set()]
    graph, node_vars, i, independent_sets = UndirectedGraph(), {}, 0, []
    for clause in cnf:
        j = i
        for var in clause:
            node_vars[Node(i)] = var
            graph.add(Node(i))
            i += 1
        i += 1
        independent_sets.append({*map(Node, range(j, i))})
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v and compatible(node_vars[u], node_vars[v]) and v.value not in independent_set(u.value):
                graph.connect(u, v)
    result, n = [], len(cnf)
    for u in min(independent_sets, key=len):
        if len((curr := graph.max_cliques_node(u))[0]) == n:
            result += [set(map(node_vars.__getitem__, clique)) for clique in curr]
    return result
