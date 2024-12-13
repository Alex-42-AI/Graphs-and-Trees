from functools import reduce

from collections import defaultdict

from itertools import permutations, combinations, product

from Graphs.General import *

from Graphs.Tree import Tree, WeightedTree


class Link:
    def __init__(self, u: Node, v: Node):
        self.__u, self.__v = u, v

    @property
    def u(self) -> Node:
        return self.__u

    @property
    def v(self) -> Node:
        return self.__v

    def __contains__(self, item: Node):
        return item in {self.u, self.v}

    def __hash__(self):
        return hash(frozenset({self.u, self.v}))

    def __eq__(self, other):
        if isinstance(other, Link):
            return {self.u, self.v} == {other.u, other.v}
        return False

    def __str__(self):
        return f"{self.u}-{self.v}"

    __repr__ = __str__


class UndirectedGraph(Graph):
    def __init__(self, neighborhood: dict[Node, Iterable[Node]] = {}):
        self.__nodes, self.__links = set(), set()
        self.__neighboring, self.__degrees = {}, {}
        for u, neighbors in neighborhood.items():
            if u not in self:
                self.add(u)
            for v in neighbors:
                if v in self:
                    self.connect(u, v)
                else:
                    self.add(v, u)

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes.copy()

    @property
    def links(self) -> set[Link]:
        return self.__links.copy()

    @property
    def leaves(self) -> set[Node]:
        return {n for n in self.nodes if self.leaf(n)}

    def leaf(self, n: Node) -> bool:
        return self.degrees(n) == 1

    def neighboring(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        return (self.__neighboring if u is None else self.__neighboring[u]).copy()

    def degrees(self, u: Node = None) -> dict[Node, int] | int:
        return self.__degrees.copy() if u is None else self.__degrees[u]

    @property
    def degrees_sum(self) -> int:
        return 2 * len(self.links)

    def add(self, u: Node, *current_nodes: Node) -> "UndirectedGraph":
        if u not in self:
            self.__nodes.add(u)
            self.__degrees[u], self.__neighboring[u] = 0, set()
            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)
        return self

    def remove(self, n: Node, *rest: Node) -> "UndirectedGraph":
        for u in (n,) + rest:
            if u in self:
                if tmp := self.neighboring(u):
                    self.disconnect(u, *tmp)
                self.__nodes.remove(u), self.__degrees.pop(u), self.__neighboring.pop(u)
        return self

    def connect(self, u: Node, v: Node, *rest: Node) -> "UndirectedGraph":
        for n in (v,) + rest:
            if u != n and n not in self.neighboring(u) and n in self:
                self.__degrees[u] += 1
                self.__degrees[n] += 1
                self.__neighboring[u].add(n), self.__neighboring[n].add(u), self.__links.add(Link(u, n))
        return self

    def disconnect(self, u: Node, v: Node, *rest: Node) -> "UndirectedGraph":
        for n in (v,) + rest:
            if n in self.neighboring(u):
                self.__degrees[u] -= 1
                self.__degrees[n] -= 1
                self.__neighboring[u].remove(n), self.__neighboring[n].remove(u), self.__links.remove(Link(u, n))
        return self

    def disconnect_all(self, n: Node, *rest: Node) -> "UndirectedGraph":
        if not rest:
            return self
        self.disconnect(n, *rest)
        return self.disconnect_all(*rest)

    def copy(self) -> "UndirectedGraph":
        return UndirectedGraph(self.neighboring())

    def excentricity(self, u: Node) -> int:
        res, total, queue = 0, {u}, [u]
        while queue:
            new = []
            while queue:
                for v in filter(lambda x: x not in total, self.neighboring(_ := queue.pop(0))):
                    new.append(v), total.add(v)
            queue = new.copy()
            res += bool(new)
        return res

    def diameter(self) -> int:
        return max(self.excentricity(u) for u in self.nodes)

    def complementary(self) -> "UndirectedGraph":
        res = UndirectedGraph({u: self.nodes for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def connection_components(self) -> "list[UndirectedGraph]":
        components, rest = [], self.nodes
        while rest:
            components.append(curr := self.component(rest.copy().pop()))
            rest -= curr.nodes
        return components

    def connected(self) -> bool:
        if len(self.links) + 1 < (n := len(self.nodes)):
            return False
        if self.degrees_sum > (n - 1) * (n - 2) or n < 2:
            return True
        queue, total = [u := self.nodes.pop()], {u}
        while queue:
            for v in filter(lambda x: x not in total, self.neighboring(queue.pop(0))):
                total.add(v), queue.append(v)
        return len(total) == len(self.nodes)

    def is_tree(self) -> bool:
        return len(self.nodes) == len(self.links) + bool(self.nodes) and self.connected()

    def tree(self, n: Node) -> Tree:
        tree = Tree(n)
        queue, total = [n], {n}
        while queue:
            for v in filter(lambda x: x not in total, self.neighboring(u := queue.pop(0))):
                tree.add(u, v), queue.append(v), total.add(v)
        return tree

    def loop_with_length_3(self) -> list[Node]:
        for l in self.links:
            if intersection := self.neighboring(u := l.u).intersection(self.neighboring(v := l.v)):
                return [u, v, intersection.pop()]
        return []

    def planar(self) -> bool:
        return all(len(tmp.links) <= (2 + bool(tmp.loop_with_length_3())) * (len(tmp.nodes) - 2) for tmp in self.connection_components())

    def reachable(self, u: Node, v: Node) -> bool:
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        if v in {u, *self.neighboring(u)}:
            return True
        return v in self.component(u)

    def component(self, u_or_s: Node | Iterable[Node]) -> "UndirectedGraph":
        if isinstance(u_or_s, Node):
            if u_or_s not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_s], UndirectedGraph({u_or_s: []})
            while queue:
                for n in self.neighboring(v := queue.pop(0)):
                    if n in res.nodes:
                        res.connect(v, n)
                    else:
                        res.add(n, v), queue.append(n)
            return res
        return UndirectedGraph({u: self.neighboring(u).intersection(u_or_s) for u in u_or_s})

    def cut_nodes(self) -> set[Node]:
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False
            for v in self.neighboring(u):
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
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back = 1, l, l
            for v in self.neighboring(u):
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
            for m in filter(lambda x: x not in total, self.neighboring(n)):
                queue.append(m), total.add(m)
                previous[m] = n

    def euler_tour_exists(self) -> bool:
        for n in self.nodes:
            if self.degrees(n) % 2:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node) -> bool:
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if u == v:
            return self.euler_tour_exists()
        for n in self.nodes:
            if self.degrees(n) % 2 - (n in {u, v}):
                return False
        return self.connected()

    def euler_tour(self) -> list[Node]:
        if self.euler_tour_exists():
            tmp = UndirectedGraph.copy(self)
            return tmp.disconnect(u := (l := tmp.links.pop()).u, v := l.v).euler_walk(u, v)
        return []

    def euler_walk(self, u: Node, v: Node) -> list[Node]:
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = UndirectedGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i], path[i + 1])
            for i, u in enumerate(path):
                while tmp.neighboring(u):
                    curr = tmp.disconnect(u, v := tmp.neighboring(u).pop()).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j], curr[j + 1])
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def links_graph(self) -> "UndirectedGraph":
        neighborhood = {Node(l0): [Node(l1) for l1 in self.links if (l1.u in l0) ^ (l1.v in l0)] for l0 in self.links}
        return UndirectedGraph(neighborhood)

    def interval_sort(self) -> list[Node]:
        def extract(i_s, pred):
            first, second = [], []
            for u in i_s:
                if pred(u):
                    first.append(u)
                else:
                    second.append(u)
            return first + second

        def consecutive_1s(sort):
            if not sort:
                return False
            for i, u in enumerate(sort):
                j = -1
                for j, v in enumerate(sort[i:-1]):
                    if u not in self.neighboring(sort[i + j + 1]) and u in {v, *self.neighboring(v)}:
                        break
                for v in sort[i + j + 2:]:
                    if u in self.neighboring(v):
                        return False
            return True

        def lex_bfs(graph, u):
            labels, order = {node: 0 for node in graph.nodes}, [u]
            labels.pop(u)
            while labels:
                if not (neighbors := graph.neighboring(u).intersection(labels)):
                    u = max(labels, key=labels.get)
                    labels.pop(u), order.append(u)
                    continue
                neighborhood = defaultdict(set)
                for v in neighbors:
                    neighborhood[v] = graph.neighboring(v) - set(order)
                    labels[v] += 1
                comps, total, final = [], set(), set()
                for v in neighbors:
                    if v not in total:
                        total.add(v)
                        stack, comp, this_final = {v}, {v}, False
                        while stack:
                            for _u in graph.neighboring(_ := stack.pop()):
                                if _u in set(labels) - total:
                                    if _u in neighbors:
                                        stack.add(_u)
                                    else:
                                        if final:
                                            return []
                                        this_final = True
                                    comp.add(_u)
                                    if _u in neighbors:
                                        total.add(_u)
                                    labels[_u] += 1
                                    for _v in graph.neighboring(_u):
                                        if _v in labels:
                                            labels[_v] += 1
                        if this_final:
                            final = comp
                        comps.append(comp)
                for v in neighborhood:
                    neighborhood[v] -= neighbors
                if final:
                    comps.remove(final)
                for comp in comps:
                    if not (curr := graph.component(comp).interval_sort()):
                        return []
                    order += curr
                    for m in curr:
                        labels.pop(m)
                if not labels:
                    return order
                final_neighbors = {n for n in neighbors.intersection(final) if (graph.neighboring(n) - {u}).issubset(neighbors)}
                if not final_neighbors:
                    final_neighbors = neighbors.intersection(final)
                found = False
                for v in final_neighbors:
                    if consecutive_1s(curr := extract(lex_bfs(graph.component(final), v), lambda x: x in neighbors)):
                        found = True
                        order += curr
                        for m in curr:
                            labels.pop(m)
                        break
                if not found:
                    return []
                if not labels:
                    return order
                u = max(labels, key=labels.get)
                labels.pop(u), order.append(u)
            return order

        if not self.connected():
            return sum(map(lambda comp: comp.interval_sort(), self.connection_components()), [])
        for n in self.nodes:
            if consecutive_1s(result := lex_bfs(self, n)):
                return result
        return []

    def is_full_k_partite(self, k: int = -1) -> bool:
        return k in {-1, len(comps := self.complementary().connection_components())} and all(c.full() for c in comps)

    def clique(self, n: Node, *nodes: Node) -> bool:
        if {n, *nodes} == self.nodes:
            return self.full()
        if not nodes:
            return True
        if any(u not in self.neighboring(n) for u in nodes):
            return False
        return self.clique(*nodes)

    def cliques(self, k: int) -> list[set[Node]]:
        return [set(p) for p in combinations(self.nodes, abs(k)) if self.clique(*p)]

    def max_cliques_node(self, u: Node) -> list[set[Node]]:
        if (tmp := self.component(u)).full():
            return [tmp.nodes]
        if not (neighbors := tmp.neighboring(u)):
            return [{u}]
        cliques = [{u, v} for v in neighbors]
        while True:
            new = set()
            for i, cl1 in enumerate(cliques):
                for cl2 in cliques[i + 1:]:
                    compatible = True
                    for x in cl1 - {u}:
                        for y in cl2 - cl1:
                            if x != y and y not in tmp.neighboring(x):
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
        if (tmp := self.component(u)).full():
            return [tmp.nodes]
        if not (neighbors := tmp.neighboring(u)):
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
                            if x != y and y not in tmp.neighboring(x):
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

    def cliques_graph(self) -> "UndirectedGraph":
        result, total = UndirectedGraph(), set()
        for n in filter(lambda x: x not in total, self.nodes):
            total.update(*(curr := self.all_maximal_cliques_node(n)))
            for clique in curr:
                result.add(Node(frozenset(clique)))
        for i, u in enumerate(result.nodes):
            for v in list(result.nodes)[i + 1:]:
                if u.value.intersection(v.value):
                    result.connect(u, v)
        return result

    def chromatic_nodes_partition(self) -> list[set[Node]]:
        def anti_cliques(curr, result=set(), ii=0):
            for j, n in enumerate(list(self.nodes)[ii:]):
                if curr.isdisjoint(self.neighboring(n)):
                    for res in anti_cliques({n, *curr}, {n, *self.neighboring(n), *result}, ii + j + 1):
                        yield res
            if result == self.nodes:
                yield curr

        def helper(partition, union=set(), ii=0):
            if union == self.nodes:
                return partition
            res, entered = list(map(lambda x: {x}, self.nodes)), False
            for j, s in enumerate(independent_sets[ii:]):
                if {s, *union} == self.nodes or s.isdisjoint(union):
                    entered = True
                    if len(curr := helper(partition + [s - union], {s, *union}, ii + j + 1)) == 2:
                        return curr
                    res = min(res, curr, key=len)
            if not entered:
                res = partition + self.component(self.nodes - union).chromatic_nodes_partition()
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
                for v in filter(lambda x: x not in total, self.neighboring(u)):
                    if flag:
                        c1.add(v), c0.remove(v)
                    queue.append(v), total.add(v)
            return [c0, c1]
        if sort := self.interval_sort():
            result = []
            while sort:
                current, last = set(), None
                for u in sort:
                    if last is None or u not in self.neighboring(last):
                        current.add(u)
                        last = u
                for u in current:
                    sort.remove(u)
                result.append(current)
            return result
        independent_sets = [i_s for i_s in anti_cliques(set())]
        return helper([])

    def chromatic_links_partition(self) -> list[set[Node]]:
        return [set(map(lambda x: x.value, s)) for s in UndirectedGraph.links_graph(self).chromatic_nodes_partition()]

    def vertex_cover(self) -> set[Node]:
        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        def helper(curr, total, i=0):
            if total == self.nodes:
                return curr.copy()
            result = self.nodes
            for j, u in enumerate(list(nodes)[i:]):
                if len((res := helper({u, *curr}, {u, *self.neighboring(u), *total}, i + j + 1))) < len(result):
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
            if (neighbors := self.neighboring(u)) and {u, *neighbors} != self.nodes:
                res.add(neighbors.pop())
            return res
        nodes, isolated = self.nodes, set()
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
        return helper(isolated, isolated)

    def independent_set(self) -> set[Node]:
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
                if self.neighboring(u).isdisjoint(result):
                    result.add(u)
            return result
        return self.complementary().max_cliques()[0]

    def loop_with_length(self, length: int) -> list[Node]:
        if (length := abs(length)) < 3:
            return []
        if length == 3:
            return self.loop_with_length_3()
        for l in (tmp := UndirectedGraph.copy(self)).links:
            res = tmp.disconnect(u := l.u, v := l.v).path_with_length(v, u, length - 1)
            tmp.connect(u, v)
            if res:
                return res
        return []

    def path_with_length(self, u: Node, v: Node, length: int) -> list[Node]:
        def dfs(x: Node, l: int, stack: list[Link]):
            if not l:
                return list(map(lambda link: link.u, stack)) + [v] if [x == v] else []
            for y in filter(lambda _x: Link(x, _x) not in stack, self.neighboring(x)):
                if res := dfs(y, l - 1, stack + [Link(x, y)]):
                    return res
            return []

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
            neighbors = tmp.neighboring(x)
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
        can_end_in = tmp.neighboring(u := self.nodes.pop())
        return dfs(u)

    def hamilton_walk_exists(self, u: Node, v: Node) -> bool:
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if v in self.neighboring(u):
            return True if all(n in {u, v} for n in self.nodes) else self.hamilton_tour_exists()
        return UndirectedGraph.copy(self).connect(u, v).hamilton_tour_exists()

    def hamilton_tour(self) -> list[Node]:
        if len(self.nodes) == 1:
            return [self.nodes.pop()]
        if self.leaves or not self or not self.connected():
            return []
        for v in self.neighboring(u := self.nodes.pop()):
            if res := self.hamilton_walk(u, v):
                return res
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
            neighbors = tmp.neighboring(x)
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
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
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
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)):
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

    def __reversed__(self) -> "UndirectedGraph":
        return self.complementary()

    def __contains__(self, u: Node):
        return u in self.nodes

    def __add__(self, other: "UndirectedGraph") -> "UndirectedGraph":
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

    def __eq__(self, other: "UndirectedGraph"):
        if type(other) == UndirectedGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self):
        return f"<{self.nodes}, {self.links}>"

    __repr__ = __str__


class WeightedNodesUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: dict[Node, tuple[float, Iterable[Node]]] = {}):
        super().__init__()
        self.__node_weights = {}
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v in p[1]:
                if v in self:
                    self.connect(u, v)
                else:
                    self.add((v, 0), u)

    def node_weights(self, u: Node = None) -> dict[Node, float] | float:
        return self.__node_weights.copy() if u is None else self.__node_weights.get(u)

    @property
    def total_nodes_weight(self) -> float:
        return sum(self.node_weights().values())

    def add(self, n_w: tuple[Node, float], *current_nodes: Node) -> "WeightedNodesUndirectedGraph":
        super().add(n_w[0], *current_nodes)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedNodesUndirectedGraph":
        for u in (n,) + rest:
            self.__node_weights.pop(u)
        return super().remove(n, *rest)

    def set_weight(self, u: Node, w: float) -> "WeightedNodesUndirectedGraph":
        if u in self:
            self.__node_weights[u] = w
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedNodesUndirectedGraph":
        if u in self.node_weights:
            self.set_weight(u, self.node_weights(u) + w)
        return self

    def copy(self) -> "WeightedNodesUndirectedGraph":
        return WeightedNodesUndirectedGraph({n: (self.node_weights(n), self.neighboring(n)) for n in self.nodes})

    def complementary(self) -> "WeightedNodesUndirectedGraph":
        res = WeightedNodesUndirectedGraph({u: (self.node_weights(u), self.nodes) for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def weighted_tree(self, n: Node) -> WeightedTree:
        tree = WeightedTree((n, self.node_weights(n)))
        queue, total = [n], {n}
        while queue:
            for v in filter(lambda x: x not in total, self.neighboring(u := queue.pop(0))):
                tree.add(u, {v: self.node_weights(v)}), queue.append(v), total.add(v)
        return tree

    def component(self, u_or_s: Node | Iterable[Node]) -> "WeightedNodesUndirectedGraph":
        if isinstance(u_or_s, Node):
            if u_or_s not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_s], WeightedNodesUndirectedGraph({u_or_s: (self.node_weights(u_or_s), [])})
            while queue:
                for v in self.neighboring(u := queue.pop(0)):
                    if v in res:
                        res.connect(u, v)
                    else:
                        res.add((v, self.node_weights(v)), u), queue.append(v)
            return res
        return WeightedNodesUndirectedGraph({u: (self.node_weights(u), self.neighboring(u).intersection(u_or_s)) for u in u_or_s})

    def links_graph(self) -> "WeightedLinksUndirectedGraph":
        result = WeightedLinksUndirectedGraph({Node(l): {} for l in self.links})
        for l0 in self.links:
            for l1 in self.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): self.node_weights(Node(s.pop()))})
        return result

    def minimal_path_nodes(self, u: Node, v: Node) -> list[Node]:
        neighborhood = {n: (self.node_weights(n), {m: 0 for m in self.neighboring(n)}) for n in self.nodes}
        return WeightedUndirectedGraph(neighborhood).minimal_path(u, v)

    def weighted_vertex_cover(self) -> set[Node]:
        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        def helper(curr, total, total_weight, i=0):
            if total == self.nodes:
                return curr.copy(), total_weight
            result, result_sum = self.nodes, self.total_nodes_weight
            for j, u in enumerate(list(nodes)[i:]):
                cover, weight = helper({u, *curr}, {u, *self.neighboring(u), *total}, total_weight + self.node_weights(u), i + j + 1)
                if weight < result_sum:
                    result, result_sum = cover, weight
            return result, result_sum

        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.weighted_dominating_set() for comp in self.connection_components()])
        if len(self.nodes) == len(self.links) + bool(self):
            if not self:
                return set()
            return self.weighted_tree(self.nodes.pop()).weighted_dominating_set()
        nodes, isolated, weights = self.nodes, set(), 0
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
                weights += self.node_weights(n)
        return helper(isolated, isolated, weights)[0]

    def weighted_independent_set(self) -> set[Node]:
        def helper(curr, res_sum=0.0, i=0):
            if not tmp.links:
                return curr.copy(), res_sum
            result, result_sum = nodes.copy(), weights
            for j, u in enumerate(list(nodes)[i:]):
                if neighbors := tmp.neighboring(u):
                    tmp.remove(u)
                    cover, weight = helper({u, *curr}, res_sum + (w := tmp.node_weights(u)), i + j + 1)
                    tmp.add((u, w), *neighbors)
                    if weight > result_sum:
                        result, result_sum = cover, weight
            return result, result_sum

        if not self.connected():
            return reduce(lambda x, y: x.union(y), [comp.independent_set() for comp in self.connection_components()])
        if len(self.nodes) == len(self.links) + bool(self):
            if not self:
                return set()
            return self.weighted_tree(self.nodes.pop()).weighted_independent_set()
        nodes, weights, tmp = self.nodes, self.total_nodes_weight, WeightedNodesUndirectedGraph.copy(self)
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
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
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
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)) or self.node_weights(m) != other.node_weights(v):
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
        return self + WeightedNodesUndirectedGraph({n: (0, other.neighboring(n)) for n in other.nodes})

    def __eq__(self, other: "WeightedNodesUndirectedGraph"):
        if type(other) == WeightedNodesUndirectedGraph:
            return (self.node_weights, self.links) == (other.node_weights, other.links)
        return False

    def __str__(self):
        return "<{" + ", ".join(f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, " + str(self.links) + ">"


class WeightedLinksUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: dict[Node, dict[Node, float]] = {}):
        super().__init__()
        self.__link_weights = {}
        for u, neighbors in neighborhood.items():
            if u not in self:
                self.add(u)
            for v, w in neighbors.items():
                if v not in self:
                    self.add(v, {u: w})
                elif v not in self.neighboring(u):
                    self.connect(u, {v: w})

    def link_weights(self, u_or_l: Node | Link = None, v: Node = None) -> dict[Node, float] | dict[Link, float] | float:
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return {n: self.__link_weights[Link(n, u_or_l)] for n in self.neighboring(u_or_l)}
            return self.__link_weights.get(Link(u_or_l, v))
        elif isinstance(u_or_l, Link):
            return self.__link_weights.get(u_or_l)

    @property
    def total_links_weight(self) -> float:
        return sum(self.link_weights().values())

    def add(self, u: Node, nodes_weights: dict[Node, float] = {}) -> "WeightedLinksUndirectedGraph":
        if u not in self:
            super().add(u, *nodes_weights.keys())
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def remove(self, n: Node, *rest: Node) -> "WeightedLinksUndirectedGraph":
        for u in (n,) + rest:
            for v in self.neighboring(u):
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

    def disconnect(self, u: Node, v: Node, *rest: Node) -> "WeightedLinksUndirectedGraph":
        super().disconnect(u, v, *rest)
        for n in (v,) + rest:
            if Link(u, n) in self.link_weights():
                self.__link_weights.pop(Link(u, n))
        return self

    def set_weight(self, l: Link, w: float) -> "WeightedLinksUndirectedGraph":
        if l in self.links:
            self.__link_weights[l] = w
        return self

    def increase_weight(self, l: Link, w: float) -> "WeightedLinksUndirectedGraph":
        if l in self.link_weights:
            self.set_weight(l, self.link_weights(l) + w)
        return self

    def copy(self) -> "WeightedLinksUndirectedGraph":
        return WeightedLinksUndirectedGraph({n: self.link_weights(n) for n in self.nodes})

    def component(self, u_or_s: Node | Iterable[Node]) -> "WeightedLinksUndirectedGraph":
        if isinstance(u_or_s, Node):
            if u_or_s not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_s], WeightedLinksUndirectedGraph({u_or_s: {}})
            while queue:
                for n in self.neighboring(v := queue.pop(0)):
                    if n in res:
                        res.connect(v, {n: self.link_weights(v, n)})
                    else:
                        res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
            return res
        return WeightedLinksUndirectedGraph({u: {k: v for k, v in self.link_weights(u).items() if k in u_or_s} for u in u_or_s})

    def minimal_spanning_tree(self) -> set[Link]:
        def insert(x):
            low, high, f_x = 0, len(bridge_links), self.link_weights(x)
            while low < high:
                mid = (low + high) // 2
                if f_x == (f_mid := self.link_weights(bridge_links[mid])):
                    bridge_links.insert(mid, x)
                    return
                if f_x < f_mid:
                    high = mid
                else:
                    if low == mid:
                        break
                    low = mid + 1
            bridge_links.insert(high, x)

        def remove(x):
            low, high, f_x = 0, len(bridge_links), self.link_weights(x)
            while low < high:
                mid = (low + high) // 2
                if f_x == (f_mid := self.link_weights(mid_el := bridge_links[mid])):
                    if x == mid_el:
                        bridge_links.pop(mid)
                        return
                    i, j, still = mid - 1, mid + 1, True
                    while still:
                        still = False
                        if i >= 0 and self.link_weights(bridge_links[i]) == f_x:
                            if x == bridge_links[i]:
                                bridge_links.pop(i)
                                return
                            i -= 1
                            still = True
                        if j < len(bridge_links) and self.link_weights(bridge_links[j]) == f_x:
                            if x == bridge_links[j]:
                                bridge_links.pop(j)
                                return
                            j += 1
                            still = True
                    return
                if f_x < f_mid:
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
        for v in self.neighboring(u):
            insert(Link(u, v))
        for _ in range(1, len(self.nodes)):
            res_links.add(l := bridge_links.pop(0))
            if (u := l.u) in total:
                total.add(v := l.v)
                for _v in self.neighboring(v):
                    l = Link(v, _v)
                    if _v in total:
                        remove(l)
                    else:
                        insert(l)
            else:
                total.add(u)
                for _u in self.neighboring(u):
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
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
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
        return self + WeightedLinksUndirectedGraph({u: {v: 0 for v in other.neighboring(u)} for u in other.nodes})

    def __eq__(self, other: "WeightedLinksUndirectedGraph"):
        if type(other) == WeightedLinksUndirectedGraph:
            return (self.nodes, self.link_weights()) == (other.nodes, other.link_weights())
        return False

    def __str__(self):
        return "<" + str(self.nodes) + ", {" + ", ".join(f"{l} -> {self.link_weights(l)}" for l in self.links) + "}>"


class WeightedUndirectedGraph(WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph):
    def __init__(self, neighborhood: dict[Node, tuple[float, dict[Node, float]]] = {}):
        WeightedNodesUndirectedGraph.__init__(self)
        WeightedLinksUndirectedGraph.__init__(self)
        for n, p in neighborhood.items():
            self.add((n, p[0]))
        for u, p in neighborhood.items():
            for v, w in p[1].items():
                if v in self:
                    self.connect(u, {v: w})
                else:
                    self.add((v, 0), {u: w})

    @property
    def total_weight(self) -> float:
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: tuple[Node, float], nodes_weights: dict[Node, float] = {}) -> "WeightedUndirectedGraph":
        WeightedLinksUndirectedGraph.add(self, n_w[0], nodes_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node) -> "WeightedUndirectedGraph":
        for n in (u,) + rest:
            if tmp := self.neighboring(u):
                WeightedLinksUndirectedGraph.disconnect(self, u, *tmp)
            super().remove(n)
        return self

    def connect(self, u: Node, nodes_weights: dict[Node, float] = {}) -> "WeightedUndirectedGraph":
        WeightedLinksUndirectedGraph.connect(self, u, nodes_weights)
        return self

    def disconnect(self, u: Node, v: Node, *rest: Node) -> "WeightedUndirectedGraph":
        WeightedLinksUndirectedGraph.disconnect(self, u, v, *rest)
        return self

    def set_weight(self, el: Node | Link, w: float) -> "WeightedUndirectedGraph":
        if el in self:
            super().set_weight(el, w)
            return self
        if el in self.links:
            WeightedLinksUndirectedGraph.set_weight(self, el, w)
        return self

    def increase_weight(self, el: Node | Link, w: float) -> "WeightedUndirectedGraph":
        if el in self.node_weights:
            return self.set_weight(el, self.node_weights(el) + w)
        if el in self.link_weights:
            self.set_weight(el, self.link_weights(el) + w)
        return self

    def copy(self) -> "WeightedUndirectedGraph":
        return WeightedUndirectedGraph({n: (self.node_weights(n), self.link_weights(n)) for n in self.nodes})

    def component(self, u_or_s: Node | Iterable[Node]) -> "WeightedUndirectedGraph":
        if isinstance(u_or_s, Node):
            if u_or_s not in self:
                raise ValueError("Unrecognized node!")
            queue, res = [u_or_s], WeightedUndirectedGraph({u_or_s: (self.node_weights(u_or_s), {})})
            while queue:
                for u_or_s in self.neighboring(v := queue.pop(0)):
                    if u_or_s in res:
                        res.connect(v, {u_or_s: self.link_weights(v, u_or_s)})
                    else:
                        res.add((u_or_s, self.node_weights(u_or_s)), {v: self.link_weights(v, u_or_s)}), queue.append(u_or_s)
            return res
        return WeightedUndirectedGraph({u: (self.node_weights(u), {k: v for k, v in self.link_weights(u).items() if k in u_or_s}) for u in u_or_s})

    def links_graph(self) -> "WeightedUndirectedGraph":
        result = WeightedUndirectedGraph({Node(l): (self.link_weights(l), {}) for l in self.links})
        for l0 in self.links:
            for l1 in self.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(Node(l0), {Node(l1): self.node_weights(Node(s.pop()))})
        return result

    def minimal_path(self, u: Node, v: Node) -> list[Node]:
        def dfs(x, current_path, current_weight, total_negative, res_path=None, res_weight=0):
            def dijkstra(s, curr_path, curr_weight):
                curr_tmp = tmp.copy()
                for l in curr_path:
                    curr_tmp.disconnect(l.u, l.v)
                paths = {n: {m: [] for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                weights_from_to = {n: {m: curr_tmp.total_weight for m in curr_tmp.nodes} for n in curr_tmp.nodes}
                for n in curr_tmp.nodes:
                    weights_from_to[n][n] = 0
                    for m in curr_tmp.neighboring(n):
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
                for y in filter(lambda _y: Link(x, _y) not in current_path, tmp.neighboring(x)):
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
            this_nodes_degrees = defaultdict(list)
            other_nodes_degrees = defaultdict(list)
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
            return self + WeightedUndirectedGraph({u: (other.node_weights(u), {v: 0 for v in other.neighboring(u)}) for u in other.nodes})
        if isinstance(other, WeightedLinksUndirectedGraph):
            return self + WeightedUndirectedGraph({n: (0, other.link_weights(n)) for n in other.nodes})
        return self + WeightedUndirectedGraph({u: (0, {v: 0 for v in other.neighboring(u)}) for u in other.nodes})

    def __eq__(self, other: "WeightedUndirectedGraph"):
        if isinstance(other, WeightedUndirectedGraph):
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False

    def __str__(self):
        return "<{" + ", ".join(f"{n} -> {self.node_weights(n)}" for n in self.nodes) + "}, {" + ", ".join(f"{l} -> {self.link_weights(l)}" for l in self.links) + "}>"
