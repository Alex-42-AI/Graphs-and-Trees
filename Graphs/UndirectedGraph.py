from itertools import permutations, combinations, product

from Graphs.General import Node

from Graphs.Tree import Tree, WeightedNodesTree


class Link:
    def __init__(self, u: Node, v: Node):
        self.__u, self.__v = u, v

    @property
    def u(self):
        return self.__u

    @property
    def v(self):
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


class UndirectedGraph:
    def __init__(self, neighborhood: dict[Node, list[Node] | set[Node]] = {}):
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
    def nodes(self):
        return self.__nodes.copy()

    @property
    def links(self):
        return self.__links.copy()

    def neighboring(self, u: Node = None) -> dict | set:
        return (self.__neighboring if u is None else self.__neighboring[u]).copy()

    def degrees(self, u: Node = None):
        return self.__degrees.copy() if u is None else self.__degrees[u]

    @property
    def degrees_sum(self):
        return 2 * len(self.links)

    def add(self, u: Node, *current_nodes: Node):
        if u not in self:
            self.__nodes.add(u)
            self.__degrees[u], self.__neighboring[u] = 0, set()
            if current_nodes:
                UndirectedGraph.connect(self, u, *current_nodes)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            if u in self:
                if tmp := self.neighboring(u):
                    self.disconnect(u, *tmp)
                self.__nodes.remove(u), self.__degrees.pop(u), self.__neighboring.pop(u)
        return self

    def connect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if u != n and n not in self.neighboring(u) and n in self:
                self.__degrees[u] += 1
                self.__degrees[n] += 1
                self.__neighboring[u].add(n), self.__neighboring[n].add(u), self.__links.add(Link(u, n))
        return self

    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if n in self.neighboring(u):
                self.__degrees[u] -= 1
                self.__degrees[n] -= 1
                self.__neighboring[u].remove(n), self.__neighboring[n].remove(u), self.__links.remove(Link(u, n))
        return self

    def copy(self):
        return UndirectedGraph(self.neighboring())

    def width(self):
        result = 0
        for n in self.nodes:
            res, total, queue = 0, {n}, [n]
            while queue:
                new = []
                while queue:
                    for v in filter(lambda x: x not in total, self.neighboring(_ := queue.pop(0))):
                        new.append(v)
                        total.add(v)
                queue = new.copy()
                res += bool(new)
            if res > result:
                result = res
        return result

    def complementary(self):
        res = UndirectedGraph({u: self.nodes for u in self.nodes})
        for l in self.links:
            res.disconnect(l.u, l.v)
        return res

    def connection_components(self):
        components, rest = [], self.nodes.copy()
        while rest:
            components.append(curr := self.component(list(rest)[0]))
            rest -= curr.nodes
        return components

    def connected(self):
        if len(self.links) + 1 < (n := len(self.nodes)):
            return False
        if self.degrees_sum > (n - 1) * (n - 2) or n < 2:
            return True
        queue, total = [u := list(self.nodes)[0]], {u}
        while queue:
            for v in filter(lambda x: x not in total, self.neighboring(queue.pop(0))):
                total.add(v), queue.append(v)
        return len(total) == len(self.nodes)

    def is_tree(self):
        return len(self.nodes) == len(self.links) + 1 and self.connected()

    def tree(self, n: Node):
        tree = Tree(n)
        queue, total = [n], {n}
        while queue:
            for v in filter(lambda x: x not in total, self.neighboring(u := queue.pop(0))):
                tree.add(u, v), queue.append(v), total.add(v)
        return tree

    def reachable(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        if u in {v}.union(self.neighboring(v)):
            return True
        return v in self.component(u)

    def component(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], UndirectedGraph({u: []})
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res.nodes:
                    res.connect(v, n)
                else:
                    res.add(n, v), queue.append(n)
        return res

    def cut_nodes(self):
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
                res.append(u)
            colors[u] = 2
            return min_back

        levels = {n: 0 for n in self.nodes}
        colors, res = levels.copy(), []
        for n in self.nodes:
            if not colors[n]:
                dfs(n, 0)
        return res

    def bridge_links(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back = 1, l, l
            for v in self.neighboring(u):
                if not colors[v]:
                    if (b := dfs(v, l + 1)) > l:
                        res.append(Link(u, v))
                    else:
                        min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            colors[u] = 2
            return min_back

        levels = {n: 0 for n in self.nodes}
        colors, res = levels.copy(), []
        for n in self.nodes:
            if not colors[n]:
                dfs(n, 0)
        return res

    def full(self):
        return self.degrees_sum == (n := len(self.nodes)) * (n - 1)

    def get_shortest_path(self, u: Node, v: Node):
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

    def euler_tour_exists(self):
        for n in self.nodes:
            if self.degrees(n) % 2:
                return False
        return self.connected()

    def euler_walk_exists(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise ValueError("Unrecognized node(s)!")
        if u == v:
            return self.euler_tour_exists()
        for n in self.nodes:
            if self.degrees(n) % 2 - (n in {u, v}):
                return False
        return self.connected()

    def euler_tour(self):
        if self.euler_tour_exists():
            tmp = UndirectedGraph.copy(self)
            return tmp.disconnect(u := (l := list(tmp.links)[0]).u, v := l.v).euler_walk(u, v)
        return []

    def euler_walk(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s)!")
        if self.euler_walk_exists(u, v):
            path = self.get_shortest_path(u, v)
            tmp = UndirectedGraph.copy(self)
            for i in range(len(path) - 1):
                tmp.disconnect(path[i], path[i + 1])
            for i, u in enumerate(path):
                while tmp.neighboring(u):
                    curr = tmp.disconnect(u, v := tmp.neighboring(u)[0]).get_shortest_path(v, u)
                    for j in range(len(curr) - 1):
                        tmp.disconnect(curr[j], curr[j + 1])
                    while curr:
                        path.insert(i + 1, curr.pop())
            return path
        return []

    def links_graph(self):
        neighborhood = {Node(l0): [Node(l1) for l1 in self.links if (l1.u in l0) ^ (l1.v in l0)] for l0 in self.links}
        return UndirectedGraph(neighborhood)

    def interval_sort(self):
        def get_path(given):
            cont = None
            if given[1:]:
                for u in (c := given[0].value):
                    if u in given[1].value:
                        cont = u
                        break
                nodes = list(c)
                nodes.remove(cont)
                result = nodes + [cont]
            else:
                result = given[0].value
            for i, c in enumerate(map(lambda x: list(x.value), given[1:])):
                start = cont
                c.remove(start)
                try:
                    cont = [x for x in c if x in given[i + 1]][0]
                    c.remove(cont)
                    result += c + [cont]
                except IndexError:
                    result += c
            return result

        if not self.links or self.full():
            return list(self.nodes)
        if not self.connected():
            interval_sorts = list(map(lambda x: x.interval_sort(), self.connection_components()))
            if any(not i_s for i_s in interval_sorts):
                return []
            return sum(interval_sorts, [])
        tmp = self.cliquesGraph()
        final_graph = tmp.cliquesGraph()
        if any(final_graph.degrees(u) > 2 for u in final_graph.nodes):
            return []
        if len([u for u in final_graph.nodes if final_graph.degrees(u) == 1]) != 2:
            return []
        return get_path(get_path(final_graph.hamiltonWalk()))

    def is_full_k_partite(self):
        return all(comp.full() for comp in self.complementary().connection_components())

    def clique(self, n: Node, *nodes: Node):
        if {n, *nodes} == self.nodes:
            return self.full()
        if not nodes:
            return True
        if any(u not in self.neighboring(n) for u in nodes):
            return False
        return self.clique(*nodes)

    def cliques(self, k: int):
        return [list(p) for p in combinations(self.nodes, abs(k)) if self.clique(*p)]

    def maxCliquesNode(self, u: Node):
        max_card = max(map(len, (cliques := self.allMaximalCliquesNode(u))))
        return [cl for cl in cliques if len(cl) == max_card]

    def maxCliques(self):
        result, low, high = [set()], 1, len(self.nodes)
        while low < high:
            if not (curr := self.cliques(mid := (low + high) // 2)):
                high = mid - 1
            else:
                low = mid + 1
                if len(curr[0]) > len(result[0]):
                    result = curr
        return result

    def allMaximalCliquesNode(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node(s)!")
        if (tmp := self.component(u)).full():
            return [set(tmp.nodes)]
        cliques = [{u, v} for v in tmp.neighboring(u)]
        while True:
            new = []
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
                        new.append(cl1.union(cl2))
            if not new:
                return cliques
            newer = new.copy()
            for cl1 in new:
                for cl2 in new:
                    if cl1 != cl2 and cl1.issubset(cl2):
                        newer.remove(cl1)
                        break
            cliques = newer.copy()

    def cliquesGraph(self):
        result, total = UndirectedGraph(), set()
        for n in filter(lambda x: x not in total, self.nodes):
            total.update(*(curr := self.allMaximalCliquesNode(n)))
            for clique in curr:
                result.add(Node(frozenset(clique)))
        for i, u in enumerate(result.nodes):
            for v in list(result.nodes)[i + 1:]:
                if u.value.intersection(v.value):
                    result.connect(u, v)
        return result

    def independentSet(self):
        if not self.connected():
            r = [comp.independentSet() for comp in self.connection_components()]
            result = r[0]
            for i_s in r[1:]:
                result = [a + b for a in result for b in i_s]
            return result
        if self.is_tree():
            if not self:
                return [[]]
            return self.tree(list(self.nodes)[0]).independent_set()
        return self.complementary().maxCliques()

    def chromaticNodesPartition(self):
        def helper(curr):
            if tmp.full():
                return curr + list(map(lambda x: [x], tmp.nodes))
            _result = max_nodes
            for anti_clique in tmp.independentSet():
                neighbors = {n: tmp.neighboring(n) for n in anti_clique}
                tmp.remove(*anti_clique)
                res = helper(curr + [anti_clique])
                for n in anti_clique:
                    tmp.add(n, *neighbors[n])
                if len(res) < len(_result):
                    _result = res
            return _result

        if not self.connected():
            r = [comp.chromaticNodesPartition() for comp in self.connection_components()]
            final = r[0]
            for c in r[1:]:
                for i in range(min(len(c), len(final))):
                    final[i] += c[i]
                if len(c) > len(final):
                    for i in range(len(final), len(c)):
                        final.append(c[i])
            return final
        if self.is_full_k_partite():
            return [comp.nodes for comp in self.complementary().connection_components()]
        if self.is_tree():
            if not self:
                return [[]]
            queue, c0, c1, total = [list(self.nodes)[0]], self.nodes.copy(), [], set()
            while queue:
                flag = (u := queue.pop(0)) in c0
                for v in filter(lambda x: x not in total, self.neighboring(u)):
                    if flag:
                        c1.append(v), c0.remove(v)
                    queue.append(v), total.add(v)
            return [list(c0), c1]
        if s := self.interval_sort():
            result = [[s[0]]]
            for u in s[1:]:
                found = False
                for r in range(len(result)):
                    if all(v not in self.neighboring(u) for v in result[r]):
                        result[r].append(u)
                        found = True
                        break
                if not found:
                    result.append([u])
            return result
        max_nodes, tmp = self.nodes.copy(), UndirectedGraph.copy(self)
        return helper([])

    def chromaticLinksPartition(self):
        return [list(map(lambda x: x.value, s)) for s in UndirectedGraph.links_graph(self).chromaticNodesPartition()]

    def vertexCover(self):
        return [list(filter(lambda x: x not in res, self.nodes)) for res in self.independentSet()]

    def dominatingSet(self):
        def helper(curr, total, i=0):
            if total == set(self.nodes):
                return [curr.copy()]
            result = [list(self.nodes)]
            for j, u in enumerate(list(nodes)[i:]):
                new = set()
                if u not in total:
                    new.add(u)
                for v in self.neighboring(u):
                    if v not in total:
                        new.add(v)
                if len((res := helper(curr + [u], total.union(new), i + j + 1))[0]) == len(result[0]):
                    result += res
                elif len(res[0]) < len(result[0]):
                    result = res
            return result

        nodes, isolated = self.nodes.copy(), set()
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
        return helper(list(isolated), isolated)

    def pathWithLength(self, u: Node, v: Node, length: int):
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

    def loopWithLength(self, length: int):
        if abs(length) < 3:
            return []
        for l in (tmp := UndirectedGraph.copy(self)).links:
            res = tmp.disconnect(u := l.u, v := l.v).pathWithLength(v, u, abs(length) - 1)
            tmp.connect(u, v)
            if res:
                return res
        return []

    def hamiltonTourExists(self):
        def dfs(x):
            if tmp.nodes == [x]:
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

        if (k := len(self.nodes)) == 1 or (self.degrees_sum > (k - 1) * (k - 2) + 2 or k > 2 and all(2 * self.degrees(n) >= k for n in self.nodes)):
            return True
        if any(self.degrees(n) < 2 for n in self.nodes) or self.is_tree() and any(self.degrees(n) > 2 for n in self.nodes) or not self.connected() or self.interval_sort():
            return False
        tmp = UndirectedGraph.copy(self)
        can_end_in = tmp.neighboring(u := list(self.nodes)[0])
        return dfs(u)

    def hamiltonWalkExists(self, u: Node, v: Node):
        if u not in self or v not in self:
            raise Exception("Unrecognized node(s).")
        if v in self.neighboring(u):
            return True if all(n in {u, v} for n in self.nodes) else self.hamiltonTourExists()
        if s := self.interval_sort():
            return {u, v}.issubset({s[0], s[-1], None})
        return UndirectedGraph.copy(self).connect(u, v).hamiltonTourExists()

    def hamiltonTour(self):
        if any(self.degrees(n) < 2 for n in self.nodes) or not self or not self.connected():
            return []
        for v in self.neighboring(u := list(self.nodes)[0]):
            if res := self.hamiltonWalk(u, v):
                return res
        return []

    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            if not tmp.degrees(x) or v is not None and not tmp.degrees(v):
                return []
            too_many = v is not None
            for n in tmp.nodes:
                if n not in {x, v} and tmp.degrees(n) < 2:
                    if too_many:
                        return []
                    too_many = True
            if not tmp.nodes:
                return stack
            neighbors = tmp.neighboring(x)
            tmp.remove(x)
            if v is None and len(tmp.nodes) == 1 and tmp.nodes == neighbors:
                tmp.add(x, neighbors[0])
                return stack + [neighbors[0]]
            for y in neighbors:
                if y == v:
                    if tmp.nodes == [v]:
                        tmp.add(x, *neighbors)
                        return stack + [v]
                    continue
                if res := dfs(y, stack + [y]):
                    tmp.add(x, *neighbors)
                    return res
            tmp.add(x, *neighbors)
            return []

        tmp = UndirectedGraph.copy(self)
        if s := self.interval_sort():
            return s if {u, v}.issubset({s[0], s[-1], None}) else []
        if u is None:
            if v is not None and v not in self:
                raise Exception("Unrecognized node.")
            for _u in self.nodes:
                if result := dfs(_u, [_u]):
                    return result
                if self.degrees(_u) == 1:
                    return []
            return []
        if u not in self or v is not None and v not in self:
            raise Exception("Unrecognized node(s).")
        return dfs(u, [u])

    def isomorphicFunction(self, other):
        if isinstance(other, UndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            if this_degrees != other_degrees:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
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

    def __reversed__(self):
        return self.complementary()

    def __contains__(self, u: Node):
        return u in self.nodes

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class UndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return other + self
        res = self.copy()
        for n in other.nodes:
            if n not in res:
                res.add(n)
        for l in other.links:
            res.connect(l.u, l.v)
        return res

    def __eq__(self, other):
        if type(other) == UndirectedGraph:
            return (self.nodes, self.links) == (other.nodes, other.links)
        return False

    def __str__(self):
        return "<{" + ", ".join(str(n) for n in self.nodes) + "}, {" + ", ".join(str(l) for l in self.links) + "}>"

    __repr__ = __str__


class WeightedNodesUndirectedGraph(UndirectedGraph):
    def __init__(self, neighborhood: dict[Node, tuple[float, list[Node] | set[Node]]] = {}):
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

    def node_weights(self, u: Node = None):
        return self.__node_weights.copy() if u is None else self.__node_weights.get(u)

    @property
    def total_nodes_weight(self):
        return sum(self.node_weights().values())

    def add(self, n_w: (Node, float), *current_nodes: Node):
        super().add(n_w[0], *current_nodes)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            self.__node_weights.pop(u)
        return super().remove(n, *rest)

    def set_weight(self, u: Node, w: float):
        if u in self:
            self.__node_weights[u] = w
        return self

    def copy(self):
        return WeightedNodesUndirectedGraph({n: (self.node_weights(n), self.neighboring(n)) for n in self.nodes})

    def weighted_tree(self, n: Node):
        tree = WeightedNodesTree((n, self.node_weights(n)))
        queue, total = [n], {n}
        while queue:
            for v in filter(lambda x: x not in total, self.neighboring(u := queue.pop(0))):
                tree.add(u, {v: self.node_weights(v)}), queue.append(v), total.add(v)
        return tree

    def component(self, n: Node):
        if n not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [n], WeightedNodesUndirectedGraph({n: (self.node_weights(n), [])})
        while queue:
            for v in self.neighboring(u := queue.pop(0)):
                if v in res:
                    res.connect(u, v)
                else:
                    res.add((v, self.node_weights(v)), u), queue.append(v)
        return res

    def links_graph(self):
        result = WeightedLinksUndirectedGraph({Node(l): {} for l in self.links})
        for l0 in self.links:
            for l1 in self.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(l0, {l1: self.node_weights(list(s)[0])})
        return result

    def minimalPathNodes(self, u: Node, v: Node):
        neighborhood = {n: (self.node_weights(n), {m: 0 for m in self.neighboring(n)}) for n in self.nodes}
        return WeightedUndirectedGraph(neighborhood).minimalPath(u, v)

    def weightedVertexCover(self):
        if not self.connected():
            r = [comp.weightedVertexCover() for comp in self.connection_components()]
            final = r[0]
            for i_s in r[1:]:
                final = [a + b for a in final for b in i_s]
            return final
        if self.is_tree():
            if not self:
                return [[]]
            return self.weighted_tree(list(self.nodes)[0]).weighted_vertex_cover()
        nodes, weights, tmp = self.nodes.copy(), self.total_nodes_weight, WeightedNodesUndirectedGraph.copy(self)

        def helper(curr, res_sum=0, i=0):
            if not tmp.links:
                return [curr.copy()], res_sum
            result, result_sum = [nodes.copy()], weights
            for j, u in enumerate(list(nodes)[i:]):
                neighbors, w = tmp.neighboring(u), tmp.node_weights(u)
                if neighbors:
                    tmp.remove(u)
                    cover, weight = helper(curr + [u], res_sum + w, i + j + 1)
                    tmp.add((u, w), *neighbors)
                    if weight == result_sum:
                        result += cover
                    elif weight < result_sum:
                        result, result_sum = cover, weight
            return result, result_sum

        for n in self.nodes:
            if not self.degrees(n):
                nodes.remove(n)
                weights -= self.node_weights(n)
        return helper([])[0]

    def weightedDominatingSet(self):
        def helper(curr, total, total_weight, i=0):
            if total == self.nodes:
                return [curr.copy()], total_weight
            result, result_sum = [self.nodes], self.total_nodes_weight
            for j, u in enumerate(list(nodes)[i:]):
                new = set()
                if u not in total:
                    new.add(u)
                for v in self.neighboring(u):
                    if v not in total:
                        new.add(v)
                cover, weight = helper(curr + [u], total.union(new), total_weight + self.node_weights(u), i + j + 1)
                if weight == result_sum:
                    result += cover
                elif weight < result_sum:
                    result, result_sum = cover, weight
            return result, result_sum

        nodes, isolated, weights = self.nodes.copy(), set(), 0
        for n in nodes:
            if not self.degrees(n):
                isolated.add(n), nodes.remove(n)
                weights += self.node_weights(n)
        return helper(list(isolated), isolated, weights)[0]

    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedNodesUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees, this_weights, other_weights = {}, {}, {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.node_weights().values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.node_weights().values():
                if w in other_weights:
                    other_weights[w] += 1
                else:
                    other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
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
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphicFunction(other)

    def __add__(self, other):
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
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links:
                res.connect(l.u, l.v)
            return res
        return self + WeightedNodesUndirectedGraph({n: (0, other.neighboring(n)) for n in other.nodes})

    def __eq__(self, other):
        if isinstance(other, WeightedNodesUndirectedGraph):
            return (self.node_weights, self.links) == (other.node_weights, other.links)
        return False

    def __str__(self):
        return "<{" + ", ".join(f"{str(n)} -> {self.node_weights(n)}" for n in self.nodes) + "}, {" + ", ".join(str(l) for l in self.links) + "}>"


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

    def link_weights(self, u_or_l: Node | Link = None, v: Node = None):
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return {n: self.__link_weights[Link(n, u_or_l)] for n in self.neighboring(u_or_l)}
            return self.__link_weights.get(Link(u_or_l, v))
        elif isinstance(u_or_l, Link):
            return self.__link_weights.get(u_or_l)

    @property
    def total_links_weight(self):
        return sum(self.link_weights().values())

    def add(self, u: Node, nodes_weights: dict[Node, float] = {}):
        if u not in self:
            super().add(u, *nodes_weights.keys())
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            for v in self.neighboring(u):
                self.__link_weights.pop(Link(u, v))
        return super().remove(n, *rest)

    def connect(self, u: Node, nodes_weights: dict[Node, float] = {}):
        if nodes_weights:
            super().connect(u, *nodes_weights.keys())
        if u in self:
            for v, w in nodes_weights.items():
                if Link(u, v) not in self.link_weights():
                    self.set_weight(Link(u, v), w)
        return self

    def disconnect(self, u: Node, v: Node, *rest: Node):
        super().disconnect(u, v, *rest)
        for n in (v,) + rest:
            if Link(u, n) in self.link_weights():
                self.__link_weights.pop(Link(u, n))
        return self

    def set_weight(self, l: Link, w: float):
        if l.u in self.neighboring(l.v):
            self.__link_weights[l] = w
        return self

    def copy(self):
        return WeightedLinksUndirectedGraph({n: self.link_weights(n) for n in self.nodes})

    def component(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], WeightedLinksUndirectedGraph({u: {}})
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res:
                    res.connect(v, {n: self.link_weights(v, n)})
                else:
                    res.add(n, {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def minimal_spanning_tree(self):
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
            return [comp.minimal_spanning_tree() for comp in self.connection_components()]
        if self.is_tree():
            return self.links, self.total_links_weight
        if not self.links:
            return [], 0
        res_links, total = [], {u := list(self.nodes)[0]}
        bridge_links = []
        for v in self.neighboring(u):
            insert(Link(u, v))
        for _ in range(1, len(self.nodes)):
            res_links.append(l := bridge_links.pop(0))
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
        return res_links, sum(map(lambda x: self.link_weights(x), res_links))

    def links_graph(self):
        neighborhood = {Node(l0): (self.link_weights(l0), [Node(l1) for l1 in self.links if (l1.u in l0) ^ (l1.v in l0)]) for l0 in self.links}
        return WeightedNodesUndirectedGraph(neighborhood)

    def minimalPathLinks(self, u: Node, v: Node):
        return WeightedUndirectedGraph({n: (0, self.link_weights(n)) for n in self.nodes}).minimalPath(u, v)

    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedLinksUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees, this_weights, other_weights = {}, {}, {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.link_weights().values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.link_weights().values():
                if w in other_weights:
                    other_weights[w] += 1
                else:
                    other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
            this_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p)))
            other_nodes_degrees = list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
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
        return super().isomorphicFunction(other)

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedLinksUndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedNodesUndirectedGraph):
            return other + self
        if isinstance(other, WeightedLinksUndirectedGraph):
            res = self.copy()
            for n in other.nodes:
                if n not in res:
                    res.add(n)
            for l in other.links:
                if l in res.links:
                    res.set_weight(l, res.link_weights(l) + other.link_weights(l))
                else:
                    res.connect(l.u, {l.v: other.link_weights(l)})
            return res
        return self + WeightedLinksUndirectedGraph({u: {v: 0 for v in other.neighboring(u)} for u in other.nodes})

    def __eq__(self, other):
        if isinstance(other, WeightedLinksUndirectedGraph):
            return (self.nodes, self.link_weights()) == (other.nodes, other.link_weights())
        return False

    def __str__(self):
        return "<{" + ", ".join(str(n) for n in self.nodes) + "}, " + str(self.link_weights()) + ">"


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
    def total_weight(self):
        return self.total_nodes_weight + self.total_links_weight

    def add(self, n_w: (Node, float), nodes_weights: dict[Node, float] = {}):
        WeightedLinksUndirectedGraph.add(self, n_w[0], nodes_weights)
        if n_w[0] not in self.node_weights():
            self.set_weight(*n_w)
        return self

    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest:
            if tmp := self.neighboring(u):
                WeightedLinksUndirectedGraph.disconnect(self, u, *tmp)
            super().remove(n)
        return self

    def connect(self, u: Node, nodes_weights: dict[Node, float] = {}):
        return WeightedLinksUndirectedGraph.connect(self, u, nodes_weights)

    def disconnect(self, u: Node, v: Node, *rest: Node):
        return WeightedLinksUndirectedGraph.disconnect(self, u, v, *rest)

    def set_weight(self, el: Node | Link, w: float):
        if el in self:
            WeightedNodesUndirectedGraph.set_weight(self, el, w)
        elif el in self.links:
            WeightedLinksUndirectedGraph.set_weight(self, el, w)
        return self

    def copy(self):
        return WeightedUndirectedGraph({n: (self.node_weights(n), self.link_weights(n)) for n in self.nodes})

    def component(self, n: Node):
        if n not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [n], WeightedUndirectedGraph({n: (self.node_weights(n), {})})
        while queue:
            for n in self.neighboring(v := queue.pop(0)):
                if n in res:
                    res.connect(v, {n: self.link_weights(v, n)})
                else:
                    res.add((n, self.node_weights(n)), {v: self.link_weights(v, n)}), queue.append(n)
        return res

    def links_graph(self):
        result = WeightedUndirectedGraph({Node(l): (self.link_weights(l), {}) for l in self.links})
        for l0 in self.links:
            for l1 in self.links:
                if l0 != l1 and (s := {l0.u, l0.v}.intersection({l1.u, l1.v})):
                    result.connect(l0, {l1: self.node_weights(list(s)[0])})
        return result

    def minimalPath(self, u: Node, v: Node):
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
                return [l.u for l in res[0]] + [res[0][-1].v], res[1]
            return [], 0
        raise ValueError('Unrecognized node(s)!')

    def isomorphicFunction(self, other: UndirectedGraph):
        if isinstance(other, WeightedUndirectedGraph):
            if len(self.links) != len(other.links) or len(self.nodes) != len(other.nodes):
                return {}
            this_degrees, other_degrees = {}, {}
            this_node_weights, other_node_weights = {}, {}
            this_link_weights, other_link_weights = {}, {}
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.link_weights().values():
                if w in this_link_weights:
                    this_link_weights[w] += 1
                else:
                    this_link_weights[w] = 1
            for w in other.link_weights().values():
                if w in other_link_weights:
                    other_link_weights[w] += 1
                else:
                    other_link_weights[w] = 1
            for w in self.node_weights().values():
                if w in this_node_weights:
                    this_node_weights[w] += 1
                else:
                    this_node_weights[w] = 1
            for w in other.node_weights().values():
                if w in other_node_weights:
                    other_node_weights[w] += 1
                else:
                    other_node_weights[w] = 1
            if this_degrees != other_degrees or this_node_weights != other_node_weights or this_link_weights != other_link_weights:
                return {}
            this_nodes_degrees = {d: [] for d in this_degrees}
            other_nodes_degrees = {d: [] for d in other_degrees}
            for n in self.nodes:
                this_nodes_degrees[self.degrees(n)].append(n)
            for n in other.nodes:
                other_nodes_degrees[other.degrees(n)].append(n)
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
                        if self.link_weights(Link(n, m)) != other.link_weights(Link(u, v)) or self.node_weights(m) != other.node_weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        if isinstance(other, (WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph)):
            return type(other).isomorphicFunction(other, self)
        return UndirectedGraph.isomorphicFunction(self, other)

    def __add__(self, other):
        if not isinstance(other, UndirectedGraph):
            raise TypeError(f"Addition not defined between class WeightedUndirectedGraph and type {type(other).__name__}!")
        if isinstance(other, WeightedUndirectedGraph):
            res = self.copy()
            for n in other.nodes:
                if n in res:
                    res.set_weight(n, res.node_weights(n) + other.node_weights(n))
                else:
                    res.add((n, other.node_weights(n)))
            for l in other.links:
                if (v := l.v) in res.neighboring(u := l.u):
                    res.set_weight(l, res.link_weights(l) + other.link_weights(l))
                else:
                    res.connect(u, {v: other.link_weights(l)})
            return res
        if isinstance(other, WeightedNodesUndirectedGraph):
            return self + WeightedUndirectedGraph({u: (other.node_weights(u), {v: 0 for v in other.neighboring(u)}) for u in other.nodes})
        if isinstance(other, WeightedLinksUndirectedGraph):
            return self + WeightedUndirectedGraph({n: (0, other.link_weights(n)) for n in other.nodes})
        return self + WeightedUndirectedGraph({u: (0, {v: 0 for v in other.neighboring(u)}) for u in other.nodes})

    def __eq__(self, other):
        if isinstance(other, WeightedUndirectedGraph):
            return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False

    def __str__(self):
        return f"<{self.node_weights()}, {self.link_weights()}>"
