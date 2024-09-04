from Personal.DiscreteMath.Graphs.General import Node, Dict, SortedKeysDict, SortedList
class DirectedGraph:
    def __init__(self, *nodes: Node):
        self.__nodes = SortedList()
        for n in nodes:
            if n not in self.__nodes:
                self.__nodes.insert(n)
        self.__links, self.__degrees, self.__prev, self.__next = [], SortedKeysDict(*[(n, [0, 0]) for n in self.__nodes]), SortedKeysDict(*[(n, SortedList()) for n in self.__nodes]), SortedKeysDict(*[(n, SortedList()) for n in self.__nodes])
    def nodes(self): return self.__nodes
    def links(self): return self.__links
    def degrees(self, u: Node = None): return self.__degrees if u is None else self.__degrees[u]
    def next(self, u: Node = None): return self.__next if u is None else self.__next[u]
    def prev(self, u: Node = None): return self.__prev if u is None else self.__prev[u]
    def add(self, u: Node, pointed_by: [Node] = None, points_to: [Node] = None):
        if u not in self.nodes():
            if pointed_by is None: pointed_by = []
            if points_to is None: points_to = []
            res_pointed_by, res_points_to = [], []
            for v in pointed_by:
                if v in self.nodes() and v not in res_pointed_by: res_pointed_by.append(v)
            for v in points_to:
                if v in self.nodes() and v not in res_points_to: res_points_to.append(v)
            self.__degrees[u], self.__next[u], self.__prev[u] = [len(res_pointed_by), len(res_points_to)], SortedList(), SortedList()
            for v in res_pointed_by:
                self.__links.append((v, u)), self.__next[v].insert(u), self.__prev[u].insert(v)
                self.__degrees[v][1] += 1
            for v in res_points_to:
                self.__links.append((u, v)), self.__next[u].insert(v), self.__prev[v].insert(u)
                self.__degrees[v][0] += 1
            self.__nodes.insert(u)
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes:
            if u in self.nodes():
                for v in self.next(u).value():
                    self.__prev[v].remove(u), self.__links.remove((u, v))
                    self.__degrees[v][0] -= 1
                for v in self.prev(u).value():
                    self.__next[v].remove(u), self.__links.remove((v, u))
                    self.__degrees[v][1] -= 1
                self.__nodes.remove(u), self.__next.pop(u), self.__degrees.pop(u), self.__prev.pop(u)
    def connect_from_to(self, u: Node, v: Node, *rest: Node):
        if u in self.nodes():
            for n in (v,) + rest:
                if u != n and n not in self.next(u) and n in self.nodes():
                    self.__links.append((u, n)), self.__next[u].insert(n), self.__prev[n].insert(u)
                    self.__degrees[u][1] += 1
                    self.__degrees[n][0] += 1
    def connect_to_from(self, u: Node, v: Node, *rest: Node):
        if u in self.nodes():
            for n in (v,) + rest:
                if u != n and u not in self.next(n) and n in self.nodes():
                    self.__links.append((n, u)), self.__next[n].insert(u), self.__prev[u].insert(n)
                    self.__degrees[u][0] += 1
                    self.__degrees[n][1] += 1
    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if n in self.next(u):
                self.__degrees[u][1] -= 1
                self.__degrees[n][0] -= 1
                self.__links.remove((u, n)), self.__next[u].remove(n), self.__prev[n].remove(u)
    def complementary(self):
        res = DirectedGraph(*self.nodes().value())
        for i, n in enumerate(self.nodes().value()):
            for j in range(i + 1, len(self.nodes())):
                if self.nodes()[j] not in self.next(n): res.connect_from_to(n, self.nodes()[j])
        return res
    def transposed(self):
        res = DirectedGraph(*self.nodes().value())
        for (u, v) in self.links(): res.connect_to_from(u, v)
        return res
    def copy(self):
        res = DirectedGraph(*self.nodes().value())
        for u in self.nodes().value():
            if self.degrees(u)[1]: res.connect_from_to(u, *self.next(u).value())
        return res
    def connected(self):
        queue, total, k, n = [self.nodes()[0]], SortedList(), 1, len(self.nodes())
        total.insert(queue[0])
        while queue:
            u = queue.pop(0)
            for v in filter(lambda x: x not in total and (u, x) in self.links() or (x, u) in self.links(), self.nodes()):
                total.insert(v), queue.append(v)
                k += 1
            if k == n: return True
        return False
    def sources(self): return [u for u in self.nodes().value() if not self.prev(u)]
    def sinks(self): return [v for v in self.nodes().value() if not self.next(v)]
    def dag(self):
        sources, total = self.sources(), SortedList()
        if not sources: return False
        stack = sources.copy()
        while stack:
            u = stack.pop()
            for v in self.next(u).value():
                if v in total: continue
                if v in stack: return False
                stack.append(v)
            total.insert(u)
        return True
    def toposort(self):
        if not self.dag(): raise ValueError("Not a dag!")
        queue, res, total = self.sources(), [], SortedList()
        while queue:
            u = queue.pop(0)
            res.append(u), total.insert(u)
            for v in filter(lambda x: x not in total, self.next(u).value()): queue.append(v), total.insert(v)
        return res
    def connection_components(self):
        if len(self.nodes()) in (0, 1): return [self.nodes()]
        components, queue, total, k, n = [[self.nodes()[0]]], [self.nodes()[0]], SortedList(), 1, len(self.nodes())
        total.insert(self.nodes()[0])
        while k < n:
            while queue:
                u = queue.pop(0)
                for v in filter(lambda x: x not in total, self.next(u).value() + self.prev(u).value()):
                    components[-1].append(v), queue.append(v), total.insert(v)
                    k += 1
                    if k == n: return components
            if k < n:
                new = [[n for n in self.nodes().value() if n not in total][0]]
                components.append(new), total.insert(new[0])
                k, queue = k + 1, [new[0]]
                if k == n: return components
        return components
    def reachable(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s).')
        total, queue = SortedList(), [u]
        total.insert(u)
        while queue:
            x = queue.pop(0)
            for y in filter(lambda _x: _x not in total, self.next(x).value()):
                if y == v: return True
                total.insert(y), queue.append(y)
        return False
    def cut_nodes(self):
        def dfs(x: Node, l: int):
            colors[x], levels[x], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False
            for y in self.next(x).value() + self.prev(x).value():
                if not colors[y]:
                    count += 1
                    b = dfs(y, l + 1)
                    if b >= l and not is_root: is_cut = True
                    min_back = min(min_back, b)
                if colors[y] == 1 and levels[y] < min_back and levels[y] + 1 != l: min_back = levels[y]
            if is_cut or is_root and count > 1: res.append(x)
            colors[x] = 2
            return min_back
        levels, colors, res = SortedKeysDict(*[(n, 0) for n in self.nodes().value()]), SortedKeysDict(*[(n, 0) for n in self.nodes().value()]), []
        dfs(self.nodes()[0], 0)
        return res
    def bridge_links(self):
        def dfs(x: Node, l: int):
            colors[x], levels[x], min_back = 1, l, l
            for y in self.next(x).value():
                if not colors[y]:
                    b = dfs(y, l + 1)
                    if b > l: res.append((x, y))
                    else: min_back = min(min_back, b)
                if colors[y] == 1 and levels[y] < min_back and levels[y] + 1 != l: min_back = levels[y]
            colors[x] = 2
            return min_back
        levels, colors, res = SortedKeysDict(*[(n, 0) for n in self.nodes().value()]), SortedKeysDict(*[(n, 0) for n in self.nodes().value()]), []
        dfs(self.nodes()[0], 0)
        return res
    def get_shortest_path(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s)!')
        previous = SortedKeysDict(*[(n, None) for n in self.nodes().value()])
        queue, total = [u], SortedList()
        total.insert(u), previous.pop(u)
        while queue:
            x = queue.pop(0)
            if x == v:
                res, curr_node = [], x
                while curr_node != u:
                    res.insert(0, (previous[curr_node], curr_node))
                    curr_node = previous[curr_node]
                return res
            for y in filter(lambda _x: _x not in total, self.next(x).value()):
                queue.append(y), total.insert(y)
                previous[y] = x
    def shortest_path_length(self, u: Node, v: Node):
        distances = SortedKeysDict(*[(n, 0) for n in self.nodes().value()])
        if u not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s).')
        queue, total = [u], SortedList()
        total.insert(u)
        while queue:
            x = queue.pop(0)
            for y in filter(lambda _x: _x not in total, self.next(x).value()):
                if y == v: return 1 + distances[x]
                total.insert(x), queue.append(y)
                distances[y] = distances[x] + 1
        return float('inf')
    def euler_tour_exists(self):
        for d in self.degrees().values():
            if d[0] != d[1]: return False
        return self.connected()
    def euler_walk_exists(self, u: Node, v: Node):
        if self.euler_tour_exists(): return u == v
        for n in self.nodes().value():
            if self.degrees(n)[1] % 2 and n != u or self.degrees(n)[0] % 2 and n != v: return False
        return not (self.degrees(u)[1] + self.degrees(v)[0]) % 2 and self.connected()
    def euler_tour(self):
        if self.euler_tour_exists():
            v, u = self.links()[0]
            self.disconnect(v, u)
            res = self.euler_walk(u, v)
            self.connect_from_to(v, u)
            return res
        return False
    def euler_walk(self, u: Node, v: Node):
        def dfs(x, stack):
            for y in self.next(x).value():
                if (x, y) not in result + stack:
                    if y == n:
                        stack.append((x, y))
                        for j in range(len(stack)): result.insert(i + j, stack[j])
                        return
                    dfs(y, stack + [(x, y)])
        if u in self.nodes() and v in self.nodes():
            if self.euler_walk_exists(u, v):
                result = self.get_shortest_path(u, v)
                for i, l in enumerate(result):
                    n = l[0]
                    dfs(n, [])
                return result
            return False
    def stronglyConnectedComponents(self):
        def helper(x, stack):
            for y in self.next(x).value():
                if y not in curr: helper(y, stack + [y])
                elif x not in curr and y in curr:
                    curr_node, new = stack.pop(), []
                    while curr_node not in curr:
                        total.insert(curr_node), curr.insert(curr_node), new.append(curr_node)
                        if not stack: break
                        curr_node = stack.pop()
                    return
        def dfs(x, stack):
            for y in self.next(x).value():
                if y not in curr and y not in stack: dfs(y, stack + [y])
                if y == n:
                    curr_node = stack.pop()
                    while stack and curr_node != n:
                        total.insert(curr_node), curr.insert(curr_node)
                        curr_node = stack.pop()
                    for curr_node in curr.value(): helper(curr_node, [curr_node])
                    return
        total, res = SortedList(), []
        for n in self.nodes().value():
            if n not in total:
                curr = SortedList()
                curr.insert(n), dfs(n, [n]), res.append(curr), total.insert(n)
        return res
    def sccDag(self):
        res = DirectedGraph()
        for c in self.stronglyConnectedComponents():
            curr = DirectedGraph(*c)
            for u in c:
                to_connect = filter(lambda x: x in c, self.next(u).value())
                if to_connect: curr.connect_from_to(u, *to_connect)
            res.add(Node(curr))
        for u in res.nodes().value():
            linked_to = SortedList(lambda x: len(x.nodes()))
            linked_to.insert(u)
            for n in u.value().nodes().value():
                for m in self.next(n).value():
                    for v in res.nodes().value():
                        if v not in linked_to and m in v.value().nodes():
                            res.connect_from_to(u, v), linked_to.insert(v)
                            break
        return res
    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack):
            if x not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s).')
            if not l: return (False, stack)[x == v]
            for y in filter(lambda _x: (x, _x) not in stack, self.next(x).value()):
                res = dfs(y, l - 1, stack + [(x, y)])
                if res: return res
            return False
        tmp = self.get_shortest_path(u, v)
        if len(tmp) > length: return False
        if length == len(tmp): return tmp
        return dfs(u, length, [])
    def loopWithLength(self, length: int):
        for u in self.nodes().value():
            for v in self.next(u).value():
                res = self.pathWithLength(v, u, length - 1)
                if res: return [(u, v)] + res
        return False
    def hamiltonTourExists(self):
        def dfs(x):
            if len(self.links()) > (len(self.nodes()) - 1) ** 2 + 1: return True
            if [*self.nodes().value()] == [x]: return x in can_end_in
            if all(y not in self.nodes() for y in can_end_in.value()): return False
            tmp0, tmp1 = self.prev(x), self.next(x)
            self.remove(x)
            for y in tmp1.value():
                res = dfs(y)
                if res:
                    self.add(x, [*tmp0.value()], [*tmp1.value()])
                    return True
            self.add(x, [*tmp0.value()], [*tmp1.value()])
            return False
        u = self.nodes()[0]
        can_end_in = self.prev(u)
        return dfs(u)
    def hamiltonWalkExists(self, u: Node, v: Node):
        if u in self.next(v): return True if all(n in (u, v) for n in self.nodes().value()) else self.hamiltonTourExists()
        self.connect_from_to(v, u)
        res = self.hamiltonTourExists()
        self.disconnect(v, u)
        return res
    def hamiltonTour(self):
        if any(not self.degrees(u)[0] or not self.degrees(u)[1] for u in self.nodes().value()) or not self.connected(): return False
        u = self.nodes()[0]
        for v in self.prev(u).value():
            result = self.hamiltonWalk(u, v)
            if result: return result
        return False
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            for n in self.nodes().value():
                if not self.degrees(n)[1] and n != v or not self.degrees(n)[0] and n != x: return False
            if not self.nodes(): return stack
            tmp0, tmp1 = self.prev(x), self.next(x)
            self.remove(x)
            for y in tmp1.value():
                if y == v:
                    if [*self.nodes().value()] == [v]: return stack
                    continue
                res = dfs(y, stack + [y])
                if res:
                    self.add(x, [*tmp0.value()], [*tmp1.value()])
                    return res
            self.add(x, [*tmp0.value()], [*tmp1.value()])
            return False
        if u is None:
            for _u in sorted(self.nodes().value(), key=lambda _x: self.degrees(_x)[1]):
                result = dfs(_u, [_u])
                if result: return result
            return False
        return dfs(u, [u])
    def isomorphic(self, other):
        if isinstance(other, DirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()): return False
            this_degrees, other_degrees = SortedKeysDict(), SortedKeysDict()
            for d in self.degrees().values():
                if d in this_degrees: this_degrees[d] += 1
                else: this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees: other_degrees[d] += 1
                else: other_degrees[d] = 1
            if this_degrees != other_degrees: return False
            this_nodes_degrees, other_nodes_degrees = SortedKeysDict(*[(d, []) for d in this_degrees.keys()]), SortedKeysDict(*[(d, []) for d in other_degrees.keys()])
            for d in this_degrees.keys():
                for n in self.nodes().value():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility): map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if (m in self.next(n)) ^ (v in other.next(u)):
                            possible = False
                            break
                    if not possible: break
                if possible: return True
            return False
        return False
    def __reversed__(self):
        return self.complementary()
    def __contains__(self, item):
        return item in self.nodes() or item in self.links()
    def __add__(self, other):
        if isinstance(other, (WeightedNodesDirectedGraph, WeightedLinksDirectedGraph)):
            return other + self
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes().value():
                if n not in res.nodes(): res.add(n)
            for (u, v) in other.links():
                if v not in res.next(u): res.connect_from_to(u, v)
            return res
        raise TypeError(f"Addition not defined between class DirectedGraph and type {type(other).__name__}!")
    def __eq__(self, other):
        if isinstance(other, DirectedGraph):
            if len(self.links()) != len(other.links()) or self.nodes() != other.nodes(): return False
            for l in self.links():
                if l not in other.links(): return False
            return True
        return False
    def __str__(self): return '({' + ', '.join(str(n) for n in self.nodes().value()) + '}, {' + ', '.join(str(l[0]) + '->' + str(l[1]) for l in self.links()) + '})'
    def __repr__(self): return str(self)
class WeightedNodesDirectedGraph(DirectedGraph):
    def __init__(self, *pairs: (Node, float)):
        super().__init__(*[p[0] for p in pairs])
        self.__node_weights = SortedKeysDict()
        for (n, w) in pairs:
            if n not in self.__node_weights: self.__node_weights[n] = w
    def node_weights(self, n: Node = None): return self.__node_weights[n] if n is not None else self.__node_weights
    def total_nodes_weight(self): return sum(self.node_weights().values())
    def copy(self):
        res = WeightedNodesDirectedGraph(*self.node_weights().items().value())
        for n in self.nodes().value():
            if self.degrees(n)[1]: res.connect_from_to(n, *self.next(n).value())
        return res
    def add(self, n_w: (Node, float), pointed_by: [Node] = None, points_to: [Node] = None):
        if n_w[0] not in self.nodes(): self.__node_weights[n_w[0]] = n_w[1]
        super().add(n_w[0], pointed_by, points_to)
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes: self.__node_weights.pop(u)
        super().remove(n, *nodes)
    def minimalPathNodes(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None: res_path = []
            for y in filter(lambda _y: (x, _y) not in curr_path, self.next(x).value()):
                if curr_w + self.node_weights(y) + total_negative >= res_w and res_path: continue
                if y == v and (curr_w + self.node_weights(y) < res_w or not res_path): res_path, res_w = curr_path + [(x, y)], curr_w + self.node_weights(y)
                curr = dfs(y, curr_path + [(x, y)], curr_w + self.node_weights(y), total_negative - self.node_weights(y) * (self.node_weights(y) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path: res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v): return dfs(u, [], 0, sum(self.node_weights(n) for n in self.nodes().value() if self.node_weights(n) < 0))
            return [], 0
        raise ValueError("Unrecognized node(s)!")
    def hamiltonTourExists(self):
        def dfs(x):
            if len(self.links()) > (len(self.nodes()) - 1) ** 2 + 1: return True
            if [*self.nodes().value()] == [x]: return x in can_end_in
            if all(y not in self.nodes() for y in can_end_in.value()): return False
            tmp0, tmp1, w = self.prev(x), self.next(x), self.node_weights(x)
            self.remove(x)
            for y in tmp1.value():
                res = dfs(y)
                if res:
                    self.add((x, w), tmp0.value(), tmp1.value())
                    return True
            self.add((x, w), tmp0.value(), tmp1.value())
            return False
        u = self.nodes()[0]
        can_end_in = self.prev(u)
        return dfs(u)
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            for n in self.nodes().value():
                if not self.degrees(n)[1] and n != v or not self.degrees(n)[0] and n != x: return False
            if not self.nodes(): return stack
            tmp0, tmp1, w = self.prev(x), self.next(x), self.node_weights(x)
            self.remove(x)
            for y in tmp1.value():
                if y == v:
                    if [*self.nodes().value()] == [v]: return stack
                    continue
                res = dfs(y, stack + [y])
                if res:
                    self.add((x, w), tmp0.value(), tmp1.value())
                    return res
            self.add((x, w), tmp0.value(), tmp1.value())
            return False
        if u is None:
            for _u in sorted(self.nodes().value(), key=lambda _x: self.degrees(_x)[1]):
                result = dfs(_u, [_u])
                if result: return result
            return False
        return dfs(u, [u])
    def isomorphic(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()): return False
            this_degrees, other_degrees, this_weights, other_weights = dict(), dict(), dict(), dict()
            for d in self.degrees().values():
                if d in this_degrees: this_degrees[d] += 1
                else: this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees: other_degrees[d] += 1
                else: other_degrees[d] = 1
            for w in self.node_weights().values():
                if w in this_weights: this_weights[w] += 1
                else: this_weights[w] = 1
            for w in other.node_weights().values():
                if w in other_weights: other_weights[w] += 1
                else: other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights: return False
            this_nodes_degrees, other_nodes_degrees = {d: [] for d in this_degrees.keys()}, {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes().value():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility): map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if (m in self.next(n)) ^ (v in other.next(u)) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible: break
                if possible: return True
            return False
        return super().isomorphic(other)
    def __add__(self, other):
        if isinstance(other, WeightedDirectedGraph): return other + self
        if isinstance(other, WeightedNodesDirectedGraph):
            res = self.copy()
            for n in other.nodes().value():
                if n in res.nodes(): res.__node_weights[n] += other.node_weights(n)
                else: res.add(n, other.node_weights(n))
            for (u, v) in other.links():
                if v not in res.next(u): res.connect_from_to(u, v)
            return res
        if isinstance(other, WeightedLinksDirectedGraph): return WeightedDirectedGraph() + self + other
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes().value():
                if n not in res.nodes(): res.add((n, 0))
            for (u, v) in other.links():
                if v not in res.next(u): res.connect_from_to(u, v)
            return res
        raise TypeError(f"Addition not defined between class WeightedNodesDirectedGraph and type {type(other).__name__}!")
    def __eq__(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links()) != len(other.links()) or self.node_weights() != other.node_weights(): return False
            for l in self.links():
                if l not in other.links(): return False
            return True
        return False
    def __str__(self): return '({' + ', '.join(f'{str(n)} -> {self.node_weights(n)}' for n in self.nodes().value()) + '}, ' + str(self.links()) + ')'
class WeightedLinksDirectedGraph(DirectedGraph):
    def __init__(self, *nodes: Node):
        super().__init__(*nodes)
        self.__link_weights = Dict()
    def link_weights(self, u_or_l: Node | tuple = None, v: Node = None):
        if u_or_l is None: return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None: return SortedKeysDict(*[(n, self.__link_weights[(u_or_l, n)]) for n in self.next(u_or_l).value()])
            return self.__link_weights[(u_or_l, v)]
        elif isinstance(u_or_l, tuple): return self.__link_weights[u_or_l]
    def total_links_weight(self): return sum(self.link_weights().values())
    def add(self, u: Node, pointed_by_weights: [(Node, float)] = None, points_to_weights: [(Node, float)] = None):
        if u not in self.nodes():
            if pointed_by_weights is None: pointed_by_weights = []
            if points_to_weights is None: points_to_weights = []
            for w in [p[1] for p in pointed_by_weights] + [p[1] for p in points_to_weights]:
                if not isinstance(w, (int, float)): raise TypeError('Real numerical values expected!')
            super().add(u, [p[0] for p in pointed_by_weights], [p[0] for p in points_to_weights])
            for v, w in pointed_by_weights:
                if v in self.prev(u) and (v, u) not in self.link_weights(): self.__link_weights[(v, u)] = w
            for v, w in points_to_weights:
                if v in self.next(u) and (u, v) not in self.link_weights(): self.__link_weights[(u, v)] = w
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes:
            for v in self.next(u).value(): self.__link_weights.pop((u, v))
            for v in self.prev(u).value(): self.__link_weights.pop((v, u))
        super().remove(n, *nodes)
    def connect_from_to(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect_from_to(u, *[p[0] for p in (v_w,) + nodes_weights])
            for v, w in (v_w,) + nodes_weights:
                if v in self.next(u) and (u, v) not in self.link_weights(): self.__link_weights[(u, v)] = w
    def connect_to_from(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect_to_from(u, *[p[0] for p in (v_w,) + nodes_weights])
            for v, w in (v_w,) + nodes_weights:
                if v in self.prev(u) and (v, u) not in self.link_weights(): self.__link_weights[(v, u)] = w
    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in (v,) + rest:
            if n in self.next(u): self.__link_weights.pop((u, n))
        super().disconnect(u, v, *rest)
    def transposed(self):
        res = WeightedLinksDirectedGraph(*self.nodes().value())
        for (u, v) in self.links(): res.connect_to_from(u, (v, self.link_weights(u, v)))
        return res
    def copy(self):
        res = WeightedLinksDirectedGraph(*self.nodes().value())
        for u in self.nodes().value():
            for v in self.next(u).value():
                if (u, v) in self.links(): res.connect_from_to(u, (v, self.link_weights(u, v)))
        return res
    def euler_tour(self):
        if self.euler_tour_exists():
            v, u = self.links()[0]
            w = self.link_weights(v, u)
            self.disconnect(v, u)
            res = self.euler_walk(u, v)
            self.connect_from_to(v, (u, w))
            return res
        return False
    def minimalPathLinks(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None: res_path = []
            for y in filter(lambda _y: (x, _y) not in curr_path, self.next(x).value()):
                if curr_w + self.link_weights((x, y)) + total_negative >= res_w and res_path: continue
                if y == v and (curr_w + self.link_weights((x, y)) < res_w or not res_path): res_path, res_w = curr_path.copy() + [(x, y)], curr_w + self.link_weights((x, y))
                curr = dfs(y, curr_path + [(x, y)], curr_w + self.link_weights((x, y)), total_negative - self.link_weights((x, y)) * (self.link_weights((x, y)) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path: res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v): return dfs(u, [], 0, sum(self.link_weights(l) for l in self.links() if self.link_weights(l) < 0))
            return [], 0
        raise ValueError('Unrecognized node(s)!')
    def hamiltonTourExists(self):
        def dfs(x):
            if len(self.links()) > (len(self.nodes()) - 1) ** 2 + 1: return True
            if [*self.nodes().value()] == [x]: return x in can_end_in
            if all(y not in self.nodes() for y in can_end_in.value()): return False
            tmp0, tmp1 = self.prev(x), self.next(x)
            tmp0weights, tmp1weights = SortedKeysDict(*[(t, self.link_weights(t, x)) for t in tmp0.value()]), SortedKeysDict(*[(t, self.link_weights(x, t)) for t in tmp1.value()])
            self.remove(x)
            for y in tmp1.value():
                res = dfs(y)
                if res:
                    self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0.value()], [(t, tmp1weights[(x, t)]) for t in tmp1.value()])
                    return True
            self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0.value()], [(t, tmp1weights[(x, t)]) for t in tmp1.value()])
            return False
        u = self.nodes()[0]
        can_end_in = self.prev(u)
        return dfs(u)
    def hamiltonWalkExists(self, u: Node, v: Node):
        if u in self.next(v): return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        self.connect_from_to(v, (u, 0))
        res = self.hamiltonTourExists()
        self.disconnect(v, u)
        return res
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            for n in self.nodes().value():
                if not self.degrees(n)[1] and n != v or not self.degrees(n)[0] and n != x: return False
            if not self.nodes(): return stack
            tmp0, tmp1 = self.prev(x), self.next(x)
            tmp0weights, tmp1weights = SortedKeysDict(*[(t, self.link_weights(t, x)) for t in tmp0.value()]), SortedKeysDict(*[(t, self.link_weights(x, t)) for t in tmp1.value()])
            self.remove(x)
            for y in tmp1.value():
                if y == v:
                    if [*self.nodes().value()] == [v]: return stack
                    continue
                res = dfs(y, stack + [y])
                if res:
                    self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0.value()], [(t, tmp1weights[(x, t)]) for t in tmp1.value()])
                    return res
                self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0.value()], [(t, tmp1weights[(x, t)]) for t in tmp1.value()])
            return False
        if u is None:
            for _u in sorted(self.nodes().value(), key=lambda _x: self.degrees(_x)[1]):
                result = dfs(_u, [_u])
                if result: return result
            return False
        return dfs(u, [u])
    def isomorphic(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()): return False
            this_degrees, other_degrees, this_weights, other_weights = SortedKeysDict(), SortedKeysDict(), dict(), dict()
            for d in self.degrees().values():
                if d in this_degrees: this_degrees[d] += 1
                else: this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees: other_degrees[d] += 1
                else: other_degrees[d] = 1
            for w in self.link_weights().values():
                if w in this_weights: this_weights[w] += 1
                else: this_weights[w] = 1
            for w in other.link_weights().values():
                if w in other_weights: other_weights[w] += 1
                else: other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights: return False
            this_nodes_degrees, other_nodes_degrees = SortedKeysDict(*[(d, []) for d in this_degrees.keys()]), SortedKeysDict(*[(d, []) for d in other_degrees.keys()])
            for d in this_degrees.keys():
                for n in self.nodes().value():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility): map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if self.link_weights((n, m)) != other.link_weights((u, v)):
                            possible = False
                            break
                    if not possible: break
                if possible: return True
            return False
        return super().isomorphic(other)
    def __add__(self, other):
        if isinstance(other, (WeightedDirectedGraph, WeightedNodesDirectedGraph)): return other + self
        res = self.copy()
        if isinstance(other, WeightedLinksDirectedGraph):
            for n in other.nodes().value():
                if n not in res.nodes(): res.add(n)
            for l in other.links():
                if l in res.links(): res.__link_weights[l] += other.link_weights(l)
                else: res.connect_from_to(l[0], (l[1], other.link_weights(l)))
            return res
        if isinstance(other, DirectedGraph):
            for n in other.nodes().value():
                if n not in res.nodes(): res.add(n)
            for l in other.links(): res.connect_from_to(l[0], (l[1], self.link_weights(l)))
            return res
        raise TypeError(f"Addition not defined between class WeightedLinksDirectedGraph and type {type(other).__name__}!")
    def __eq__(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            if self.nodes() != other.nodes() or self.link_weights() != other.link_weights(): return False
            for n in self.nodes().value():
                if n not in other.nodes(): return False
            return True
        return False
    def __str__(self): return '({' + ', '.join(str(n) for n in self.nodes().value()) + '}, ' + f'{self.link_weights()}' + ')'
class WeightedDirectedGraph(WeightedNodesDirectedGraph, WeightedLinksDirectedGraph):
    def __init__(self, *pairs: (Node, float)): WeightedNodesDirectedGraph.__init__(self, *pairs)
    def total_weight(self): return self.total_nodes_weight() + self.total_links_weight()
    def add(self, n_w: (Node, float), *nodes_weights: (Node, float)):
        if n_w[0] not in self.nodes(): self._WeightedNodesDirectedGraph__node_weights[n_w[0]] = n_w[1]
        WeightedLinksDirectedGraph.add(self, n_w[0], *nodes_weights)
    def remove(self, u: Node, *rest: Node):
        for n in (u,) + rest: self.__node_weights.pop(n)
        WeightedLinksDirectedGraph.remove(self, u, *rest)
    def connect_from_to(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)): WeightedLinksDirectedGraph.connect_from_to(self, u, v_w, *nodes_weights)
    def connect_to_from(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)): WeightedLinksDirectedGraph.connect_to_from(self, u, v_w, *nodes_weights)
    def disconnect(self, u: Node, v: Node, *rest: Node): WeightedLinksDirectedGraph.disconnect(self, u, v, *rest)
    def copy(self):
        res = WeightedDirectedGraph(*self.node_weights().items().value())
        for n in self.nodes().value():
            if self.degrees(n)[1]: res.connect_from_to(n, *self.link_weights(n).items().value())
        return res
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None: res_path = []
            for y in filter(lambda _y: (x, _y) not in curr_path, self.next(x).value()):
                if curr_w + self.link_weights(x, y) + total_negative >= res_w and res_path: continue
                if y == v and (curr_w + self.link_weights(x, y) + self.node_weights(y) < res_w or not res_path): res_path, res_w = curr_path + [(x, y)], curr_w + self.link_weights(x, y) + self.node_weights(y)
                curr = dfs(y, curr_path + [(x, y)], curr_w + self.link_weights(x, y) + self.node_weights(y), total_negative - self.link_weights(x, y) * (self.link_weights(x, y) < 0) - self.node_weights(y) * (self.node_weights(y) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path: res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v): return dfs(u, [], self.node_weights(u), sum(self.link_weights(l) for l in self.links() if self.link_weights(l) < 0) + sum(self.node_weights(n) for n in self.nodes().value() if self.node_weights(n) < 0))
            return [], 0
        raise ValueError('Unrecognized node(s)!')
    def isomorphic(self, other):
        if isinstance(other, WeightedDirectedGraph):
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()): return False
            this_degrees, other_degrees, this_node_weights, other_node_weights, this_link_weights, other_link_weights = SortedKeysDict(), SortedKeysDict(), dict(), dict(), dict(), dict()
            for d in self.degrees().values():
                if d in this_degrees: this_degrees[d] += 1
                else: this_degrees[d] = 1
            for d in other.degrees().values():
                if d in other_degrees: other_degrees[d] += 1
                else: other_degrees[d] = 1
            for w in self.link_weights().values():
                if w in this_link_weights: this_link_weights[w] += 1
                else: this_link_weights[w] = 1
            for w in other.link_weights().values():
                if w in other_link_weights: other_link_weights[w] += 1
                else: other_link_weights[w] = 1
            for w in self.node_weights().values():
                if w in this_node_weights: this_node_weights[w] += 1
                else: this_node_weights[w] = 1
            for w in other.node_weights().values():
                if w in other_node_weights: other_node_weights[w] += 1
                else: other_node_weights[w] = 1
            if this_degrees != other_degrees or this_node_weights != other_node_weights or this_link_weights != other_link_weights: return False
            this_nodes_degrees, other_nodes_degrees = SortedKeysDict(*[(d, []) for d in this_degrees.keys()]), SortedKeysDict(*[(d, []) for d in other_degrees.keys()])
            for d in this_degrees.keys():
                for n in self.nodes().value():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility): map_dict += [*zip(group, other_nodes_degrees[i])]
                possible = True
                for n, u in map_dict:
                    for m, v in map_dict:
                        if self.link_weights(n, m) != other.link_weights(u, v) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible: break
                if possible: return True
            return False
        if isinstance(other, WeightedNodesDirectedGraph): return WeightedNodesDirectedGraph.isomorphic(self, other)
        if isinstance(other, WeightedLinksDirectedGraph): return WeightedLinksDirectedGraph.isomorphic(self, other)
        return DirectedGraph.isomorphic(self, other)
    def __add__(self, other):
        res = self.copy()
        if isinstance(other, WeightedDirectedGraph):
            for n in other.nodes().value():
                if n in res.nodes(): res.__node_weights[n] += other.node_weights(n)
                else: res.add((n, other.node_weights(n)))
            for l in other.links():
                if l in res.links(): res.__link_weights += other.link_weights(l)
                else: res.connect_from_to(l[0], (l[1], other.link_weights(l)))
        elif isinstance(other, WeightedNodesDirectedGraph):
            for n in other.nodes().value():
                if n in res.nodes(): res._WeightedNodesDirectedGraph__node_weights[n] += other.node_weights(n)
                else: res.add((n, other.node_weights(n)))
            for (u, v) in other.links():
                if v not in res.next(u): res.connect_from_to(u, (v, 0))
        elif isinstance(other, WeightedLinksDirectedGraph):
            for n in other.nodes().value():
                if n not in res.nodes(): res.add((n, 0))
            for l in other.links():
                if l in res.links(): res.__link_weights += other.link_weights(l)
                else: res.connect_from_to(l[0], (l[1], other.link_weights(l)))
        elif isinstance(other, DirectedGraph):
            for n in other.nodes().value():
                if n not in res.nodes(): res.add((n, 0))
            for (u, v) in other.links():
                if v not in res.next(u): res.connect_from_to(u, (v, 0))
        else: raise TypeError(f"Addition not defined between class WeightedDirectedGraph and type {type(other).__name__}!")
        return res
    def __eq__(self, other):
        if isinstance(other, WeightedDirectedGraph): return (self.node_weights(), self.link_weights()) == (other.node_weights(), other.link_weights())
        return False
    def __str__(self): return f"({self.node_weights()}, {self.link_weights()})"
