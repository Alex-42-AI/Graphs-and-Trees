from Personal.DiscreteMath.Graphs.General import Node, Dict, SortedKeysDict, SortedList
class DirectedGraph:
    def __init__(self, *nodes: Node):
        self.__nodes = SortedList()
        for n in nodes:
            if n not in self.__nodes:
                self.__nodes.insert(n)
        self.__links, self.__degrees, self.__prev, self.__next = [], SortedKeysDict(*[(n, [0, 0]) for n in self.__nodes]), SortedKeysDict(*[(n, []) for n in self.__nodes]), SortedKeysDict(*[(n, []) for n in self.__nodes])
    def nodes(self):
        return self.__nodes
    def links(self):
        return self.__links
    def degrees(self, u: Node = None):
        if u is None: return self.__degrees
        elif isinstance(u, Node):
            if u in self.__degrees: return self.__degrees[u]
            raise ValueError('No such node in the graph!')
        raise TypeError('Node expected!')
    def next(self, u: Node = None):
        if u is None: return self.__next
        if isinstance(u, Node):
            if u in self.nodes(): return self.__next[u]
            raise ValueError('Node not in graph!')
        raise TypeError('Node expected!')
    def prev(self, u: Node = None):
        if u is None: return self.__prev
        if isinstance(u, Node):
            if u in self.nodes(): return self.__prev[u]
            raise ValueError('Node not in graph!')
        raise TypeError('Node expected!')
    def add(self, u: Node, pointed_by: list = None, points_to: list = None):
        if u not in self.nodes():
            if pointed_by is None: pointed_by = []
            if points_to is None: points_to = []
            res_pointed_by, res_points_to = [], []
            for v in pointed_by:
                if v in self.nodes() and v not in res_pointed_by: res_pointed_by.append(v)
            for v in points_to:
                if v in self.nodes() and v not in res_points_to: res_points_to.append(v)
            self.__degrees[u], self.__next[u], self.__prev[u] = [len(res_points_to), len(res_pointed_by)], [], []
            for v in res_pointed_by:
                self.__links.append((v, u)), self.__next[v].append(u), self.__prev[u].append(v)
                self.__degrees[v][0] += 1
            for v in res_points_to:
                self.__links.append((u, v)), self.__next[u].append(v), self.__prev[v].append(u)
                self.__degrees[v][1] += 1
            self.__nodes.insert(u)
    def remove(self, node: Node, *nodes: Node):
        for u in (node,) + nodes:
            if u in self.nodes():
                for v in self.next(u):
                    self.__prev[v].remove(u), self.__links.remove((u, v))
                    self.__degrees[v][1] -= 1
                for v in self.__prev[u]:
                    self.__next[v].remove(u), self.__links.remove((v, u))
                    self.__degrees[v][0] -= 1
                self.__nodes.remove(u), self.__next.pop(u), self.__degrees.pop(u), self.__prev.pop(u)
    def connect_from_to(self, u: Node, v: Node, *rest: Node):
        if u in self.nodes():
            for n in [v] + list(rest):
                if (u, n) not in self.links() and u != n and n in self.nodes():
                    self.__links.append((u, n)), self.__next[u].append(n), self.__prev[n].append(u)
                    self.__degrees[u][0] += 1
                    self.__degrees[n][1] += 1
    def connect_to_from(self, u: Node, v: Node, *rest: Node):
        if u in self.nodes():
            for n in [v] + list(rest):
                if (n, u) not in self.links() and u != n and n in self.nodes():
                    self.__links.append((n, u)), self.__next[n].append(u), self.__prev[u].append(n)
                    self.__degrees[u][1] += 1
                    self.__degrees[n][0] += 1
    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in [v] + [*rest]:
            if n in self.next(u):
                self.__degrees[u][0] -= 1
                self.__degrees[n][1] -= 1
                self.__links.remove((u, n)), self.__next[u].remove(n), self.__prev[n].remove(u)
    def complementary(self):
        res = DirectedGraph(*self.nodes())
        for i, n in enumerate(self.nodes()):
            for j in range(i + 1, len(self.nodes())):
                if (n, self.nodes()[j]) not in self.links(): res.connect_from_to(n, self.nodes()[j])
        return res
    def transposed(self):
        res = DirectedGraph(*self.nodes())
        for l in self.links(): res.connect_to_from(l[0], l[1])
        return res
    def copy(self):
        res = DirectedGraph(*self.nodes())
        for u in self.nodes():
            if self.degrees(u)[0]: res.connect_from_to(u, *self.next(u))
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
    def sources(self):
        return [u for u in self.nodes() if not self.prev(u)]
    def sinks(self):
        return [v for v in self.nodes() if not self.next(v)]
    def dag(self):
        sources, total = self.sources(), SortedList()
        if not sources: return False
        stack = sources.copy()
        while stack:
            u = stack.pop()
            for v in self.next(u):
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
            for v in filter(lambda x: x not in total, self.next(u)): queue.append(v), total.insert(v)
        return res
    def connection_components(self):
        if len(self.nodes()) in (0, 1): return [self.nodes()]
        components, queue, total, k, n = [[self.nodes()[0]]], [self.nodes()[0]], SortedList(), 1, len(self.nodes())
        total.insert(self.nodes()[0])
        while k < n:
            while queue:
                u = queue.pop(0)
                for v in filter(lambda x: x not in total, self.next(u) + self.prev(u)):
                    components[-1].append(v), queue.append(v), total.insert(v)
                    k += 1
                    if k == n: return components
            if k < n:
                new = [[n for n in self.nodes() if n not in total][0]]
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
            for y in filter(lambda _x: _x not in total, self.next(x)):
                if y == v: return True
                total.insert(y), queue.append(y)
        return False
    def cut_nodes(self):
        def dfs(x: Node, l: int):
            colors[x], levels[x], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False
            for y in self.next(x) + self.prev(x):
                if not colors[y]:
                    count += 1
                    b = dfs(y, l + 1)
                    if b >= l and not is_root: is_cut = True
                    min_back = min(min_back, b)
                if colors[y] == 1 and levels[y] < min_back and levels[y] + 1 != l: min_back = levels[y]
            if is_cut or is_root and count > 1: res.append(x)
            colors[x] = 2
            return min_back
        levels = SortedKeysDict(*[(n, 0) for n in self.nodes()])
        colors, res = levels.copy(), []
        dfs(self.nodes()[0], 0)
        return res
    def bridge_links(self):
        def dfs(x: Node, l: int):
            colors[x], levels[x], min_back = 1, l, l
            for y in self.next(x):
                if not colors[y]:
                    b = dfs(y, l + 1)
                    if b > l: res.append((x, y))
                    else: min_back = min(min_back, b)
                if colors[y] == 1 and levels[y] < min_back and levels[y] + 1 != l: min_back = levels[y]
            colors[x] = 2
            return min_back
        levels, res = SortedKeysDict(*[(n, 0) for n in self.nodes()]), []
        colors = levels.copy()
        dfs(self.nodes()[0], 0)
        return res
    def get_shortest_path(self, u: Node, v: Node):
        if u not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s)!')
        previous = SortedKeysDict(*[(n, None) for n in self.nodes()])
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
            for y in filter(lambda _x: _x not in total, self.next(x)):
                queue.append(y), total.insert(y)
                previous[y] = x
    def shortest_path_length(self, u: Node, v: Node):
        distances = SortedKeysDict(*[(n, 0) for n in self.nodes()])
        if u not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s).')
        queue, total = [u], SortedList()
        total.insert(u)
        while queue:
            x = queue.pop(0)
            for y in filter(lambda _x: _x not in total, self.next(x)):
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
        for n in self.nodes():
            if self.degrees(n)[0] % 2 and n != u or self.degrees(n)[1] % 2 and n != v: return False
        return self.degrees(u)[0] % 2 + self.degrees(v)[1] % 2 in [0, 2] and self.connected()
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
            for y in self.next(x):
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
            for y in self.next(x):
                if y not in curr: helper(y, stack + [y])
                elif x not in curr and y in curr:
                    curr_node, new = stack.pop(), []
                    while curr_node not in curr:
                        total.insert(curr_node), curr.insert(curr_node), new.append(curr_node)
                        if not stack: break
                        curr_node = stack.pop()
                    return
        def dfs(x, stack):
            for y in self.next(x):
                if y not in curr and y not in stack: dfs(y, stack + [y])
                if y == n:
                    curr_node = stack.pop()
                    while stack and curr_node != n:
                        total.insert(curr_node), curr.insert(curr_node)
                        curr_node = stack.pop()
                    for curr_node in curr: helper(curr_node, [curr_node])
                    return
        total, res = SortedList(), []
        for n in self.nodes():
            if n not in total:
                curr = SortedList()
                curr.insert(n), dfs(n, [n]), res.append(curr), total.insert(n)
        return res
    def sccDag(self):
        res = DirectedGraph()
        for c in self.stronglyConnectedComponents():
            curr = DirectedGraph(*c)
            for u in c:
                to_connect = filter(lambda x: x in c, self.next(u))
                if to_connect: curr.connect_from_to(u, *to_connect)
            res.add(Node(curr))
        for u in res.nodes():
            linked_to = SortedList(lambda x: len(x.nodes()))
            linked_to.insert(u)
            for n in u.value().nodes():
                for m in self.next(n):
                    for v in res.nodes():
                        if v not in linked_to and m in v.value().nodes():
                            res.connect_from_to(u, v), linked_to.insert(v)
                            break
        return res
    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack):
            if x not in self.nodes() or v not in self.nodes(): raise Exception('Unrecognized node(s).')
            if not l: return (False, stack)[x == v]
            for y in filter(lambda _x: (x, _x) not in stack, self.next(x)):
                res = dfs(y, l - 1, stack + [(x, y)])
                if res: return res
            return False
        tmp = self.get_shortest_path(u, v)
        if len(tmp) > length: return False
        if length == len(tmp): return tmp
        return dfs(u, length, [])
    def loopWithLength(self, length: int):
        for u in self.nodes():
            for v in self.next(u):
                res = self.pathWithLength(v, u, length - 1)
                if res: return [(u, v)] + res
        return False
    def hamiltonTourExists(self):
        def dfs(x):
            if len(self.links()) > (len(self.nodes()) - 1) ** 2 + 1: return True
            if [*self.nodes()] == [x]: return x in can_end_in
            if all(y not in self.nodes() for y in can_end_in): return False
            tmp0, tmp1 = self.prev(x), self.next(x)
            self.remove(x)
            for y in tmp1:
                res = dfs(y)
                if res:
                    self.add(x, tmp0, tmp1)
                    return True
            self.add(x, tmp0, tmp1)
            return False
        u = self.nodes()[0]
        can_end_in = self.prev(u)
        return dfs(u)
    def hamiltonWalkExists(self, u: Node, v: Node):
        if u in self.next(v): return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        self.connect_from_to(v, u)
        res = self.hamiltonTourExists()
        self.disconnect(v, u)
        return res
    def hamiltonTour(self):
        if any(not self.degrees(u)[0] or not self.degrees(u)[1] for u in self.nodes()) or not self.connected(): return False
        u = self.nodes()[0]
        for v in self.prev(u):
            result = self.hamiltonWalk(u, v)
            if result: return result
        return False
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            for n in self.nodes():
                if not self.degrees(n)[0] and n != v or not self.degrees(n)[1] and n != x: return False
            if not self.nodes(): return stack
            tmp0, tmp1 = self.prev(x), self.next(x)
            self.remove(x)
            for y in tmp1:
                if y == v:
                    if [*self.nodes()] == [v]: return stack
                    continue
                res = dfs(y, stack + [y])
                if res:
                    self.add(x, tmp0, tmp1)
                    return res
            self.add(x, tmp0, tmp1)
            return False
        if u is None:
            for _u in sorted(self.nodes(), key=lambda _x: self.degrees(_x)[0]):
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
                for n in self.nodes():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility): map_dict += Dict(*zip(group, other_nodes_degrees[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
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
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes(): res.add(n)
            for l in other.links():
                if l not in res.links(): res.connect_from_to(l[0], l[1])
            return res
        raise TypeError(f'Can\'t add class DirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, DirectedGraph):
            for l in self.links():
                if l not in other.links(): return False
            return len(self.links()) == len(other.links()) and self.nodes() == other.nodes()
        return False
    def __str__(self):
        return '({' + ', '.join(str(n) for n in self.nodes()) + '}, {' + ', '.join(str(l[0]) + '->' + str(l[1]) for l in self.links()) + '})'
    def __repr__(self):
        return str(self)
class WeightedNodesDirectedGraph(DirectedGraph):
    def __init__(self, *pairs: (Node, float)):
        super().__init__(*[p[0] for p in pairs])
        self.__node_weights = SortedKeysDict()
        for (n, w) in pairs:
            if n not in self.__node_weights: self.__node_weights[n] = w
    def node_weights(self, n: Node = None):
        return self.__node_weights[n] if n is not None else self.__node_weights
    def total_weight(self):
        return sum(self.node_weights().values())
    def copy(self):
        res = WeightedNodesDirectedGraph(*self.node_weights().items())
        for n in self.nodes():
            if self.degrees(n): res.connect_from_to(n, *self.next(n))
        return res
    def add(self, n_w: (Node, float), *current_nodes: Node):
        if n_w[0] not in self.nodes(): self.__node_weights[n_w[0]] = n_w[1]
        super().add(n_w[0], *current_nodes)
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes:
            if u in self.nodes(): self.__node_weights.pop(u)
        super().remove(n, *nodes)
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None: res_path = []
            for y in [n for n in self.next(x) if (x, n) not in curr_path]:
                if curr_w + self.node_weights(y) + total_negative >= res_w and res_path: continue
                if y == v and (curr_w + self.node_weights(y) < res_w or not res_path): res_path, res_w = curr_path + [(x, y)], curr_w + self.node_weights(y)
                curr = dfs(y, curr_path + [(x, y)], curr_w + self.node_weights(y), total_negative - self.node_weights(y) * (self.node_weights(y) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path: res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v): return dfs(u, [], 0, sum(self.node_weights(n) for n in self.nodes() if self.node_weights(n) < 0))
            return [], 0
        raise ValueError("Unrecognized node(s)!")
    def hamiltonTourExists(self):
        def dfs(x):
            if len(self.links()) > (len(self.nodes()) - 1) ** 2 + 1: return True
            if [*self.nodes()] == [x]: return x in can_end_in
            if all(y not in self.nodes() for y in can_end_in): return False
            tmp0, tmp1, w = self.prev(x), self.next(x), self.node_weights(x)
            self.remove(x)
            for y in tmp1:
                res = dfs(y)
                if res:
                    self.add((x, w), tmp0, tmp1)
                    return True
            self.add((x, w), tmp0, tmp1)
            return False
        u = self.nodes()[0]
        can_end_in = self.prev(u)
        return dfs(u)
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x, stack):
            for n in self.nodes():
                if not self.degrees(n)[0] and n != v or not self.degrees(n)[1] and n != x: return False
            if not self.nodes(): return stack
            tmp0, tmp1, w = self.prev(x), self.next(x), self.node_weights(x)
            self.remove(x)
            for y in tmp1:
                if y == v:
                    if [*self.nodes()] == [v]: return stack
                    continue
                res = dfs(y, stack + [y])
                if res:
                    self.add((x, w), tmp0, tmp1)
                    return res
            self.add((x, w), tmp0, tmp1)
            return False
        if u is None:
            for _u in sorted(self.nodes(), key=lambda _x: self.degrees(_x)[0]):
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
                for n in self.nodes():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility): map_dict += Dict(*zip(group, other_nodes_degrees[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.next(n)) ^ (v in other.next(u)) or self.node_weights(n) != other.node_weights(u):
                            possible = False
                            break
                    if not possible: break
                if possible: return True
            return False
        return super().isomorphic(other)
    def __add__(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n in res.nodes(): res.__node_weights[n] += other.node_weights(n)
                else: res.add(n, other.node_weights(n))
            for l in other.links(): res.connect_from_to(l[0], l[1])
            return res
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes(): res.add(n, 0)
            for l in other.links(): res.connect_from_to(l[0], l[1])
            return res
        raise TypeError(f'Can\'t add class WeightedUndirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links()) != len(other.links()) or self.node_weights() != other.node_weights(): return False
            for l in self.links():
                if l not in other.links(): return False
            return True
        return super().__eq__(other)
    def __str__(self):
        return '({' + ', '.join(f'{str(n)} -> {self.node_weights(n)}' for n in self.nodes()) + '}, ' + str(self.links()) + ')'
class WeightedLinksDirectedGraph(DirectedGraph):
    def __init__(self, *nodes: Node):
        super().__init__(*nodes)
        self.__link_weights = Dict()
    def link_weights(self, u_or_l: Node | tuple = None, v: Node = None):
        if u_or_l is None: return ', '.join([str(k) + ' -> ' + str(v) for k, v in self.__link_weights.items()])
        elif isinstance(u_or_l, Node):
            if v is None: return ', '.join(str((u_or_l, n)) + ' -> ' + str(self.__link_weights[(u_or_l, n)]) for n in [m for m in self.nodes() if (u_or_l, m) in self.links()])
            if isinstance(v, Node):
                if v in self.nodes():
                    if (u_or_l, v) in self.links(): return self.__link_weights[(u_or_l, v)]
                    raise KeyError(f'No link from {u_or_l} to {v}!')
                raise ValueError('No such node exists in this graph!')
            raise TypeError('Node expected!')
        elif isinstance(u_or_l, tuple):
            if u_or_l in self.links(): return self.__link_weights[u_or_l]
            raise KeyError('Link not in graph!')
        raise TypeError('Node or link expected!')
    def total_weight(self):
        return sum(self.link_weights().values())
    def add(self, u: Node, pointed_by_weights: list = None, points_to_weights: list = None):
        if u not in self.nodes():
            if pointed_by_weights is None: pointed_by_weights = []
            if points_to_weights is None: points_to_weights = []
            for p in points_to_weights + pointed_by_weights:
                if len(p) < 2: raise ValueError('Node-value pairs expected!')
            for w in [p[1] for p in pointed_by_weights] + [p[1] for p in points_to_weights]:
                if not isinstance(w, (int, float)): raise TypeError('Real numerical values expected!')
            pointed_by_res, points_to_res = [], []
            for v, w in pointed_by_weights:
                if v in self.nodes() and v not in [p[0] for p in pointed_by_res]: pointed_by_res.append((v, w))
            for v, w in points_to_weights:
                if v in self.nodes() and v not in [p[0] for p in points_to_res]: points_to_res.append((v, w))
            super().add(u, [p[0] for p in pointed_by_res], [p[0] for p in points_to_res])
            for v, w in pointed_by_res: self.__link_weights[(v, u)] = w
            for v, w in points_to_res: self.__link_weights[(u, v)] = w
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes:
            for v in self.next(u): self.__link_weights.pop((u, v))
            for v in self.prev(u): self.__link_weights.pop((v, u))
        super().remove(n, *nodes)
    def connect_from_to(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect_from_to(u, *[p[0] for p in [v_w] + list(nodes_weights)])
            for v, w in [v_w] + list(nodes_weights):
                if (u, v) not in self.link_weights(): self.__link_weights[(u, v)] = w
    def connect_to_from(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect_to_from(u, *[p[0] for p in [v_w] + list(nodes_weights)])
            for v, w in [v_w] + list(nodes_weights):
                if (v, u) not in self.link_weights(): self.__link_weights[(v, u)] = w
    def disconnect(self, u: Node, v: Node, *rest: Node):
        super().disconnect(u, v, *rest)
        for n in [v] + [*rest]:
            if (u, n) in [l for l in self.link_weights().keys()]: self.__link_weights.pop((u, n))
    def copy(self):
        res = WeightedLinksDirectedGraph(*self.nodes())
        for u in self.nodes():
            for v in self.nodes():
                if (u, v) in self.links(): res.connect_from_to(u, (v, self.link_weights(u, v)))
        return res
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None: res_path = []
            for y in [n for n in self.next(x) if (x, n) not in curr_path]:
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
            if [*self.nodes()] == [x]: return x in can_end_in
            if all(y not in self.nodes() for y in can_end_in): return False
            tmp0, tmp1 = self.prev(x), self.next(x)
            tmp0weights, tmp1weights = SortedKeysDict(*[(t, self.link_weights(t, x)) for t in tmp0]), SortedKeysDict(*[(t, self.link_weights(x, t)) for t in tmp1])
            self.remove(x)
            for y in tmp1:
                res = dfs(y)
                if res:
                    self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0], [(t, tmp1weights[(x, t)]) for t in tmp1])
                    return True
            self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0], [(t, tmp1weights[(x, t)]) for t in tmp1])
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
            for n in self.nodes():
                if not self.degrees(n)[0] and n != v or not self.degrees(n)[1] and n != x: return False
            if not self.nodes(): return stack
            tmp0, tmp1 = self.prev(x), self.next(x)
            tmp0weights, tmp1weights = SortedKeysDict(*[(t, self.link_weights(t, x)) for t in tmp0]), SortedKeysDict(*[(t, self.link_weights(x, t)) for t in tmp1])
            self.remove(x)
            for y in tmp1:
                if y == v:
                    if [*self.nodes()] == [v]: return stack
                    continue
                res = dfs(y, stack + [y])
                if res:
                    self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0], [(t, tmp1weights[(x, t)]) for t in tmp1])
                    return res
                self.add(x, [(t, tmp0weights[(t, x)]) for t in tmp0], [(t, tmp1weights[(x, t)]) for t in tmp1])
            return False
        if u is None:
            for _u in sorted(self.nodes(), key=lambda _x: self.degrees(_x)[0]):
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
                for n in self.nodes():
                    if self.degrees(n) == d: this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d: other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility): map_dict += Dict(*zip(group, other_nodes_degrees[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if self.link_weights((n, m)) != other.link_weights((u, v)):
                            possible = False
                            break
                    if not possible: break
                if possible: return True
            return False
        return super().isomorphic(other)
    def __add__(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes(): res.add(n)
            for l in other.links(): res.connect_from_to(l[0], (l[1], self.link_weights(l)))
            return res
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes(): res.add(n, [], [])
            for l in other.links(): res.connect_from_to(l[0], (l[1], self.link_weights(l)))
            return res
        raise TypeError(f'Can\'t add class WeightedDirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            for n in self.nodes():
                if n not in other.nodes(): return False
            return self.nodes() == len(other.nodes()) and self.link_weights() == other.link_weights()
        return super().__eq__(other)
    def __str__(self):
        return '({' + ', '.join(str(n) for n in self.nodes()) + '}, ' + f'{self.link_weights()}' + ')'
