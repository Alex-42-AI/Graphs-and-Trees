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
        if u is None:
            return self.__degrees
        elif isinstance(u, Node):
            if u in self.__degrees:
                return self.__degrees[u]
            raise ValueError('No such node in the graph!')
        raise TypeError('Node expected!')
    def next(self, u: Node = None):
        if u is None:
            return self.__next
        if isinstance(u, Node):
            if u in self.__nodes:
                return self.__next[u]
            raise ValueError('Node not in graph!')
        raise TypeError('Node expected!')
    def prev(self, u: Node = None):
        if u is None:
            return self.__prev
        if isinstance(u, Node):
            if u in self.__nodes:
                return self.__prev[u]
            raise ValueError('Node not in graph!')
        raise TypeError('Node expected!')
    def add(self, u: Node, pointed_by: list = None, points_to: list = None):
        if u not in self.__nodes:
            if pointed_by is None:
                pointed_by = []
            if points_to is None:
                points_to = []
            res_pointed_by, res_points_to = [], []
            for v in pointed_by:
                if v in self.__nodes and v not in res_pointed_by:
                    res_pointed_by.append(v)
            for v in points_to:
                if v in self.__nodes and v not in res_points_to:
                    res_points_to.append(v)
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
            if u in self.__nodes:
                for v in self.__next[u]:
                    self.__prev[v].remove(u), self.__links.remove((u, v))
                    self.__degrees[v][1] -= 1
                for v in self.__prev[u]:
                    self.__next[v].remove(u), self.__links.remove((v, u))
                    self.__degrees[v][0] -= 1
                self.__nodes.remove(u), self.__next.pop(u), self.__degrees.pop(u), self.__prev.pop(u)
    def connect_from_to(self, u: Node, v: Node, *rest: Node):
        if u in self.__nodes:
            for n in [v] + list(rest):
                if (u, n) not in self.__links and u != n and n in self.__nodes:
                    self.__links.append((u, n))
                    self.__next[u].append(n)
                    self.__prev[n].append(u)
                    self.__degrees[u][0] += 1
                    self.__degrees[n][1] += 1
    def connect_to_from(self, u: Node, v: Node, *rest: Node):
        if u in self.__nodes:
            for n in [v] + list(rest):
                if (n, u) not in self.__links and u != n and n in self.__nodes:
                    self.__links.append((n, u)), self.__next[n].append(u), self.__prev[u].append(n)
                    self.__degrees[u][1] += 1
                    self.__degrees[n][0] += 1
    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in [v] + [*rest]:
            if n in self.__next[u]:
                self.__degrees[u][0] -= 1
                self.__degrees[n][1] -= 1
                self.__links.remove((u, n)), self.__next[u].remove(n), self.__prev[n].remove(u)
    def complementary(self):
        res = DirectedGraph(*self.__nodes)
        for i, n in enumerate(self.__nodes):
            for j in range(i + 1, len(self.__nodes)):
                if (n, self.__nodes[j]) not in self.__links:
                    res.connect_from_to(n, self.__nodes[j])
        return res
    def transposed(self):
        res = DirectedGraph(*self.__nodes)
        for l in self.__links:
            res.connect_to_from(l[0], l[1])
        return res
    def copy(self):
        res = DirectedGraph(*self.__nodes)
        for u in self.__nodes:
            if self.degrees(u)[0]:
                res.connect_from_to(u, *self.next(u))
        return res
    @staticmethod
    def __connected(nodes: [Node], links: [(Node, Node)]):
        queue, total, k, n = [nodes[0]], SortedList(), 1, len(nodes)
        total.insert(nodes[0])
        while queue:
            u = queue.pop(0)
            for v in filter(lambda x: x not in total and (u, x) in links or (x, u) in links, nodes):
                total.insert(v), queue.append(v)
                k += 1
            if k == n:
                return True
        return False
    def connected(self):
        return self.__connected(self.__nodes, self.__links)
    def sources(self):
        return [u for u in self.__nodes if not self.__prev[u]]
    def sinks(self):
        return [v for v in self.__nodes if not self.__next[v]]
    def dag(self):
        sources, total = self.sources(), SortedList()
        if not sources:
            return False
        stack = sources.copy()
        while stack:
            u = stack.pop()
            for v in self.__next[u]:
                if v in total:
                    continue
                if v in stack:
                    return False
                stack.append(v)
            total.insert(u)
        return True
    def toposort(self):
        if not self.dag():
            raise ValueError("Not a dag!")
        queue, res, total = self.sources(), [], SortedList()
        while queue:
            u = queue.pop(0)
            res.append(u), total.insert(u)
            for v in filter(lambda x: x not in total, self.__next[u]):
                queue.append(v), total.insert(v)
        return res
    def connection_components(self):
        if len(self.__nodes) in (0, 1):
            return [self.__nodes]
        components, queue, total, k, n = [[self.__nodes[0]]], [self.__nodes[0]], SortedList(), 1, len(self.__nodes)
        total.insert(self.__nodes[0])
        while k < n:
            while queue:
                u = queue.pop(0)
                for v in filter(lambda x: x not in total, self.__next[u] + self.__prev[u]):
                    components[-1].append(v), queue.append(v), total.insert(v)
                    k += 1
                    if k == n:
                        return components
            if k < n:
                new = [[n for n in self.__nodes if n not in total][0]]
                components.append(new), total.insert(new[0])
                k, queue = k + 1, [new[0]]
                if k == n:
                    return components
        return components
    def strongly_connected_components(self):
        def helper(x, stack):
            for y in self.__next[x]:
                if y not in curr:
                    helper(y, stack + [y])
                elif x not in curr and y in curr:
                    curr_node, new = stack.pop(), []
                    while curr_node not in curr:
                        total.insert(curr_node), curr.insert(curr_node), new.append(curr_node)
                        if not stack:
                            break
                        curr_node = stack.pop()
                    for curr_node in new:
                        helper(curr_node, stack + [curr_node])
                    return
        def dfs(x, stack):
            for y in self.__next[x]:
                if y not in stack and y not in curr:
                    dfs(y, stack + [y])
                if y == n:
                    curr_node = stack.pop()
                    while stack and curr_node != n:
                        total.insert(curr_node), curr.insert(curr_node)
                        curr_node = stack.pop()
                    for curr_node in curr:
                        helper(curr_node, [curr_node])
                    return
        total, res = SortedList(), []
        for n in self.__nodes:
            if n not in total:
                curr = SortedList()
                curr.insert(n), dfs(n, [n]), res.append(curr), total.insert(n)
        return res
    def SCC_DAG(self):
        res = DirectedGraph()
        for c in self.strongly_connected_components():
            curr = DirectedGraph(*c)
            for u in c:
                to_connect = filter(lambda x: x in c, self.__next[u])
                if to_connect:
                    curr.connect_from_to(u, *to_connect)
            res.add(Node(curr))
        for u in res.nodes():
            linked_to = SortedList(lambda x: len(x.nodes()))
            linked_to.insert(u)
            for n in u.value().nodes():
                for m in self.__next[n]:
                    for v in res.nodes():
                        if v not in linked_to and m in v.value().nodes():
                            res.connect_from_to(u, v), linked_to.insert(v)
                            break
        return res
    def reachable(self, u: Node, v: Node):
        if u not in self.__nodes or v not in self.__nodes:
            raise Exception('Unrecognized node(s).')
        total, queue = SortedList(), [u]
        total.insert(u)
        while queue:
            x = queue.pop(0)
            for y in filter(lambda _x: _x not in total, self.__next[x]):
                if y == v:
                    return True
                total.insert(y), queue.append(y)
        return False
    def cut_nodes(self):
        def dfs(x: Node, l: int):
            colors[x], levels[x], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False
            for y in self.__next[x] + self.__prev[x]:
                if not colors[y]:
                    count += 1
                    b = dfs(y, l + 1)
                    if b >= l and not is_root:
                        is_cut = True
                    min_back = min(min_back, b)
                if colors[y] == 1 and levels[y] < min_back and levels[y] + 1 != l:
                    min_back = levels[y]
            if is_cut or is_root and count > 1:
                res.append(x)
            colors[x] = 2
            return min_back
        levels = SortedKeysDict(*[(n, 0) for n in self.__nodes])
        colors, res = levels.copy(), []
        dfs(self.__nodes[0], 0)
        return res
    def bridge_links(self):
        def dfs(x: Node, l: int):
            colors[x], levels[x], min_back = 1, l, l
            for y in self.__next[x]:
                if not colors[y]:
                    b = dfs(y, l + 1)
                    if b > l:
                        res.append((x, y))
                    else:
                        min_back = min(min_back, b)
                if colors[y] == 1 and levels[y] < min_back and levels[y] + 1 != l:
                    min_back = levels[y]
            colors[x] = 2
            return min_back
        levels, res = SortedKeysDict(*[(n, 0) for n in self.__nodes]), []
        colors = levels.copy()
        dfs(self.__nodes[0], 0)
        return res
    def get_shortest_path(self, u: Node, v: Node):
        if u not in self.__nodes or v not in self.__nodes:
            raise Exception('Unrecognized node(s)!')
        previous = SortedKeysDict(*[(n, None) for n in self.__nodes])
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
            for y in filter(lambda _x: _x not in total, self.__next[x]):
                queue.append(y), total.insert(y)
                previous[y] = x
    def shortest_path_length(self, u: Node, v: Node):
        distances = SortedKeysDict(*[(n, 0) for n in self.__nodes])
        if u not in self.__nodes or v not in self.__nodes:
            raise Exception('Unrecognized node(s).')
        queue, total = [u], SortedList()
        total.insert(u)
        while queue:
            x = queue.pop(0)
            for y in filter(lambda _x: _x not in total, self.__next[x]):
                if y == v:
                    return 1 + distances[x]
                total.insert(x), queue.append(y)
                distances[y] = distances[x] + 1
        return float('inf')
    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack):
            if x not in self.__nodes or v not in self.__nodes:
                raise Exception('Unrecognized node(s).')
            if not l:
                return (False, stack)[x == v]
            for y in filter(lambda _x: (x, _x) not in stack, self.__next[x]):
                res = dfs(y, l - 1, stack + [(x, y)])
                if res:
                    return res
            return False
        tmp = self.get_shortest_path(u, v)
        if len(tmp) > length:
            return False
        if length == len(tmp):
            return tmp
        return dfs(u, length, [])
    def loopWithLength(self, length: int):
        for u in self.__nodes:
            for v in self.__next[u]:
                res = self.pathWithLength(v, u, length - 1)
                if res:
                    return [(u, v)] + res
        return False
    def euler_tour_exists(self):
        for d in self.__degrees.values():
            if d[0] != d[1]:
                return False
        return self.connected()
    def euler_walk_exists(self, u: Node, v: Node):
        if self.euler_tour_exists():
            return u == v
        for n in self.__nodes:
            if self.degrees(n)[0] % 2 and n != u or self.degrees(n)[1] % 2 and n != v:
                return False
        return self.degrees(u)[0] % 2 + self.degrees(v)[1] % 2 in [0, 2] and self.connected()
    def euler_tour(self):
        if self.euler_tour_exists():
            u, v = self.links()[0]
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
                        for j in range(len(stack)):
                            result.insert(i + j, stack[j])
                        return
                    dfs(y, stack + [(x, y)])
        if u in self.__nodes and v in self.__nodes:
            if self.euler_walk_exists(u, v):
                result = self.get_shortest_path(u, v)
                for i, l in enumerate(result):
                    n = l[0]
                    dfs(n, [])
                return result
            return False
    def hamiltonTourExists(self):
        def dfs(nodes: [Node], links: [(Node, Node)], can_continue_from: [Node] = None, can_end_in: [Node] = None, end_links: [(Node, Node)] = None):
            if can_continue_from is None:
                can_continue_from = nodes
            curr_degrees = SortedKeysDict(*[(n, [0, 0]) for n in nodes])
            for n in nodes:
                for l in links:
                    curr_degrees[n][0] += n == l[0]
                    curr_degrees[n][1] += n == l[1]
            if len(links) == len(nodes) ** 2 - len(nodes) or all(sum(curr_degrees[n]) >= len(can_continue_from) for n in can_continue_from):
                return True
            if can_end_in is not None:
                can_continue = False
                for n in nodes:
                    if n in [l[1] for l in end_links]:
                        can_continue = True
                        break
                if not can_continue:
                    return False
            for n in can_continue_from:
                if can_end_in is None:
                    can_end_in = [m for m in nodes if (m, n) in links]
                    if not can_end_in:
                        continue
                    end_links = [(m, n) for m in can_end_in]
                if dfs([_n for _n in nodes if _n != n], [l for l in links if n not in l], [_n for _n in nodes if (n, _n) in links], can_end_in, end_links):
                    return True
            return False
        return dfs(self.__nodes, self.__links)
    def hamiltonWalkExists(self, u: Node, v: Node = None):
        if u in self.next(v):
            return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        self.connect_from_to(v, u)
        res = self.hamiltonTourExists()
        self.disconnect(v, u)
        return res
    def hamiltonTour(self):
        if any(not self.__degrees[u][0] or not self.__degrees[u][1] for u in self.__nodes) or not self.__connected(self.__nodes, self.__links):
            return False
        u = self.__nodes[0]
        for v in self.next(u):
            result = self.hamiltonWalk(u, v)
            if result:
                return result + [u]
        return False
    def hamiltonWalk(self, u: Node, v: Node = None):
        def dfs(x: Node, y: Node = None, nodes: [Node] = None, links: [(Node, Node)] = None, can_continue_from: [Node] = None, res_stack: [Node] = None):
            if nodes is None:
                nodes = self.__nodes
            if links is None:
                links = self.__links
            if x in nodes and (y is None or y in nodes):
                if not self.__connected(nodes, links):
                    return False
                if res_stack is None:
                    res_stack = [x]
                curr_degrees = SortedKeysDict()
                for n in nodes:
                    curr_degrees[n] = [0, 0]
                    for m in nodes:
                        if (n, m) in links:
                            curr_degrees[n][0] += 1
                            if m in curr_degrees:
                                curr_degrees[m][1] += 1
                            else:
                                curr_degrees[m] = [0, 1]
                        elif (m, n) in links:
                            curr_degrees[n][1] += 1
                            if m in curr_degrees:
                                curr_degrees[m][0] += 1
                            else:
                                curr_degrees[m] = [1, 0]
                if can_continue_from is None:
                    can_continue_from = sorted([n for n in nodes if (x, n) in links and n != y], key=lambda _x: (curr_degrees[_x][1], curr_degrees[_x][0]))
                for n in nodes:
                    if not curr_degrees[n][0] and n != y or not curr_degrees[n][1] and n != x:
                        return False
                if y is None:
                    if len(nodes) == 1:
                        return nodes
                elif len(nodes) == 2 and (x, y) in links:
                    return [x, y]
                for n in can_continue_from:
                    res = dfs(n, y, [_n for _n in nodes if _n != x], [l for l in links if x not in l], sorted([_n for _n in nodes if (_n, n) in links and _n not in [x, y]], key=lambda _x: self.__degrees[_x][0]), [n])
                    if res:
                        return res_stack + res
                return False
            raise ValueError('Unrecognized nodes!')
        return dfs(u, v)
    def isomorphic(self, other):
        if type(other) == DirectedGraph:
            if len(self.__links) != len(other.__links):
                return False
            if len(self.__nodes) != len(other.__nodes):
                return False
            this_degrees, other_degrees = SortedKeysDict(), SortedKeysDict()
            for d in self.__degrees.values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.__degrees.values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            if this_degrees != other_degrees:
                return False
            this_nodes_degrees, other_nodes_degrees = SortedKeysDict(*[(d, []) for d in this_degrees.keys()]), SortedKeysDict(*[(d, []) for d in other_degrees.keys()])
            for d in this_degrees.keys():
                for n in self.__nodes:
                    if self.__degrees[n] == d:
                        this_nodes_degrees[d].append(n)
                    if other.__degrees[n] == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility):
                    map_dict += Dict(*zip(group, other_nodes_degrees[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.next(n)) ^ (v in other.next(u)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __reversed__(self):
        return self.complementary()
    def __contains__(self, item):
        return item in self.__nodes or item in self.__links
    def __add__(self, other):
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.__nodes:
                    res.add(n)
            for l in other.links():
                if l not in res.links():
                    res.connect_from_to(l[0], l[1])
            return res
        raise TypeError(f'Can\'t add class DirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, DirectedGraph):
            for l in self.__links:
                if l not in other.links():
                    return False
            return len(self.__links) == len(other.links()) and self.__nodes == other.nodes()
        return False
    def __str__(self):
        return '({' + ', '.join(str(n) for n in self.__nodes) + '}, {' + ', '.join(str(l[0]) + '->' + str(l[1]) for l in self.__links) + '})'
    def __repr__(self):
        return str(self)
class WeightedNodesDirectedGraph(DirectedGraph):
    def __init__(self, *pairs: (Node, float)):
        super().__init__(*[p[0] for p in pairs])
        self.__weights = SortedKeysDict()
        for (n, w) in pairs:
            if n not in self.__weights:
                self.__weights[n] = w
    def weights(self, n: Node = None):
        return self.__weights[n] if n is not None else self.__weights
    def total_weight(self):
        return sum(self.__weights.values())
    def copy(self):
        res = WeightedNodesDirectedGraph(*self.__weights.items())
        for n in self.nodes():
            if self.degrees(n):
                res.connect_from_to(n, *self.next(n))
        return res
    def add(self, n_w: (Node, float), *current_nodes: Node):
        super().add(n_w[0], *current_nodes)
        if n_w[0] not in self.nodes():
            self.__weights[n_w[0]] = n_w[1]
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes:
            if u in self.nodes():
                self.__weights.pop(u)
        super().remove(n, *nodes)
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None:
                res_path = []
            for y in [n for n in self.next(x) if (x, n) not in curr_path]:
                if curr_w + self.weights(y) + total_negative >= res_w and res_path:
                    continue
                if y == v:
                    if curr_w + self.weights(y) < res_w or not res_path:
                        res_path, res_w = curr_path + [(x, y)], curr_w + self.weights(y)
                curr = dfs(y, curr_path + [(x, y)], curr_w + self.weights(y), total_negative - self.weights(y) * (self.weights(y) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v):
                return dfs(u, [], 0, sum(self.weights(n) for n in self.nodes() if self.weights(n) < 0))
            return [], 0
        raise ValueError("Unrecognized node(s)!")
    def isomorphic(self, other):
        if type(other) == WeightedNodesDirectedGraph:
            if len(self.links()) != len(other.links()) or len(self.nodes()) != len(other.nodes()):
                return False
            this_degrees, other_degrees, this_weights, other_weights = dict(), dict(), dict(), dict()
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.__degrees.values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.__weights.values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.__weights.values():
                if w in other_weights:
                    other_weights[w] += 1
                else:
                    other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights:
                return False
            this_nodes_degrees, other_nodes_degrees = {d: [] for d in this_degrees.keys()}, {d: [] for d in other_degrees.keys()}
            for d in this_degrees.keys():
                for n in self.nodes():
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                    if other.degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility):
                    map_dict += Dict(*zip(group, other_nodes_degrees[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.next(n)) ^ (v in other.next(u)) or self.__weights[n] != other.__weights[u]:
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __add__(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n in res.nodes():
                    res.__weights[n] += other.__weights[n]
                else:
                    res.add(n, other.__weights[n])
            for l in other.links():
                res.connect_from_to(l[0], l[1])
            return res
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n, 0)
            for l in other.links():
                res.connect_from_to(l[0], l[1])
            return res
        raise TypeError(f'Can\'t add class WeightedUndirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, WeightedNodesDirectedGraph):
            if len(self.links()) != len(other.links()) or self.__weights != other.__weights:
                return False
            for l in self.links():
                if l not in other.links():
                    return False
            return True
        return super().__eq__(other)
    def __str__(self):
        return '({' + ', '.join(f'{str(n)} -> {self.__weights[n]}' for n in self.nodes()) + '}, ' + str(self.__links) + ')'
class WeightedLinksDirectedGraph(DirectedGraph):
    def __init__(self, *nodes: Node):
        super().__init__(*nodes)
        self.__weights = Dict()
    def weights(self, u_or_l: Node | tuple = None, v: Node = None):
        if u_or_l is None:
            return ', '.join([str(k) + ' -> ' + str(v) for k, v in self.__weights.items()])
        elif isinstance(u_or_l, Node):
            if v is None:
                return ', '.join(str((u_or_l, n)) + ' -> ' + str(self.__weights[(u_or_l, n)]) for n in [m for m in self.nodes() if (u_or_l, m) in self.links()])
            if isinstance(v, Node):
                if v in self.nodes():
                    if (u_or_l, v) in self.links():
                        return self.__weights[(u_or_l, v)]
                    raise KeyError(f'No link from {u_or_l} to {v}!')
                raise ValueError('No such node exists in this graph!')
            raise TypeError('Node expected!')
        elif isinstance(u_or_l, tuple):
            if u_or_l in self.links():
                return self.__weights[u_or_l]
            raise KeyError('Link not in graph!')
        raise TypeError('Node or link expected!')
    def total_weight(self):
        return sum(self.__weights.values())
    def add(self, u: Node, pointed_by_weights: list = None, points_to_weights: list = None):
        if u not in self.nodes():
            if pointed_by_weights is None:
                pointed_by_weights = []
            if points_to_weights is None:
                points_to_weights = []
            for p in points_to_weights + pointed_by_weights:
                if len(p) < 2:
                    raise ValueError('Node-value pairs expected!')
            for w in [p[1] for p in pointed_by_weights] + [p[1] for p in points_to_weights]:
                if not isinstance(w, (int, float)):
                    raise TypeError('Real numerical values expected!')
            pointed_by_res, points_to_res = [], []
            for v, w in pointed_by_weights:
                if v in self.nodes() and v not in [p[0] for p in pointed_by_res]:
                    pointed_by_res.append((v, w))
            for v, w in points_to_weights:
                if v in self.nodes() and v not in [p[0] for p in points_to_res]:
                    points_to_res.append((v, w))
            super().add(u, [p[0] for p in pointed_by_res], [p[0] for p in points_to_res])
            for v, w in pointed_by_res:
                self.__weights[(v, u)] = w
            for v, w in points_to_res:
                self.__weights[(u, v)] = w
    def remove(self, n: Node, *nodes: Node):
        for u in (n,) + nodes:
            for v in self.__next[u]:
                self.__weights.pop((u, v))
            for v in self.__prev[u]:
                self.__weights.pop((v, u))
        super().remove(n, *nodes)
    def connect_from_to(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect_from_to(u, *[p[0] for p in [v_w] + list(nodes_weights)])
            for v, w in [v_w] + list(nodes_weights):
                if (u, v) not in self.__weights:
                    self.__weights[(u, v)] = w
    def connect_to_from(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect_to_from(u, *[p[0] for p in [v_w] + list(nodes_weights)])
            for v, w in [v_w] + list(nodes_weights):
                if (v, u) not in self.__weights:
                    self.__weights[(v, u)] = w
    def disconnect(self, u: Node, v: Node, *rest: Node):
        super().disconnect(u, v, *rest)
        for n in [v] + [*rest]:
            if (u, n) in [l for l in self.__weights.keys()]:
                self.__weights.pop((u, n))
    def copy(self):
        res = WeightedLinksDirectedGraph(*self.nodes())
        for u in self.nodes():
            for v in self.nodes():
                if (u, v) in self.links():
                    res.connect_from_to(u, (v, self.weights(u, v)))
        return res
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None:
                res_path = []
            for y in [n for n in self.next(x) if (x, n) not in curr_path]:
                if curr_w + self.weights((x, y)) + total_negative >= res_w and res_path:
                    continue
                if y == v:
                    if curr_w + self.weights((x, y)) < res_w or not res_path:
                        res_path, res_w = curr_path.copy() + [(x, y)], curr_w + self.weights((x, y))
                curr = dfs(y, curr_path + [(x, y)], curr_w + self.weights((x, y)), total_negative - self.weights((x, y)) * (self.weights((x, y)) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v):
                return dfs(u, [], 0, sum(self.weights(l) for l in self.links() if self.weights(l) < 0))
            return [], 0
        raise ValueError('Unrecognized node(s)!')
    def hamiltonWalkExists(self, u: Node, v: Node = None):
        if u in self.next(v):
            return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        self.connect_from_to(v, (u, 0))
        res = self.hamiltonTourExists()
        self.disconnect(v, u)
        return res
    def hamiltonTour(self):
        if any(not self.degrees(n)[0] or not self.degrees(n)[1] for n in self.nodes()) or not self.connected():
            return False
        for u in self.nodes():
            for v in self.next(u):
                res = self.hamiltonWalk(u, v)
                if res:
                    return res[0] + [u], res[1] + sum(self.weights(res[0][i], res[0][i + 1]) for i in range(len(res[0]) - 1)) + self.weights(res[0][-1], res[0][0])
        return False
    def hamiltonWalk(self, u: Node, v: Node = None):
        def dfs(x: Node, y: Node = None, nodes: [Node] = None, links: [(Node, Node)] = None, can_continue_from: [Node] = None, res_stack: [Node] = None):
            if nodes is None:
                nodes = self.nodes()
            if links is None:
                links = self.links()
            if x in self.nodes() and y in nodes + [None]:
                if not self._DirectedGraph__connected(nodes, links):
                    return False
                if res_stack is None:
                    res_stack = [x]
                curr_degrees = SortedKeysDict()
                for n in nodes:
                    curr_degrees[n] = [0, 0]
                    for m in nodes:
                        if (n, m) in links:
                            curr_degrees[n][0] += 1
                            curr_degrees[m][1] += 1
                        elif (m, n) in links:
                            curr_degrees[n][1] += 1
                            curr_degrees[m][0] += 1
                if can_continue_from is None:
                    can_continue_from = sorted((n for n in nodes if (x, n) in links and n != y), key=lambda _x: (curr_degrees[_x][1], curr_degrees[_x][0]))
                for n in nodes:
                    if not curr_degrees[n][0] and n != y or not curr_degrees[n][1] and n != x:
                        return False
                if y is None:
                    if len(nodes) == 1:
                        return nodes, 0
                elif len(nodes) == 2 and (x, y) in links:
                    return [x, y], self.__weights[(x, y)]
                for n in sorted(can_continue_from, key=lambda _x: self.__weights[(_x, _x)]):
                    res = dfs(n, y, [_n for _n in nodes if _n != x], [l for l in links if x not in l], [_n for _n in nodes if (n, _n) in links and _n not in [x, y]], [n])
                    if res:
                        return res_stack + res[0], res[1] + sum(self.weights(res[0][i], res[0][i + 1]) for i in range(len(res[0]) - 1))
                return False
        return dfs(u, v)
    def isomorphic(self, other):
        if type(other) == DirectedGraph:
            if len(self.links()) != len(other.links()):
                return False
            if len(self.nodes()) != len(other.nodes()):
                return False
            this_degrees, other_degrees, this_weights, other_weights = SortedKeysDict(), SortedKeysDict(), dict(), dict()
            for d in self.degrees().values():
                if d in this_degrees:
                    this_degrees[d] += 1
                else:
                    this_degrees[d] = 1
            for d in other.__degrees.values():
                if d in other_degrees:
                    other_degrees[d] += 1
                else:
                    other_degrees[d] = 1
            for w in self.__weights.values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.__weights.values():
                if w in other_weights:
                    other_weights[w] += 1
                else:
                    other_weights[w] = 1
            if this_degrees != other_degrees or this_weights != other_weights:
                return False
            this_nodes_degrees, other_nodes_degrees = SortedKeysDict(*[(d, []) for d in this_degrees.keys()]), SortedKeysDict(*[(d, []) for d in other_degrees.keys()])
            for d in this_degrees.keys():
                for n in self.nodes():
                    if self.degrees(n) == d:
                        this_nodes_degrees[d].append(n)
                    if other.__degrees(n) == d:
                        other_nodes_degrees[d].append(n)
            this_nodes_degrees, other_nodes_degrees = list(sorted(this_nodes_degrees.values(), key=lambda _p: len(_p))), list(sorted(other_nodes_degrees.values(), key=lambda _p: len(_p)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_degrees]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility):
                    map_dict += Dict(*zip(group, other_nodes_degrees[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if self.__weights[(n, m)] != other.__weights[(u, v)]:
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __add__(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n)
            for l in other.links():
                res.connect_from_to(l[0], (l[1], self.__weights[l]))
            return res
        if isinstance(other, DirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n, [], [])
            for l in other.links():
                res.connect_from_to(l[0], (l[1], self.__weights[l]))
            return res
        raise TypeError(f'Can\'t add class WeightedDirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, WeightedLinksDirectedGraph):
            for n in self.nodes():
                if n not in other.nodes():
                    return False
            return self.nodes() == len(other.nodes()) and self.__weights == other.__weights
        return super().__eq__(other)
    def __str__(self):
        return '({' + ', '.join(str(n) for n in self.nodes()) + '}, ' + f'{self.__weights}' + ')'
