from Personal.DiscreteMath.Graphs.General import Node, Link, Dict, SortedKeysDict, SortedList
class UndirectedGraph:
    def __init__(self, *nodes: Node):
        self.__nodes = SortedList()
        for n in nodes:
            if n not in self.__nodes:
                self.__nodes.insert(n)
        self.__links, self.__neighboring, self.__degrees = [], SortedKeysDict(*[(n, []) for n in self.__nodes]), SortedKeysDict(*[(n, 0) for n in self.__nodes])
    def nodes(self):
        return self.__nodes
    def links(self):
        return self.__links
    def neighboring(self, u: Node = None):
        if u is None:
            return self.__neighboring
        if isinstance(u, Node):
            if u in self.__nodes:
                return self.__neighboring[u]
            raise KeyError('No such node in the graph!')
        raise TypeError('Node expected!')
    def degrees(self, u: Node = None):
        if u is None:
            return self.__degrees
        if isinstance(u, Node):
            if u in self.__nodes:
                return self.__degrees[u]
            raise KeyError('No such node in the graph!')
        raise TypeError('Node expected!')
    def degrees_sum(self):
        return 2 * len(self.__links)
    def add(self, u: Node, *current_nodes: Node):
        if u not in self.__nodes:
            res = []
            for c in current_nodes:
                if c in self.__nodes and c not in res:
                    res.append(c)
            self.__degrees[u] = len(res)
            for v in res:
                self.__degrees[v] += 1
                self.__links.append(Link(v, u)), self.__neighboring[v].append(u)
            self.__nodes.insert(u)
            self.__neighboring[u] = res
    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            if u in self.__nodes:
                for v in self.__neighboring[u]:
                    self.__degrees[v] -= 1
                    self.__links.remove(Link(v, u)), self.__neighboring[v].remove(u)
                self.__nodes.remove(u), self.__degrees.pop(u), self.__neighboring.pop(u)
    def connect(self, u: Node, v: Node, *rest: Node):
        for n in [v] + [*rest]:
            if Link(u, n) not in self.__links and u != n and n in self.__nodes:
                self.__degrees[u] += 1
                self.__degrees[n] += 1
                self.__neighboring[u].append(n), self.__neighboring[n].append(u), self.__links.append(Link(u, n))
    def disconnect(self, u: Node, v: Node, *rest: Node):
        for n in [v] + [*rest]:
            if Link(u, n) in self.__links:
                self.__degrees[u] -= 1
                self.__degrees[n] -= 1
                self.__neighboring[u].remove(n), self.__neighboring[n].remove(u), self.__links.remove(Link(u, n))
    def width(self):
        res = 0
        for u in self.__nodes:
            _res, total, queue = 0, SortedList(), [u]
            total.insert(u)
            while queue:
                u = queue.pop(0)
                for v in filter(lambda x: x not in total, self.__neighboring[u]):
                    total.insert(v), queue.append(v)
                    if len(total) == len(self.__nodes):
                        break
                _res += 1
            if _res > res:
                res = _res
        return res
    def complementary(self):
        res = UndirectedGraph(*self.__nodes)
        for i, n in enumerate(self.__nodes):
            for j in range(i + 1, len(self.__nodes)):
                if self.__nodes[j] not in self.__neighboring[n]:
                    res.connect(n, self.__nodes[j])
        return res
    def copy(self):
        res = UndirectedGraph(*self.__nodes)
        for n in self.__nodes:
            if self.degrees(n):
                res.connect(n, *self.neighboring(n))
        return res
    def connection_components(self):
        if len(self.__nodes) in (0, 1):
            return [self.__nodes]
        components, queue, total, k, n = [[self.__nodes[0]]], [self.__nodes[0]], SortedList(), 1, len(self.__nodes)
        total.insert(self.__nodes[0])
        while k < n:
            while queue:
                u = queue.pop(0)
                for v in filter(lambda x: x not in total, self.__neighboring[u]):
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
    @staticmethod
    def __connected(nodes: [Node], links: [Link]):
        if len(links) < len(nodes) - 1:
            return False
        if 2 * len(links) > (len(nodes) - 1) * (len(nodes) - 2) or len(nodes) == 1:
            return True
        queue, total, k, n = [nodes[0]], SortedList(), 1, len(nodes)
        total.insert(nodes[0])
        while queue:
            u = queue.pop(0)
            for v in filter(lambda x: x not in total and Link(u, x) in links, nodes):
                total.insert(v), queue.append(v)
                k += 1
            if k == n:
                return True
        return False
    def connected(self):
        return self.__connected(self.__nodes, self.__links)
    def tree(self):
        if len(self.__nodes) != len(self.__links) + 1:
            return False
        nodes, duplicates = SortedList(), SortedList()
        for l in self.__links:
            if l[0] not in nodes:
                nodes.insert(l[0])
            else:
                if l[1] in duplicates:
                    return False
                duplicates.insert(l[0])
            if l[1] not in nodes:
                nodes.insert(l[1])
            else:
                if l[0] in duplicates:
                    return False
                duplicates.insert(l[1])
        return True
    def reachable(self, u: Node, v: Node):
        if u not in self.__nodes or v not in self.__nodes:
            raise Exception('Unrecognized node(s).')
        if u == v:
            return True
        total, queue = SortedList(), [u]
        total.insert(u)
        while queue:
            n = queue.pop(0)
            for m in filter(lambda x: x not in total, self.__neighboring[n]):
                if m == v:
                    return True
                total.insert(m), queue.append(m)
        return False
    def cut_nodes(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back, is_root, count, is_cut = 1, l, l, not l, 0, False
            for v in self.__neighboring[u]:
                if not colors[v]:
                    count += 1
                    b = dfs(v, l + 1)
                    if b >= l and not is_root:
                        is_cut = True
                    min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            if is_cut or is_root and count > 1:
                res.append(u)
            colors[u] = 2
            return min_back
        levels = SortedKeysDict(*[(n, 0) for n in self.__nodes])
        colors, res = levels.copy(), []
        dfs(self.__nodes[0], 0)
        return res
    def bridge_links(self):
        def dfs(u: Node, l: int):
            colors[u], levels[u], min_back = 1, l, l
            for v in self.__neighboring[u]:
                if not colors[v]:
                    b = dfs(v, l + 1)
                    if b > l:
                        res.append(Link(u, v))
                    else:
                        min_back = min(min_back, b)
                if colors[v] == 1 and levels[v] < min_back and levels[v] + 1 != l:
                    min_back = levels[v]
            colors[u] = 2
            return min_back
        levels, res = SortedKeysDict(*[(n, 0) for n in self.__nodes]), []
        colors = levels.copy()
        dfs(self.__nodes[0], 0)
        return res
    @staticmethod
    def __full(nodes: [Node], links: [Link]):
        return 2 * len(links) == len(nodes) * (len(nodes) - 1)
    def full(self):
        return self.__full(self.__nodes, self.__links)
    def get_shortest_path(self, u: Node, v: Node):
        if u not in self.__nodes or v not in self.__nodes:
            raise Exception('Unrecognized node(s)!')
        previous = SortedKeysDict(*[(n, None) for n in self.__nodes])
        queue, total = [u], SortedList()
        total.insert(u), previous.pop(u)
        while queue:
            n = queue.pop(0)
            if n == v:
                res, curr_node = [], n
                while curr_node != u:
                    res.insert(0, Link(previous[curr_node], curr_node))
                    curr_node = previous[curr_node]
                return res
            for m in filter(lambda x: x not in total, self.__neighboring[n]):
                queue.append(m), total.insert(m)
                previous[m] = n
    def shortest_path_length(self, u: Node, v: Node):
        if u not in self.__nodes or v not in self.__nodes:
            raise Exception('Unrecognized node(s).')
        distances = SortedKeysDict(*[(n, 0) for n in self.__nodes])
        queue, total = [u], SortedList()
        total.insert(u)
        while queue:
            n = queue.pop(0)
            for m in filter(lambda x: x not in total, self.__neighboring[n]):
                if m == v:
                    return 1 + distances[n]
                queue.append(m), total.insert(m)
                distances[m] = distances[n] + 1
        return float('inf')
    def euler_tour_exists(self):
        for n in self.__nodes:
            if self.__degrees[n] % 2:
                return False
        return self.connected()
    def euler_walk_exists(self, u: Node, v: Node):
        for n in self.__nodes:
            if self.degrees(n) % 2 and n not in [u, v]:
                return False
        return self.degrees(u) % 2 + self.degrees(v) % 2 == [2, 0][u == v] and self.connected()
    def euler_tour(self):
        if self.euler_tour_exists():
            u, v = self.links()[0][0], self.links()[0][1]
            self.disconnect(u, v)
            res = self.euler_walk(u, v)
            self.connect(u, v)
            return res
        return False
    def euler_walk(self, u: Node, v: Node):
        def dfs(x, stack):
            for y in self.neighboring(x):
                if Link(x, y) not in result + stack:
                    if y == n:
                        stack.append(Link(x, y))
                        for j in range(len(stack)):
                            result.insert(i + j, stack[j])
                        return
                    dfs(y, stack + [Link(x, y)])
        if u in self.__nodes and v in self.__nodes:
            if self.euler_walk_exists(u, v):
                result = self.get_shortest_path(u, v)
                for i, l in enumerate(result):
                    n = l[0]
                    dfs(n, [])
                return result
            return False
        raise Exception('Unrecognized nodes!')
    def cliques(self, k: int):
        from itertools import permutations
        k = abs(k)
        if not k:
            return [[]]
        if k > len(self.__nodes):
            return []
        if k == 1:
            return list(map(list, self.__nodes))
        if k == len(self.__nodes):
            return [[], [self.__nodes]][self.full()]
        result = SortedList()
        for p in permutations(self.__nodes, k):
            can = True
            for i, _n in enumerate(p):
                for j in range(i + 1, len(p)):
                    if Link(_n, p[j]) not in self.__links:
                        can = False
                        break
                if not can:
                    break
            if can:
                exists = False
                for clique in result:
                    if all(_n in clique for _n in p):
                        exists = True
                        break
                if not exists:
                    result.insert(list(p))
        return result
    def chromaticNumberNodes(self):
        def helper(nodes: [Node] = None, curr=0, so_far: SortedList = None, _except: [Node] = None):
            if len(self.__links) > 30:
                nodes = sorted(self.__nodes, key=lambda _n: self.__degrees[_n])
                colors, total, queue, k, n = SortedKeysDict((nodes[0], 0)), SortedList(), [nodes[0]], 1, len(nodes)
                total.insert(nodes[0])
                while k < n and queue:
                    u = queue.pop(0)
                    for v in filter(lambda x: x not in total, self.__neighboring[u]):
                        total.insert(v), queue.append(v)
                        cols, k = [colors[_n] for _n in filter(lambda x: x in total, self.__neighboring[v])], k + 1
                        for i in range(len(cols) + 1):
                            if i not in cols:
                                colors[v] = i
                                break
                return max(colors.values())
            if so_far is None:
                so_far = SortedList()
            if _except is None:
                _except = []
            if nodes is None:
                nodes = sorted(self.__nodes, key=lambda _n: self.__degrees[_n])
            if not nodes:
                return curr
            if self.__full(nodes, [l for l in self.__links if l[0] in nodes and l[1] in nodes]):
                return len(nodes) + curr
            curr_degrees = SortedKeysDict(*[(n, 0) for n in nodes])
            for u in nodes:
                for v in nodes:
                    if Link(u, v) in self.__links:
                        curr_degrees[u] += 1
            nodes = [nodes[0]] + sorted([_n for _n in nodes if _n != nodes[0]], key=lambda _n: curr_degrees[_n])
            so_far.insert(nodes[0])
            rest = [_n for _n in nodes if _n not in self.__neighboring[nodes[0]] and _n != nodes[0] and _n not in _except]
            if not rest:
                _res = helper([n for n in nodes if n not in so_far], curr + 1, so_far)
                so_far.pop()
                return _res
            res = len(self.__nodes)
            for u in rest:
                _res = helper([u] + [_n for _n in nodes if _n not in (nodes[0], u)], curr, so_far, _except + [_n for _n in nodes if Link(nodes[0], _n) in self.__links or Link(u, _n) in self.__links])
                if u in so_far:
                    so_far.remove(u)
                res = min(res, _res)
            return res
        return helper()
    def chromaticNumberLinks(self):
        if not self.__links:
            return 0
        res_graph = UndirectedGraph(*[Node(l) for l in self.__links])
        for n in res_graph.nodes():
            for m in res_graph.nodes():
                if m != n and (n.value()[0] in m.value() or n.value()[1] in m.value()):
                    res_graph.connect(n, m)
        return res_graph.chromaticNumberNodes()
    def pathWithLength(self, u: Node, v: Node, length: int):
        def dfs(x: Node, l: int, stack: [Link]):
            if not l:
                return (False, stack)[x == v]
            for y in filter(lambda _x: Link(x, _x) not in stack, self.__neighboring[x]):
                res = dfs(y, l - 1, stack + [Link(x, y)])
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
        if abs(length) < 3:
            raise False
        for l in self.__links:
            u, v = l
            self.disconnect(u, v)
            res = self.pathWithLength(v, u, abs(length) - 1)
            self.connect(u, v)
            if res:
                return [l] + res
        return False
    def vertexCover(self):
        def helper(curr):
            if not self.nodes():
                return curr
            result = [*self.nodes()]
            for u in self.nodes():
                neighbors = self.neighboring(u)
                self.remove(u)
                res = helper(curr + [u])
                self.add(u, *neighbors)
                if len(res) < len(result):
                    result = res
            return result
        return helper([])
    def dominatingSet(self):
        def helper(curr):
            if not self.nodes():
                return curr
            result = [*self.nodes()]
            for u in self.nodes():
                neighbors = self.neighboring(u)
                rest = SortedKeysDict(*[(n, self.neighboring(n)) for n in neighbors])
                self.remove(u, *neighbors)
                res = helper(curr + [u])
                for n in neighbors:
                    self.add(n, *rest[n])
                self.add(u, *neighbors)
                if len(res) < len(result):
                    result = res
            return result
        return helper([])
    def hamiltonTourExists(self):
        def dfs(nodes: [Node], links: [Link], can_continue_from: [Node] = None, can_end_in: [Node] = None, end_links: [Link] = None):
            if nodes is None:
                nodes = self.__nodes
            if links is None:
                links = self.__links
            if can_continue_from is None:
                can_continue_from = nodes
            if len(links) >= (len(nodes) - 2) * (len(nodes) - 1) // 2 + 2:
                return True
            curr_degrees = SortedKeysDict(*[(n, 0) for n in nodes])
            for u in nodes:
                for v in self.__neighboring[u]:
                    curr_degrees[u] += Link(u, v) in links
            if any(curr_degrees[n] <= 1 for n in nodes) or not self.__connected(nodes, links):
                return False
            if can_end_in is not None:
                can_continue = False
                for u in [n for n in nodes if n not in can_continue_from + can_end_in]:
                    for l in end_links:
                        if Link(u, l[0]) in links or Link(u, l[1]) in links:
                            can_continue = True
                            break
                    if can_continue:
                        break
                if not can_continue:
                    return False
            if len(nodes) > 2:
                if all(2 * curr_degrees[n] >= len(nodes) for n in nodes) or 2 * len(links) > (len(nodes) - 1) * (len(nodes) - 2) + 2:
                    return True
                res = True
                for u in nodes:
                    for v in [m for m in nodes if m != u and Link(m, u) not in links]:
                        if curr_degrees[u] + curr_degrees[v] < len(nodes):
                            res = False
                            break
                    if not res:
                        break
                if res:
                    return True
            for u in can_continue_from:
                if can_end_in is None:
                    can_end_in = [m for m in nodes if Link(m, u) in links]
                    if not can_end_in:
                        continue
                    end_links = [Link(u, m) for m in can_end_in]
                if dfs([_n for _n in nodes if _n != u], [l for l in links if u not in l], [_n for _n in nodes if _n in self.__neighboring[u]], can_end_in, end_links):
                    return True
            return False
        return dfs(self.__nodes, self.__links)
    def hamiltonWalkExists(self, u: Node, v: Node):
        if v in self.neighboring(u):
            return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        self.connect(u, v)
        res = self.hamiltonTourExists()
        self.disconnect(u, v)
        return res
    def hamiltonTour(self):
        if any(self.__degrees[n] <= 1 for n in self.__nodes) or not self.connected():
            return False
        u = self.__nodes[0]
        for v in self.neighboring(u):
            res = self.hamiltonWalk(u, v)
            if res:
                return res + [u]
        return False
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x: Node, y: Node, nodes: [Node] = None, links: [Link] = None, can_continue_from: [Node] = None, res_stack: [Node] = None):
            if nodes is None:
                nodes = self.__nodes
            if links is None:
                links = self.__links
            if x in nodes + [None] and y in nodes + [None]:
                if not self.__connected(nodes, links):
                    return False
                if res_stack is None:
                    res_stack = [x]
                curr_degrees = Dict(*[(n, 0) for n in nodes])
                for n in nodes:
                    for m in self.__neighboring[n]:
                        curr_degrees[n] += Link(n, m) in links
                if can_continue_from is None:
                    can_continue_from = sorted([n for n in nodes if Link(x, n) in links and n != y], key=lambda _x: curr_degrees[_x])
                leaf_nodes = list(filter(lambda _x: curr_degrees[_x] == 1, curr_degrees.keys()))
                if len(leaf_nodes) > (x in leaf_nodes + [None]) + (y in leaf_nodes + [None]) or len(leaf_nodes) == 1 and leaf_nodes[0] not in (x, y) and None not in (x, y):
                    return False
                if len(leaf_nodes) == 1 and x not in leaf_nodes + [None]:
                    y = leaf_nodes[0]
                if len(leaf_nodes) == 1 and y not in leaf_nodes + [None]:
                    x = leaf_nodes[0]
                if y is None:
                    if len(nodes) == 1:
                        return nodes
                elif len(nodes) == 2 and Link(x, y) in links:
                    return [x, y]
                for n in can_continue_from:
                    res = dfs(n, y, [_n for _n in nodes if _n != x], [l for l in links if x not in l], sorted([_n for _n in nodes if Link(_n, n) in links and _n not in [x, y]], key=lambda _x: self.__degrees[_x]), [n])
                    if res:
                        return res_stack + res
                return False
            raise ValueError('Unrecognized nodes!')
        return dfs(u, v)
    def isomorphic(self, other):
        if type(other) == UndirectedGraph:
            if len(self.__links) != len(other.__links):
                return False
            if len(self.__nodes) != len(other.__nodes):
                return False
            this_degrees, other_degrees = dict(), dict()
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
            this_nodes_degrees, other_nodes_degrees = {d: [] for d in this_degrees.keys()}, {d: [] for d in other_degrees.keys()}
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
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)):
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
        return item in self.__nodes  or item in self.__links
    def __add__(self, other):
        if isinstance(other, UndirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n)
            for l in other.links():
                if l not in res.links():
                    res.connect(l[0], l[1])
            return res
        raise TypeError(f'Can\'t add class UndirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, UndirectedGraph):
            for l in self.__links:
                if l not in other.links():
                    return False
            return len(self.__links) == len(other.links()) and self.__nodes == other.nodes()
        return False
    def __str__(self):
        return '({' + ', '.join(str(n) for n in self.__nodes) + '}, {' + ', '.join(str(l) for l in self.__links) + '})'
    def __repr__(self):
        return str(self)
class WeightedNodesUndirectedGraph(UndirectedGraph):
    def __init__(self, *pairs: (Node, float)):
        super().__init__(*[p[0] for p in pairs])
        self.__node_weights = Dict()
        for (n, w) in pairs:
            if n not in self.__node_weights:
                self.__node_weights[n] = w
    def node_weights(self, u: Node = None):
        return self.__node_weights[u] if u is not None else self.__node_weights
    def total_weight(self):
        return sum(self.__node_weights.values())
    def copy(self):
        res = WeightedNodesUndirectedGraph(*[(n, self.__node_weights[n]) for n in self.nodes()])
        for n in self.nodes():
            if self.degrees(n):
                res.connect(n, *self.neighboring(n))
        return res
    def add(self, n_w: (Node, float), *current_nodes: Node):
        super().add(n_w[0], *current_nodes)
        if n_w[0] not in self.nodes():
            self.__node_weights[n_w[0]] = n_w[1]
    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            if u in self.nodes():
                self.__node_weights.pop(u)
        super().remove(n, *rest)
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None:
                res_path = []
            for y in [y for y in self.neighboring(x) if Link(y, x) not in curr_path]:
                if curr_w + self.node_weights(y) + total_negative >= res_w and res_path:
                    continue
                if y == v:
                    if curr_w + self.node_weights(y) < res_w or not res_path:
                        res_path, res_w = curr_path + [Link(x, y)], curr_w + self.node_weights(y)
                curr = dfs(y, curr_path + [Link(x, y)], curr_w + self.node_weights(y), total_negative - self.node_weights(y) * (self.node_weights(y) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v):
                return dfs(u, [], 0, sum(self.node_weights(n) for n in self.nodes() if self.node_weights(n) < 0))
            return [], 0
        raise ValueError("Unrecognized node(s)!")
    def vertexCover(self):
        def helper(curr):
            if not self.nodes():
                return curr
            result, result_sum = [*self.nodes()], self.total_weight()
            for u in self.nodes():
                neighbors, w = self.neighboring(u), self.node_weights(u)
                self.remove(u)
                res = helper(curr + [u])
                self.add((u, w), *neighbors)
                res_sum = sum(self.node_weights(n) for n in res)
                if res_sum < result_sum:
                    result, result_sum = res, res_sum
            return result
        return helper([])
    def dominatingSet(self):
        def helper(curr):
            if not self.nodes():
                return curr
            result, result_sum = [*self.nodes()], self.total_weight()
            for u in self.nodes():
                neighbors, w = self.neighboring(u), self.node_weights(u)
                rest = SortedKeysDict(*[(n, (self.node_weights(n), self.neighboring(n))) for n in neighbors])
                self.remove(u, *neighbors)
                res = helper(curr + [u])
                for n in neighbors:
                    p = rest[n]
                    self.add((n, p[0]), *p[1])
                self.add((u, w), *neighbors)
                res_sum = sum(self.node_weights(n) for n in res)
                if res_sum < result_sum:
                    result, result_sum = res, res_sum
            return result
        return helper([])
    def isomorphic(self, other):
        if type(other) == WeightedNodesUndirectedGraph:
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
            for w in self.__node_weights.values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.__node_weights.values():
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
                        if (m in self.neighboring(n)) ^ (v in other.neighboring(u)) or self.__node_weights[n] != other.__node_weights[u]:
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __add__(self, other):
        if isinstance(other, WeightedNodesUndirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n in res.nodes():
                    res.__node_weights[n] += other.__node_weights[n]
                else:
                    res.add(n, other.__node_weights[n])
            for l in other.links():
                res.connect(l[0], l[1])
            return res
        if isinstance(other, UndirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n, 0)
            for l in other.links():
                res.connect(l[0], l[1])
            return res
        raise TypeError(f'Can\'t add class WeightedUndirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, WeightedNodesUndirectedGraph):
            for l in self.links():
                if l not in other.links():
                    return False
            return self.__node_weights == other.__node_weights and len(self.__links) == len(other.links())
        return super().__eq__(other)
    def __str__(self):
        return '({' + ', '.join(f'{str(n)} -> {self.__node_weights[n]}' for n in self.nodes()) + '}, ' + str(self.__links) + ')'
class WeightedLinksUndirectedGraph(UndirectedGraph):
    def __init__(self, *nodes: Node):
        super().__init__(*nodes)
        self.__link_weights = Dict()
    def link_weights(self, u_or_l: Node | Link = None, v: Node = None):
        if u_or_l is None:
            return self.__link_weights
        elif isinstance(u_or_l, Node):
            if v is None:
                return Dict(*[(n, self.__link_weights[Link(n, u_or_l)]) for n in self.neighboring(u_or_l)])
            if isinstance(v, Node):
                if v in self.nodes():
                    if Link(u_or_l, v) in self.links():
                        return self.__link_weights[Link(u_or_l, v)]
                    raise KeyError(f'No link between {u_or_l} and {v}!')
                raise ValueError('No such node exists in this graph!')
            raise TypeError('Node expected!')
        elif isinstance(u_or_l, Link):
            if u_or_l in self.links():
                return self.__link_weights[u_or_l]
            raise KeyError('Link not in graph!')
        raise TypeError('Node or link expected!')
    def total_weight(self):
        return sum(self.__link_weights.values())
    def add(self, u: Node, *nodes_weights: (Node, float)):
        if u not in self.nodes():
            res = []
            for v, w in nodes_weights:
                if v in self.nodes() and v not in [p[0] for p in res]:
                    res.append((v, w))
            for v, w in res:
                if not isinstance(w, (int, float)):
                    raise TypeError('Real numerical values expected!')
            super().add(u, *[p[0] for p in res])
            for v, w in res:
                self.__link_weights[Link(u, v)] = w
    def remove(self, n: Node, *rest: Node):
        for u in (n,) + rest:
            for v in self.__neighboring[u]:
                self.__link_weights.pop(Link(u, v))
        super().remove(n, *rest)
    def connect(self, u: Node, v_w: (Node, float), *nodes_weights: (Node, float)):
        if u in self.nodes():
            super().connect(u, *[p[0] for p in [v_w] + list(nodes_weights)])
            for v, w in [v_w] + list(nodes_weights):
                if Link(u, v) not in self.__link_weights:
                    self.__link_weights[Link(u, v)] = w
    def disconnect(self, u: Node, v: Node, *rest: Node):
        super().disconnect(u, v, *rest)
        for n in [v] + [*rest]:
            if Link(u, n) in [l for l in self.__link_weights.keys()]:
                self.__link_weights.pop(Link(u, n))
    def copy(self):
        res = WeightedLinksUndirectedGraph(*self.nodes())
        for u in self.nodes():
            for v in self.nodes():
                if Link(u, v) in self.links():
                    res.connect(u, (v, self.link_weights(u, v)))
        return res
    def euler_tour(self):
        if self.euler_tour_exists():
            u, v, w = self.links()[0][0], self.links()[0][1], self.link_weights(self.links()[0])
            self.disconnect(u, v)
            res = self.euler_walk(u, v)
            self.connect(u, (v, w))
            return res
        return False
    def minimal_spanning_tree(self):
        if self.tree():
            return self.links(), self.total_weight()
        if not self.connected():
            res = []
            for comp in self.connection_components():
                curr = WeightedLinksUndirectedGraph(*comp)
                for u in comp:
                    for v in self.neighboring(u):
                        curr.connect(u, (v, self.link_weights(u, v)))
                res.append(curr.minimal_spanning_tree())
            return res
        if not self.nodes():
            return [], 0
        res_links, total, bridge_links = [], [self.nodes()[0]], SortedList(lambda x: self.link_weights(x))
        for u in self.neighboring(self.nodes()[0]):
            bridge_links.insert(Link(self.nodes()[0], u))
        k, n = 1, len(self.nodes())
        while k < n:
            u, v, k = bridge_links[0][0], bridge_links[0][1], k + 1
            res_links.append(bridge_links.pop(0))
            if u in total:
                total.append(v)
                for _v in self.neighboring(v):
                    if _v in total:
                        bridge_links.remove(Link(v, _v))
                    else:
                        bridge_links.insert(Link(v, _v))
            else:
                total.append(u)
                for _u in self.neighboring(u):
                    if _u in total:
                        bridge_links.remove(Link(u, _u))
                    else:
                        bridge_links.insert(Link(u, _u))
        return res_links, sum(map(lambda x: self.__link_weights[x], res_links))
    def minimalPath(self, u: Node, v: Node):
        def dfs(x, curr_path, curr_w, total_negative, res_path=None, res_w=0):
            if res_path is None:
                res_path = []
            for y in [y for y in self.neighboring(x) if Link(y, x) not in curr_path]:
                if curr_w + self.link_weights(Link(y, x)) + total_negative >= res_w and res_path:
                    continue
                if y == v:
                    if curr_w + self.link_weights(Link(x, y)) < res_w or not res_path:
                        res_path, res_w = curr_path + [Link(x, y)], curr_w + self.link_weights(Link(x, y))
                curr = dfs(y, curr_path + [Link(x, y)], curr_w + self.link_weights(Link(x, y)), total_negative - self.link_weights(Link(x, y)) * (self.link_weights(Link(x, y)) < 0), res_path, res_w)
                if curr[1] < res_w or not res_path:
                    res_path, res_w = curr
            return res_path, res_w
        if u in self.nodes() and v in self.nodes():
            if self.reachable(u, v):
                return dfs(u, [], 0, sum(self.link_weights(l) for l in self.links() if self.link_weights(l) < 0))
            return [], 0
        raise ValueError('Unrecognized node(s)!')
    def hamiltonWalkExists(self, u: Node, v: Node):
        if v in self.neighboring(u):
            return True if all(n in (u, v) for n in self.nodes()) else self.hamiltonTourExists()
        self.connect(u, (v, 0))
        res = self.hamiltonTourExists()
        self.disconnect(u, v)
        return res
    def hamiltonTour(self):
        if any(self.degrees(n) <= 1 for n in self.nodes()) or not self.connected():
            return False
        u = self.nodes()[0]
        for v in self.neighboring(u):
            res = self.hamiltonWalk(u, v)
            if res:
                return res[0] + [u], res[1] + sum(self.link_weights(res[0][i], res[0][i + 1]) for i in range(len(res[0]) - 1)) + self.link_weights(res[0][-1], res[0][0])
        return False
    def hamiltonWalk(self, u: Node = None, v: Node = None):
        def dfs(x: Node, y: Node = None, nodes: [Node] = None, links: [Link] = None, can_continue_from: [Node] = None, res_stack: [Node] = None):
            if nodes is None:
                nodes = self.nodes()
            if links is None:
                links = self.links()
            if x in self.nodes() and y in nodes + [None]:
                if not self._UndirectedGraph__connected(nodes, links):
                    return False
                if res_stack is None:
                    res_stack = [x]
                curr_degrees = SortedKeysDict(*[(n, 0) for n in nodes])
                for n in nodes:
                    for m in self.neighboring(n):
                        curr_degrees[n] += Link(n, m) in links
                if can_continue_from is None:
                    can_continue_from = sorted([n for n in nodes if Link(x, n) in links and n != y], key=lambda _x: curr_degrees[_x])
                nodes_with_degree_1 = list(filter(lambda _x: curr_degrees[_x] == 1, curr_degrees.keys()))
                if len(nodes_with_degree_1) > (x in nodes_with_degree_1) + (y in nodes_with_degree_1 + [None]) or len(nodes_with_degree_1) == 1 and nodes_with_degree_1[0] not in (x, y):
                    return False
                if len(nodes_with_degree_1) == 1 and x not in nodes_with_degree_1:
                    y = nodes_with_degree_1[0]
                if y is None:
                    if len(nodes) == 1:
                        return nodes, 0
                elif len(nodes) == 2 and Link(x, y) in links:
                    return [x, y], self.__link_weights[Link(x, y)]
                for n in sorted(can_continue_from, key=lambda _x: self.__link_weights[Link(_x, _x)]):
                    res = dfs(n, y, [_n for _n in nodes if _n != x], [l for l in links if x not in l], [_n for _n in nodes if Link(_n, n) in links and _n not in [x, y]], [n])
                    if res:
                        return res_stack + res[0], res[1] + sum(self.link_weights(res[0][i], res[0][i + 1]) for i in range(len(res[0]) - 1))
                return False
            raise ValueError('Unrecognized nodes!')
        return dfs(u, v)
    def isomorphic(self, other):
        if type(other) == WeightedLinksUndirectedGraph:
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
            for w in self.__link_weights.values():
                if w in this_weights:
                    this_weights[w] += 1
                else:
                    this_weights[w] = 1
            for w in other.__node_weights.values():
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
                        if self.__link_weights[Link(n, m)] != other.__node_weights[Link(u, v)]:
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __add__(self, other):
        if isinstance(other, WeightedLinksUndirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n)
            for l in other.links():
                if l in res.links():
                    res.__link_weights[l] += other.__link_weights[l]
                else:
                    res.connect(l[0], (l[1], other.__link_weights[l]))
            return res
        if isinstance(other, UndirectedGraph):
            res = self.copy()
            for n in other.nodes():
                if n not in res.nodes():
                    res.add(n)
            for l in other.links():
                if l not in res.links():
                    res.connect(l[0], (l[1], 0))
            return res
        raise TypeError(f'Can\'t add class WeightedUndirectedGraph to class {type(other)}!')
    def __eq__(self, other):
        if isinstance(other, WeightedLinksUndirectedGraph):
            return self.nodes() == other.nodes() and self.__link_weights == other.__link_weights
        return super().__eq__(other)
    def __str__(self):
        return '({' + ', '.join(str(n) for n in self.nodes()) + '}, ' + str(self.__link_weights) + ')'
