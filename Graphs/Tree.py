from Graphs.General import Node, SortedList


class BinTree:
    def __init__(self, root=None, left=None, right=None):
        self.__root = Node(root)
        if isinstance(root, Node):
            self.__root = root
        self.__left, self.__right = None, None
        if root is not None:
            self.__left, self.__right = BinTree(), BinTree()
        if isinstance(left, BinTree):
            self.__left = left
        elif left is not None:
            self.__left = BinTree(left)
        if isinstance(right, BinTree):
            self.__right = right
        elif right is not None:
            self.__right = BinTree(right)

    @property
    def root(self):
        return self.__root

    @property
    def left(self):
        return self.__left

    @property
    def right(self):
        return self.__right

    def copy(self):
        if not self:
            return
        return BinTree(self.root, self.left.copy(), self.right.copy())

    def subtree(self, u: Node):
        def dfs(tree):
            if not tree:
                return
            if tree.root == u:
                return tree
            l = dfs(tree.left)
            if l is not None:
                return l
            r = dfs(tree.right)
            if r is not None:
                return r

        return dfs(self)

    def nodes_on_level(self, level: int):
        def dfs(l, tree):
            if not tree:
                return []
            if not l:
                return [tree.root]
            return dfs(l - 1, tree.left) + dfs(l - 1, tree.right)

        return dfs(abs(level), self)

    @property
    def width(self):
        res = len(self.nodes_on_level(self.height))
        for i in range(self.height - 1, -1, -1):
            curr = len(self.nodes_on_level(i))
            if curr <= res and res >= 2 ** i:
                return res
            res = max(res, curr)

    @property
    def height(self):
        def dfs(tree):
            if not tree:
                return -1
            return 1 + max(dfs(tree.left), dfs(tree.right))

        return dfs(self)

    def count_leaves(self):
        def dfs(tree):
            if not tree:
                return 0
            if not (tree.left or tree.right):
                return 1
            return dfs(tree.left) + dfs(tree.right)

        return dfs(self)

    def count_nodes(self):
        def dfs(tree):
            if not tree:
                return 0
            if not (tree.left or tree.right):
                return 1
            return (tree.root.value is not None) + dfs(tree.left) + dfs(tree.right)

        return dfs(self)

    def code_in_morse(self, u: Node):
        def dfs(tree):
            if not tree:
                return
            if tree.left and tree.left.root == u:
                return "."
            res = dfs(tree.left)
            if res: return ". " + res
            if tree.right and tree.right.root == u:
                return "-"
            res = dfs(tree.right)
            if res:
                return "- " + res

        return dfs(self)

    def encode(self, message: str):
        res = ""
        for c in message.upper():
            if Node(c) in self:
                res += self.code_in_morse(Node(c)) + "   "
            else:
                res += c + "  "
        return res[:-2]

    def inverted(self):
        return ~self.copy()

    def __preorder_print(self):
        def dfs(tree, traversal):
            if tree:
                traversal += [tree.root]
                traversal = dfs(tree.left, traversal)
                traversal = dfs(tree.right, traversal)
            return traversal

        return dfs(self, [])

    def __in_order_print(self):
        def dfs(tree, traversal):
            if tree:
                traversal = dfs(tree.left, traversal)
                traversal += [tree.root]
                traversal = dfs(tree.right, traversal)
            return traversal

        return dfs(self, [])

    def __post_order_print(self):
        def dfs(tree, traversal):
            if tree:
                traversal = dfs(tree.left, traversal)
                traversal = dfs(tree.right, traversal)
                traversal += [tree.root]
            return traversal

        return dfs(self, [])

    def print(self, traversal_type: str = "in-order"):
        if traversal_type.lower() == "preorder":
            print(self.__preorder_print())
        elif traversal_type.lower() == "in-order":
            print(self.__in_order_print())
        elif traversal_type.lower() == "post-order":
            print(self.__post_order_print())
        else:
            raise ValueError("Traversal type " + str(traversal_type) + " is not supported!")

    def __invert__(self):
        def dfs(tree):
            if not tree:
                return
            dfs(tree.left), dfs(tree.right)
            tree.__left, tree.__right = tree.right, tree.left

        dfs(self)

    def __contains__(self, item):
        if self.root == item:
            return True
        if self.left and item in self.left:
            return True
        if self.right:
            return item in self.right

    def __bool__(self):
        return self.root.value is not None or bool(self.left) or bool(self.right)

    def __eq__(self, other):
        if isinstance(other, BinTree):
            return (self.root, self.left, self.right) == (other.root, other.left, other.right)
        return False

    def __str__(self):
        return str(self.__in_order_print())

    __repr__ = __str__


class Tree:
    def __init__(self, root: Node, inheritance: dict, f=lambda x: x):
        self.__root, self.__f = root, f
        sources, self.__links = SortedList(root, *inheritance.keys(), f=f), []
        self.__hierarchy = dict([(root, SortedList(f=f))])
        self.__nodes, self.__leaves = SortedList(*inheritance.keys(), f=f), SortedList(root, f=f)
        for u, desc in inheritance.items():
            self.__leaves.remove(root)
            if u not in self.nodes:
                self.__hierarchy[u] = SortedList(*desc, f=self.f())
                for v in desc:
                    if v not in self.nodes and v not in self.hierarchy(u):
                        sources.remove(v), self.__leaves.remove(u), self.__leaves.insert(v), self.__nodes.insert(v), self.__links.append((u, v))
        self.__hierarchy[root] = SortedList(*sources, f=self.f())
        for s in sources:
            self.__links.append((root, s)), self.__leaves.remove(s)

    @property
    def root(self):
        return self.__root

    @property
    def nodes(self):
        return self.__nodes

    @property
    def links(self):
        return self.__links

    @property
    def leaves(self):
        return self.__leaves

    def hierarchy(self, u: Node = None):
        return self.__hierarchy if u is None else self.__hierarchy[u]

    def descendants(self, n: Node):
        return self.hierarchy(n)

    def f(self, x=None):
        return self.__f if x is None else self.__f(x)

    def add_nodes_to(self, curr: Node, u: Node, *rest: Node):
        if curr not in self.nodes:
            raise Exception("Unrecognized node")
        if curr in self.leaves: self.__leaves.remove(curr)
        for v in [u] + [*rest]:
            if v not in self.nodes:
                self.__nodes.insert(v)
                self.__hierarchy[curr].insert(v)
                self.__links.append((curr, v))
                self.__leaves.insert(v)
                self.__hierarchy[v] = SortedList(f=self.f())
        return self

    def copy(self):
        return Tree(self.root, dict([(u, self.descendants(u)) for u in self.nodes if u != self.root]), self.f())

    def subtree(self, u: Node):
        if u not in self.nodes:
            raise ValueError("Unrecognized node!")
        queue, res = [u], Tree(u, dict(), self.f())
        while queue:
            v = queue.pop(0)
            for n in self.descendants(v):
                res.add_nodes_to(v, n), queue.append(n)
        return res

    def extend_tree_at(self, tree):
        if not isinstance(tree, Tree):
            raise TypeError("Tree expected!")
        if tree.root not in self.nodes:
            raise Exception("Unrecognized node!")
        queue = [tree.root]
        while queue:
            u = queue.pop(0)
            res = list(filter(lambda x: x not in self.nodes, tree.descendants(u)))
            if res:
                if res:
                    self.add_nodes_to(u, *res)
                queue += res
        return self

    def move_node(self, u: Node, at_new: Node):
        if u in self:
            tmp = self.subtree(u)
            self.remove(u)
            self.add_nodes_to(at_new, dict([(u, tmp.weights(u))]) if isinstance(tmp, WeightedNodesTree) else u)
            self.extend_tree_at(tmp)
        return self

    def remove(self, u: Node):
        if u not in self.nodes:
            raise ValueError("Unrecognized node!")
        if u == self.root:
            raise ValueError("Can't remove root!")
        self.__nodes.remove(u)
        v, l = self.parent(u), 0
        while l < len(self.links):
            if u in self.links[l]:
                self.__links.remove(self.links[l])
                l -= 1
            l += 1
        self.__hierarchy[v] += self.hierarchy(u)
        if u in self.leaves:
            self.__leaves.remove(u)
            if not self.hierarchy(v):
                self.__leaves.insert(v)
        self.__hierarchy.pop(u)
        return self

    def parent(self, u: Node):
        if u in self.nodes:
            for l in self.links:
                if u == l[1]:
                    return l[0]
        raise ValueError("Unrecognized node")

    def node_depth(self, u: Node):
        if u in self.nodes:
            d = 0
            while u != self.root:
                u = self.parent(u)
                d += 1
            return d
        raise ValueError("Unrecognized node")

    @property
    def height(self):
        def helper(x: Node):
            return 1 + max([0, *map(helper, self.descendants(x))])

        return helper(self.root)

    def path_to(self, u: Node):
        x, res = u, []
        while x != self.root:
            res = [x] + res
            x = self.parent(x)
        return res

    def vertex_cover(self):
        return list(filter(lambda x: x not in self.independent_set(), self.nodes))

    def dominating_set(self):
        dp = dict([(n, [[n], []]) for n in self.nodes])

        def dfs(r):
            if r in self.leaves:
                return
            only_leaves, min_no_root = True, None
            for d in self.descendants(r):
                if d in self.leaves:
                    dp[r][1].append(d)
                    min_no_root = d
                else:
                    only_leaves = False
            if only_leaves:
                return
            for d in self.descendants(r):
                if d not in self.leaves:
                    dfs(d)
                    dp[r][0] += dp[d][0] if len(dp[d][0]) < len(dp[d][1]) else dp[d][1]
                    if min_no_root is None or len(dp[d][0]) < len(dp[min_no_root][0]):
                        min_no_root = d
            for d in self.descendants(r):
                dp[r][1] += dp[d][1] if len(dp[d][1]) < len(dp[d][0]) and d != min_no_root else dp[d][0]

        dfs(self.root)
        return dp[self.root][0] if len(dp[self.root][0]) <= len(dp[self.root][1]) else dp[self.root][1]

    def independent_set(self):
        dp = dict([(n, [[n], []]) for n in self.nodes])

        def dfs(x: Node):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0] += dp[y][1]
                dp[x][1] += dp[y][0] if len(dp[y][0]) >= len(dp[y][1]) else dp[y][1]

        dfs(self.root)
        return dp[self.root][0] if len(dp[self.root][0]) >= len(dp[self.root][1]) else dp[self.root][1]

    def isomorphicFunction(self, other):
        if isinstance(other, Tree):
            if len(self.nodes) != len(other.nodes) or len(self.links) != len(other.links) or len(self.leaves) != len(other.leaves) or len(self.descendants(self.root)) != len(other.descendants(other.root)):
                return dict()
            this_hierarchies, other_hierarchies = dict(), dict()
            for n in self.nodes:
                descendants = len(self.descendants(n))
                if descendants not in this_hierarchies:
                    this_hierarchies[descendants] = 1
                else:
                    this_hierarchies[descendants] += 1
            for n in other.nodes:
                descendants = len(other.descendants(n))
                if descendants not in other_hierarchies:
                    other_hierarchies[descendants] = 1
                else:
                    other_hierarchies[descendants] += 1
            if this_hierarchies != other_hierarchies:
                return dict()
            this_nodes_descendants = {d: [] for d in this_hierarchies.keys()}
            other_nodes_descendants = {d: [] for d in other_hierarchies.keys()}
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].append(n)
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x)))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_descendants]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility):
                    map_dict += [*zip(group, other_nodes_descendants[i])]
                map_dict = dict(map_dict)
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.descendants(n)) ^ (v in other.descendants(u)) or (n in self.descendants(m)) ^ (u in other.descendants(v)):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return dict()
        return dict()

    def __bool__(self):
        return bool(self.nodes)

    def __call__(self, x):
        return self.f(x)

    def __contains__(self, item):
        return item in self.nodes or item in self.links

    def __eq__(self, other):
        if isinstance(other, Tree):
            for u in self.nodes:
                if u not in other.nodes:
                    return False
            if len(self.nodes) != len(other.nodes):
                return False
            for u in self.nodes:
                if len(self.hierarchy(u)) != len(other.hierarchy(u)):
                    return False
                for v in self.hierarchy(u):
                    if v not in other.hierarchy(u):
                        return False
            return True
        return False

    def __str__(self):
        return "\n".join(str(k) + " -- " + str(v) for k, v in filter(lambda p: p[1], self.hierarchy().items()))

    __repr__ = __str__


class WeightedNodesTree(Tree):
    def __init__(self, root_and_weight: (Node, float), inheritance: dict, f=lambda x: x):
        super().__init__(root_and_weight[0], dict([(k, v[1].keys()) for k, v in inheritance.items()]), f)
        self.__weights = dict([root_and_weight])
        for u, p in self.hierarchy():
            if u not in self.__weights:
                self.__weights[u] = p[0]
                for v_w in p[1]:
                    if v_w[0] not in self.__weights:
                        self.__weights[v_w[0]] = v_w[1]

    def weights(self, u: Node = None):
        return self.__weights if u is None else self.__weights.get(u)

    def set_weight(self, u: Node, w: float):
        if u in self.nodes:
            self.__weights[u] = w
        return self

    def copy(self):
        return WeightedNodesTree((self.root, self.weights(self.root)), dict([(u, (self.weights(u), dict([(v, self.weights(v)) for v in self.descendants(u)]))) for u in self.nodes]), self.f())

    def subtree(self, u: Node):
        queue, res = [u], WeightedNodesTree((u, self.weights(u)), dict(), self.f())
        while queue:
            v = queue.pop(0)
            for n in self.descendants(v):
                res.add_nodes_to(v, (n, self.weights(n))), queue.append(n)
        return res

    def add_nodes_to(self, u: Node, rest: dict = dict()):
        if u not in self.nodes:
            raise Exception("Unrecognized node")
        for v, w in rest.items():
            if v not in self.nodes:
                self.__weights[v] = w
        return super().add_nodes_to(u, *rest.keys()) if rest else self

    def extend_tree_at(self, tree):
        super().extend_tree_at(tree)
        queue = [tree.root]
        while queue:
            u = queue.pop(0)
            self.set_weight(u, tree.weights(u) * isinstance(tree, WeightedNodesTree))
            queue += self.descendants(u)
        return self

    def remove(self, u: Node):
        self.__weights.pop(u)
        return super().remove(u)

    def weighted_vertex_cover(self):
        dp = dict([(n, [[n], []]) for n in self.nodes])

        def dfs(u: Node):
            for v in self.descendants(u):
                dfs(v)
                dp[u][0] += dp[v][0] if sum(map(self.weights, dp[v][0])) <= sum(map(self.weights, dp[v][1])) else dp[v][1]
                dp[u][1] += dp[v][0]

        dfs(self.root)
        return dp[self.root][0] if sum(map(self.weights, dp[self.root][0])) <= sum(map(self.weights, dp[self.root][1])) else dp[self.root][1]

    def weighted_dominating_set(self):
        dp = dict([(n, [[n], []]) for n in self.nodes])

        def dfs(r):
            if r in self.leaves:
                return
            only_leaves, min_no_root = True, None
            for d in self.descendants(r):
                if d in self.leaves:
                    dp[r][1].append(d)
                    min_no_root = d
                else:
                    only_leaves = False
            if only_leaves:
                return
            for d in self.descendants(r):
                if d not in self.leaves:
                    dfs(d)
                    dp[r][0] += dp[d][0] if sum(map(self.weights, dp[d][0])) < sum(map(self.weights, dp[d][1])) else dp[d][1]
                    if min_no_root is None or sum(map(self.weights, dp[d][0])) < sum(map(self.weights, dp[min_no_root][0])):
                        min_no_root = d
            for d in self.descendants(r):
                dp[r][1] += dp[d][1] if sum(map(self.weights, dp[d][1])) < sum(map(self.weights, dp[d][0])) and d != min_no_root else dp[d][0]

        dfs(self.root)
        return dp[self.root][0] if sum(map(self.weights, dp[self.root][0])) <= sum(map(self.weights, dp[self.root][1])) else dp[self.root][1]

    def isomorphicFunction(self, other):
        if isinstance(other, WeightedNodesTree):
            if len(self.nodes) != len(other.nodes) or len(self.links) != len(other.links) or len(self.leaves) != len(other.leaves) or len(self.descendants(self.root)) != len(other.descendants(other.root)):
                return dict()
            this_hierarchies, other_hierarchies = dict(), dict()
            for n in self.nodes:
                descendants = len(self.descendants(n))
                if descendants not in this_hierarchies:
                    this_hierarchies[descendants] = 1
                else:
                    this_hierarchies[descendants] += 1
            for n in other.nodes:
                descendants = len(other.descendants(n))
                if descendants not in other_hierarchies:
                    other_hierarchies[descendants] = 1
                else:
                    other_hierarchies[descendants] += 1
            if this_hierarchies != other_hierarchies:
                return dict()
            this_nodes_descendants = {d: [] for d in this_hierarchies.keys()}
            other_nodes_descendants = {d: [] for d in other_hierarchies.keys()}
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].append(n)
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x)))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_descendants]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = []
                for i, group in enumerate(possibility):
                    map_dict += [*zip(group, other_nodes_descendants[i])]
                map_dict = dict(map_dict)
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if (m in self.descendants(n)) ^ (v in other.descendants(u)) or (n in self.descendants(m)) ^ (u in other.descendants(v)) or self.weights(n) != other.weights(u) or self.weights(m) != other.weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return dict()
        return super().isomorphicFunction(other)

    def __eq__(self, other):
        if isinstance(other, WeightedNodesTree):
            if self.weights() != other.weights():
                return False
        return super().__eq__(other)

    def __str__(self):
        return "\n".join(f"{k}, {self.weights(k)} -- {v}" for k, v in self.hierarchy().items())
