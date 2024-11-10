from functools import reduce

from itertools import permutations, product

from Graphs.General import Node, SortedList


class BinTree:
    def __init__(self, root=None, left=None, right=None):
        self.__root = root if isinstance(root, Node) else Node(root)
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
            if (l := dfs(tree.left)) is not None:
                return l
            if (r := dfs(tree.right)) is not None:
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
            if (curr := len(self.nodes_on_level(i))) <= res and res >= 2 ** i:
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
            if res := dfs(tree.left):
                return ". " + res
            if tree.right and tree.right.root == u:
                return "-"
            if res := dfs(tree.right):
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

    def print(self, traversal_type: str = "in-order"):
        def preorder_print():
            def dfs(tree, traversal):
                traversal += [tree.root]
                if tree.left:
                    traversal = dfs(tree.left, traversal)
                if tree.right:
                    traversal = dfs(tree.right, traversal)
                return traversal

            return dfs(self, [])

        def in_order_print():
            def dfs(tree, traversal):
                if tree.left:
                    traversal = dfs(tree.left, traversal)
                traversal += [tree.root]
                if tree.right:
                    traversal = dfs(tree.right, traversal)
                return traversal

            return dfs(self, [])

        def post_order_print():
            def dfs(tree, traversal):
                if tree.left:
                    traversal = dfs(tree.left, traversal)
                if tree.right:
                    traversal = dfs(tree.right, traversal)
                return traversal + [tree.root]

            return dfs(self, [])

        if traversal_type.lower() == "preorder":
            print(preorder_print())
        elif traversal_type.lower() == "in-order":
            print(in_order_print())
        elif traversal_type.lower() == "post-order":
            print(post_order_print())
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
        def helper(t, i=0):
            res = str(t.root)
            if t.left:
                res += "\n|" + "-" * i + "L" + "-" * 4
                res += helper(t.left, i + 4)
            if t.right:
                res += "\n|" + "-" * i + "R" + "-" * 4
                res += helper(t.right, i + 4)
            return res

        return helper(self)

    __repr__ = __str__


class Tree:
    def __init__(self, root: Node, inheritance: dict = {}, f=lambda x: x):
        self.__root, self.__f = root, f
        self.__hierarchy, self.__parents = {root: SortedList(f=f)}, {}
        self.__nodes, self.__leaves = SortedList(root, f=f), SortedList(root, f=f)
        for u, desc in inheritance.items():
            if u not in self:
                self.add(root, u)
            if desc:
                self.add(u, *desc)

    @property
    def root(self):
        return self.__root

    @property
    def nodes(self):
        return self.__nodes

    @property
    def leaves(self):
        return self.__leaves

    @property
    def height(self):
        def helper(x: Node):
            return 1 + max([0, *map(helper, self.descendants(x))])

        return helper(self.root)

    def parents(self, u: Node = None):
        return self.__parents if u is None else self.__parents[u]

    def hierarchy(self, u: Node = None):
        return self.__hierarchy if u is None else self.__hierarchy[u]

    def descendants(self, n: Node):
        return self.hierarchy(n)

    @property
    def f(self):
        return self.__f

    def copy(self):
        return Tree(self.root, {u: self.descendants(u) for u in self.nodes if u != self.root}, self.f)

    def subtree(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        queue, res = [u], Tree(u, {}, self.f)
        while queue:
            for n in self.descendants(v := queue.pop(0)):
                res.add(v, n), queue.append(n)
        return res

    def add(self, curr: Node, u: Node, *rest: Node):
        if curr not in self:
            raise Exception("Unrecognized node")
        if curr in self.leaves: self.__leaves.remove(curr)
        for v in [u] + [*rest]:
            if v not in self:
                self.__nodes.insert(v)
                self.__hierarchy[curr].insert(v)
                self.__parents[v] = curr
                self.__leaves.insert(v)
                self.__hierarchy[v] = SortedList(f=self.f)
        return self

    def add_tree(self, tree):
        if not isinstance(tree, Tree):
            raise TypeError("Tree expected!")
        if tree.root not in self:
            raise Exception("Unrecognized node!")
        queue = [tree.root]
        while queue:
            if res := list(filter(lambda x: x not in self, tree.descendants(u := queue.pop(0)))):
                if res:
                    self.add(u, *res)
                queue += res
        return self

    def remove(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        if u == self.root:
            raise ValueError("Can't remove root!")
        self.__nodes.remove(u), self.__parents.pop(u)
        v = self.parents(u)
        for n in self.descendants(u):
            self.__parents[n] = v
        self.__hierarchy[v] += self.hierarchy(u)
        if u in self.leaves:
            self.__leaves.remove(u)
            if not self.hierarchy(v):
                self.__leaves.insert(v)
        self.__hierarchy.pop(u)
        return self

    def remove_tree(self, u: Node):
        if u not in self:
            raise ValueError("Unrecognized node!")
        if u == self.root:
            raise ValueError("Can't remove root!")
        for v in self.descendants(u):
            self.remove_tree(v)
        self.remove(u)
        return self

    def move_node(self, u: Node, at_new: Node):
        if u in self:
            tmp = self.subtree(u)
            self.remove(u)
            self.add(at_new, {u: tmp.weights(u)} if isinstance(tmp, WeightedNodesTree) else u)
            self.add_tree(tmp)
        return self

    def node_depth(self, u: Node):
        if u in self:
            d = 0
            while u != self.root:
                u = self.parents(u)
                d += 1
            return d
        raise ValueError("Unrecognized node")

    def path_to(self, u: Node):
        x, res = u, []
        while x != self.root:
            res = [x] + res
            x = self.parents(x)
        return res

    def vertex_cover(self):
        return [list(filter(lambda x: x not in res, self.nodes)) for res in self.independent_set()]

    def dominating_set(self):
        dp = {n: [[n], []] for n in self.nodes}

        def dfs(r):
            if r in self.leaves:
                return
            only_leaves, min_no_root = True, None
            for d in self.descendants(r):
                if d in self.leaves:
                    dp[r][1].append(min_no_root := d)
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
        dp = {n: [[[n]], [[]]] for n in self.nodes}

        def dfs(x: Node):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0] = [c + d for c in dp[x][0] for d in dp[y][1]]
                optimal = dp[y][0] + dp[y][1]
                if (c := len(dp[y][0][0])) != (d := len(dp[y][1][0])):
                    optimal = dp[y][0] if c > d else dp[y][1]
                dp[x][1] = [c + d for c in dp[x][1] for d in optimal]

        dfs(self.root)
        if (a := len(dp[self.root][0][0])) == (b := len(dp[self.root][1][0])):
            return dp[self.root][0] + dp[self.root][1]
        return dp[self.root][0] if a > b else dp[self.root][1]

    def isomorphicFunction(self, other):
        if isinstance(other, Tree):
            if len(self.nodes) != len(other.nodes) or len(self.leaves) != len(other.leaves) or len(self.descendants(self.root)) != len(other.descendants(other.root)):
                return {}
            this_hierarchies, other_hierarchies = {}, {}
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
                return {}
            this_nodes_descendants = {d: [] for d in this_hierarchies}
            other_nodes_descendants = {d: [] for d in other_hierarchies}
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].append(n)
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x)))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            for possibility in product(*map(permutations, this_nodes_descendants)):
                flatten_self = reduce(lambda x, y: x + list(y), possibility, [])
                flatten_other = reduce(lambda x, y: x + y, other_nodes_descendants, [])
                map_dict = dict(zip(flatten_self, flatten_other))
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
            return {}
        return {}

    def __bool__(self):
        return bool(self.nodes)

    def __call__(self, x):
        return self.f(x)

    def __contains__(self, item):
        return item in self.nodes

    def __eq__(self, other):
        if isinstance(other, Tree):
            for u in self.nodes:
                if u not in other:
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
        def helper(r, i=4):
            res = str(r)
            for d in self.descendants(r):
                res += "\n|" + "-" * i
                res += helper(d, i + 4)
            return res

        return helper(self.root)

    __repr__ = __str__


class WeightedNodesTree(Tree):
    def __init__(self, root_and_weight: (Node, float), inheritance: dict = {}, f=lambda x: x):
        super().__init__(root_and_weight[0], {}, f)
        self.__weights = dict([root_and_weight])
        for u, (w, desc) in inheritance.items():
            if u not in self:
                self.add(root_and_weight[0], {u: w})
            if desc:
                self.add(u, {v: inheritance[v][0] if v in inheritance else 0 for v in desc})

    def weights(self, u: Node = None):
        return self.__weights if u is None else self.__weights.get(u)

    def set_weight(self, u: Node, w: float):
        if u in self:
            self.__weights[u] = w
        return self

    def copy(self):
        return WeightedNodesTree((self.root, self.weights(self.root)), {u: (self.weights(u),
                            {v: self.weights(v) for v in self.descendants(u)}) for u in self.nodes}, self.f)

    def subtree(self, u: Node):
        queue, res = [u], WeightedNodesTree((u, self.weights(u)), {}, self.f)
        while queue:
            for n in self.descendants(v := queue.pop(0)):
                res.add(v, {n: self.weights(n)}), queue.append(n)
        return res

    def add(self, curr: Node, rest: dict = {}):
        if curr not in self:
            raise Exception("Unrecognized node")
        for u, w in rest.items():
            if u not in self:
                self.__weights[u] = w
        return super().add(curr, *rest.keys()) if rest else self

    def add_tree(self, tree):
        super().add_tree(tree)
        queue = [tree.root]
        while queue:
            self.set_weight((u := queue.pop(0)), tree.weights(u) * isinstance(tree, WeightedNodesTree))
            queue += self.descendants(u)
        return self

    def remove(self, u: Node):
        self.__weights.pop(u)
        return super().remove(u)

    def weighted_vertex_cover(self):
        dp = {n: [[[n]], [[]]] for n in self.nodes}

        def dfs(x: Node):
            for y in self.descendants(x):
                dfs(y)
                optimal = dp[y][0] + dp[y][1]
                if (c := sum(map(self.weights, dp[y][0][0]))) != (d := sum(map(self.weights, dp[y][1][0]))):
                    optimal = dp[y][0] if c < d else dp[y][1]
                dp[x][0] = [c + d for c in dp[x][0] for d in optimal]
                dp[x][1] = [c + d for c in dp[x][1] for d in dp[y][0]]

        dfs(self.root)
        if (a := sum(map(self.weights, dp[self.root][0][0]))) == (b := sum(map(self.weights, dp[self.root][1][0]))):
            return dp[self.root][0] + dp[self.root][1]
        return dp[self.root][0] if a < b else dp[self.root][1]

    def weighted_dominating_set(self):
        dp = {n: [[n], []] for n in self.nodes}

        def dfs(r):
            if r in self.leaves:
                return
            only_leaves, min_no_root = True, None
            for d in self.descendants(r):
                if d in self.leaves:
                    dp[r][1].append(min_no_root := d)
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
            if len(self.nodes) != len(other.nodes) or len(self.leaves) != len(other.leaves) or len(self.descendants(self.root)) != len(other.descendants(other.root)):
                return {}
            this_hierarchies, other_hierarchies = {}, {}
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
                return {}
            this_nodes_descendants = {d: [] for d in this_hierarchies}
            other_nodes_descendants = {d: [] for d in other_hierarchies}
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].append(n)
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x)))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            for possibility in product(*map(permutations, this_nodes_descendants)):
                flatten_self = reduce(lambda x, y: x + list(y), possibility, [])
                flatten_other = reduce(lambda x, y: x + y, other_nodes_descendants, [])
                map_dict = dict(zip(flatten_self, flatten_other))
                possible = True
                for n, u in map_dict.items():
                    if self.weights(n) != other.weights(u):
                        possible = False
                        break
                    for m, v in map_dict.items():
                        if (m in self.descendants(n)) ^ (v in other.descendants(u)) or (n in self.descendants(m)) ^ (u in other.descendants(v)) or self.weights(m) != other.weights(v):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return map_dict
            return {}
        return super().isomorphicFunction(other)

    def __eq__(self, other):
        if isinstance(other, WeightedNodesTree):
            if self.weights() != other.weights():
                return False
        return super().__eq__(other)

    def __str__(self):
        def helper(r, i=4):
            res = f"{r}->{self.weights(r)}"
            for d in self.descendants(r):
                res += "\n|" + "-" * i
                res += helper(d, i + 4)
            return res

        return helper(self.root)
