from collections import defaultdict

from itertools import permutations, product

from ..__init__ import Node, Iterable


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
    def root(self) -> Node:
        return self.__root

    @property
    def left(self) -> "BinTree":
        return self.__left

    @property
    def right(self) -> "BinTree":
        return self.__right

    def copy(self) -> "BinTree":
        if self:
            return BinTree(self.root, self.left.copy(), self.right.copy())

    def rotate_left(self) -> "BinTree":
        self.__root, self.__left, self.__right = self.right.root, BinTree(self.root, self.left, self.right.left), self.right.right
        return self

    def rotate_right(self) -> "BinTree":
        self.__root, self.__left, self.__right = self.left.root, self.left.left, BinTree(self.root, self.left.right, self.right)
        return self

    def subtree(self, u: Node) -> "BinTree":
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

    def nodes_on_level(self, level: int) -> set[Node]:
        def dfs(l, tree):
            if not tree:
                return set()
            if not l:
                return {tree.root}
            return dfs(l - 1, tree.left).union(dfs(l - 1, tree.right))

        return dfs(abs(level), self)

    def width(self) -> int:
        res = len(self.nodes_on_level(self.height()))
        for i in range(self.height() - 1, -1, -1):
            if (curr := len(self.nodes_on_level(i))) <= res and res >= 2 ** (i - 1):
                return res
            res = max(res, curr)

    def height(self) -> int:
        def dfs(tree):
            if not tree:
                return -1
            return 1 + max(dfs(tree.left), dfs(tree.right))

        return dfs(self)

    def leaves(self) -> set[Node]:
        def dfs(tree):
            if not tree:
                return set()
            if not (tree.left or tree.right):
                return {tree.root}
            return dfs(tree.left).union(dfs(tree.right))

        return dfs(self)

    def count_nodes(self) -> int:
        def dfs(tree):
            if not tree:
                return 0
            if not (tree.left or tree.right):
                return 1
            return (tree.root.value is not None) + dfs(tree.left) + dfs(tree.right)

        return dfs(self)

    def code_in_morse(self, x) -> str:
        def dfs(tree):
            if not tree:
                return
            if (l := tree.left) and l.root == u:
                return "."
            if res := dfs(l):
                return ". " + res
            if (r := tree.right) and r.root == u:
                return "-"
            if res := dfs(r):
                return "- " + res

        u = Node(x)
        return dfs(self)

    def encode(self, message: str) -> str:
        res = ""
        for c in message:
            if Node(c) in self:
                res += self.code_in_morse(c) + "   "
            else:
                res += c + "  "
        return res[:-2]

    def inverted(self) -> "BinTree":
        return ~self.copy()

    def traverse(self, traversal_type: str = "in-order") -> list[Node]:
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
            return preorder_print()
        elif traversal_type.lower() == "in-order":
            return in_order_print()
        elif traversal_type.lower() == "post-order":
            return post_order_print()
        else:
            raise ValueError("Traversal type " + str(traversal_type) + " is not supported!")

    def __invert__(self) -> "BinTree":
        def dfs(tree):
            if not tree:
                return
            dfs(tree.left), dfs(tree.right)
            tree.__left, tree.__right = tree.right, tree.left

        dfs(self)
        return self

    def __contains__(self, item):
        if self.root == item:
            return True
        if self.left and item in self.left:
            return True
        return self.right and item in self.right

    def __bool__(self):
        return self.root.value is not None or bool(self.left) or bool(self.right)

    def __eq__(self, other: "BinTree"):
        if type(other) == BinTree:
            return (self.root, self.left, self.right) == (other.root, other.left, other.right)
        return False

    def __str__(self):
        def helper(t, i=0, flags=()):
            res = str(t.root)
            if t.left or t.right:
                line = "".join([" │"[not j % 4 and (flags + (True,))[j // 4]] for j in range(i)])
                res += f"\n {line}├──"
                res += helper(t.left, i + 4, flags + (True,)) if t.left else "\\"
                res += f"\n {line}└──"
                res += helper(t.right, i + 4, flags + (False,)) if t.right else "\\"
            return res

        return helper(self)

    __repr__ = traverse


class Tree:
    def __init__(self, root: Node, inheritance: dict[Node, Iterable[Node]] = {}):
        self.__root = root
        self.__hierarchy, self.__parent = {root: set()}, {}
        self.__nodes, self.__leaves = {root}, {root}
        for u, desc in inheritance.items():
            if u not in self:
                self.add(root, u)
            if desc:
                self.add(u, *desc)

    @property
    def root(self) -> Node:
        return self.__root

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes

    @property
    def leaves(self) -> set[Node]:
        return self.__leaves

    def leaf(self, n) -> bool:
        return n in self.leaves

    def height(self) -> int:
        def helper(x: Node):
            return 1 + max([0, *map(helper, self.descendants(x))])

        return helper(self.root)

    def parent(self, u: Node = None) -> dict[Node, Node] | Node:
        return self.__parent if u is None else self.__parent[u]

    def hierarchy(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        return self.__hierarchy if u is None else self.__hierarchy[u]

    def descendants(self, n: Node) -> set[Node]:
        return self.hierarchy(n)

    def copy(self) -> "Tree":
        return Tree(self.root, {u: self.descendants(u) for u in self.nodes if u != self.root})

    def subtree(self, u: Node) -> "Tree":
        if u not in self:
            raise ValueError("Unrecognized node!")
        if u == self.root:
            return self
        queue, res = [u], Tree(u)
        while queue:
            for n in self.descendants(v := queue.pop(0)):
                res.add(v, n), queue.append(n)
        return res

    def add(self, curr: Node, u: Node, *rest: Node) -> "Tree":
        if curr not in self:
            raise Exception("Unrecognized node")
        if self.leaf(curr): self.__leaves.remove(curr)
        for v in [u] + [*rest]:
            if v not in self:
                self.__nodes.add(v)
                self.__hierarchy[curr].add(v)
                self.__parent[v] = curr
                self.__leaves.add(v)
                self.__hierarchy[v] = set()
        return self

    def add_tree(self, tree: "Tree") -> "Tree":
        if not isinstance(tree, Tree):
            raise TypeError("Tree expected!")
        if tree.root not in self:
            raise Exception("Unrecognized node!")
        queue = [tree.root]
        while queue:
            if res := list(filter(lambda x: x not in self, tree.descendants(u := queue.pop(0)))):
                self.add(u, *res)
                queue += res
        return self

    def remove(self, u: Node, subtree: bool = False) -> "Tree":
        if u not in self:
            raise ValueError("Unrecognized node!")
        if u == self.root:
            raise ValueError("Can't remove root!")
        if subtree:
            for d in self.descendants(u):
                self.remove(d, True)
        self.__nodes.remove(u), self.__parent.pop(u)
        v = self.parent(u)
        for n in self.descendants(u):
            self.__parent[n] = v
        self.__hierarchy[v] += self.hierarchy(u)
        if self.leaf(u):
            self.__leaves.remove(u)
            if not self.hierarchy(v):
                self.__leaves.add(v)
        self.__hierarchy.pop(u)
        return self

    def move_node(self, u: Node, at_new: Node, subtree: bool = False) -> "Tree":
        if u in self:
            tmp = self.subtree(u) if subtree else None
            self.remove(u, subtree)
            self.add(at_new, {u: self.weights(u)} if isinstance(self, WeightedTree) else u)
            if subtree:
                self.add_tree(tmp)
        return self

    def node_depth(self, u: Node) -> int:
        if u in self:
            d = 0
            while u != self.root:
                u = self.parent(u)
                d += 1
            return d
        raise ValueError("Unrecognized node")

    def path_to(self, u: Node) -> list[Node]:
        x, res = u, []
        while x != self.root:
            res = [x] + res
            x = self.parent(x)
        return [self.root] + res

    def vertex_cover(self) -> set[Node]:
        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        def dfs(r):
            if self.leaf(r):
                return
            only_leaves, min_no_root = True, None
            for d in self.descendants(r):
                if self.leaf(d):
                    dp[r][1].add(min_no_root := d)
                else:
                    only_leaves = False
            if only_leaves:
                return
            for d in self.descendants(r):
                if not self.leaf(d):
                    dfs(d)
                    dp[r][0].update(dp[d][0] if len(dp[d][0]) < len(dp[d][1]) else dp[d][1])
                    if min_no_root is None or len(dp[d][0]) < len(dp[min_no_root][0]):
                        min_no_root = d
            for d in self.descendants(r):
                dp[r][1].update(dp[d][1] if len(dp[d][1]) < len(dp[d][0]) and d != min_no_root else dp[d][0])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return root_val[0] if len((root_val := dp[self.root])[0]) <= len(root_val[1]) else root_val[1]

    def independent_set(self) -> set[Node]:
        def dfs(x: Node):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0].update(dp[y][1])
                dp[x][1].update(dp[y][0] if len(dp[y][0]) > len(dp[y][1]) else dp[y][1])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return dp[self.root][0] if len(dp[self.root][0]) > len(dp[self.root][1]) else dp[self.root][1]

    def isomorphicFunction(self, other: "Tree") -> dict[Node, Node]:
        if isinstance(other, Tree):
            if len(self.nodes) != len(other.nodes) or len(self.leaves) != len(other.leaves) or len(self.descendants(self.root)) != len(other.descendants(other.root)):
                return {}
            this_nodes_descendants, other_nodes_descendants = defaultdict(list), defaultdict(list)
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].append(n)
            if any(len(this_nodes_descendants[d]) != len(other_nodes_descendants[d]) for d in this_nodes_descendants):
                return {}
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x)))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            for possibility in product(*map(permutations, this_nodes_descendants)):
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_descendants, [])
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

    def __contains__(self, item: Node):
        return item in self.nodes

    def __eq__(self, other: "Tree"):
        if type(other) == Tree:
            return self.hierarchy == other.hierarchy
        return False

    def __str__(self):
        def helper(r, i=0, flags=()):
            res, total_descendants = str(r), len(self.descendants(r))
            line = "".join([" │"[not j % 4 and (flags + (True,))[j // 4]] for j in range(i)])
            for j, d in enumerate(self.descendants(r)):
                res += f"\n {line + "├└"[j + 1 == total_descendants]}──"
                res += helper(d, i + 4, flags + (j + 1 < total_descendants,))
            return res

        return helper(self.root)

    def __repr__(self):
        inheritance = self.hierarchy().copy()
        inheritance.pop(self.root)
        return f"Tree({self.root}, {inheritance})"


class WeightedTree(Tree):
    def __init__(self, root_and_weight: tuple[Node, float], inheritance: dict[Node, tuple[float, Iterable[Node]]] = {}):
        super().__init__(root_and_weight[0])
        self.__weights = dict([root_and_weight])
        for u, (w, desc) in inheritance.items():
            if u not in self:
                self.add(root_and_weight[0], {u: w})
            if desc:
                self.add(u, {v: inheritance[v][0] if v in inheritance else 0 for v in desc})

    def weights(self, u: Node = None) -> dict[Node, float] | float:
        return self.__weights if u is None else self.__weights.get(u)

    def set_weight(self, u: Node, w: float) -> "WeightedTree":
        if u in self:
            self.__weights[u] = w
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedTree":
        if u in self.weights:
            self.set_weight(u, self.weights(u) + w)
        return self

    def copy(self) -> "WeightedTree":
        inheritance = {u: (self.weights(u), {v: self.weights(v) for v in self.descendants(u)}) for u in self.nodes}
        return WeightedTree((self.root, self.weights(self.root)), inheritance)

    def subtree(self, u: Node) -> "WeightedTree":
        if u not in self:
            raise ValueError("Unrecognized node!")
        if u == self.root:
            return self
        queue, res = [u], WeightedTree((u, self.weights(u)))
        while queue:
            for n in self.descendants(v := queue.pop(0)):
                res.add(v, {n: self.weights(n)}), queue.append(n)
        return res

    def add(self, curr: Node, rest: dict[Node, float] = {}) -> "WeightedTree":
        if curr not in self:
            raise Exception("Unrecognized node")
        for u, w in rest.items():
            if u not in self:
                self.__weights[u] = w
        if rest:
            super().add(curr, *rest.keys())
        return self

    def add_tree(self, tree) -> "WeightedTree":
        super().add_tree(tree)
        queue = [tree.root]
        while queue:
            self.set_weight((u := queue.pop(0)), tree.weights(u) if isinstance(tree, WeightedTree) else 0)
            queue += self.descendants(u)
        return self

    def remove(self, u: Node, subtree: bool = False) -> "WeightedTree":
        if subtree:
            for d in self.descendants(u):
                self.remove(d, True)
        self.__weights.pop(u)
        super().remove(u)
        return self

    def weighted_vertex_cover(self) -> set[Node]:
        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        def dfs(r):
            if r in self.leaves:
                return
            only_leaves, min_no_root = True, None
            for d in self.descendants(r):
                if d in self.leaves:
                    dp[r][1].add(min_no_root := d)
                else:
                    only_leaves = False
            if only_leaves:
                return
            for d in self.descendants(r):
                if d not in self.leaves:
                    dfs(d)
                    dp[r][0].update(dp[d][0] if (d_weights_sum := sum(map(self.weights, dp[d][0]))) < sum(map(self.weights, dp[d][1])) else dp[d][1])
                    if min_no_root is None or d_weights_sum < sum(map(self.weights, dp[min_no_root][0])):
                        min_no_root = d
            for d in self.descendants(r):
                dp[r][1].update(dp[d][1] if sum(map(self.weights, dp[d][1])) < sum(map(self.weights, dp[d][0])) and d != min_no_root else dp[d][0])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return root_val[0] if sum(map(self.weights, (root_val := dp[self.root])[0])) <= sum(map(self.weights, root_val[1])) else root_val[1]

    def weighted_independent_set(self) -> set[Node]:
        def dfs(x):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0].update(dp[y][1])
                dp[x][1].update(dp[y][0] if sum(map(self.weights, dp[y][0])) > sum(map(self.weights, dp[y][1])) else dp[y][1])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return root_val[0] if sum(map(self.weights, (root_val := dp[self.root])[0])) > sum(map(self.weights, root_val[1])) else root_val[1]

    def isomorphicFunction(self, other: Tree) -> dict[Node, Node]:
        if isinstance(other, WeightedTree):
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
                flatten_self = sum(map(list, possibility), [])
                flatten_other = sum(other_nodes_descendants, [])
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

    def __eq__(self, other: "WeightedTree"):
        if type(other) == WeightedTree:
            return (self.hierarchy, self.weights()) == (other.hierarchy, other.weights())
        return False

    def __str__(self):
        def helper(r, i=0, flags=()):
            res, total_descendants = f"{r}->{self.weights(r)}", len(self.descendants(r))
            line = "".join([" │"[not j % 4 and (flags + (True,))[j // 4]] for j in range(i + 1)])
            for j, d in enumerate(self.descendants(r)):
                res += f"\n {line + "├└"[j + 1 == total_descendants]}──"
                res += helper(d, i + 4, flags + (j + 1 < total_descendants,))
            return res

        return helper(self.root)

    def __repr__(self):
        inheritance = {k: (self.weights(k), v) for k, v in self.hierarchy().items()}
        inheritance.pop(self.root)
        return f"WeightedNodesTree({(self.root, self.weights(self.root))}, {inheritance})"
