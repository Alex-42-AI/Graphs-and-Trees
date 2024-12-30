"""
Module for implementing trees and working with them.
"""

from typing import Iterable

from collections import defaultdict

from itertools import permutations, product

from Graphs.src.implementation.directed_graph import DirectedGraph, WeightedNodesDirectedGraph

from Graphs.src.implementation.undirected_graph import Node, UndirectedGraph, WeightedNodesUndirectedGraph


def build_heap(ll: list[float]):
    """
    Args:
        ll: A list of real values.
    Sort list ll such, that it could represent a binary heap.
    """

    def heapify(low: int, high: int, ind: int, f=max):
        left, right = 2 * ind - low, 2 * ind - low + 1
        res = ind
        if left <= high and (el := ll[ind - 1]) != f(ll[left - 1], el):
            res = left
        if right <= high and (el := ll[res - 1]) != f(ll[right - 1], el):
            res = right
        if res != ind:
            ll[ind - 1], ll[res - 1] = ll[res - 1], ll[ind - 1]
            heapify(res - low - 1, high, res, f)

    for i in range((h := len(ll)) // 2, 0, -1):
        heapify(0, h, i)


class BinTree:
    """
    Class for implementing a binary tree.
    """

    def __init__(self, root=None, left=None, right=None) -> None:
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
        """
        Returns:
            Root.
        """
        return self.__root

    @property
    def left(self) -> "BinTree":
        """
        Returns:
            Left subtree.
        """
        return self.__left

    @property
    def right(self) -> "BinTree":
        """
        Returns:
            Left subtree.
        """
        return self.__right

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Leaves.
        """

        def dfs(tree):
            if not tree:
                return set()
            if not (tree.left or tree.right):
                return {tree.root}
            return dfs(tree.left).union(dfs(tree.right))

        return dfs(self)

    def copy(self) -> "BinTree":
        """
        Returns:
            An identical copy of the tree.
        """
        if self:
            return BinTree(self.root, self.left.copy(), self.right.copy())

    def rotate_left(self) -> "BinTree":
        """
        Left rotation of the tree makes it so, that the root is replaced by the current right subtree root.
        """
        self.__root, self.__left, self.__right = self.right.root, BinTree(self.root, self.left, self.right.left), self.right.right
        return self

    def rotate_right(self) -> "BinTree":
        """
        Right rotation of the tree makes it so, that the root is replaced by the current left subtree root.
        """
        self.__root, self.__left, self.__right = self.left.root, self.left.left, BinTree(self.root, self.left.right, self.right)
        return self

    def subtree(self, u: Node) -> "BinTree":
        """
        Args:
            u: A present node.
        Returns:
            The subtree rooted at node u.
        """

        def dfs(tree):
            if not tree:
                return
            if tree.root == u:
                return tree
            if (l := dfs(tree.left)) is not None:
                return l
            if (r := dfs(tree.right)) is not None:
                return r

        if not isinstance(u, Node):
            u = Node(u)
        return dfs(self)

    def tree(self) -> "Tree":
        """
        Returns:
            The Tree class version of the tree.
        """
        res = Tree(self.root, {self.root: {self.left.root, self.right.root}})
        if self.left:
            res.add_tree(self.left.tree())
        if self.right:
            res.add_tree(self.right.tree())
        return res

    def nodes_on_level(self, level: int) -> list[Node]:
        """
        Args:
            level: Distance of a node from the root.
        Returns:
            A list of all nodes on the given level.
        """

        def dfs(l, tree):
            if l < 0:
                raise ValueError("Non-negative value expected!")
            if not tree:
                return []
            if not l:
                return [tree.root]
            return dfs(l - 1, tree.left) + dfs(l - 1, tree.right)

        try:
            return dfs(int(level), self)
        except TypeError:
            raise TypeError("Integer expected!")

    def width(self) -> int:
        """
        Returns:
            The greatest number of nodes on any level.
        """
        res = len(self.nodes_on_level(self.height()))
        for i in range(self.height() - 1, -1, -1):
            if (curr := len(self.nodes_on_level(i))) <= res and res >= 2 ** (i - 1):
                return res
            res = max(res, curr)

    def height(self) -> int:
        """
        Returns:
            Tree height.
        """

        def dfs(tree):
            if not tree:
                return -1
            return 1 + max(dfs(tree.left), dfs(tree.right))

        return dfs(self)

    def count_nodes(self) -> int:
        def dfs(tree):
            if not tree:
                return 0
            if not (tree.left or tree.right):
                return 1
            return (tree.root.value is not None) + dfs(tree.left) + dfs(tree.right)

        return dfs(self)

    def code_in_morse(self, u: Node) -> str:
        """
        Args:
            u: A present node.
        Returns:
            The morse code of node u, based on its position in the tree. Left is '.' and right is '-'.
        """

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

        if not isinstance(u, Node):
            u = Node(u)
        return dfs(self)

    def encode(self, message: str) -> str:
        """
        Args:
            message: string type parameter.
        Each character in message needs to be a node value in the tree. Otherwise, instead of being encoded,
        it's passed as it is. Finally, each character encoding is concatenated and the result is returned.
        """
        res = ""
        for c in message:
            if isinstance(c, Node):
                c = c.value
            if c != " ":
                if Node(c) in self:
                    res += self.code_in_morse(c)
                else:
                    res += c
            res += "   "
        return res[:-2]

    def inverted(self) -> "BinTree":
        """
        Returns:
            An inverted copy of the tree.
        """
        return ~self.copy()

    def preorder(self) -> list[Node]:
        """
        List out the nodes in a preorder traversal type (root, left, right).
        """

        def dfs(tree, traversal):
            traversal += [tree.root]
            if tree.left:
                traversal = dfs(tree.left, traversal)
            if tree.right:
                traversal = dfs(tree.right, traversal)
            return traversal

        return dfs(self, [])

    def in_order(self) -> list[Node]:
        """
        List out the nodes in an in-order traversal type (left, root, right).
        """

        def dfs(tree, traversal):
            if tree.left:
                traversal = dfs(tree.left, traversal)
            traversal += [tree.root]
            if tree.right:
                traversal = dfs(tree.right, traversal)
            return traversal

        return dfs(self, [])

    def post_order(self) -> list[Node]:
        """
        List out the nodes in a post-order traversal type (left, right, root).
        """

        def dfs(tree, traversal):
            if tree.left:
                traversal = dfs(tree.left, traversal)
            if tree.right:
                traversal = dfs(tree.right, traversal)
            return traversal + [tree.root]

        return dfs(self, [])

    def traverse(self, traversal_type: str = "in-order") -> list[Node]:
        """
        Args:
            traversal_type: in-order, preorder, or postorder.
        Traverse the tree nodes in any traversal type (in-order by default).
        """
        if traversal_type.lower() == "preorder":
            return self.preorder()
        if traversal_type.lower() == "in-order":
            return self.in_order()
        if traversal_type.lower() == "post-order":
            return self.post_order()
        raise ValueError(f"Traversal type {traversal_type} is not supported!")

    def __invert__(self) -> "BinTree":
        """
        Invert the tree left to right.
        """

        def dfs(tree):
            if not tree:
                return
            dfs(tree.left), dfs(tree.right)
            tree.__left, tree.__right = tree.right, tree.left

        dfs(self)
        return self

    def __contains__(self, u: Node) -> bool:
        """
        Args:
            u: A Node object.
        Returns:
            Whether u is a node, present in the tree.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if self.root == u:
            return True
        if self.left and u in self.left:
            return True
        return self.right and u in self.right

    def __bool__(self):
        """
        Returns:
            Whether the tree is not empty (if it has a root value and has left or right subtrees).
        """
        return self.root.value is not None or bool(self.left) or bool(self.right)

    def __eq__(self, other: "BinTree") -> bool:
        """
        Args:
            other: a BinTree object.
        Returns:
            Whether both trees are equal.
        """
        if type(other) == BinTree:
            return (self.root, self.left, self.right) == (other.root, other.left, other.right)
        return False

    def __str__(self) -> str:
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

    def __repr__(self) -> str:
        if self:
            return f"BinTree({self.root}, {self.left}, {self.right})"
        return "None"


def binary_heap(l: list[float]) -> BinTree:
    """
    Args:
        l: A list of real values.
    Returns:
        A binary heap of list l.
    """
    build_heap(l)

    def helper(curr_root, rest, i=1):
        left = helper(rest[0], rest[(2 ** i):], i + 1) if rest else None
        right = helper(rest[1], rest[2 * 2 ** i:], i + 1) if rest[1:] else None
        res = BinTree(curr_root, left, right)
        return res

    return BinTree(helper(l[0], l[1:]))


def print_zig_zag(b_t: BinTree):
    """
    Args:
        b_t: BinTree
    Print the nodes of b_t zigzag.
    """

    def bfs(from_left: bool, *trees: BinTree):
        new = []
        if from_left:
            for t in trees:
                if t.left and (t.left.left is not None or t.left.right is not None):
                    new.insert(0, t.left), print(t.left.root, end=" ")
                if t.right and (t.right.left is not None or t.right.right is not None):
                    new.insert(0, t.right), print(t.right.root, end=" ")
        else:
            for t in trees:
                if t.right and (t.right.left is not None or t.right.right is not None):
                    new.insert(0, t.right), print(t.right.root, end=" ")
                if t.left and (t.left.left is not None or t.left.right is not None):
                    new.insert(0, t.left), print(t.left.root, end=" ")
        if not new:
            return
        print(), bfs(not from_left, *new)

    print(b_t.root), bfs(True, b_t)


class Tree:
    """
    Class for implementing a tree with multiple descendants.
    """

    def __init__(self, root: Node, inheritance: dict[Node, Iterable[Node]] = {}) -> None:
        if not isinstance(root, Node):
            root = Node(root)
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
        """
        Returns:
            Root.
        """
        return self.__root

    @property
    def nodes(self) -> set[Node]:
        """
        Returns:
            Nodes.
        """
        return self.__nodes.copy()

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Leaves.
        """
        return self.__leaves.copy()

    def leaf(self, n) -> bool:
        """
        Args:
            n: A present node.
        Returns:
            Whether node n is a leaf.
        """
        if not isinstance(n, Node):
            n = Node(n)
        return n in self.leaves

    def height(self) -> int:
        """
        Returns:
            Tree height.
        """

        def helper(x):
            return 1 + max([0, *map(helper, self.descendants(x))])

        return helper(self.root)

    def parent(self, u: Node = None) -> dict[Node, Node] | Node:
        """
        Args:
            u: A present node or None.
        Returns:
            The parent node of node u if it's given, otherwise the parent of each node.
        """
        if u is None:
            return self.__parent.copy()
        if not isinstance(u, Node):
            u = Node(u)
        return self.__parent.get(u)

    def hierarchy(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None.
        Returns:
            Descendants of node u if it's given, otherwise the descendants of each node.
        """
        if u is None:
            return self.__hierarchy.copy()
        if not isinstance(u, Node):
            u = Node(u)
        return self.__hierarchy[u].copy()

    def descendants(self, u: Node) -> set[Node]:
        """
        Args:
            u: A present node.
        Returns:
            The descendants of node u.
        """
        if not isinstance(u, Node):
            u = Node(u)
        return self.hierarchy(u)

    def copy(self) -> "Tree":
        """
        Returns:
            An identical copy of the tree.
        """
        return Tree(self.root, {u: self.descendants(u) for u in self.nodes if u != self.root})

    def subtree(self, u: Node) -> "Tree":
        """
        Args:
            u: A present node.
        Returns:
            The subtree, rooted in node u.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        if u == self.root:
            return self
        queue, res = [u], Tree(u)
        while queue:
            for n in self.descendants(v := queue.pop(0)):
                res.add(v, n), queue.append(n)
        return res

    def undirected_graph(self) -> "UndirectedGraph":
        """
        Returns:
            The undirected graph version of the tree.
        """
        return UndirectedGraph(self.hierarchy())

    def directed_graph(self, from_root: bool = True) -> DirectedGraph:
        """
        Args:
            from_root: Boolean flag, answering to whether the links point to or from the root.
        Returns:
            A directed graph version of the tree.
        """
        if from_root:
            return DirectedGraph({k: ([], v) for k, v in self.hierarchy().items()})
        return DirectedGraph({k: (v, []) for k, v in self.hierarchy().items()})

    def add(self, curr: Node, u: Node, *rest: Node) -> "Tree":
        """
        Args:
            curr: A present node.
            u: A new node.
            rest: Other new nodes.
        Add new nodes as descendants of a given present node.
        """
        if not isinstance(curr, Node):
            curr = Node(curr)
        if curr not in self:
            raise KeyError("Unrecognized node")
        if self.leaf(curr): self.__leaves.remove(curr)
        for v in (u, *rest):
            if not isinstance(v, Node):
                v = Node(v)
            if v not in self:
                self.__nodes.add(v)
                self.__hierarchy[curr].add(v)
                self.__parent[v] = curr
                self.__leaves.add(v)
                self.__hierarchy[v] = set()
        return self

    def add_tree(self, tree: "Tree") -> "Tree":
        """
        Args:
            tree: A Tree object, the root of which is a present node.
        Add an entire tree, the root of which is an already present node. It expands the tree from there.
        """
        if isinstance(tree, BinTree):
            tree = tree.tree()
        if not isinstance(tree, Tree):
            raise TypeError("Tree expected!")
        if tree.root not in self:
            raise KeyError("Unrecognized node!")
        queue = [tree.root]
        while queue:
            if res := tree.descendants(u := queue.pop(0)) - self.nodes:
                self.add(u, *res)
                queue += list(res)
        return self

    def remove(self, u: Node, subtree: bool = False) -> "Tree":
        """
        Args:
            u: A present node.
            subtree: Boolean flag, answering to whether to remove the subtree, rooted in node u.
        Remove a node and make its descendants direct descendants of its parent node if subtree is False.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
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

    def move_node(self, u: Node, new_parent: Node, subtree: bool = False) -> "Tree":
        """
        Args:
            u: A present node.
            new_parent: A present node.
            subtree: A boolean flag, answering to whether the entire subtree, rooted in node u, should be moved.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            tmp = self.subtree(u) if subtree else None
            self.remove(u, subtree)
            self.add(new_parent, {u: self.weights(u)} if isinstance(self, WeightedTree) else u)
            if subtree:
                self.add_tree(tmp)
        return self

    def node_depth(self, u: Node) -> int:
        """
        Args:
            u: A present node.
        Returns:
            The distance (in links) from the root to a node u.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node")
        d = 0
        while u != self.root:
            u = self.parent(u)
            d += 1
        return d

    def path_to(self, u: Node) -> list[Node]:
        """
        Args:
            u: A present node.
        Returns:
            The path from the root to node u.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node")
        x, res = u, []
        while x != self.root:
            res = [x] + res
            x = self.parent(x)
        return [self.root] + res

    def vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimum set of nodes, that cover all links in the tree.
        """
        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimum set of nodes, that cover all nodes in the tree.
        """

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
        """
        Returns:
            A maximum set of nodes, none of which is a parent of any other.
        """

        def dfs(x: Node):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0].update(dp[y][1])
                dp[x][1].update(dp[y][0] if len(dp[y][0]) > len(dp[y][1]) else dp[y][1])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return dp[self.root][0] if len(dp[self.root][0]) > len(dp[self.root][1]) else dp[self.root][1]

    def isomorphic_bijection(self, other: "Tree") -> dict[Node, Node]:
        """
        Args:
            other: another Tree object
        Returns:
            An isomorphic function between the nodes of the tree and those of the given tree,
            if such exists, otherwise empty dictionary. Let f be such a bijection and u and v be nodes
            in the tree. f(u) and f(v) are nodes in the other tree and f(u) is parent of f(v) exactly
            when the same applies for u and v. For weighted trees, the weights are considered.
        """
        if isinstance(other, Tree):
            if len(self.nodes) != len(other.nodes) or len(self.leaves) != len(other.leaves) or len(self.descendants(self.root)) != len(other.descendants(other.root)):
                return {}
            this_nodes_descendants, other_nodes_descendants = defaultdict(set), defaultdict(set)
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].add(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].add(n)
            if any(len(this_nodes_descendants[d]) != len(other_nodes_descendants[d]) for d in this_nodes_descendants):
                return {}
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=len))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=len))
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

    def __bool__(self) -> bool:
        """
        Returns:
            Whether the tree has nodes.
        """
        return bool(self.nodes)

    def __contains__(self, u: Node) -> bool:
        """
        Args:
            u: A Node object.
        Returns:
            Whether u is a node in the tree.
        """
        if not isinstance(u, Node):
            u = Node(u)
        return u in self.nodes

    def __eq__(self, other: "Tree") -> bool:
        """
        Args:
            other: a Tree object.
        Returns:
            Whether both trees are equal.
        """
        if type(other) == Tree:
            return self.hierarchy == other.hierarchy
        return False

    def __str__(self) -> str:
        def helper(r, i=0, flags=()):
            res, total_descendants = str(r), len(self.descendants(r))
            line = "".join([" │"[not j % 4 and (flags + (True,))[j // 4]] for j in range(i)])
            for j, d in enumerate(self.descendants(r)):
                res += f"\n {line + "├└"[j + 1 == total_descendants]}──"
                res += helper(d, i + 4, flags + (j + 1 < total_descendants,))
            return res

        return helper(self.root)

    def __repr__(self) -> str:
        inheritance = self.hierarchy().copy()
        inheritance.pop(self.root)
        return f"Tree({self.root}, {inheritance})"


class WeightedTree(Tree):
    """
    Class for implementing a tree with node weights.
    """

    def __init__(self, root_and_weight: tuple[Node, float], inheritance: dict[Node, tuple[float, Iterable[Node]]] = {}) -> None:
        super().__init__(root_and_weight[0])
        self.__weights = dict([root_and_weight])
        for u, (w, desc) in inheritance.items():
            if u not in self:
                self.add(root_and_weight[0], {u: w})
            if desc:
                self.add(u, {v: inheritance[v][0] if v in inheritance else 0 for v in desc})

    def weights(self, u: Node = None) -> dict[Node, float] | float:
        """
        Args:
            u: A present node or None.
        Returns:
            The weight of node n or the dictionary with all node weights.
        """
        if u is None:
            return self.__weights.copy()
        if not isinstance(u, Node):
            u = Node(u)
        return self.__weights.get(u)

    def set_weight(self, u: Node, w: float) -> "WeightedTree":
        """
        Args:
            u: A present node.
            w: The new weight of node u.
        Set the weight of node u to w.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self.weights():
            try:
                self.__weights[u] = float(w)
            except TypeError:
                raise TypeError("Real value expected!")
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedTree":
        """
        Args:
            u: A present node.
            w: A real value.
        Increase the weight of node u by w.
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self.weights:
            try:
                self.set_weight(u, self.weights(u) + float(w))
            except TypeError:
                raise TypeError("Real value expected!")
        return self

    def copy(self) -> "WeightedTree":
        inheritance = {u: (self.weights(u), {v: self.weights(v) for v in self.descendants(u)}) for u in self.nodes}
        return WeightedTree((self.root, self.weights(self.root)), inheritance)

    def subtree(self, u: Node) -> "WeightedTree":
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        if u == self.root:
            return self
        queue, res = [u], WeightedTree((u, self.weights(u)))
        while queue:
            for n in self.descendants(v := queue.pop(0)):
                res.add(v, {n: self.weights(n)}), queue.append(n)
        return res

    def undirected_graph(self) -> "WeightedNodesUndirectedGraph":
        return WeightedNodesUndirectedGraph({n: (self.weights(n), self.descendants(n)) for n in self.nodes})

    def directed_graph(self, from_root: bool = True) -> WeightedNodesDirectedGraph:
        if from_root:
            return WeightedNodesDirectedGraph({k: (self.weights(k), ([], v)) for k, v in self.hierarchy().items()})
        return WeightedNodesDirectedGraph({k: (self.weights(k), (v, [])) for k, v in self.hierarchy().items()})

    def add(self, curr: Node, rest: dict[Node, float] = {}) -> "WeightedTree":
        if not isinstance(curr, Node):
            curr = Node(curr)
        if curr not in self:
            raise KeyError("Unrecognized node")
        for u, w in rest.items():
            if u not in self:
                self.set_weight(u, w)
        if rest:
            super().add(curr, *rest.keys())
        return self

    def add_tree(self, tree: Tree) -> "WeightedTree":
        super().add_tree(tree)
        queue = [tree.root]
        while queue:
            self.set_weight((u := queue.pop(0)), tree.weights(u) if isinstance(tree, WeightedTree) else 0)
            queue += self.descendants(u)
        return self

    def remove(self, u: Node, subtree: bool = False) -> "WeightedTree":
        if not isinstance(u, Node):
            u = Node(u)
        if subtree:
            for d in self.descendants(u):
                self.remove(d, True)
        self.__weights.pop(u)
        super().remove(u)
        return self

    def weighted_vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimum by sum of the weights set of nodes, that cover all links in the tree.
        """
        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimum by sum of the weights set of nodes, that cover all nodes in the tree.
        """

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
        """
        Returns:
            A set of nodes with a maximum possible sum of the weights, none of which is a parent of any other.
        """

        def dfs(x):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0].update(dp[y][1])
                dp[x][1].update(dp[y][0] if sum(map(self.weights, dp[y][0])) > sum(map(self.weights, dp[y][1])) else dp[y][1])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return root_val[0] if sum(map(self.weights, (root_val := dp[self.root])[0])) > sum(map(self.weights, root_val[1])) else root_val[1]

    def isomorphic_bijection(self, other: Tree) -> dict[Node, Node]:
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
            this_nodes_descendants = defaultdict(set)
            other_nodes_descendants = defaultdict(set)
            for n in self.nodes:
                this_nodes_descendants[len(self.descendants(n))].add(n)
            for n in other.nodes:
                other_nodes_descendants[len(self.descendants(n))].add(n)
            this_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=len))
            other_nodes_descendants = list(sorted(other_nodes_descendants.values(), key=len))
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
        return super().isomorphic_bijection(other)

    def __eq__(self, other: "WeightedTree") -> bool:
        if type(other) == WeightedTree:
            return (self.hierarchy, self.weights()) == (other.hierarchy, other.weights())
        return False

    def __str__(self) -> str:
        def helper(r, i=0, flags=()):
            res, total_descendants = f"{r}->{self.weights(r)}", len(self.descendants(r))
            line = "".join([" │"[not j % 4 and (flags + (True,))[j // 4]] for j in range(i + 1)])
            for j, d in enumerate(self.descendants(r)):
                res += f"\n {line + "├└"[j + 1 == total_descendants]}──"
                res += helper(d, i + 4, flags + (j + 1 < total_descendants,))
            return res

        return helper(self.root)

    def __repr__(self) -> str:
        inheritance = {k: (self.weights(k), v) for k, v in self.hierarchy().items()}
        inheritance.pop(self.root)
        return f"WeightedTree({(self.root, self.weights(self.root))}, {inheritance})"
