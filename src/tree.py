"""
Module for implementing trees and functions for working with them
"""

__all__ = ["BinTree", "print_zig_zag", "build_heap", "binary_heap", "Tree", "WeightedTree"]

from typing import Callable

from collections import defaultdict

from itertools import permutations, product

from .directed_graph import DirectedGraph, WeightedNodesDirectedGraph

from .undirected_graph import Node, UndirectedGraph, WeightedNodesUndirectedGraph, Iterable, reduce


def build_heap(ll: list[float], f: Callable = max):
    """
    Args:
        ll: A list of real values
        f: Optimizing function, max by default
    Sort list ll such, that it could represent a binary heap
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
        heapify(0, h, i, f)


def isomorphic_bijection(tree0: "Tree", tree1: "Tree") -> dict[Node, Node]:
    if not isinstance(tree1, Tree):
        return {}
    weights = isinstance(tree0, WeightedTree) and isinstance(tree1, WeightedTree)
    if weights:
        this_weights, other_weights = defaultdict(int), defaultdict(int)
        for w in tree0.weights().values():
            this_weights[w] += 1
        for w in tree1.weights().values():
            other_weights[w] += 1
        if this_weights != other_weights:
            return {}
    elif len(tree0.nodes) != len(tree1.nodes):
        return {}
    if len(tree0.nodes) != len(tree1.nodes) or len(tree0.leaves) != len(tree1.leaves) or len(
            tree0.descendants(tree0.root)) != len(tree1.descendants(tree1.root)):
        return {}
    this_nodes_descendants, other_nodes_descendants = defaultdict(set), defaultdict(set)
    for n in tree0.nodes:
        this_nodes_descendants[len(tree0.descendants(n))].add(n)
    for n in tree1.nodes:
        other_nodes_descendants[len(tree1.descendants(n))].add(n)
    if any(len(this_nodes_descendants[d]) != len(other_nodes_descendants[d]) for d in this_nodes_descendants):
        return {}
    this_nodes_descendants = list(sorted(map(list, this_nodes_descendants.values()), key=len))
    other_nodes_descendants = list(sorted(map(list, other_nodes_descendants.values()), key=len))
    for possibility in product(*map(permutations, this_nodes_descendants)):
        flatten_self = sum(map(list, possibility), [])
        flatten_other = sum(other_nodes_descendants, [])
        map_dict = dict(zip(flatten_self, flatten_other))
        possible = True
        for n, u in map_dict.items():
            for m, v in map_dict.items():
                if weights and tree0.weights(n) != tree1.weights(u):
                    possible = False
                    break
                if (m in tree0.descendants(n)) ^ (v in tree1.descendants(u)) or (n in tree0.descendants(m)) ^ (
                        u in tree1.descendants(v)) or weights and tree0.weights(m) != tree1.weights(v):
                    possible = False
                    break
            if not possible:
                break
        if possible:
            return map_dict
    return {}


def compare(tree0: "Tree", tree1: "Tree") -> bool:
    if type(tree0) != type(tree1):
        return False
    if isinstance(tree0, WeightedTree) and tree0.weights() != tree1.weights():
        return False
    return tree0.hierarchy() == tree1.hierarchy()


def string(tree: "Tree") -> str:
    def helper(r, f, i=0, flags=()):
        res, total_descendants = f(r), len(tree.descendants(r))
        line = "".join([" │"[not j % 4 and (flags + (True,))[j // 4]] for j in range(i)])
        for j, d in enumerate(tree.descendants(r)):
            res += f"\n {line + "├└"[j + 1 == total_descendants]}──"
            res += helper(d, f, i + 4, flags + (j + 1 < total_descendants,))
        return res

    return helper(tree.root, lambda x: f"{x}->{tree.weights(x)}" if isinstance(tree, WeightedTree) else str(x))


class BinTree:
    """
    Class for implementing a binary tree
    """

    def __init__(self, root=None, left=None, right=None) -> None:
        """
        Args:
            root: The root of the binary tree
            left: The left subtree of the binary tree
            right: The right subtree of the binary tree
        """
        self.__root = root if isinstance(root, Node) else Node(root)
        self.__left, self.__right = None, None
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
            Tree root
        """
        return self.__root

    @property
    def left(self) -> "BinTree":
        """
        Returns:
            Left subtree
        """
        return self.__left

    @property
    def right(self) -> "BinTree":
        """
        Returns:
            Right subtree
        """
        return self.__right

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Tree leaves
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
            An identical copy of the tree
        """
        return BinTree(self.root, self.left if self.left is None else self.left.copy(),
                       self.right if self.right is None else self.right.copy())

    def rotate_left(self) -> "BinTree":
        """
        Left rotation of the tree makes it so, that the root is replaced by the current right subtree root
        """
        self.__root, self.__left, self.__right = self.right.root, BinTree(self.root, self.left,
                                                                          self.right.left), self.right.right
        return self

    def rotate_right(self) -> "BinTree":
        """
        Right rotation of the tree makes it so, that the root is replaced by the current left subtree root
        """
        self.__root, self.__left, self.__right = self.left.root, self.left.left, BinTree(self.root, self.left.right,
                                                                                         self.right)
        return self

    def subtree(self, u: Node) -> "BinTree":
        """
        Args:
            u: A present node
        Returns:
            The subtree rooted at node u
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
        res = dfs(self)
        if res is None:
            raise KeyError("Unrecognized node!")
        return res

    def tree(self) -> "Tree":
        """
        Returns:
            The Tree class version of the tree
        """
        res = Tree(self.root)
        if self.left:
            res.add(self.root, self.left.root)
            res.add_tree(self.left.tree())
        if self.right:
            res.add(self.root, self.right.root)
            res.add_tree(self.right.tree())
        return res

    def nodes_on_level(self, level: int) -> set[Node]:
        """
        Args:
            level: Distance of a node from the root
        Returns:
            A list of all nodes on the given level
        """

        def dfs(l, tree):
            if l < 0 or not tree:
                return []
            if not l:
                return [tree.root]
            return dfs(l - 1, tree.left) + dfs(l - 1, tree.right)

        try:
            return set(dfs(int(level), self))
        except TypeError:
            raise TypeError("Integer expected!")

    def width(self) -> int:
        """
        Returns:
            The greatest number of nodes on any level
        """
        res = len(self.nodes_on_level(self.height()))
        for i in range(self.height() - 1, -1, -1):
            if (curr := len(self.nodes_on_level(i))) <= res and res >= 2 ** (i - 1):
                return res
            res = max(res, curr)
        return 1

    def height(self) -> int:
        """
        Returns:
            Tree height
        """

        def dfs(tree):
            if not tree:
                return -1
            return 1 + max(dfs(tree.left), dfs(tree.right))

        return dfs(self)

    def count_nodes(self) -> int:
        """
        Returns:
            The number of nodes
        """

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
            u: A present node
        Returns:
            The morse code of node u, based on its position in the tree. Left is '.' and right is '-'
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
        res = dfs(self)
        if res is None:
            raise KeyError("Unrecognized node!")
        return res

    def encode(self, message: str) -> str:
        """
        Args:
            message: string type parameter
        Each character in message needs to be a node value in the tree. Otherwise, instead of being encoded, it's passed as it is. Finally, each character encoding is concatenated and the result is returned
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
            An inverted copy of the tree
        """
        return ~self.copy()

    def preorder(self) -> list[Node]:
        """
        List out the nodes in a preorder traversal type (root, left, right)
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
        List out the nodes in an in-order traversal type (left, root, right)
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
        List out the nodes in a post-order traversal type (left, right, root)
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
            traversal_type: in-order, preorder, or postorder
        Traverse the tree nodes in any traversal type (in-order by default)
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
        Invert the tree left to right
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
            u: A Node object
        Returns:
            Whether u is a node, present in the tree
        """
        if not isinstance(u, Node):
            u = Node(u)
        if self.root == u:
            return True
        if self.left and u in self.left:
            return True
        return self.right and u in self.right

    def __eq__(self, other: "BinTree") -> bool:
        """
        Args:
            other: Another binary tree
        Returns:
            Whether both trees are equal
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
        return f"BinTree({self.root}, {repr(self.left)}, {repr(self.right)})"


def print_zig_zag(b_t: BinTree):
    """
    Args:
        b_t: BinTree
    Print the nodes of b_t zigzag
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


def binary_heap(l: list[float], f: Callable = max) -> BinTree:
    """
    Args:
        l: A list of real values
        f: Optimizing function, max by default
    Returns:
        A binary heap of list l
    """

    def helper(curr_root, i=0):
        left = helper(l[2 * i + 1], 2 * i + 1) if 2 * i + 1 < n else None
        right = helper(l[2 * i + 2], 2 * i + 2) if 2 * i + 2 < n else None
        res = BinTree(curr_root, left, right)
        return res

    build_heap(l, f)
    n = len(l)
    return helper(l[0])


class Tree:
    """
    Class for implementing a tree with multiple descendants
    """

    def __init__(self, root: Node, inheritance: dict[Node, Iterable[Node]] = {}) -> None:
        """
        Args:
            root: Root node
            inheritance: Inheritance dictionary
        """
        if not isinstance(root, Node):
            root = Node(root)
        self.__root = root
        self.__hierarchy, self.__parent = {root: set()}, {}
        self.__nodes, self.__leaves = {root}, {root}
        if root in inheritance:
            inheritance.pop(root)
        remaining = reduce(lambda x, y: x.union(y), inheritance.values(), set())
        remaining = set(map(lambda x: x if isinstance(x, Node) else Node(x), remaining))
        if not (root_desc := set(inheritance) - remaining) and inheritance:
            raise ValueError("This dictionary doesn't represent a tree!")
        root_desc = set(map(lambda x: x if isinstance(x, Node) else Node(x), root_desc))
        for u, desc in inheritance.items():
            if not isinstance(u, Node):
                u = Node(u)
            if u in root_desc:
                self.add(root, u)
            if desc:
                self.add(u, *desc)

    @property
    def root(self) -> Node:
        """
        Returns:
            Tree root
        """
        return self.__root

    @property
    def nodes(self) -> set[Node]:
        """
        Returns:
            Nodes
        """
        return self.__nodes.copy()

    @property
    def leaves(self) -> set[Node]:
        """
        Returns:
            Leaves
        """
        return self.__leaves.copy()

    def leaf(self, n: Node) -> bool:
        """
        Args:
            n: A present node
        Returns:
            Whether node n is a leaf
        """
        if not isinstance(n, Node):
            n = Node(n)
        if n not in self:
            raise KeyError("Unrecognized node!")
        return n in self.leaves

    def add(self, curr: Node, u: Node, *rest: Node) -> "Tree":
        """
        Args:
            curr: A present node
            u: A new node
            rest: Other new nodes
        Add new nodes as descendants of a given present node
        """
        if not isinstance(curr, Node):
            curr = Node(curr)
        if curr in self:
            new_nodes = False
            for v in {u, *rest}:
                if not isinstance(v, Node):
                    v = Node(v)
                if v not in self:
                    new_nodes = True
                    self.__nodes.add(v)
                    self.__hierarchy[curr].add(v)
                    self.__parent[v] = curr
                    self.__leaves.add(v)
                    self.__hierarchy[v] = set()
            if self.leaf(curr) and new_nodes:
                self.__leaves.remove(curr)
        return self

    def add_tree(self, tree: "Tree") -> "Tree":
        """
        Args:
            tree: A Tree object, the root of which is a present node
        Add an entire tree, the root of which is an already present node. It expands the tree from there
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
                Tree.add(self, u, *res)
                queue += list(res)
        return self

    def remove(self, u: Node, subtree: bool = True) -> "Tree":
        """
        Args:
            u: A present node
            subtree: Boolean flag, answering to whether to remove the subtree, rooted in node u
        Remove a node and make its descendants direct descendants of its parent node if subtree is False
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            if u == self.root:
                raise ValueError("Can't remove root!")
            if subtree:
                for d in self.descendants(u):
                    self.remove(d)
            v = self.parent(u)
            leaf = self.leaf(u)
            self.__nodes.remove(u), self.__parent.pop(u)
            for n in self.descendants(u):
                self.__parent[n] = v
            self.__hierarchy[v].update(self.hierarchy(u)), self.__hierarchy[v].remove(u)
            if leaf:
                self.__leaves.remove(u)
                if not self.hierarchy(v):
                    self.__leaves.add(v)
            self.__hierarchy.pop(u)
        return self

    def height(self) -> int:
        """
        Returns:
            Tree height
        """

        def helper(x):
            return 1 + max([-1, *map(helper, self.descendants(x))])

        return helper(self.root)

    def parent(self, u: Node = None) -> dict[Node, Node] | Node:
        """
        Args:
            u: A present node or None
        Returns:
            The parent node of node u if it's given, otherwise the parent of each node
        """
        if u is None:
            return self.__parent.copy()
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        return self.__parent.get(u)

    def hierarchy(self, u: Node = None) -> dict[Node, set[Node]] | set[Node]:
        """
        Args:
            u: A present node or None
        Returns:
            Descendants of node u if it's given, otherwise the descendants of each node
        """
        if u is None:
            return {n: self.hierarchy(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return self.__hierarchy[u].copy()

    def descendants(self, u: Node) -> set[Node]:
        """
        Args:
            u: A present node
        Returns:
            The descendants of node u
        """
        return self.hierarchy(u)

    def copy(self) -> "Tree":
        """
        Returns:
            An identical copy of the tree
        """
        return Tree(self.root, {u: self.descendants(u) for u in self.nodes - {self.root}})

    def subtree(self, u: Node) -> "Tree":
        """
        Args:
            u: A present node
        Returns:
            The subtree, rooted in node u
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
            The undirected graph version of the tree
        """
        return UndirectedGraph(self.hierarchy())

    def directed_graph(self, from_root: bool = True) -> DirectedGraph:
        """
        Args:
            from_root: Boolean flag, answering to whether the links point to or from the root
        Returns:
            A directed graph version of the tree
        """
        return DirectedGraph({k: ([], v) if from_root else (v, []) for k, v in self.hierarchy().items()})

    def weighted_tree(self, weights: dict[Node, float] = None) -> "WeightedTree":
        """
        Args:
            weights: A dictionary, mapping nodes to weights
        Returns:
            The weighted tree version of the tree
        """
        if weights is None:
            weights = {n: 0 for n in self.nodes}
        for n in self.nodes - set(weights):
            weights[n] = 0
        return WeightedTree((self.root, weights[self.root]),
                            {n: (weights[n], self.descendants(n)) for n in self.nodes})

    def node_depth(self, u: Node) -> int:
        """
        Args:
            u: A present node
        Returns:
            The distance (in links) from the root to a node u
        """
        if not isinstance(u, Node):
            u = Node(u)
        if u not in self:
            raise KeyError("Unrecognized node!")
        d = 0
        while u != self.root:
            u = self.parent(u)
            d += 1
        return d

    def path_to(self, u: Node) -> list[Node]:
        """
        Args:
            u: A present node
        Returns:
            The path from the root to node u
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
            A minimum set of nodes, that cover all links in the tree
        """
        return self.nodes - self.independent_set()

    def dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimum set of nodes, that cover all nodes in the tree
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

        if self.nodes == {self.root}:
            return {self.root}
        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return root_val[0] if len((root_val := dp[self.root])[0]) <= len(root_val[1]) else root_val[1]

    def independent_set(self) -> set[Node]:
        """
        Returns:
            A maximum set of nodes, none of which is a parent of any other
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
            An isomorphic function between the nodes of the tree and those of the given tree, if such exists, otherwise empty dictionary. Let f be such a bijection and u and v be tree nodes. f(u) and f(v) are nodes in the other tree and f(u) is parent of f(v) exactly when the same applies for u and v. For weighted trees, the weights are considered.
        """
        return isomorphic_bijection(self, other)

    def __contains__(self, u: Node) -> bool:
        """
        Args:
            u: A Node object
        Returns:
            Whether u is a node in the tree
        """
        if not isinstance(u, Node):
            u = Node(u)
        return u in self.nodes

    def __eq__(self, other: "Tree") -> bool:
        """
        Args:
            other: a Tree object
        Returns:
            Whether both trees are equal
        """
        return compare(self, other)

    def __str__(self) -> str:
        return string(self)

    def __repr__(self) -> str:
        inheritance = self.hierarchy().copy()
        inheritance.pop(self.root)
        return f"Tree({self.root}, {inheritance})"


class WeightedTree(Tree):
    """
    Class for implementing a tree with node weights
    """

    def __init__(self, root_and_weight: tuple[Node, float],
                 inheritance: dict[Node, tuple[float, Iterable[Node]]] = {}) -> None:
        """
        Args:
            root_and_weight: A tuple of the root node and its weight
            inheritance: An inheritance dictionary. Each node is mapped to a tuple, where the first element is its weight and the second element is the set of its descendants.
        """
        super().__init__(root := root_and_weight[0])
        if not isinstance(root, Node):
            root = Node(root)
        self.__weights = {root: float(root_and_weight[1])}
        remaining = reduce(lambda x, y: x.union(y[1]), inheritance.values(), set())
        if not (root_descendants := set(inheritance) - remaining) and inheritance:
            raise ValueError("This dictionary doesn't represent a tree!")
        for u in root_descendants:
            self.add(root, {u: inheritance[u][0]})
        for u, (_, desc) in inheritance.items():
            if desc:
                self.add(u, {d: inheritance[d][0] if d in inheritance else 0 for d in desc})

    def weights(self, u: Node = None) -> dict[Node, float] | float:
        """
        Args:
            u: A present node or None
        Returns:
            The weight of node n or the dictionary with all node weights
        """
        if u is None:
            return {n: self.weights(n) for n in self.nodes}
        if not isinstance(u, Node):
            u = Node(u)
        return self.__weights[u]

    def add(self, curr: Node, rest: dict[Node, float] = {}) -> "WeightedTree":
        if not isinstance(curr, Node):
            curr = Node(curr)
        if curr in self:
            for u, w in rest.items():
                if u not in self:
                    self.set_weight(u, w)
            if rest:
                super().add(curr, *rest.keys())
        return self

    def add_tree(self, tree: Tree) -> "WeightedTree":
        super().add_tree(tree)
        if not isinstance(tree, WeightedTree):
            tree = tree.weighted_tree()
        queue = [*tree.descendants(root := tree.root)]
        self.increase_weight(root, tree.weights(root))
        while queue:
            self.set_weight((u := queue.pop(0)), tree.weights(u))
            queue += self.descendants(u)
        return self

    def remove(self, u: Node, subtree: bool = True) -> "WeightedTree":
        if not isinstance(u, Node):
            u = Node(u)
        if u in self:
            if subtree:
                for d in self.descendants(u):
                    self.remove(d)
            self.__weights.pop(u)
            super().remove(u, subtree)
        return self

    def set_weight(self, u: Node, w: float) -> "WeightedTree":
        """
        Args:
            u: A present node
            w: The new weight of node u
        Set the weight of node u to w
        """
        if not isinstance(u, Node):
            u = Node(u)
        try:
            self.__weights[u] = float(w)
        except ValueError:
            raise TypeError("Real value expected!")
        return self

    def increase_weight(self, u: Node, w: float) -> "WeightedTree":
        """
        Args:
            u: A present node
            w: A real value
        Increase the weight of node u by w
        """
        if not isinstance(u, Node):
            u = Node(u)
        try:
            try:
                self.set_weight(u, self.weights(u) + float(w))
            except ValueError:
                raise TypeError("Real value expected!")
        except KeyError:
            ...
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
        return WeightedNodesDirectedGraph(
            {k: (self.weights(k), ([], v) if from_root else (v, [])) for k, v in self.hierarchy().items()})

    def weighted_vertex_cover(self) -> set[Node]:
        """
        Returns:
            A minimum by sum of the weights set of nodes, that cover all links in the tree
        """
        return self.nodes - self.weighted_independent_set()

    def weighted_dominating_set(self) -> set[Node]:
        """
        Returns:
            A minimum by sum of the weights set of nodes, that cover all nodes in the tree
        """

        def dfs(r):
            if self.leaf(r):
                if r == self.root:
                    dp[r][1] = {r}
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
                    dp[r][0].update(dp[d][0] if (d_weights_sum := sum(map(self.weights, dp[d][0]))) < sum(
                        map(self.weights, dp[d][1])) else dp[d][1])
                    if min_no_root is None or d_weights_sum < sum(map(self.weights, dp[min_no_root][0])):
                        min_no_root = d
            for d in self.descendants(r):
                dp[r][1].update(dp[d][1] if sum(map(self.weights, dp[d][1])) < sum(
                    map(self.weights, dp[d][0])) and d != min_no_root else dp[d][0])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        root_val = dp[self.root]
        return root_val[0] if sum(map(self.weights, root_val[0])) <= sum(map(self.weights, root_val[1])) else root_val[
            1]

    def weighted_independent_set(self) -> set[Node]:
        """
        Returns:
            A set of nodes with a maximum possible sum of the weights, none of which is a parent of any other
        """

        def dfs(x):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0].update(dp[y][1])
                dp[x][1].update(
                    dp[y][0] if sum(map(self.weights, dp[y][0])) > sum(map(self.weights, dp[y][1])) else dp[y][1])

        dp = {n: [{n}, set()] for n in self.nodes}
        dfs(self.root)
        return root_val[0] if (
                sum(map(self.weights, (root_val := dp[self.root])[0])) > sum(map(self.weights, root_val[1]))) else \
            root_val[1]

    def __repr__(self) -> str:
        inheritance = {k: (self.weights(k), v) for k, v in self.hierarchy().items()}
        inheritance.pop(self.root)
        return f"WeightedTree({(self.root, self.weights(self.root))}, {inheritance})"
