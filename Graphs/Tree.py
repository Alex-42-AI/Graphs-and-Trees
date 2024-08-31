from Graphs.General import Node, Dict, SortedKeysDict, SortedList, BinNode
class BinTree:
    def __init__(self, root=None):
        self.root = root if isinstance(root, BinNode) else BinNode(root)
    def copy(self, curr_from: BinNode, curr_to: BinNode):
        if curr_from is None or curr_to is None:
            return
        if curr_from.left is not None:
            self.copy(curr_from.left, curr_to.left)
            curr_to.left = curr_from.left
        if curr_from.right is not None:
            self.copy(curr_from.right, curr_to.right)
            curr_to.right = curr_from.right
    def left(self):
        if self.root.left is not None:
            res = BinTree(self.root.left.value())
            self.copy(self.root.left, res.root)
            return res
        return BinTree(False)
    def right(self):
        if self.root.right is not None:
            res = BinTree(self.root.right.value())
            self.copy(self.root.right, res.root)
            return res
        return BinTree(False)
    def nodes_on_level(self, level: int, curr_node: BinNode = ''):
        if curr_node == '':
            curr_node = self.root
        if level > self.get_height() or level < 0:
            return []
        if not level:
            return [curr_node]
        if curr_node.left is None and curr_node.right is None:
            return []
        left, right = [], []
        if curr_node.left is not None:
            left = self.nodes_on_level(level - 1, curr_node.left)
        if curr_node.right is not None:
            right = self.nodes_on_level(level - 1, curr_node.right)
        return left + right
    def width(self):
        Max = 0
        for i in range(self.get_height()):
            Max = max(len(self.nodes_on_level(i)), Max)
        return Max
    def get_height_recursive(self):
        def helper(curr_node=''):
            if curr_node == '':
                curr_node = self.root
            if curr_node.left is None and curr_node.right is None:
                return 0
            left, right = 0, 0
            if curr_node.left is not None:
                left = helper(curr_node.left)
            if curr_node.right is not None:
                right = helper(curr_node.right)
            return 1 + max(left, right)
        return helper()
    def get_height(self):
        Last_Node = self.root
        while Last_Node.right is not None:
            Last_Node = Last_Node.right
        node = self.root
        current, Max, so_far, chain = 0, 0, [None], [node]
        while True:
            if node.left not in so_far:
                node = node.left
                chain.append(node)
                current += 1
            elif node.right not in so_far:
                node = node.right
                chain.append(node)
                current += 1
            else:
                Max = max(Max, current)
                if node == Last_Node:
                    break
                current -= 1
                node = chain[-2]
                so_far.append(chain.pop())
        return Max
    def count_leaves(self, curr_node=''):
        if curr_node == '':
            curr_node = self.root
        if curr_node is None:
            return 0
        if curr_node.left is None and curr_node.right is None:
            return 1
        return self.count_leaves(curr_node.left) + self.count_leaves(curr_node.right)
    def count_nodes(self, curr_node=''):
        if curr_node == '':
            curr_node = self.root
        if curr_node is None:
            return 0
        return (curr_node.value() is not None) + self.count_nodes(curr_node.left) + self.count_nodes(curr_node.right)
    def code_in_morse(self, v):
        def helper(tree=None):
            if tree is None:
                tree = self
            if tree.root.value() is False:
                return
            if tree.root.left is not None:
                if tree.root.left.value() == v:
                    return '.'
            res = helper(tree.left())
            if res:
                return '. ' + res
            if tree.root.right is not None:
                if tree.root.right.value() == v:
                    return '-'
            res = helper(tree.right())
            if res:
                return '- ' + res
        return helper(v)
    def encode(self, message: str):
        res = ''
        for c in message.upper():
            if c in self:
                res += self.code_in_morse(c) + '   '
            else:
                res += c + '  '
        return res[:-2]
    def invert(self):
        def helper(node=''):
            if node == '':
                node = self.root
            if node is None:
                return
            helper(node.left)
            helper(node.right)
            node.left, node.right = node.right, node.left
        helper()
    def __invert__(self):
        self.invert()
    def __contains__(self, item):
        if self.root.value() == item:
            return True
        if self.root.left is not None:
            if item in self.left():
                return True
        if self.root.right is not None:
            return item in self.right()
    def __eq__(self, other):
        if self.root == other.root:
            if self.root.left is not None and other.root.left is not None:
                if self.left() == other.left():
                    return True
                return self.root.left == other.root.left
            if self.root.right is not None and other.root.right is not None:
                return self.right() == other.right()
            return self.root.right == other.root.right
        return False
    def __bool__(self):
        return self.root is not None
    def __preorder_print(self, start: BinNode, traversal: [BinNode]):
        if start is not None:
            traversal += [start]
            traversal = self.__preorder_print(start.left, traversal)
            traversal = self.__preorder_print(start.right, traversal)
        return traversal
    def __in_order_print(self, start: BinNode, traversal: [BinNode]):
        if start is not None:
            traversal = self.__in_order_print(start.left, traversal)
            traversal += [start]
            traversal = self.__in_order_print(start.right, traversal)
        return traversal
    def __post_order_print(self, start: BinNode, traversal: [BinNode]):
        if start is not None:
            traversal = self.__post_order_print(start.left, traversal)
            traversal = self.__post_order_print(start.right, traversal)
            traversal += [start]
        return traversal
    def print(self, traversal_type: str = 'in-order'):
        if traversal_type.lower() == 'preorder':
            print(self.__preorder_print(self.root, []))
        elif traversal_type.lower() == 'in-order':
            print(self.__in_order_print(self.root, []))
        elif traversal_type.lower() == 'post-order':
            print(self.__post_order_print(self.root, []))
        else:
            print('Traversal type ' + str(traversal_type) + ' is not supported!')
    def __str__(self):
        return str(self.__in_order_print(self.root, []))
    def __repr__(self):
        return str(self)
class Tree:
    def __init__(self, root: Node, *descendants: Node):
        self.__root = root
        if root in descendants:
            raise ValueError('Can\'t have a node twice in a tree!')
        for i in range(len(descendants)):
            for j in range(i + 1, len(descendants)):
                if descendants[i] == descendants[j]:
                    raise ValueError('Can\'t have a node twice in a tree!')
        self.__hierarchy, self.__nodes, self.__links, self.__leaves = SortedKeysDict((self.__root, list(descendants))) + SortedKeysDict(*[(n, []) for n in descendants]), SortedList(), [(root, n) for n in descendants], SortedList()
        for n in [root] + [*descendants]:
            self.__nodes.insert(n)
        if descendants:
            for d in descendants:
                self.__leaves.insert(d)
        else:
            self.__leaves.insert(root)
    def root(self):
        return self.__root
    def nodes(self):
        return self.__nodes
    def links(self):
        return self.__links
    def leaves(self):
        return self.__leaves
    def hierarchy(self):
        return SortedKeysDict(*filter(lambda p: p[1], self.__hierarchy.items()))
    def descendants(self, n: Node):
        return self.__hierarchy[n]
    def add_nodes_to(self, curr: Node, u: Node, *rest: Node):
        if curr not in self.__nodes:
            raise Exception('Node not found!')
        if curr in self.__leaves:
            self.__leaves.remove(curr)
        for v in [u] + [*rest]:
            if v not in self.nodes():
                self.__nodes.insert(v), self.__hierarchy[curr].append(v), self.__links.append((curr, v)), self.__leaves.insert(v)
                self.__hierarchy[v] = []
    def copy(self):
        res = Tree(self.root(), *self.descendants(self.__root))
        queue = self.descendants(res.root())
        while queue:
            u = queue.pop(0)
            res_descendants = self.descendants(u)
            if res_descendants:
                res.add_nodes_to(u, *res_descendants)
                queue += res_descendants
        return res
    def extend_tree_at(self, n: Node, tree):
        if n not in self.__nodes:
            raise Exception('Node not found!')
        if not isinstance(tree, Tree):
            raise TypeError('Tree expected!')
        self.add_nodes_to(n, tree.__root)
        queue = [tree.__root]
        while queue:
            u = queue.pop(0)
            res = list(filter(lambda x: x not in self.__nodes, tree.descendants(u)))
            if res:
                self.add_nodes_to(u, *res)
                queue += res
    def move_node(self, u: Node, at_new: Node):
        if u not in self.__nodes:
            return
        if at_new not in self.__nodes:
            raise ValueError("Unrecognized node!")
        descendants, p = self.descendants(u), self.parent(u)
        self.__hierarchy[p].remove(u)
        if not self.__hierarchy[p]:
            self.__leaves.insert(p)
        self.__links.remove((p, u)), self.__nodes.remove(u), self.add_nodes_to(at_new, u)
        self.__hierarchy[u] = descendants
    def remove_node(self, u: Node):
        if u not in self.__nodes:
            raise ValueError("Node not in tree!")
        if u == self.__root:
            raise ValueError("Can't remove root!")
        self.__nodes.remove(u)
        v, l = self.parent(u), 0
        while l < len(self.__links):
            if u in self.__links[l]:
                self.__links.remove(self.__links[l])
                l -= 1
            l += 1
        self.__hierarchy[v] += self.__hierarchy[u]
        if u in self.__leaves:
            self.__leaves.remove(u)
            if not self.__hierarchy[v]:
                self.__leaves.insert(v)
        self.__hierarchy.pop(u)
    def parent(self, u: Node):
        if u in self.__nodes:
            for l in self.__links:
                if u == l[1]:
                    return l[0]
        raise ValueError('Node not in tree!')
    def node_depth(self, u: Node):
        if u in self.__nodes:
            d = 0
            while u != self.__root:
                u = self.parent(u)
                d += 1
            return d
        raise ValueError('Node not in graph!')
    def height(self):
        def helper(x: Node):
            return 1 + max([0, *map(helper, self.descendants(x))])
        return helper(self.__root)
    def path_to(self, u: Node):
        x, res = u, []
        while x != self.__root:
            res = [x] + res
            x = self.parent(x)
        return res
    def vertexCover(self):
        dp = SortedKeysDict(*[(n, [1, 0]) for n in self.nodes()])
        def dfs(x: Node):
            for y in self.descendants(x):
                dfs(y)
                dp[x][0] += dp[y][1]
                dp[x][1] += max(dp[y])
        dfs(self.root())
        return len(self.nodes()) - max(dp[self.root()])
    def dominatingSet(self):
        dp = SortedKeysDict(*[(n, [1, 0] for n in self.__nodes)])
        def dfs(r):
            if not self.descendants(r):
                return
            s0, s1, s2 = 0, 0, 0
            for d in self.descendants(r):
                dfs(d)
                s0 += dp[d][1]
                s1, s2 = s1 + dp[d][0], min(s2, sum(dp[w][1] for w in self.descendants(d)) + sum(dp[_d][0] for _d in self.descendants(r)) - dp[d][0])
            dp[r][0] += min(s0, s2)
            dp[r][1] = min(s0 + 1, s1)
        dfs(self.__root)
        return dp[self.__root][0]
    def isomorphic(self, other):
        if isinstance(other, Tree):
            if len(self.__nodes) != len(other.__nodes) or len(self.__links) != len(other.__links) or len(self.leaves()) != len(other.leaves()) or len(self.descendants(self.__root)) != len(other.descendants(other.root())):
                return False
            this_hierarchies, other_hierarchies = dict(), dict()
            for n in self.__nodes:
                descendants = len(self.descendants(n))
                if descendants not in this_hierarchies:
                    this_hierarchies[descendants] = 1
                else:
                    this_hierarchies[descendants] += 1
            for n in other.__nodes:
                descendants = len(other.descendants(n))
                if descendants not in other_hierarchies:
                    other_hierarchies[descendants] = 1
                else:
                    other_hierarchies[descendants] += 1
            if this_hierarchies != other_hierarchies:
                return False
            this_nodes_descendants, other_nodes_descendants = {d: [] for d in this_hierarchies.keys()}, {d: [] for d in other_hierarchies.keys()}
            for n in self.__nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.__nodes:
                other_nodes_descendants[len(self.descendants(n))].append(n)
            this_nodes_descendants, other_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x))), list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_descendants]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility):
                    map_dict += Dict(*zip(group, other_nodes_descendants[i]))
                possible = True
                for n, v0 in map_dict.items():
                    for m, v1 in map_dict.items():
                        if ((n, m) in self.links()) ^ ((v0, v1) in other.links()) or ((m, n) in self.links()) ^ ((v1, v0) in other.links()):
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __contains__(self, item):
        return item in self.__nodes + self.__links
    def __eq__(self, other):
        for u in self.nodes():
            if u not in other.nodes():
                return False
        if len(self.__nodes) != len(other.__nodes):
            return False
        for u in self.__nodes:
            if len(self.__hierarchy[u]) != len(other.__hierarchy[u]):
                return False
            for v in self.__hierarchy[u]:
                if v not in other.__hierarchy[u]:
                    return False
            return True
    def __str__(self):
        return '\n'.join(str(k) + ' -- ' + str(v) for k, v in filter(lambda p: p[1], self.__hierarchy.items()))
    def __repr__(self):
        return str(self)
class WeightedNodesTree(Tree):
    def __init__(self, root_and_weight: (Node, float), *pairs: (Node, float)):
        super().__init__(root_and_weight[0], *[p[0] for p in pairs])
        self.__weights = SortedKeysDict(root_and_weight)
        for d, w in pairs:
            if d not in self.__weights:
                self.__weights[d] = w
    def copy(self):
        res = WeightedNodesTree((self.root(), self.__weights[self.root()]), *[(d, self.__weights[d]) for d in self.descendants(self.__root)])
        queue = self.descendants(res.root())
        while queue:
            u = queue.pop(0)
            res_descendants = self.descendants(u)
            if res_descendants:
                res.add_nodes_to(u, *res_descendants)
                for d in res_descendants:
                    res.__weights[d] = self.__weights[d]
                queue += res_descendants
        return res
    def add_nodes_to(self, u: Node, new_p: (Node, float), *rest_p: (Node, float)):
        if u not in self.__nodes:
            raise Exception('Node not found!')
        for v, w in [new_p] + [*rest_p]:
            if v not in self.nodes():
                self.__weights[v] = w
        super().add_nodes_to(u, new_p[0], *[p[0] for p in rest_p])
    def extend_tree_at(self, n: Node, tree):
        super().extend_tree_at(n, tree)
        if isinstance(tree, WeightedNodesTree):
            queue = [tree.__root]
            while queue:
                u = queue.pop(0)
                res = list(filter(lambda _n: _n not in self.__nodes, tree.descendants(u)))
                if res:
                    self.__weights[u] = tree.__weights[u]
                    queue += res
    def remove_node(self, u: Node):
        self.__weights.pop(u), super().remove_node(u)
    def vertexCover(self):
        dp = SortedKeysDict(*[(n, [self.__weights[n], 0]) for n in self.nodes()])
        def dfs(u: Node):
            for v in self.descendants(u):
                dfs(v)
                dp[u][0] += dp[v][1]
                dp[u][1] += max(dp[v])
        dfs(self.root())
        return len(self.nodes()) - max(dp[self.root()])
    def dominatingSet(self):
        dp = SortedKeysDict(*[(n, [self.__weights[n], 0] for n in self.__nodes)])
        def dfs(r):
            if not self.descendants(r):
                return
            s0, s1, s2 = 0, 0, 0
            for d in self.descendants(r):
                dfs(d)
                s0 += dp[d][1]
                s1, s2 = s1 + dp[d][0], min(s2, self.__weights[d] + sum(dp[w][1] for w in self.descendants(d)) + sum(dp[_d][0] for _d in self.descendants(r)) - dp[d][0])
            dp[r][0] += min(s0, s2)
            dp[r][1] = min(s0 + self.__weights[r], s1)
        dfs(self.__root)
        return dp[self.__root][0]
    def isomorphic(self, other):
        if isinstance(other, Tree):
            if len(self.__nodes) != len(other.nodes()) or len(self.__links) != len(other.links()) or len(self.leaves()) != len(other.leaves()) or len(self.descendants(self.__root)) != len(other.descendants(other.root())):
                return False
            this_hierarchies, other_hierarchies = dict(), dict()
            for n in self.__nodes:
                descendants = len(self.descendants(n))
                if descendants not in this_hierarchies:
                    this_hierarchies[descendants] = 1
                else:
                    this_hierarchies[descendants] += 1
            for n in other.nodes():
                descendants = len(other.descendants(n))
                if descendants not in other_hierarchies:
                    other_hierarchies[descendants] = 1
                else:
                    other_hierarchies[descendants] += 1
            if this_hierarchies != other_hierarchies:
                return False
            this_nodes_descendants, other_nodes_descendants = {d: [] for d in this_hierarchies.keys()}, {d: [] for d in other_hierarchies.keys()}
            for n in self.__nodes:
                this_nodes_descendants[len(self.descendants(n))].append(n)
            for n in other.nodes():
                other_nodes_descendants[len(self.descendants(n))].append(n)
            this_nodes_descendants, other_nodes_descendants = list(sorted(this_nodes_descendants.values(), key=lambda x: len(x))), list(sorted(other_nodes_descendants.values(), key=lambda x: len(x)))
            from itertools import permutations, product
            _permutations = [list(permutations(this_nodes)) for this_nodes in this_nodes_descendants]
            possibilities = product(*_permutations)
            for possibility in possibilities:
                map_dict = Dict()
                for i, group in enumerate(possibility):
                    map_dict += Dict(*zip(group, other_nodes_descendants[i]))
                possible = True
                for n, u in map_dict.items():
                    for m, v in map_dict.items():
                        if ((n, m) in self.links()) ^ ((u, v) in other.links()) or ((m, n) in self.links()) ^ ((v, u) in other.links()) or isinstance(other, WeightedNodesTree) and self.__weights[n] != other.__weights[u]:
                            possible = False
                            break
                    if not possible:
                        break
                if possible:
                    return True
            return False
        return False
    def __eq__(self, other):
        if isinstance(other, WeightedNodesTree):
            if self.__weights != other.__weights:
                return False
        return super().__eq__(other)
    def __str__(self):
        return '\n'.join(f'{k}, {self.__weights[k]} -- {v}' for k, v in self.__hierarchy.items())
