from unittest import TestCase, main

from undirected_graph import Node, UndirectedGraph, WeightedNodesUndirectedGraph

from directed_graph import DirectedGraph, WeightedNodesDirectedGraph

from tree import *

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = map(Node, range(16))

eps, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = Node(""), *map(Node,
                                                                                                   "abcdefghijklmnopqrstuvwxyz")


def test_binary_heap():
    l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    tree = binary_heap(l)
    assert tree.root == n9
    assert tree.leaves == {n0, n1, n2, n3, n4}
    for n in {n5, n6, n7, n8}:
        if tree.subtree(n).left:
            assert tree.subtree(n).left.root < n
        if tree.subtree(n).right:
            assert tree.subtree(n).right.root < n


class TestBinTree(TestCase):
    def setUp(self):
        self.tree = BinTree(eps,
                            BinTree(e, BinTree(i, BinTree(s, h, v), BinTree(u, f)),
                                    BinTree(a, BinTree(r, l), BinTree(w, p, j))),
                            BinTree(t,
                                    BinTree(n, BinTree(d, b, x), BinTree(k, c, y)), BinTree(m, BinTree(g, z, q), o)))

    def test_init(self):
        t = BinTree(0, BinTree(1), BinTree(2))
        self.assertEqual(t.root, n0)
        self.assertEqual(t.left, BinTree(1))
        self.assertEqual(t.right, BinTree(2))

    def test_get_root(self):
        self.assertEqual(self.tree.root, eps)

    def test_get_left(self):
        self.assertEqual(self.tree.left, BinTree(e, BinTree(i, BinTree(s, h, v), BinTree(u, f)),
                                                 BinTree(a, BinTree(r, l), BinTree(w, p, j))))

    def test_get_right(self):
        self.assertEqual(self.tree.right, BinTree(t, BinTree(n, BinTree(d, b, x), BinTree(k, c, y)),
                                                  BinTree(m, BinTree(g, z, q), o)))

    def test_get_leaves(self):
        self.assertSetEqual(self.tree.leaves, {h, v, f, l, p, j, b, x, c, y, z, q, o})

    def test_copy(self):
        self.assertTrue(self.tree == self.tree.copy())

    def test_rotate_left(self):
        tree = self.tree.copy().rotate_left()
        self.assertEqual(tree, BinTree(t, BinTree(eps, BinTree(e, BinTree(i, BinTree(s, h, v), BinTree(u, f)),
                                                               BinTree(a, BinTree(r, l), BinTree(w, p, j))),
                                                  BinTree(n, BinTree(d, b, x), BinTree(k, c, y))),
                                       BinTree(m, BinTree(g, z, q), o)))

    def test_rotate_right(self):
        tree = self.tree.copy().rotate_right()
        self.assertEqual(tree, BinTree(e, BinTree(i, BinTree(s, h, v), BinTree(u, f)),
                                       BinTree(eps, BinTree(a, BinTree(r, l), BinTree(w, p, j)),
                                               BinTree(t, BinTree(n, BinTree(d, b, x), BinTree(k, c, y)),
                                                       BinTree(m, BinTree(g, z, q), o)))))

    def test_subtree(self):
        self.assertEqual(self.tree.subtree(a), BinTree(a, BinTree(r, l), BinTree(w, p, j)))

    def test_subtree_missing_node(self):
        with self.assertRaises(KeyError):
            self.tree.subtree(4)

    def test_tree(self):
        tree = BinTree(0, BinTree(1), BinTree(2))
        self.assertEqual(tree.tree(), Tree(0, {1: [], 2: []}))

    def test_nodes_on_level(self):
        self.assertSetEqual(self.tree.nodes_on_level(2), {a, i, m, n})

    def test_nodes_on_level_bad_level_value(self):
        self.assertSetEqual(self.tree.nodes_on_level(-2), set())

    def test_nodes_on_level_bad_level_type(self):
        with self.assertRaises(TypeError):
            self.tree.nodes_on_level([3])

    def test_width(self):
        self.assertEqual(self.tree.width(), 12)

    def test_height(self):
        self.assertEqual(self.tree.height(), 4)

    def test_count_nodes(self):
        self.assertEqual(self.tree.count_nodes(), 27)

    def test_code_in_morse(self):
        self.assertEqual(self.tree.code_in_morse(w), ". - -")

    def test_code_in_morse_missing_node(self):
        with self.assertRaises(KeyError):
            self.tree.code_in_morse(4)

    def test_encode(self):
        self.assertEqual(self.tree.encode("testing encode"),
                         "-   .   . . .   -   . .   - .   - - .      .   - .   - . - .   - - -   - . .   . ")

    def test_inverted(self):
        tree = self.tree.inverted()
        self.assertEqual(tree, BinTree(eps,
                                       BinTree(t, BinTree(m, o, BinTree(g, q, z)),
                                               BinTree(n, BinTree(k, y, c), BinTree(d, x, b))),
                                       BinTree(e,
                                               BinTree(a, BinTree(w, j, p), BinTree(r, right=l)),
                                               BinTree(i, BinTree(u, right=f), BinTree(s, v, h)))))

    def test_traverse(self):
        self.assertListEqual(self.tree.traverse("preorder"),
                             [eps, e, i, s, h, v, u, f, a, r, l, w, p, j, t, n, d, b, x, k, c, y, m, g, z, q, o])
        self.assertListEqual(self.tree.traverse("in-order"),
                             [h, s, v, i, f, u, e, l, r, a, p, w, j, eps, b, d, x, n, c, k, y, t, z, g, q, m, o])
        self.assertListEqual(self.tree.traverse("post-order"),
                             [h, v, s, f, u, i, l, r, p, j, w, a, e, b, x, d, c, y, k, n, z, q, g, o, m, t, eps])

    def test_unique_structure_hash(self):
        res = self.tree.unique_structure_hash()
        leaf_hash = hash((None, None))
        self.assertSetEqual({leaf_hash}, {res[var] for var in {h, v, f, l, p, j, b, x, z, q, o}})
        left_descendant = hash((leaf_hash, None))
        two_descendants = hash((leaf_hash, leaf_hash))
        self.assertSetEqual({left_descendant}, {res[var] for var in {u, r}})
        self.assertSetEqual({two_descendants}, {res[var] for var in {s, w, d, k, g}})
        self.assertEqual(res[i], hash((res[s], res[u])))
        self.assertEqual(res[a], hash((res[r], res[w])))
        self.assertEqual(res[n], hash((res[d], res[k])))
        self.assertEqual(res[m], hash((res[g], res[o])))
        self.assertEqual(res[e], hash((res[i], res[a])))
        self.assertEqual(res[t], hash((res[n], res[m])))
        self.assertEqual(res[eps], hash((res[e], res[t])))

    def test_isomorphic_bijection(self):
        tree = BinTree(eps,
                       BinTree("ee", BinTree("ii", BinTree("ss", "hh", "vv"), BinTree("uu", "ff")),
                               BinTree("aa", BinTree("rr", "ll"), BinTree("ww", "pp", "jj"))),
                       BinTree("tt",
                               BinTree("nn", BinTree("dd", "bb", "xx"), BinTree("kk", "cc", "yy")),
                               BinTree("mm", BinTree("gg", "zz", "qq"), "oo")))
        res = self.tree.isomorphic_bijection(tree)
        self.assertDictEqual(res,
                             {eps: eps, a: Node("aa"), b: Node("bb"), c: Node("cc"), d: Node("dd"), e: Node("ee"), f: Node("ff"),
                              g: Node("gg"), h: Node("hh"), i: Node("ii"), j: Node("jj"), k: Node("kk"), l: Node("ll"),
                              m: Node("mm"), n: Node("nn"), o: Node("oo"), p: Node("pp"), q: Node("qq"), r: Node("rr"),
                              s: Node("ss"), t: Node("tt"), u: Node("uu"), v: Node("vv"), w: Node("ww"), x: Node("xx"),
                              y: Node("yy"), z: Node("zz")})

    def test_contains(self):
        self.assertIn(d, self.tree)
        self.assertNotIn(4, self.tree)

    def test_equal(self):
        self.assertNotEqual(self.tree.left, self.tree.right)

    def test_str(self):
        res = "()\n ├──(e)\n │   ├──(i)\n │   │   ├──(s)\n │   │   │   ├──(h)\n │   │   │   └──(v)\n │   │   └──(u)\n │   │       ├──(f)\n │   │       └──\\\n │   └──(a)\n │       ├──(r)\n │       │   ├──(l)\n │       │   └──\\\n │       └──(w)\n │           ├──(p)\n │           └──(j)\n └──(t)\n     ├──(n)\n     │   ├──(d)\n     │   │   ├──(b)\n     │   │   └──(x)\n     │   └──(k)\n     │       ├──(c)\n     │       └──(y)\n     └──(m)\n         ├──(g)\n         │   ├──(z)\n         │   └──(q)\n         └──(o)"
        self.assertEqual(str(res), res)


class TestTree(TestCase):
    def setUp(self):
        self.t0 = Tree(0, {1: {3, 4, 5}, 2: {6, 7}, 3: {8, 9}, 5: {10, 11}})
        self.t1 = Tree(0, {1: [], 2: [], 3: [], 4: [], 5: []})

    def test_init(self):
        t = Tree(0, {1: {3, 4}, 2: {5}})
        self.assertEqual(t.root, n0)
        self.assertSetEqual(t.nodes, {n0, n1, n2, n3, n4, n5})
        self.assertDictEqual(t.hierarchy(), {n0: {n1, n2}, n1: {n3, n4}, n2: {n5}, n3: set(), n4: set(), n5: set()})
        self.assertDictEqual(t.parent(), {n1: n0, n2: n0, n3: n1, n4: n1, n5: n2})
        self.assertSetEqual(t.leaves, {n3, n4, n5})

    def test_root(self):
        self.assertEqual(self.t0.root, n0)

    def test_get_nodes(self):
        self.assertSetEqual(self.t0.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11})
        self.assertSetEqual(self.t1.nodes, {n0, n1, n2, n3, n4, n5})

    def test_get_leaves(self):
        self.assertSetEqual(self.t0.leaves, {n4, n6, n7, n8, n9, n10, n11})
        self.assertSetEqual(self.t1.leaves, {n1, n2, n3, n4, n5})

    def test_leaf(self):
        self.assertTrue(self.t0.leaf(4))
        self.assertFalse(self.t0.leaf(0))

    def test_leaf_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.leaf(6)

    def test_add_nodes(self):
        t1 = self.t1.copy()
        t1.add(3, 6, 7)
        self.assertSetEqual(t1.nodes, {n0, n1, n2, n3, n4, n5, n6, n7})
        self.assertSetEqual(t1.leaves, {n1, n2, n4, n5, n6, n7})
        self.assertDictEqual(t1.hierarchy(), {n0: {n1, n2, n3, n4, n5}, n1: set(), n2: set(), n3: {n6, n7},
                                              n4: set(), n5: set(), n6: set(), n7: set()})
        self.assertSetEqual(t1.descendants(3), {n6, n7})

    def test_add_to_missing_node(self):
        self.assertEqual(self.t1, self.t1.add(6, 7))

    def test_add_already_present_nodes(self):
        self.assertEqual(self.t1, self.t1.copy().add(2, 3, 4))

    def test_add_tree(self):
        t = Tree(1, {6: {7}, 8: {9}})
        t1 = self.t1.copy()
        self.assertEqual(t1.add_tree(t), Tree(0, {1: {6, 8}, 2: set(), 3: set(), 4: set(), 5: set(), 6: {7}, 8: {9}}))

    def test_remove(self):
        t0 = self.t0.copy().remove(3, False)
        self.assertEqual(t0, Tree(0, {1: {4, 5, 8, 9}, 2: {6, 7}, 5: {10, 11}}))

    def test_remove_missing_node(self):
        self.assertEqual(self.t0, self.t0.copy().remove(-2))

    def test_remove_subtree(self):
        t0 = self.t0.copy().remove(3)
        self.assertEqual(t0, Tree(0, {1: {4, 5}, 2: {6, 7}, 5: {10, 11}}))

    def test_height(self):
        self.assertEqual(self.t0.height(), 3)
        self.assertEqual(self.t1.height(), 1)

    def test_parent(self):
        self.assertEqual(self.t0.parent(3), n1)
        self.assertDictEqual(self.t1.parent(), {n1: n0, n2: n0, n3: n0, n4: n0, n5: n0})

    def test_root_parent(self):
        self.assertIsNone(self.t0.parent(0))

    def test_parent_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.parent(6)

    def test_hierarchy(self):
        self.assertSetEqual(self.t0.descendants(0), {n1, n2})
        self.assertDictEqual(self.t1.hierarchy(),
                             {n0: {n1, n2, n3, n4, n5}, n1: set(), n2: set(), n3: set(), n4: set(), n5: set()})

    def test_hierarchy_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.hierarchy(6)

    def test_copy(self):
        self.assertEqual(self.t0.copy(), self.t0)
        self.assertEqual(self.t1.copy(), self.t1)

    def test_subtree(self):
        self.assertEqual(self.t0.subtree(1), Tree(1, {3: {8, 9}, 4: set(), 5: {10, 11}}))
        self.assertEqual(self.t1.subtree(0), self.t1)

    def test_subtree_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.subtree(6)

    def test_undirected_graph(self):
        self.assertEqual(self.t0.undirected_graph(), UndirectedGraph(
            {n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11]}))

    def test_directed_graph_from_root(self):
        self.assertEqual(self.t0.directed_graph(), DirectedGraph(
            {n1: ([n0], [n3, n4, n5]), n2: ([n0], [n6, n7]), n3: ([], [n8, n9]), n5: ([], [n10, n11])}))

    def test_directed_graph_to_root(self):
        self.assertEqual(self.t0.directed_graph(False), DirectedGraph(
            {n1: ([n3, n4, n5], [n0]), n2: ([n6, n7], [n0]), n3: ([n8, n9], []), n5: ([n10, n11], [])}))

    def test_weighted_tree(self):
        self.assertEqual(self.t0.weighted_tree(
            {n0: 7, n1: 4, n2: 3, n3: 5, n4: 6, n5: 2, n6: 2, n7: 1, n8: 6, n9: 4, n10: 5, n11: 8}),
            WeightedTree((0, 7),
                         {1: (4, {3, 4, 5}), 2: (3, {6, 7}), 3: (5, {8, 9}), 4: (6, []), 5: (2, {10, 11}),
                          6: (2, []), 7: (1, []), 8: (6, []), 9: (4, []), 10: (5, []), 11: (8, [])}))
        self.assertEqual(self.t1.weighted_tree(),
                         WeightedTree((0, 0), {1: (0, []), 2: (0, []), 3: (0, []), 4: (0, []), 5: (0, [])}))

    def test_node_depth(self):
        self.assertEqual(self.t0.node_depth(4), 2)

    def test_node_depth_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.node_depth(6)

    def test_path_to(self):
        self.assertListEqual(self.t0.path_to(10), [n0, n1, n5, n10])

    def test_path_to_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.path_to(6)

    def test_vertex_cover(self):
        self.assertSetEqual(self.t0.vertex_cover(), {n1, n2, n3, n5})
        self.assertSetEqual(self.t1.vertex_cover(), {n0})

    def test_dominating_set(self):
        self.assertSetEqual(self.t0.dominating_set(), {n1, n2, n3, n5})
        self.assertSetEqual(self.t1.dominating_set(), {n0})

    def test_unique_structure_hash(self):
        res0 = self.t0.unique_structure_hash()
        res1 = self.t1.unique_structure_hash()
        leaf_hash = hash(frozenset())
        self.assertSetEqual({leaf_hash}, {res0[var] for var in {n4, n6, n7, n8, n9, n10, n11}}.union(
            {res1[var] for var in {n1, n2, n3, n4, n5}}))
        self.assertEqual(hash(frozenset({leaf_hash: 5}.items())), res1[n0])
        two_descendants = hash(frozenset({leaf_hash: 2}.items()))
        self.assertSetEqual({two_descendants}, {res0[n2], res0[n3], res0[n5]})
        three_descendants = hash(frozenset({res0[n3]: 2, leaf_hash: 1}.items()))
        self.assertEqual(three_descendants, res0[n1])
        self.assertEqual(hash(frozenset({res0[n1]: 1, res0[n2]: 1}.items())), res0[n0])

    def test_isomorphic_bijection(self):
        t1 = Tree(10, {11: [], 12: [], 13: [], 14: [], 15: []})
        func = self.t1.isomorphic_bijection(t1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})

    def test_contains(self):
        self.assertIn(0, self.t0)
        self.assertNotIn(n6, self.t1)

    def test_equal(self):
        self.assertNotEqual(self.t0, self.t1)

    def test_str(self):
        self.assertEqual(str(self.t1), "(0)\n ├──(1)\n ├──(2)\n ├──(3)\n ├──(4)\n └──(5)")

    def test_repr(self):
        inheritance = self.t0.hierarchy().copy()
        inheritance.pop(self.t0.root)
        self.assertEqual(repr(self.t0), f"Tree({self.t0.root}, {inheritance})")


class TestWeightedTree(TestCase):
    def setUp(self):
        self.t0 = WeightedTree((0, 7), {1: (4, {3, 4, 5}), 2: (3, {6, 7}), 3: (5, {8, 9}), 4: (6, []),
                                        5: (2, {10, 11}), 6: (2, []), 7: (1, []), 8: (6, []), 9: (4, []),
                                        10: (5, []), 11: (8, [])})
        self.t1 = WeightedTree((0, 6), {1: (1, []), 2: (1, []), 3: (0, []), 4: (2, []), 5: (1, [])})

    def test_init(self):
        t = WeightedTree((0, 6), {1: (4, {2}), 3: (2, [])})
        self.assertDictEqual(t.weights(), {n0: 6, n1: 4, n2: 0, n3: 2})
        self.assertDictEqual(t.hierarchy(), {n0: {n1, n3}, n1: {n2}, n2: set(), n3: set()})
        self.assertSetEqual(t.leaves, {n2, n3})

    def test_weights(self):
        self.assertDictEqual(self.t1.weights(), {n0: 6, n1: 1, n2: 1, n3: 0, n4: 2, n5: 1})

    def test_weights_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.weights(6)

    def test_add_nodes(self):
        t1 = self.t1.copy()
        t1.add(3, {6: 3, 7: 4})
        self.assertDictEqual(t1.weights(), {n0: 6, n1: 1, n2: 1, n3: 0, n4: 2, n5: 1, n6: 3, n7: 4})
        self.assertSetEqual(t1.leaves, {n1, n2, n4, n5, n6, n7})
        self.assertDictEqual(t1.hierarchy(),
                             {n0: {n1, n2, n3, n4, n5}, n1: set(), n2: set(), n3: {n6, n7}, n4: set(), n5: set(),
                              n6: set(), n7: set()})
        self.assertSetEqual(t1.descendants(3), {n6, n7})

    def test_add_to_missing_node(self):
        self.assertEqual(self.t1, self.t1.copy().add(6, {7: 4}))

    def test_add_already_present_nodes(self):
        self.assertEqual(self.t1, self.t1.copy().add(2, {3: 2, 4: 1}))

    def test_add_tree(self):
        t = Tree(1, {6: {7}, 8: {9}})
        t1 = self.t1.copy()
        self.assertEqual(t1.add_tree(t), WeightedTree((0, 6),
                                                      {1: (1, {6, 8}), 2: (1, set()), 3: (0, set()), 4: (2, set()),
                                                       5: (1, set()), 6: (0, {7}), 8: (0, {9})}))

    def test_remove(self):
        t0 = self.t0.copy().remove(3, False)
        self.assertEqual(t0, WeightedTree((0, 7), {1: (4, {4, 5, 8, 9}), 2: (3, {6, 7}), 4: (6, []), 5: (2, {10, 11}),
                                                   6: (2, []), 7: (1, []), 8: (6, []), 9: (4, []), 10: (5, []),
                                                   11: (8, [])}))

    def test_remove_missing_node(self):
        self.assertEqual(self.t0, self.t0.copy().remove(-2))

    def test_remove_subtree(self):
        t0 = self.t0.copy().remove(3)
        self.assertEqual(t0, WeightedTree((0, 7),
                                          {1: (4, {4, 5}), 2: (3, {6, 7}), 4: (6, []), 5: (2, {10, 11}), 6: (2, []),
                                           7: (1, []), 10: (5, []), 11: (8, [])}))

    def test_set_weight(self):
        t0 = self.t0.copy().set_weight(4, 7)
        self.assertEqual(t0.weights(4), 7)

    def test_set_weight_missing_node(self):
        self.assertEqual(self.t0, self.t0.copy().set_weight(-4, 7))

    def test_set_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.t0.set_weight(4, "5-")

    def test_increase_weight(self):
        t0 = self.t0.copy().increase_weight(4, 2)
        self.assertEqual(t0.weights(4), 8)

    def test_increase_weight_missing_node(self):
        self.assertEqual(self.t0, self.t0.copy().increase_weight(-4, 3))

    def test_increase_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.t0.increase_weight(4, "5-")

    def test_copy(self):
        self.assertEqual(self.t0.copy(), self.t0)
        self.assertEqual(self.t1.copy(), self.t1)

    def test_subtree(self):
        self.assertEqual(self.t0.subtree(1), WeightedTree((1, 4),
                                                          {3: (5, {8, 9}), 4: (6, []), 5: (2, {10, 11}), 8: (6, []),
                                                           9: (4, []), 10: (5, []), 11: (8, [])}))
        self.assertEqual(self.t1.subtree(0), self.t1)

    def test_subtree_missing_node(self):
        with self.assertRaises(KeyError):
            self.t1.subtree(6)

    def test_undirected_graph(self):
        self.assertEqual(self.t0.undirected_graph(), WeightedNodesUndirectedGraph(
            {n0: (7, []), n1: (4, [n0, n3, n4, n5]), n2: (3, [n0, n6, n7]), n3: (5, [n8, n9]), n4: (6, []),
             n5: (2, [n10, n11]), n6: (2, []), n7: (1, []), n8: (6, []), n9: (4, []), n10: (5, []), n11: (8, [])}))

    def test_directed_graph_from_root(self):
        self.assertEqual(self.t0.directed_graph(), WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (4, ({n0}, {n3, n4, n5})), n2: (3, ({n0}, {n6, n7})), n3: (5, ([], {n8, n9})),
             n4: (6, ([], [])), n5: (2, ([], {n10, n11})), n6: (2, ([], [])), n7: (1, ([], [])), n8: (6, ([], [])),
             n9: (4, ([], [])), n10: (5, ([], [])), n11: (8, ([], []))}))

    def test_directed_graph_to_root(self):
        self.assertEqual(self.t0.directed_graph(False), WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (4, ([n3, n4, n5], [n0])), n2: (3, ([n6, n7], [n0])), n3: (5, ([n8, n9], [])),
             n4: (6, ([], [])), n5: (2, ([n10, n11], [])), n6: (2, ([], [])), n7: (1, ([], [])), n8: (6, ([], [])),
             n9: (4, ([], [])), n10: (5, ([], [])), n11: (8, ([], []))}))

    def test_weighted_vertex_cover(self):
        self.assertSetEqual(self.t0.weighted_vertex_cover(), {n1, n2, n3, n5})
        self.assertSetEqual(self.t1.weighted_vertex_cover(), {n1, n2, n3, n4, n5})

    def test_weighted_dominating_set(self):
        self.assertSetEqual(self.t0.weighted_vertex_cover(), {n1, n2, n3, n5})
        self.assertSetEqual(self.t1.weighted_vertex_cover(), {n1, n2, n3, n4, n5})

    def test_unique_structure_hash(self):
        res = self.t0.unique_structure_hash()
        self.assertEqual(hash((1.0, frozenset())), res[n7])
        self.assertEqual(hash((2.0, frozenset())), res[n6])
        self.assertEqual(hash((4.0, frozenset())), res[n9])
        self.assertEqual(hash((5.0, frozenset())), res[n10])
        self.assertEqual(hash((6.0, frozenset())), res[n4])
        self.assertEqual(hash((6.0, frozenset())), res[n8])
        self.assertEqual(hash((8.0, frozenset())), res[n11])
        self.assertEqual(res[n3], hash((5.0, frozenset({res[n8]: 1, res[n9]: 1}.items()))))
        self.assertEqual(res[n5], hash((2.0, frozenset({res[n10]: 1, res[n11]: 1}.items()))))
        self.assertEqual(res[n2], hash((3.0, frozenset({res[n6]: 1, res[n7]: 1}.items()))))
        self.assertEqual(res[n1], hash((4.0, frozenset({res[n3]: 1, res[n4]: 1, res[n5]: 1}.items()))))
        self.assertEqual(res[n0], hash((7.0, frozenset({res[n1]: 1, res[n2]: 1}.items()))))
        res = self.t1.unique_structure_hash()
        self.assertEqual(hash((0, frozenset())), res[n3])
        self.assertSetEqual({hash((1, frozenset()))}, {res[n1], res[n2], res[n5]})
        self.assertEqual(hash((2, frozenset())), res[n4])
        self.assertEqual(hash((6.0, frozenset({res[n1]: 3, res[n3]: 1, res[n4]: 1}.items()))), res[n0])

    def test_isomorphic_bijection(self):
        t1 = WeightedTree((10, 6), {11: (1, []), 12: (1, []), 13: (0, []), 14: (2, []), 15: (1, [])})
        func = self.t1.isomorphic_bijection(t1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        t1.set_weight(10, 5)
        self.assertDictEqual(self.t1.isomorphic_bijection(t1), {})
        self.assertDictEqual(self.t1.isomorphic_bijection(Tree.copy(t1)), func)

    def test_equal(self):
        self.assertNotEqual(self.t0, self.t1)

    def test_str(self):
        self.assertEqual(str(self.t1),
                         "(0)->6.0\n ├──(1)->1.0\n ├──(2)->1.0\n ├──(3)->0.0\n ├──(4)->2.0\n └──(5)->1.0")

    def test_repr(self):
        inheritance = {k: (self.t1.weights(k), v) for k, v in self.t1.hierarchy().items()}
        inheritance.pop(self.t1.root)
        self.assertEqual(repr(self.t1), f"WeightedTree({(n0, 6.0)}, {inheritance})")


if __name__ == "__main__":
    main()
