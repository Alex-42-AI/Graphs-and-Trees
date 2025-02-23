from unittest import TestCase, main

from undirected_graph import *


def consecutive_1s(g, sort):
    if not sort:
        return False
    for i, u in enumerate(sort):
        j = -1
        for j, v in enumerate(sort[i:-1]):
            if u not in g.neighbors(sort[i + j + 1]) and u in {v, *g.neighbors(v)}:
                break
        for v in sort[i + j + 2:]:
            if u in g.neighbors(v):
                return False
    return True


n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = map(Node, range(16))


class TestUndirectedGraph(TestCase):
    def setUp(self):
        self.g0 = UndirectedGraph(
            {n0: [n1, n2], n2: [n1, n3, n4, n5], n3: [n6], n4: [n5, n6], n7: [n6, n8, n9], n8: [n5, n6],
             n11: [n10, n12, n13], n12: [n13], n14: []})
        self.g1 = UndirectedGraph({n1: [n2, n3, n4], n2: [n0, n5], n5: [n0, n3, n4]})
        self.g2 = UndirectedGraph({n1: [n0, n2, n3], n3: [n2, n4], n5: [n0, n4, n6], n0: [n2]})
        self.g3 = UndirectedGraph({n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11]})
        self.g4 = UndirectedGraph(
            {n1: [n0, n2, n3], n2: [n0, n3], n4: [n3, n5, n6, n7, n8, n9, n10], n5: [n6], n8: [n7, n9],
             n10: [n9, n11, n12, n13, n14], n13: [n14, n15]})
        self.g5 = UndirectedGraph(
            {n0: [n1, n2], n2: [n1, n3, n4], n4: [n3, n5], n5: [n6, n7, n8], n6: [n7, n8], n7: [n8, n9]})
        self.g6 = UndirectedGraph({n0: [n1, n2, n3, n4, n5]})
        self.g7 = UndirectedGraph(
            {n1: [n0, n2, n3, n4, n5, n6, n8], n2: [n0, n3, n4, n5, n6, n7, n8], n8: [n6, n7, n9, n10]})
        self.g8 = UndirectedGraph({n1: [n0, n2, n3, n4, n5, n7], n2: [n0, n3, n4, n5, n6, n7], n7: [n6, n8, n9]})

    def test_init(self):
        g = UndirectedGraph({0: {2, 4, 5, 6}, 1: {0, 2, 6}, 2: {0, 3, 5}})
        self.assertSetEqual(g.nodes, {n0, n1, n2, n3, n4, n5, n6})
        self.assertSetEqual(g.links,
                            {Link(0, 1), Link(0, 2), Link(0, 4), Link(0, 5), Link(0, 6), Link(1, 2), Link(1, 6),
                             Link(2, 3), Link(2, 5)})
        self.assertDictEqual(g.neighbors(),
                             {n0: {n1, n2, n4, n5, n6}, n1: {n0, n2, n6}, n2: {n0, n1, n3, n5}, n3: {n2}, n4: {n0},
                              n5: {n0, n2}, n6: {n0, n1}})

    def test_get_nodes(self):
        self.assertSetEqual(self.g0.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14})
        self.assertSetEqual(self.g1.nodes, {n0, n1, n2, n3, n4, n5})
        self.assertSetEqual(self.g2.nodes, {n0, n1, n2, n3, n4, n5, n6})
        self.assertSetEqual(self.g3.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11})
        self.assertSetEqual(self.g4.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15})
        self.assertSetEqual(self.g5.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9})
        self.assertSetEqual(self.g6.nodes, {n0, n1, n2, n3, n4, n5})
        self.assertSetEqual(self.g7.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10})
        self.assertSetEqual(self.g8.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9})

    def test_get_links(self):
        self.assertSetEqual(self.g0.links,
                            {Link(n0, n1), Link(n0, n2), Link(n1, n2), Link(n2, n3), Link(n2, n4), Link(n2, n5),
                             Link(n6, n3), Link(n4, n5), Link(n4, n6), Link(n8, n5), Link(n6, n8), Link(n6, n7),
                             Link(n7, n8), Link(n7, n9), Link(n10, n11), Link(n11, n13), Link(n11, n12),
                             Link(n12, n13)})
        self.assertSetEqual(self.g1.links,
                            {Link(n1, n2), Link(n1, n3), Link(n4, n1), Link(n2, n5), Link(n0, n5), Link(n4, n5),
                             Link(n3, n5), Link(n0, n2)})
        self.assertSetEqual(self.g2.links,
                            {Link(n0, n1), Link(n1, n2), Link(n1, n3), Link(n2, n3), Link(n3, n4), Link(n0, n5),
                             Link(n4, n5), Link(n0, n2), Link(n5, n6)})
        self.assertSetEqual(self.g3.links,
                            {Link(n0, n1), Link(n0, n2), Link(n1, n3), Link(n1, n4), Link(n1, n5), Link(n2, n6),
                             Link(n2, n7), Link(n3, n8), Link(n3, n9), Link(n5, n10), Link(n5, n11)})
        self.assertSetEqual(self.g4.links,
                            {Link(n0, n1), Link(n0, n2), Link(n1, n2), Link(n1, n3), Link(n2, n3), Link(n3, n4),
                             Link(n4, n5), Link(n4, n6), Link(n5, n6), Link(n4, n7), Link(n4, n8), Link(n4, n9),
                             Link(n7, n8), Link(n4, n10), Link(n8, n9), Link(n9, n10), Link(n10, n11), Link(n10, n12),
                             Link(n10, n13), Link(n10, n14), Link(n13, n14), Link(n13, n15)})
        self.assertSetEqual(self.g5.links,
                            {Link(n0, n1), Link(n0, n2), Link(n1, n2), Link(n2, n3), Link(n2, n4), Link(n3, n4),
                             Link(n4, n5), Link(n5, n6), Link(n5, n7), Link(n5, n8), Link(n6, n7), Link(n6, n8),
                             Link(n7, n8), Link(n7, n9)})
        self.assertSetEqual(self.g6.links, {Link(n0, n1), Link(n0, n2), Link(n0, n3), Link(n0, n4), Link(n0, n5)})
        self.assertSetEqual(self.g7.links,
                            {Link(n0, n1), Link(n1, n2), Link(n1, n3), Link(n1, n4), Link(n1, n5), Link(n1, n6),
                             Link(n1, n8), Link(n0, n2), Link(n2, n3), Link(n2, n4), Link(n2, n5), Link(n2, n6),
                             Link(n2, n7), Link(n2, n8), Link(n6, n8), Link(n7, n8), Link(n8, n9), Link(n8, n10)})
        self.assertSetEqual(self.g8.links,
                            {Link(n0, n1), Link(n1, n2), Link(n1, n3), Link(n1, n4), Link(n1, n5), Link(n1, n7),
                             Link(n0, n2), Link(n2, n3), Link(n2, n4), Link(n2, n5), Link(n2, n6), Link(n2, n7),
                             Link(n6, n7), Link(n7, n8), Link(n7, n9)})

    def test_get_degrees_sum(self):
        self.assertEqual(self.g0.degrees_sum, 36)
        self.assertEqual(self.g1.degrees_sum, 16)
        self.assertEqual(self.g2.degrees_sum, 18)
        self.assertEqual(self.g3.degrees_sum, 22)
        self.assertEqual(self.g4.degrees_sum, 44)
        self.assertEqual(self.g5.degrees_sum, 28)
        self.assertEqual(self.g6.degrees_sum, 10)
        self.assertEqual(self.g7.degrees_sum, 36)
        self.assertEqual(self.g8.degrees_sum, 30)

    def test_leaves(self):
        self.assertSetEqual(self.g0.leaves, {n9, n10})
        self.assertFalse(self.g1.leaves)
        self.assertSetEqual(self.g2.leaves, {n6})
        self.assertSetEqual(self.g3.leaves, {n4, n6, n7, n8, n9, n10, n11})
        self.assertSetEqual(self.g4.leaves, {n11, n12, n15})
        self.assertSetEqual(self.g5.leaves, {n9})
        self.assertSetEqual(self.g6.leaves, {n1, n2, n3, n4, n5})
        self.assertSetEqual(self.g7.leaves, {n9, n10})
        self.assertSetEqual(self.g8.leaves, {n8, n9})

    def test_get_neighbors(self):
        self.assertDictEqual(self.g6.neighbors(),
                             {n0: {n1, n2, n3, n4, n5}, n1: {n0}, n2: {n0}, n3: {n0}, n4: {n0}, n5: {n0}})

    def test_get_neighbors_missing_node(self):
        with self.assertRaises(KeyError):
            self.g0.neighbors(-2)

    def test_get_degrees(self):
        self.assertDictEqual(self.g0.degrees(),
                             {n0: 2, n1: 2, n2: 5, n3: 2, n4: 3, n5: 3, n6: 4, n7: 3, n8: 3, n9: 1, n10: 1, n11: 3,
                              n12: 2, n13: 2, n14: 0})

    def test_get_degrees_missing_node(self):
        with self.assertRaises(KeyError):
            self.g0.degrees(-2)

    def test_leaf(self):
        self.assertTrue(all(map(lambda x: self.g0.leaf(x) == (x in {n9, n10}), self.g0.nodes)))
        self.assertFalse(any(map(self.g1.leaf, self.g1.nodes)))

    def test_leaf_missing_node(self):
        with self.assertRaises(KeyError):
            self.g6.leaf(n6)

    def test_add_node(self):
        g0 = self.g0.copy().add(-1, 0, 1, 2)
        for n in {0, 1, 2}:
            self.assertIn(Node(-1), g0.neighbors(n))
        self.assertSetEqual(g0.neighbors(-1), {n0, n1, n2})
        self.assertSetEqual({Link(-1, 0), Link(-1, 1), Link(-1, 2)}.union(self.g0.links), g0.links)
        self.assertIn(Node(-1), g0.nodes)

    def test_add_already_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add(0, 3, 4))

    def test_remove(self):
        g0 = self.g0.copy().remove(n2, n6)
        self.assertSetEqual(g0.nodes, {n0, n1, n3, n4, n5, n7, n8, n9, n10, n11, n12, n13, n14})
        self.assertSetEqual(g0.links,
                            {Link(0, 1), Link(4, 5), Link(5, 8), Link(7, 8), Link(7, 9), Link(10, 11), Link(11, 12),
                             Link(11, 13), Link(12, 13)})
        self.assertDictEqual(g0.neighbors(),
                             {n0: {n1}, n1: {n0}, n3: set(), n4: {n5}, n5: {n4, n8}, n7: {n8, n9}, n8: {n5, n7},
                              n9: {n7}, n10: {n11}, n11: {n10, n12, n13}, n12: {n11, n13}, n13: {n11, n12},
                              n14: set()})

    def test_remove_missing_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-3))

    def test_connect(self):
        g0 = self.g0.copy().connect(3, 4, 9)
        self.assertSetEqual(g0.nodes, self.g0.nodes)
        self.assertTrue(self.g0.links.isdisjoint({Link(3, 4), Link(3, 9)}))
        self.assertSetEqual({Link(3, 4), Link(3, 9)}.union(self.g0.links), g0.links)
        self.assertSetEqual(g0.neighbors(3), {n2, n4, n6, n9})
        self.assertSetEqual(g0.neighbors(4), {n2, n3, n5, n6})
        self.assertSetEqual(g0.neighbors(9), {n3, n7})

    def test_connect_connected_nodes(self):
        g00, g01 = self.g0.copy(), self.g0.copy()
        self.assertEqual(g00.connect(3, 2, 4, 6, 8), g01.connect(3, 4, 8))

    def test_connect_all(self):
        g0 = self.g0.copy().connect_all(3, 4, 6, 8)
        self.assertTrue(g0.clique(3, 4, 6, 8))

    def test_disconnect(self):
        g0 = self.g0.copy().disconnect(n6, n7, n8)
        self.assertSetEqual(g0.nodes, self.g0.nodes)
        self.assertTrue(g0.links.isdisjoint({Link(6, 7), Link(6, 8)}))
        self.assertSetEqual(g0.links.union({Link(6, 7), Link(6, 8)}), self.g0.links)
        self.assertDictEqual(g0.neighbors(),
                             {n0: {n1, n2}, n1: {n0, n2}, n2: {n0, n1, n3, n4, n5}, n3: {n2, n6}, n4: {n2, n5, n6},
                              n5: {n2, n4, n8}, n6: {n3, n4}, n7: {n8, n9}, n8: {n5, n7}, n9: {n7}, n10: {n11},
                              n11: {n10, n12, n13}, n12: {n11, n13}, n13: {n11, n12}, n14: set()})

    def test_disconnect_disconnected_nodes(self):
        g00, g01 = self.g0.copy(), self.g0.copy()
        self.assertEqual(g00.disconnect(4, 2, 3, 6, 10), g01.disconnect(4, 2, 6))

    def test_disconnect_all(self):
        g0 = self.g0.copy().disconnect_all(4, 6, 7, 9)
        self.assertTrue(g0.complementary().clique(4, 6, 7, 9))

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())
        self.assertEqual(self.g4, self.g4.copy())
        self.assertEqual(self.g5, self.g5.copy())
        self.assertEqual(self.g6, self.g6.copy())
        self.assertEqual(self.g7, self.g7.copy())
        self.assertEqual(self.g8, self.g8.copy())

    def test_excentricity(self):
        self.assertEqual(self.g0.excentricity(0), 5)
        self.assertEqual(self.g0.excentricity(2), 4)
        self.assertEqual(self.g0.excentricity(3), 3)
        self.assertEqual(self.g0.excentricity(8), 3)
        self.assertEqual(self.g0.excentricity(11), 1)
        self.assertEqual(self.g0.excentricity(14), 0)

    def test_excentricity_missing_node(self):
        with self.assertRaises(KeyError):
            self.g0.excentricity(-1)

    def test_diameter(self):
        self.assertEqual(self.g0.diameter(), 5)
        self.assertEqual(self.g1.diameter(), 2)
        self.assertEqual(self.g2.diameter(), 3)
        self.assertEqual(self.g3.diameter(), 5)
        self.assertEqual(self.g4.diameter(), 6)
        self.assertEqual(self.g5.diameter(), 5)
        self.assertEqual(self.g6.diameter(), 2)
        self.assertEqual(self.g7.diameter(), 3)
        self.assertEqual(self.g8.diameter(), 3)

    def test_complementary(self):
        self.assertEqual(self.g0.complementary(), UndirectedGraph(
            {0: {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 1: {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
             2: {6, 7, 8, 9, 10, 11, 12, 13, 14}, 3: {4, 5, 7, 8, 9, 10, 11, 12, 13, 14},
             4: {7, 8, 9, 10, 11, 12, 13, 14}, 5: {6, 7, 9, 10, 11, 12, 13, 14}, 6: {9, 10, 11, 12, 13, 14},
             7: {10, 11, 12, 13, 14}, 8: {9, 10, 11, 12, 13, 14}, 9: {10, 11, 12, 13, 14}, 10: {12, 13, 14}, 11: {14},
             12: {14}, 13: {14}}))

    def test_connected(self):
        self.assertFalse(self.g0.connected())
        self.assertTrue(self.g1.connected())
        self.assertTrue(self.g2.connected())
        self.assertTrue(self.g3.connected())
        self.assertTrue(self.g4.connected())
        self.assertTrue(self.g5.connected())
        self.assertTrue(self.g6.connected())
        self.assertTrue(self.g7.connected())
        self.assertTrue(self.g8.connected())

    def test_is_tree(self):
        self.assertFalse(self.g0.is_tree())
        self.assertFalse(self.g1.is_tree())
        self.assertFalse(self.g2.is_tree())
        self.assertTrue(self.g3.is_tree())
        self.assertFalse(self.g4.is_tree())
        self.assertFalse(self.g5.is_tree())
        self.assertTrue(self.g6.is_tree())
        self.assertFalse(self.g7.is_tree())
        self.assertFalse(self.g8.is_tree())

    def test_tree_over_tree_graph(self):
        t = self.g3.tree(0)
        self.assertEqual(t.root, n0)
        self.assertSetEqual(t.descendants(n0), {n1, n2})
        self.assertSetEqual(t.descendants(n1), {n3, n4, n5})
        self.assertSetEqual(t.descendants(n2), {n6, n7})
        self.assertSetEqual(t.descendants(n3), {n8, n9})
        self.assertSetEqual(t.descendants(n5), {n10, n11})
        self.assertSetEqual(t.leaves, {n4, n6, n7, n8, n9, n10, n11})

    def test_tree_over_non_tree_graph_bfs(self):
        def bfs(u, g):
            result = UndirectedGraph({u: []})
            queue, total = [u], {u}
            while queue:
                v = queue.pop(0)
                new = g.neighbors(v) - total
                for n in new:
                    result.add(n, v)
                total.update(new)
                queue += new
            return result

        t0 = self.g0.tree()
        r0 = t0.root
        self.assertEqual(bfs(r0, self.g0).tree(), t0)
        t1 = self.g1.tree()
        r1 = t1.root
        self.assertEqual(bfs(r1, self.g1).tree(), t1)

    def test_tree_over_non_tree_graph_dfs(self):
        def dfs(u, g):
            result = UndirectedGraph({u: []})
            stack, total = [u], {u}
            while stack:
                v = stack.pop()
                new = g.neighbors(v) - total
                for n in new:
                    result.add(n, v)
                total.update(new)
                stack += new
            return result

        t0 = self.g0.tree(dfs=True)
        r0 = t0.root
        self.assertEqual(dfs(r0, self.g0).tree(dfs=True), t0)
        t1 = self.g1.tree(dfs=True)
        r1 = t1.root
        self.assertEqual(dfs(r1, self.g1).tree(dfs=True), t1)

    def test_reachable(self):
        for u in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
            for v in [n10, n11, n12, n13, n14]:
                self.assertFalse(self.g0.reachable(u, v))
            for v in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
                self.assertTrue(self.g0.reachable(u, v))
        for u in [n10, n11, n12, n13]:
            self.assertFalse(self.g0.reachable(n14, u))

    def test_reachable_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.reachable(-2, 3)

    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph({n3, n4, n5, n6, n8, n11, n12, n14}),
                         UndirectedGraph({3: {6}, 4: {5, 6}, 5: {8}, 6: {8}, 11: {12}, 14: set()}))

    def test_subgraph_missing_nodes(self):
        self.assertEqual(self.g0.subgraph({n3, n4, n5, n6, n8, n11, n12}),
                         self.g0.subgraph({Node(-1), Node(-2), n3, n4, n5, n6, n8, n11, n12}))

    def test_component(self):
        r0 = UndirectedGraph({0: [1, 2], 2: [1, 3, 4, 5], 3: [6], 4: [5, 6], 7: [6, 8, 9], 8: [5, 6]})
        r1 = UndirectedGraph({11: [10, 12, 13], 12: [13]})
        self.assertEqual(self.g0.component(0), r0)
        self.assertEqual(self.g0.component(10), r1)
        self.assertEqual(self.g0.component(14), UndirectedGraph({14: []}))
        self.assertEqual(self.g1.component(0), self.g1)
        self.assertEqual(self.g2.component(0), self.g2)
        self.assertEqual(self.g3.component(0), self.g3)
        self.assertEqual(self.g4.component(0), self.g4)
        self.assertEqual(self.g5.component(0), self.g5)
        self.assertEqual(self.g6.component(0), self.g6)
        self.assertEqual(self.g7.component(0), self.g7)
        self.assertEqual(self.g8.component(0), self.g8)

    def test_component_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.component(10)

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(UndirectedGraph({0: [1, 2], 2: [1, 3, 4, 5], 3: [6], 4: [5, 6], 7: [6, 8, 9], 8: [5, 6]}), res)
        self.assertIn(UndirectedGraph({11: [10, 12, 13], 12: [13]}), res)
        self.assertIn(UndirectedGraph({14: []}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        self.assertListEqual(self.g3.connection_components(), [self.g3])
        self.assertListEqual(self.g4.connection_components(), [self.g4])
        self.assertListEqual(self.g5.connection_components(), [self.g5])
        self.assertListEqual(self.g6.connection_components(), [self.g6])
        self.assertListEqual(self.g7.connection_components(), [self.g7])
        self.assertListEqual(self.g8.connection_components(), [self.g8])

    def test_cut_nodes(self):
        self.assertSetEqual(self.g0.cut_nodes(), {n2, n7, n11})
        self.assertSetEqual(self.g1.cut_nodes(), set())
        self.assertSetEqual(self.g2.cut_nodes(), {n5})
        self.assertSetEqual(self.g3.cut_nodes(), {n0, n1, n2, n3, n5})
        self.assertSetEqual(self.g4.cut_nodes(), {n3, n4, n10, n13})
        self.assertSetEqual(self.g5.cut_nodes(), {n2, n4, n5, n7})
        self.assertSetEqual(self.g6.cut_nodes(), {n0})
        self.assertSetEqual(self.g7.cut_nodes(), {n8})
        self.assertSetEqual(self.g8.cut_nodes(), {n7})

    def test_bridge_links(self):
        self.assertSetEqual(self.g0.bridge_links(), {Link(n7, n9), Link(n10, n11)})
        self.assertSetEqual(self.g1.bridge_links(), set())
        self.assertSetEqual(self.g2.bridge_links(), {Link(n5, n6)})
        self.assertSetEqual(self.g3.bridge_links(), self.g3.links)
        self.assertSetEqual(self.g4.bridge_links(), {Link(n3, n4), Link(n13, n15), Link(n10, n11), Link(n10, n12)})
        self.assertSetEqual(self.g5.bridge_links(), {Link(n4, n5), Link(n7, n9)})
        self.assertSetEqual(self.g6.bridge_links(), self.g6.links)
        self.assertSetEqual(self.g7.bridge_links(), {Link(n8, n9), Link(n8, n10)})
        self.assertSetEqual(self.g8.bridge_links(), {Link(n7, n8), Link(n7, n9)})

    def test_full(self):
        self.assertFalse(any(map(lambda g: g.full(),
                                 [self.g0, self.g1, self.g2, self.g3, self.g4, self.g5, self.g6, self.g7, self.g8])))
        g0 = self.g0.component(10).connect(10, 12, 13)
        g6 = self.g6.copy().connect_all(1, 2, 3, 4, 5)
        self.assertTrue(all(map(lambda g: g.full(), [g0, g6, self.g0.component(14)])))

    def test_get_shortest_path(self):
        self.assertListEqual(self.g0.get_shortest_path(0, 6), [n0, n2, n3, n6])
        self.assertListEqual(self.g0.get_shortest_path(7, 5), [n7, n8, n5])
        self.assertListEqual(self.g4.get_shortest_path(7, 15), [n7, n4, n10, n13, n15])

    def test_get_shortest_path_unreachable_nodes(self):
        self.assertListEqual(self.g0.get_shortest_path(3, 11), [])

    def test_get_shortest_path_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g2.get_shortest_path(n0, n7)

    def test_euler_tour_exist(self):
        g0 = self.g0.copy().connect(4, 9).connect(5, 7).connect(2, 8).disconnect(10, 11)
        self.assertFalse(g0.euler_tour_exists())
        self.assertTrue(g0.component(0).euler_tour_exists())
        self.assertTrue(g0.component(10).euler_tour_exists())
        self.assertTrue(g0.component(11).euler_tour_exists())
        self.assertTrue(g0.component(14).euler_tour_exists())

    def test_euler_walk_exists(self):
        g0 = self.g0.copy().connect(4, 9).connect(5, 7)
        self.assertFalse(g0.euler_walk_exists(2, 8))
        self.assertTrue(g0.component(0).euler_walk_exists(2, 8))

    def test_euler_walk_exists_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.euler_walk_exists(3, -4)

    def test_euler_tour(self):
        g0 = self.g0.component(0).connect(4, 9).connect(5, 7).connect(2, 8)
        res = g0.euler_tour()
        n = len(res)
        self.assertEqual(n, len(g0.links) + 1)
        self.assertEqual(res[0], res[-1])
        for i in range(n - 1):
            self.assertIn(res[i], g0.neighbors(res[i + 1]))

    def test_euler_walk(self):
        g0 = self.g0.component(n0).disconnect(4, 5).disconnect(7, 8)
        res = g0.euler_walk(2, 9)
        n = len(res)
        self.assertEqual(n, len(g0.links) + 1)
        for i in range(n - 1):
            self.assertIn(res[i], g0.neighbors(res[i + 1]))

    def test_euler_walk_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.euler_walk(3, -2)

    def test_links_graph(self):
        links_g1 = self.g1.links_graph()
        l02 = Node(Link(0, 2))
        l05 = Node(Link(0, 5))
        l12 = Node(Link(1, 2))
        l13 = Node(Link(1, 3))
        l14 = Node(Link(1, 4))
        l25 = Node(Link(2, 5))
        l35 = Node(Link(3, 5))
        l45 = Node(Link(4, 5))
        self.assertSetEqual(links_g1.nodes, {l02, l05, l12, l13, l14, l25, l35, l45})
        self.assertSetEqual(links_g1.links,
                            {Link(l02, l05), Link(l02, l25), Link(l02, l12), Link(l05, l25), Link(l12, l13),
                             Link(l12, l14), Link(l12, l25), Link(l13, l14), Link(l13, l35), Link(l14, l45),
                             Link(l25, l35), Link(l25, l45), Link(l35, l45), Link(l05, l35), Link(l05, l45)})

    def test_weighted_nodes_graph(self):
        g1 = self.g1.weighted_nodes_graph()
        self.assertDictEqual(g1.node_weights(), {n0: 0, n1: 0, n2: 0, n3: 0, n4: 0, n5: 0})
        g2 = self.g2.weighted_nodes_graph({n0: 7, n1: 6, n2: 2})
        self.assertDictEqual(g2.node_weights(), {n0: 7, n1: 6, n2: 2, n3: 0, n4: 0, n5: 0, n6: 0})

    def test_weighted_links_graph(self):
        g1 = self.g1.weighted_links_graph()
        self.assertDictEqual(g1.link_weights(), dict(zip(self.g1.links, [0] * 8)))
        g2 = self.g2.weighted_links_graph({Link(0, 1): 1, Link(0, 2): 2})
        link_weights = dict(zip(self.g2.links, [0] * 9))
        link_weights[Link(0, 1)] = 1
        link_weights[Link(0, 2)] = 2
        self.assertDictEqual(g2.link_weights(), link_weights)

    def test_weighted_graph(self):
        g1 = self.g1.weighted_graph()
        self.assertDictEqual(g1.node_weights(), {n0: 0, n1: 0, n2: 0, n3: 0, n4: 0, n5: 0})
        self.assertDictEqual(g1.link_weights(), dict(zip(self.g1.links, [0] * 8)))
        g2 = self.g2.weighted_graph({n0: 7, n1: 6, n2: 2}, {Link(0, 1): 1, Link(0, 2): 2})
        self.assertDictEqual(g2.node_weights(), {n0: 7, n1: 6, n2: 2, n3: 0, n4: 0, n5: 0, n6: 0})
        link_weights = dict(zip(self.g2.links, [0] * 9))
        link_weights[Link(0, 1)] = 1
        link_weights[Link(0, 2)] = 2
        self.assertDictEqual(g2.link_weights(), link_weights)

    def test_interval_sort(self):
        self.assertTrue(consecutive_1s(self.g4, self.g4.interval_sort()))
        self.assertTrue(consecutive_1s(self.g5, self.g5.interval_sort()))
        self.assertTrue(consecutive_1s(self.g6, self.g6.interval_sort()))
        self.assertTrue(consecutive_1s(self.g7, self.g7.interval_sort()))
        self.assertTrue(consecutive_1s(self.g8, self.g8.interval_sort()))

    def test_interval_sort_given_start(self):
        self.assertTrue(consecutive_1s(self.g4, self.g4.interval_sort(0)))
        self.assertTrue(consecutive_1s(self.g5, self.g5.interval_sort(1)))
        self.assertTrue(consecutive_1s(self.g8, self.g8.interval_sort(0)))

    def test_interval_sort_on_disconnected_graph(self):
        g0 = self.g0.copy()
        self.assertListEqual(g0.interval_sort(), [])
        g0.disconnect(2, 3), g0.disconnect(5, 8), g0.connect(3, 7, 8)
        self.assertTrue(consecutive_1s(g0, g0.interval_sort()))
        self.assertTrue(consecutive_1s(g0, g0.interval_sort(11)))

    def test_full_k_partite(self):
        self.assertTrue(all(map(lambda g: g.is_full_k_partite() == (g == self.g6),
                                [self.g0, self.g1, self.g2, self.g3, self.g4, self.g5, self.g6, self.g7, self.g8])))
        self.assertTrue(self.g6.is_full_k_partite(2))
        self.assertFalse(self.g6.is_full_k_partite(3))

    def test_full_k_partite_bad_k_type(self):
        with self.assertRaises(TypeError):
            self.g0.is_full_k_partite("3.0")

    def test_clique(self):
        self.assertTrue(self.g0.clique(n0, n1, n2))
        self.assertTrue(self.g0.clique(n7, n9))
        self.assertFalse(self.g0.clique(n10, n11, n12, n13))
        self.assertTrue(self.g0.clique(n6, n7, n8))

    def test_cliques(self):
        res = self.g0.cliques(3)
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n2, n4, n5}, res)
        self.assertIn({n6, n7, n8}, res)
        self.assertIn({n11, n12, n13}, res)
        self.assertEqual(len(res), 4)
        res = sorted(map(frozenset, self.g0.cliques(2)), key=hash)
        self.assertListEqual(res, sorted([frozenset({l.u, l.v}) for l in self.g0.links], key=hash))

    def test_cliques_bad_k_value(self):
        self.assertListEqual(self.g0.cliques(-3), [])
        self.assertListEqual(self.g0.cliques(0), [set()])

    def test_cliques_bad_k_type(self):
        with self.assertRaises(TypeError):
            self.g0.cliques("a")

    def test_max_cliques(self):
        res = self.g0.max_cliques()
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n2, n4, n5}, res)
        self.assertIn({n6, n7, n8}, res)
        self.assertIn({n11, n12, n13}, res)
        self.assertEqual(len(res), 4)
        self.assertListEqual(self.g1.max_cliques(), [{n0, n2, n5}])
        res = self.g2.max_cliques()
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n1, n2, n3}, res)
        self.assertEqual(len(res), 2)
        res = self.g4.max_cliques()
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n1, n2, n3}, res)
        self.assertIn({n4, n5, n6}, res)
        self.assertIn({n4, n7, n8}, res)
        self.assertIn({n4, n8, n9}, res)
        self.assertIn({n4, n9, n10}, res)
        self.assertIn({n10, n13, n14}, res)
        self.assertEqual(len(res), 7)
        self.assertListEqual(self.g5.max_cliques(), [{n5, n6, n7, n8}])
        self.assertListEqual(self.g7.max_cliques(), [{n1, n2, n6, n8}])
        res = self.g8.max_cliques()
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n1, n2, n3}, res)
        self.assertIn({n1, n2, n4}, res)
        self.assertIn({n1, n2, n5}, res)
        self.assertIn({n1, n2, n7}, res)
        self.assertIn({n2, n6, n7}, res)
        self.assertEqual(len(res), 6)

    def test_max_cliques_node(self):
        res = self.g2.max_cliques_node(2)
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n1, n2, n3}, res)
        self.assertEqual(len(res), 2)

    def test_max_cliques_node_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.max_cliques_node(-2)

    def test_all_maximal_cliques_node(self):
        res = self.g7.all_maximal_cliques_node(1)
        self.assertIn({n0, n1, n2}, res)
        self.assertIn({n1, n2, n3}, res)
        self.assertIn({n1, n2, n4}, res)
        self.assertIn({n1, n2, n5}, res)
        self.assertIn({n1, n2, n6, n8}, res)
        self.assertEqual(len(res), 5)

    def test_all_maximal_cliques_node_missing_node(self):
        with self.assertRaises(KeyError):
            self.g4.all_maximal_cliques_node(-2)

    def test_maximal_independent_sets(self):
        res = self.g1.maximal_independent_sets()
        self.assertIn({n0, n3, n4}, res)
        self.assertIn({n0, n1}, res)
        self.assertIn({n1, n5}, res)
        self.assertIn({n2, n3, n4}, res)
        self.assertEqual(len(res), 4)

    def test_cliques_graph(self):
        n012, n234, n45, n5678, n79 = Node(frozenset({n0, n1, n2})), Node(frozenset({n2, n3, n4})), Node(
            frozenset({n4, n5})), Node(frozenset({n5, n6, n7, n8})), Node(frozenset({n7, n9}))
        res = self.g5.cliques_graph()
        self.assertSetEqual(res.nodes, {n012, n234, n45, n5678, n79})
        self.assertSetEqual(res.links, {Link(n012, n234), Link(n234, n45), Link(n45, n5678), Link(n5678, n79)})

    def test_chromatic_nodes_partition(self):
        res = self.g0.chromatic_nodes_partition()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), 15)
        for subset in res:
            self.assertTrue(self.g0.complementary().clique(*subset))

    def test_chromatic_nodes_partition_on_full_k_partite_graph(self):
        res = self.g6.chromatic_nodes_partition()
        self.assertIn({n0}, res)
        self.assertIn({n1, n2, n3, n4, n5}, res)
        self.assertEqual(len(res), 2)

    def test_chromatic_nodes_partition_on_tree_graph(self):
        res = self.g3.chromatic_nodes_partition()
        self.assertIn({n0, n3, n4, n5, n6, n7}, res)
        self.assertIn({n1, n2, n8, n9, n10, n11}, res)
        self.assertEqual(len(res), 2)

    def test_chromatic_nodes_partition_on_interval_graph(self):
        res = self.g4.chromatic_nodes_partition()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), 16)
        for partition in res:
            self.assertTrue(self.g4.complementary().clique(*partition))

    def test_chromatic_links_partition(self):
        res = self.g1.chromatic_links_partition()
        self.assertEqual(len(res), 4)
        self.assertEqual(sum(map(len, res)), 8)
        for subset in res:
            for l0 in subset:
                for l1 in subset:
                    self.assertEqual(l0.u in l1, l0 == l1)
                    self.assertEqual(l0.v in l1, l0 == l1)
                for d in res:
                    self.assertEqual(l0 in d, subset == d)

    def test_vertex_cover(self):
        res = self.g1.vertex_cover()
        self.assertEqual(len(res), 3)
        for l in self.g1.links:
            self.assertTrue({l.u, l.v}.intersection(res))

    def test_vertex_cover_on_full_k_partite_graph(self):
        self.assertSetEqual(self.g6.vertex_cover(), {n0})

    def test_vertex_cover_on_tree_graph(self):
        self.assertSetEqual(self.g3.vertex_cover(), {n1, n2, n3, n5})

    def test_vertex_cover_on_interval_graph(self):
        res = self.g4.vertex_cover()
        self.assertEqual(len(res), 7)
        for l in self.g4.links:
            self.assertTrue({l.u, l.v}.intersection(res))

    def test_vertex_cover_on_disconnected_graph(self):
        res = self.g0.vertex_cover()
        self.assertEqual(len(res), 7)
        for l in self.g0.links:
            self.assertTrue({l.u, l.v}.intersection(res))

    def test_dominating_set(self):
        res = self.g1.dominating_set()
        self.assertEqual(len(res), 2)
        for u in self.g1.nodes:
            self.assertTrue(u in res or any(v in res for v in self.g1.neighbors(u)))
        res = self.g2.dominating_set()
        self.assertEqual(len(res), 2)
        for u in self.g2.nodes:
            self.assertTrue(u in res or any(v in res for v in self.g2.neighbors(u)))

    def test_dominating_set_on_full_k_partite_graph(self):
        self.assertSetEqual(self.g6.dominating_set(), {n0})

    def test_dominating_set_on_tree_graph(self):
        self.assertSetEqual(self.g3.dominating_set(), {n1, n2, n3, n5})

    def test_dominating_set_on_disconnected_graph(self):
        self.assertSetEqual(self.g0.dominating_set(), {n2, n7, n11, n14})

    def test_cycle_with_length(self):
        res = self.g0.cycle_with_length(6)
        self.assertEqual(len(res), 7)
        for i in range(6):
            self.assertIn(res[i], self.g0.neighbors(res[i + 1]))
        self.assertListEqual(self.g1.cycle_with_length(6), [])

    def test_cycle_with_length_bad_length_value(self):
        self.assertListEqual(self.g1.cycle_with_length(2), [])
        self.assertListEqual(self.g1.cycle_with_length(-2), [])

    def test_cycle_with_length_bad_length_type(self):
        with self.assertRaises(TypeError):
            self.g0.cycle_with_length("6.")

    def test_path_with_length(self):
        res = self.g0.path_with_length(0, 5, 7)
        self.assertEqual(len(res), 8)
        self.assertEqual((res[0], res[-1]), (n0, n5))
        for i in range(6):
            self.assertIn(res[i], self.g0.neighbors(res[i + 1]))

    def test_path_with_length_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.path_with_length(0, -5, 7)

    def test_path_with_length_bad_length_value(self):
        self.assertListEqual(self.g0.path_with_length(0, 3, 1), [])
        self.assertListEqual(self.g0.path_with_length(0, 3, -1), [])

    def test_path_with_length_bad_length_type(self):
        with self.assertRaises(TypeError):
            self.g0.path_with_length(0, 4, "6.")

    def test_hamilton_tour_exist(self):
        self.assertFalse(self.g0.hamilton_tour_exists())
        g0 = self.g0.component(n0)
        self.assertFalse(g0.hamilton_tour_exists())
        g0.connect(n0, n9)
        self.assertTrue(g0.hamilton_tour_exists())

    def test_hamilton_walk_exist(self):
        g0 = self.g0.component(0)
        self.assertFalse(g0.hamilton_walk_exists(2, 9))
        self.assertTrue(g0.hamilton_walk_exists(0, 9))
        self.assertTrue(g0.hamilton_walk_exists(1, 9))

    def test_hamilton_walk_exist_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.hamilton_walk_exists(2, -1)

    def test_hamilton_tour(self):
        g0 = self.g0.component(n0).disconnect(n4, n5).connect(n4, n5).connect(n0, n9)
        res = g0.hamilton_tour()
        self.assertEqual(len(res), len(g0.nodes) + 1)
        for i in range(len(g0.nodes)):
            self.assertIn(res[i], g0.neighbors(res[i + 1]))

    def test_hamilton_walk(self):
        g0 = self.g0.component(n0)
        res = g0.hamilton_walk()
        self.assertEqual(len(res), len(g0.nodes))
        self.assertIn(n9, {res[0], res[-1]})
        for i in range(len(res) - 1):
            self.assertIn(res[i], g0.neighbors(res[i + 1]))
        g0.disconnect(n4, n5)
        self.assertListEqual(g0.hamilton_walk(), [])

    def test_hamilton_walk_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.hamilton_walk(2, -1)

    def test_isomorphic_bijection(self):
        g1 = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})
        func = self.g1.isomorphic_bijection(g1)
        self.assertDictEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        g1.disconnect(n11, n13)
        self.assertDictEqual(self.g1.isomorphic_bijection(g1), {})

    def test_bool(self):
        g = self.g0.component(14)
        self.assertTrue(g)
        g.remove(14)
        self.assertFalse(g)

    def test_contains(self):
        self.assertTrue(13 in self.g0)
        self.assertFalse(15 in self.g0)
        self.assertFalse(Link(1, 2) in self.g0)

    def test_add(self):
        tmp = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})
        expected = UndirectedGraph(
            {n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11], n11: [n12, n13, n14],
             n10: [n12, n15], n15: [n12, n13, n14]})
        self.assertEqual(self.g3 + tmp, expected)

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n5, n2, n3).disconnect(n1, n4).connect(n3, n2, n4)
        self.assertEqual(g1.connect(n0, n1), self.g2.copy().remove(n6))

    def test_str(self):
        self.assertEqual(str(self.g1), f"<{self.g1.nodes}, {self.g1.links}>")

    def test_repr(self):
        self.assertEqual(repr(self.g0), str(self.g0))


class TestWeightedNodesUndirectedGraph(TestCase):
    def setUp(self):
        self.g0 = WeightedNodesUndirectedGraph(
            {n0: (7, [n1, n2]), n1: (3, []), n2: (5, [n1, n3, n4, n5]), n3: (2, [n6]), n4: (8, [n5, n6]), n5: (4, []),
             n6: (6, []), n7: (2, [n6, n8, n9]), n8: (0, [n5, n6]), n9: (5, []), n10: (4, []),
             n11: (2, [n10, n12, n13]), n12: (1, [n13]), n13: (3, []), n14: (6, [])})
        self.g1 = WeightedNodesUndirectedGraph(
            {n0: (3, []), n1: (2, [n2, n3, n4]), n2: (4, [n0, n5]), n3: (6, []), n4: (5, []), n5: (1, [n0, n3, n4])})
        self.g2 = WeightedNodesUndirectedGraph(
            {n0: (7, [n2]), n1: (6, [n0, n2, n3]), n2: (2, []), n3: (4, [n2, n4]), n4: (3, []), n5: (5, [n0, n4, n6]),
             n6: (4, [])})
        self.g3 = WeightedNodesUndirectedGraph(
            {n0: (7, []), n1: (4, [n0, n3, n4, n5]), n2: (3, [n0, n6, n7]), n3: (5, [n8, n9]), n4: (6, []),
             n5: (2, [n10, n11]), n6: (2, []), n7: (1, []), n8: (6, []), n9: (4, []), n10: (5, []), n11: (8, [])})

    def test_init(self):
        g = WeightedNodesUndirectedGraph({0: (3, {2, 3}), 1: (2, {2, 3}), 2: (4, [])})
        self.assertDictEqual(g.node_weights(), {n0: 3, n1: 2, n2: 4, n3: 0})
        self.assertSetEqual(g.links, {Link(0, 2), Link(0, 3), Link(1, 2), Link(1, 3)})

    def test_node_weights(self):
        self.assertEqual(self.g0.node_weights(),
                         {n0: 7, n1: 3, n2: 5, n3: 2, n4: 8, n5: 4, n6: 6, n7: 2, n8: 0, n9: 5, n10: 4, n11: 2, n12: 1,
                          n13: 3, n14: 6})

    def test_node_weights_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.node_weights(n9)

    def test_total_node_weights(self):
        self.assertEqual(self.g0.total_nodes_weight, 58)

    def test_add_node(self):
        g0 = self.g0.copy().add((-1, 3), 2, 3, 5)
        self.assertSetEqual(g0.nodes, {Node(-1), n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14})
        self.assertEqual(g0.node_weights(-1), 3)
        self.assertSetEqual(g0.neighbors(-1), {n2, n3, n5})

    def test_add_already_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add((0, 3), 3))

    def test_remove(self):
        g0 = self.g0.copy().remove(2, 6, 11, 14)
        self.assertEqual(g0, WeightedNodesUndirectedGraph(
            {0: (7, {1}), 1: (3, []), 3: (2, []), 4: (8, {5}), 5: (4, {8}), 7: (2, {8, 9}), 9: (5, []), 10: (4, []),
             12: (1, {13}), 13: (3, {})}))

    def test_remove_missing_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-1, 15))

    def test_set_weight(self):
        g0 = self.g0.copy().set_weight(4, 6)
        self.assertEqual(g0.node_weights(4), 6)

    def test_set_weight_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight(-4, 0))

    def test_set_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.set_weight(4, "5-")

    def test_increase_weight(self):
        g0 = self.g0.copy().increase_weight(4, 2)
        self.assertEqual(g0.node_weights(4), 10)

    def test_increase_weight_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight(-4, 3))

    def test_increase_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.increase_weight(4, "5-")

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_complementary(self):
        self.assertEqual(self.g1.complementary(), WeightedNodesUndirectedGraph(
            {0: (3, {1, 3, 4}), 1: (2, {5}), 2: (4, {3, 4}), 3: (6, {4}), 4: (5, []), 5: (1, [])}))

    def test_weighted_tree_over_tree_graph(self):
        t = self.g3.weighted_tree(0)
        r = t.root
        self.assertEqual(r, n0)
        self.assertSetEqual(t.descendants(1), {n3, n4, n5})
        self.assertSetEqual(t.descendants(2), {n6, n7})
        self.assertSetEqual(t.descendants(3), {n8, n9})
        self.assertSetEqual(t.descendants(5), {n10, n11})
        self.assertSetEqual(t.leaves, {n4, n6, n7, n8, n9, n10, n11})

    def test_weighted_tree_over_non_tree_graph_bfs(self):
        def bfs(u, g):
            result = WeightedNodesUndirectedGraph({u: (g.node_weights(u), [])})
            queue, total = [u], {u}
            while queue:
                v = queue.pop(0)
                new = g.neighbors(v) - total
                for n in new:
                    result.add((n, g.node_weights(n)), v)
                total.update(new)
                queue += new
            return result

        t0 = self.g0.weighted_tree()
        r0 = t0.root
        self.assertEqual(bfs(r0, self.g0).weighted_tree(), t0)
        t1 = self.g1.weighted_tree()
        r1 = t1.root
        self.assertEqual(bfs(r1, self.g1).weighted_tree(), t1)

    def test_weighted_tree_over_non_tree_graph_dfs(self):
        def dfs(u, g):
            result = WeightedNodesUndirectedGraph({u: (g.node_weights(u), [])})
            stack, total = [u], {u}
            while stack:
                v = stack.pop()
                new = g.neighbors(v) - total
                for n in new:
                    result.add((n, g.node_weights(n)), v)
                total.update(new)
                stack += new
            return result

        t0 = self.g0.weighted_tree(dfs=True)
        r0 = t0.root
        self.assertEqual(dfs(r0, self.g0).weighted_tree(dfs=True), t0)
        t1 = self.g1.weighted_tree(dfs=True)
        r1 = t1.root
        self.assertEqual(dfs(r1, self.g1).weighted_tree(dfs=True), t1)

    def test_subgraph(self):
        self.assertEqual(self.g1.subgraph({n1, n2, n4, n5}),
                         WeightedNodesUndirectedGraph({1: (2, {4}), 4: (5, {5}), 5: (1, {2}), 2: (4, {1})}))

    def test_subgraph_missing_nodes(self):
        self.assertEqual(self.g0.subgraph({n3, n4, n5, n6, n8, n11, n12}),
                         self.g0.subgraph({Node(-1), Node(-2), n3, n4, n5, n6, n8, n11, n12}))

    def test_component(self):
        r0 = WeightedNodesUndirectedGraph(
            {0: (7, {1, 2}), 1: (3, []), 2: (5, {1, 3, 4, 5}), 3: (2, {6}), 4: (8, {5, 6}), 5: (4, {8}), 6: (6, {8}),
             7: (2, {6, 8, 9}), 9: (5, [])})
        r1 = WeightedNodesUndirectedGraph({10: (4, []), 11: (2, {10, 12, 13}), 12: (1, {13}), 13: (3, [])})
        self.assertEqual(self.g0.component(0), r0)
        self.assertEqual(self.g0.component(10), r1)
        self.assertEqual(self.g0.component(14), WeightedNodesUndirectedGraph({14: (6, [])}))
        self.assertEqual(self.g1.component(0), self.g1)
        self.assertEqual(self.g2.component(0), self.g2)
        self.assertEqual(self.g3.component(0), self.g3)

    def test_component_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.component(10)

    def test_links_graph(self):
        l02 = Node(Link(0, 2))
        l05 = Node(Link(0, 5))
        l12 = Node(Link(1, 2))
        l13 = Node(Link(1, 3))
        l14 = Node(Link(1, 4))
        l25 = Node(Link(2, 5))
        l35 = Node(Link(3, 5))
        l45 = Node(Link(4, 5))
        self.assertEqual(self.g1.links_graph(), WeightedLinksUndirectedGraph(
            {l02: {l05: 3, l25: 4, l12: 4}, l25: {l05: 1, l12: 4, l35: 1, l45: 1}, l12: {l14: 2},
             l13: {l12: 2, l14: 2, l35: 6}, l35: {l05: 1, l45: 1}, l05: {l45: 1}, l14: {l45: 5}}))

    def test_cliques_graph(self):
        n012, n123, n05, n34, n45, n56 = Node(frozenset({n0, n1, n2})), Node(frozenset({n1, n2, n3})), Node(
            frozenset({n0, n5})), Node(frozenset({n3, n4})), Node(frozenset({n4, n5})), Node(frozenset({n5, n6}))
        res = self.g2.cliques_graph()
        self.assertDictEqual(res.node_weights(), {n012: 15, n123: 12, n05: 12, n34: 7, n45: 8, n56: 9})
        self.assertDictEqual(res.link_weights(),
                             {Link(n012, n123): 8, Link(n123, n34): 4, Link(n05, n012): 7, Link(n34, n45): 3,
                              Link(n05, n45): 5, Link(n05, n56): 5, Link(n45, n56): 5})

    def test_weighted_graph(self):
        g1 = self.g1.weighted_graph()
        self.assertDictEqual(self.g1.node_weights(), g1.node_weights())
        self.assertDictEqual(g1.link_weights(), dict(zip(self.g1.links, [0] * 8)))
        g2 = self.g2.weighted_graph({Link(0, 1): 1, Link(0, 2): 2})
        self.assertDictEqual(self.g2.node_weights(), g2.node_weights())
        link_weights = dict(zip(self.g2.links, [0] * 9))
        link_weights[Link(0, 1)] = 1
        link_weights[Link(0, 2)] = 2
        self.assertDictEqual(g2.link_weights(), link_weights)

    def test_minimal_path_nodes(self):
        res = self.g0.minimal_path_nodes(2, 7)
        self.assertEqual(sum(map(self.g0.node_weights, res)), 11)
        self.assertListEqual(res, [n2, n5, n8, n7])

    def test_minimal_path_nodes_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.minimal_path_nodes(2, -1)

    def test_weighted_vertex_cover(self):
        res = (g0 := self.g0.component(0)).weighted_vertex_cover()
        self.assertEqual(sum(map(g0.node_weights, res)), 20)
        for l in g0.links:
            self.assertTrue({l.u, l.v}.intersection(res))

    def test_weighted_vertex_cover_on_tree_graph(self):
        self.assertSetEqual(self.g3.weighted_vertex_cover(), {n1, n2, n3, n5})

    def test_weighted_vertex_cover_on_disconnected_graph(self):
        res = self.g0.weighted_vertex_cover()
        self.assertEqual(sum(map(self.g0.node_weights, res)), 23)
        for l in self.g0.links:
            self.assertTrue({l.u, l.v}.intersection(res))

    def test_weighted_dominating_set(self):
        self.assertSetEqual(self.g1.weighted_dominating_set(), {n1, n5})

    def test_weighted_dominating_set_on_tree_graph(self):
        self.assertEqual(sum(map(self.g3.node_weights, self.g3.weighted_dominating_set())), 14)

    def test_weighted_dominating_set_on_disconnected_graph(self):
        res = self.g0.weighted_dominating_set()
        for u in self.g0.nodes:
            self.assertTrue(u in res or not self.g0.neighbors(u).isdisjoint(res))
        self.assertEqual(sum(map(self.g0.node_weights, res)), 15)

    def test_isomorphic_bijection(self):
        g1 = WeightedNodesUndirectedGraph({10: (3, [12, 15]), 11: (2, [12, 13, 14]), 12: (4, []),
                                           13: (6, []), 14: (5, []), 15: (2, [12, 13, 14])})
        self.assertFalse(self.g1.isomorphic_bijection(g1))
        g1.set_weight(15, 1)
        func = self.g1.isomorphic_bijection(g1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        g1 = UndirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1, g2 = self.g1.copy().connect(0, 1).disconnect(1, 4).connect(3, 2, 4), self.g2.copy()
        self.assertNotEqual(g1.disconnect(2, 5), g2.remove(6).connect(3, 5))
        g2.set_weight(n0, 3), g2.set_weight(n1, 2), g2.set_weight(n2, 4)
        g2.set_weight(n3, 6), g2.set_weight(n4, 5)
        self.assertEqual(g1, g2.set_weight(n5, 1))

    def test_add(self):
        self.assertEqual(self.g3 + WeightedNodesUndirectedGraph(
            {10: (3, [12, 15]), 11: (2, [12, 13, 14]), 12: (4, []), 13: (6, []), 14: (5, []), 15: (1, [12, 13, 14])}),
                         WeightedNodesUndirectedGraph(
                             {0: (7, []), 1: (4, [0, 3, 4, 5]), 2: (3, [0, 6, 7]), 3: (5, [8, 9]), 4: (6, []),
                              5: (2, []), 6: (2, []), 7: (1, []), 8: (6, []), 9: (4, []), 10: (8, [5, 12, 15]),
                              11: (10, [5, 12, 13, 14]), 12: (4, []), 13: (6, []), 14: (5, []),
                              15: (1, [12, 13, 14])}))
        self.assertEqual(self.g3 + UndirectedGraph({11: [12, 13, 14], 12: [10, 15], 15: [10, 13, 14]}),
                         WeightedNodesUndirectedGraph(
                             {0: (7, []), 1: (4, [0, 3, 4, 5]), 2: (3, [0, 6, 7]), 3: (5, [8, 9]), 4: (6, []),
                              5: (2, [10, 11]), 6: (2, []), 7: (1, []), 8: (6, []), 9: (4, []), 10: (5, []),
                              11: (8, [12, 13, 14]), 12: (0, [10, 15]), 15: (0, [10, 13, 14])}))

    def test_str(self):
        self.assertEqual(str(self.g1), "<{" + ", ".join(
            f"{n} -> {self.g1.node_weights(n)}" for n in self.g1.nodes) + "}, " + str(self.g1.links) + ">")

    def test_repr(self):
        self.assertEqual(repr(self.g0), str(self.g0))


class TestWeightedLinksUndirectedGraph(TestCase):
    def setUp(self):
        self.g0 = WeightedLinksUndirectedGraph(
            {n0: {n1: 3, n2: 1}, n2: {n1: -4, n3: 6, n4: 2, n5: 4}, n3: {n6: 3}, n4: {n5: 3, n6: 7},
             n7: {n6: 2, n8: 1, n9: 3}, n8: {n5: 5, n6: 4}, n11: {n10: 2, n12: 3, n13: 4}, n12: {n13: 1}, n14: {}})
        self.g1 = WeightedLinksUndirectedGraph(
            {n1: {n2: 5, n3: 2, n4: 4}, n2: {n0: 2, n5: 1}, n5: {n0: 4, n3: 3, n4: 2}})
        self.g2 = WeightedLinksUndirectedGraph(
            {n0: {n2: 2}, n1: {n0: 1, n2: -4, n3: -6}, n3: {n2: 1, n4: 2}, n5: {n0: 3, n4: 4, n6: 5}})
        self.g3 = WeightedLinksUndirectedGraph(
            {n1: {n0: 2, n3: 4, n4: 3, n5: -1}, n2: {n0: 1, n6: 5, n7: 3}, n3: {n8: 6, n9: 2}, n5: {n10: 4, n11: 1}})

    def test_init(self):
        g = WeightedLinksUndirectedGraph({0: {1: 4, 2: 3}, 1: {0: 3, 3: 1}})
        self.assertSetEqual(g.nodes, {n0, n1, n2, n3})
        self.assertDictEqual(g.link_weights(), {Link(0, 1): 4, Link(0, 2): 3, Link(1, 3): 1})
        self.assertDictEqual(g.neighbors(), {n0: {n1, n2}, n1: {n0, n3}, n2: {n0}, n3: {n1}})

    def test_link_weights(self):
        self.assertDictEqual(self.g0.link_weights(),
                             {Link(n0, n1): 3, Link(n0, n2): 1, Link(n1, n2): -4, Link(n3, n2): 6, Link(n4, n2): 2,
                              Link(n5, n2): 4, Link(n3, n6): 3, Link(n4, n6): 7, Link(n4, n5): 3, Link(n5, n8): 5,
                              Link(n6, n7): 2, Link(n6, n8): 4, Link(n7, n8): 1, Link(n7, n9): 3, Link(n10, n11): 2,
                              Link(n11, n12): 3, Link(n11, n13): 4, Link(n13, n12): 1})
        self.assertDictEqual(self.g0.link_weights(n0), {n1: 3, n2: 1})
        self.assertDictEqual(self.g0.link_weights(n1), {n0: 3, n2: -4})
        self.assertDictEqual(self.g0.link_weights(n2), {n0: 1, n1: -4, n3: 6, n4: 2, n5: 4})
        self.assertDictEqual(self.g0.link_weights(n3), {n2: 6, n6: 3})
        self.assertDictEqual(self.g0.link_weights(n4), {n2: 2, n5: 3, n6: 7})
        self.assertDictEqual(self.g0.link_weights(n5), {n2: 4, n4: 3, n8: 5})
        self.assertDictEqual(self.g0.link_weights(n6), {n3: 3, n4: 7, n7: 2, n8: 4})
        self.assertDictEqual(self.g0.link_weights(n7), {n6: 2, n8: 1, n9: 3})
        self.assertDictEqual(self.g0.link_weights(n8), {n5: 5, n6: 4, n7: 1})
        self.assertDictEqual(self.g0.link_weights(n9), {n7: 3})
        self.assertDictEqual(self.g0.link_weights(n10), {n11: 2})
        self.assertDictEqual(self.g0.link_weights(n11), {n10: 2, n12: 3, n13: 4})
        self.assertDictEqual(self.g0.link_weights(n12), {n11: 3, n13: 1})
        self.assertDictEqual(self.g0.link_weights(n13), {n11: 4, n12: 1})
        self.assertDictEqual(self.g0.link_weights(n14), {})
        self.assertEqual(self.g0.link_weights(Link(n2, n5)), 4)

    def test_link_weights_missing_link(self):
        with self.assertRaises(KeyError):
            self.g0.link_weights(0, 4)

    def test_total_link_weights(self):
        self.assertEqual(self.g0.total_links_weight, 50)
        self.assertEqual(self.g1.total_links_weight, 23)
        self.assertEqual(self.g2.total_links_weight, 8)
        self.assertEqual(self.g3.total_links_weight, 30)

    def test_add_node(self):
        g0 = self.g0.copy().add(-1, {0: 2, 1: 2, 2: 1})
        for n in {0, 1, 2}:
            self.assertIn(Node(-1), g0.neighbors(n))
        self.assertSetEqual(g0.neighbors(-1), {n0, n1, n2})
        link_weights = self.g0.link_weights()
        link_weights[Link(0, -1)] = 2
        link_weights[Link(1, -1)] = 2
        link_weights[Link(2, -1)] = 1
        self.assertDictEqual(g0.link_weights(), link_weights)
        self.assertSetEqual(g0.nodes, self.g0.nodes.union({Node(-1)}))

    def test_add_already_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add(0, {n3: 2, n4: 1}))

    def test_remove(self):
        g0 = self.g0.copy().remove(n2, n6)
        self.assertSetEqual(g0.nodes, {n0, n1, n3, n4, n5, n7, n8, n9, n10, n11, n12, n13, n14})
        link_weights = self.g0.link_weights()
        link_weights.pop(Link(0, 2)), link_weights.pop(Link(1, 2)), link_weights.pop(Link(2, 3))
        link_weights.pop(Link(2, 4)), link_weights.pop(Link(2, 5)), link_weights.pop(Link(3, 6))
        link_weights.pop(Link(4, 6)), link_weights.pop(Link(6, 7)), link_weights.pop(Link(6, 8))
        self.assertDictEqual(g0.link_weights(), link_weights)
        self.assertDictEqual(g0.neighbors(),
                             {n0: {n1}, n1: {n0}, n3: set(), n4: {n5}, n5: {n4, n8}, n7: {n8, n9}, n8: {n5, n7},
                              n9: {n7}, n10: {n11}, n11: {n10, n12, n13}, n12: {n11, n13}, n13: {n11, n12},
                              n14: set()})

    def test_remove_missing_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-3))

    def test_connect(self):
        g0 = self.g0.copy().connect(9, {10: 3, 14: 2})
        self.assertSetEqual(g0.nodes, self.g0.nodes)
        res_link_weights = self.g0.link_weights()
        res_link_weights[Link(9, 10)] = 3
        res_link_weights[Link(9, 14)] = 2
        self.assertDictEqual(g0.link_weights(), res_link_weights)
        self.assertSetEqual(g0.neighbors(9), {n7, n10, n14})
        self.assertSetEqual(g0.neighbors(10), {n9, n11})
        self.assertSetEqual(g0.neighbors(14), {n9})

    def test_connect_connected_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().connect(0, {2: 3}))

    def test_connect_all(self):
        g0 = self.g0.copy().connect_all(3, 4, 6, 8)
        self.assertTrue(g0.clique(3, 4, 6, 8))
        self.assertEqual(
            list(map(g0.link_weights, [Link(3, 4), Link(3, 6), Link(4, 6), Link(3, 8), Link(4, 8), Link(6, 8)])),
            [0, 3, 7, 0, 0, 4])

    def test_disconnect(self):
        g0 = self.g0.copy().disconnect(7, 9)
        self.assertSetEqual(g0.nodes, self.g0.nodes)
        self.assertSetEqual(g0.links, self.g0.links - {Link(7, 9)})
        link_weights = self.g0.link_weights()
        link_weights.pop(Link(7, 9))
        self.assertDictEqual(g0.link_weights(), link_weights)
        self.assertSetEqual(g0.neighbors(7), {n6, n8})
        self.assertSetEqual(g0.neighbors(9), set())

    def test_disconnect_disconnected_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().disconnect(6, 9, 10))

    def test_disconnect_all(self):
        g0 = self.g0.copy().disconnect_all(4, 6, 7, 9)
        self.assertTrue(g0.complementary().clique(4, 6, 7, 9))

    def test_set_weight(self):
        g0 = self.g0.copy().set_weight(Link(2, 4), 3)
        res_link_weights = self.g0.link_weights()
        res_link_weights[Link(2, 4)] = 3
        self.assertDictEqual(g0.link_weights(), res_link_weights)

    def test_set_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight(Link(2, 6), 3))

    def test_set_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().set_weight(Link(2, 3), [3])

    def test_increase_weight(self):
        g0 = self.g0.copy().increase_weight(Link(2, 4), 1)
        res_link_weights = self.g0.link_weights()
        res_link_weights[Link(2, 4)] = 3
        self.assertDictEqual(g0.link_weights(), res_link_weights)

    def test_increase_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().increase_weight(Link(2, 6), 1))

    def test_increase_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().increase_weight(Link(2, 3), [3])

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_weighted_graph(self):
        g1 = self.g1.weighted_graph()
        self.assertDictEqual(g1.node_weights(), {n0: 0, n1: 0, n2: 0, n3: 0, n4: 0, n5: 0})
        self.assertDictEqual(self.g1.link_weights(), g1.link_weights())
        g2 = self.g2.weighted_graph({n0: 7, n1: 6, n2: 2})
        self.assertDictEqual(g2.node_weights(), {n0: 7, n1: 6, n2: 2, n3: 0, n4: 0, n5: 0, n6: 0})
        self.assertDictEqual(self.g2.link_weights(), g2.link_weights())

    def test_component(self):
        r0 = WeightedLinksUndirectedGraph(
            {0: {1: 3, 2: 1}, 2: {1: -4, 3: 6, 4: 2, 5: 4}, 3: {6: 3}, 4: {5: 3, 6: 7}, 7: {6: 2, 8: 1, 9: 3},
             8: {5: 5, 6: 4}})
        r1 = WeightedLinksUndirectedGraph({11: {10: 2, 12: 3, 13: 4}, 12: {13: 1}})
        self.assertEqual(self.g0.component(0), r0)
        self.assertEqual(self.g0.component(10), r1)
        self.assertEqual(self.g0.component(14), WeightedLinksUndirectedGraph({14: {}}))
        self.assertEqual(self.g1.component(0), self.g1)
        self.assertEqual(self.g2.component(0), self.g2)
        self.assertEqual(self.g3.component(0), self.g3)

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedLinksUndirectedGraph(
            {0: {1: 3, 2: 1}, 2: {1: -4, 3: 6, 4: 2, 5: 4}, 3: {6: 3}, 4: {5: 3, 6: 7}, 7: {6: 2, 8: 1, 9: 3},
             8: {5: 5, 6: 4}}), res)
        self.assertIn(WeightedLinksUndirectedGraph({11: {10: 2, 12: 3, 13: 4}, 12: {13: 1}}), res)
        self.assertIn(WeightedLinksUndirectedGraph({14: {}}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        self.assertListEqual(self.g3.connection_components(), [self.g3])

    def test_subgraph(self):
        self.assertEqual(self.g2.subgraph({n0, n1, n2, n3}),
                         WeightedLinksUndirectedGraph({1: {0: 1, 2: -4, 3: -6}, 2: {0: 2, 3: 1}}))

    def test_subgraph_missing_nodes(self):
        self.assertEqual(self.g0.subgraph({n3, n4, n5, n6, n8, n11, n12}),
                         self.g0.subgraph({Node(-1), Node(-2), n3, n4, n5, n6, n8, n11, n12}))

    def test_minimal_spanning_tree(self):
        def build_graph(links):
            result = UndirectedGraph()
            for l in links:
                result.add(l.u), result.add(l.v), result.connect(l.u, l.v)
            return result

        res = self.g0.minimal_spanning_tree()
        self.assertEqual(sum(map(self.g0.link_weights, res)), 22)
        g = build_graph(res)
        self.assertTrue(all(_g.is_tree(True) for _g in g.connection_components()))
        res = self.g1.minimal_spanning_tree()
        self.assertEqual(sum(map(self.g1.link_weights, res)), 10)
        g = build_graph(res)
        self.assertTrue(g.is_tree())
        res = self.g2.minimal_spanning_tree()
        self.assertEqual(sum(map(self.g2.link_weights, res)), 1)
        g = build_graph(res)
        self.assertTrue(g.is_tree())
        res = self.g3.minimal_spanning_tree()
        self.assertSetEqual(res, self.g3.links)
        g = build_graph(res)
        self.assertTrue(g.is_tree())

    def test_links_graph(self):
        l02 = Node(Link(0, 2))
        l05 = Node(Link(0, 5))
        l12 = Node(Link(1, 2))
        l13 = Node(Link(1, 3))
        l14 = Node(Link(1, 4))
        l25 = Node(Link(2, 5))
        l35 = Node(Link(3, 5))
        l45 = Node(Link(4, 5))
        expected = WeightedNodesUndirectedGraph(
            {l02: (2, {l05, l12, l25}), l05: (4, {l25, l35, l45}), l12: (5, {l13, l14, l25}), l13: (2, {l14, l35}),
             l14: (4, {l45}), l25: (1, {l35, l45}), l35: (3, {l45}), l45: (2, {})})
        self.assertTrue(self.g1.links_graph() == expected)

    def test_minimal_path_links(self):
        res = self.g0.minimal_path_links(0, 8)
        u = res[0]
        self.assertEqual(u, n0)
        for v in res[1:]:
            self.assertIn(u, self.g0.neighbors(v))
            u = v
        total_weight = 0
        for i in range(len(res) - 1):
            total_weight += self.g0.link_weights(res[i], res[i + 1])
        self.assertEqual(total_weight, 8)
        res = self.g1.minimal_path_links(0, 4)
        u = res[0]
        self.assertEqual(u, n0)
        for v in res[1:]:
            self.assertIn(u, self.g1.neighbors(v))
            u = v
        total_weight = 0
        for i in range(len(res) - 1):
            total_weight += self.g1.link_weights(res[i], res[i + 1])
        self.assertEqual(total_weight, 5)
        res = self.g2.minimal_path_links(0, 5)
        u = res[0]
        self.assertEqual(u, n0)
        for v in res[1:]:
            self.assertIn(u, self.g2.neighbors(v))
            u = v
        total_weight = 0
        for i in range(len(res) - 1):
            total_weight += self.g2.link_weights(res[i], res[i + 1])
        self.assertEqual(total_weight, -2)
        res = self.g2.minimal_path_links(0, 1)
        u = res[0]
        self.assertEqual(u, n0)
        for v in res[1:]:
            self.assertIn(u, self.g2.neighbors(v))
            u = v
        total_weight = 0
        for i in range(len(res) - 1):
            total_weight += self.g2.link_weights(res[i], res[i + 1])
        self.assertEqual(total_weight, -8)

    def test_isomorphic_bijection(self):
        g1 = WeightedLinksUndirectedGraph(
            {n11: {n12: 5, n13: 2, n14: 4}, n12: {n10: 2, n15: 1}, n15: {n10: 4, n13: 3, n14: 2}})
        func = self.g1.isomorphic_bijection(g1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        g1 = UndirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n5, n2, n3).disconnect(n1, n4).connect(n3, {n2: 1, n4: 2})
        self.assertNotEqual(g1.connect(n0, {n1: 1}), self.g2)
        g1.set_weight(Link(1, 3), -6), g1.set_weight(Link(1, 2), -4), g1.set_weight(Link(0, 5), 3)
        self.assertEqual(g1.increase_weight(Link(4, 5), 2), self.g2.copy().remove(n6))

    def test_add(self):
        self.assertEqual(self.g0 + self.g2, WeightedLinksUndirectedGraph(
            {0: {1: 4, 2: 3, 5: 3}, 2: {1: -8, 3: 7, 4: 2, 5: 4}, 3: {1: -6, 4: 2, 6: 3}, 4: {5: 7, 6: 7}, 5: {6: 5},
             7: {6: 2, 8: 1, 9: 3}, 8: {5: 5, 6: 4}, 11: {10: 2, 12: 3, 13: 4}, 12: {13: 1}, 14: {}}))
        g10 = UndirectedGraph.copy(self.g1)
        g11 = self.g1.weighted_nodes_graph({n0: 3, n1: 2, n2: 4, n3: 6, n4: 5, n5: 1})
        self.assertEqual(g10 + self.g2, WeightedLinksUndirectedGraph(
            {0: {1: 1, 2: 2, 5: 3}, 1: {2: -4, 3: -6, 4: 0}, 3: {2: 1, 4: 2, 5: 0}, 5: {0: 3, 2: 0, 4: 4, 6: 5}}))
        self.assertEqual(g11 + self.g2, WeightedUndirectedGraph(
            {n0: (3, {n2: 2}), n1: (2, {n0: 1, n2: -4, n3: -6, n4: 0}), n2: (4, {n5: 0}),
             n3: (6, {n2: 1, n4: 2, n5: 0}), n4: (5, {}), n5: (1, {n0: 3, n4: 4, n6: 5}), n6: (0, {})}))

    def test_str(self):
        self.assertEqual(str(self.g0), "<" + str(self.g0.nodes) + ", {" + ", ".join(
            f"{l} -> {self.g0.link_weights(l)}" for l in self.g0.links) + "}>")

    def test_repr(self):
        self.assertEqual(repr(self.g0), str(self.g0))


class TestWeightedUndirectedGraph(TestCase):
    def setUp(self):
        self.g0 = WeightedUndirectedGraph(
            {n0: (7, {n1: 3, n2: 1}), n1: (3, {}), n2: (5, {n1: -4, n3: 6, n4: 2, n5: 4}), n3: (2, {n6: 3}),
             n4: (8, {n5: 3, n6: 7}), n5: (4, {}), n6: (6, {}), n7: (2, {n6: 2, n8: 1, n9: 3}),
             n8: (0, {n5: 5, n6: 4}), n9: (5, {}), n10: (4, {}), n11: (2, {n10: 2, n12: 3, n13: 4}),
             n12: (1, {n13: 1}), n13: (3, {}), n14: (6, {})})
        self.g1 = WeightedUndirectedGraph(
            {n0: (3, {}), n1: (2, {n2: 5, n3: 2, n4: 4}), n2: (4, {n0: 2, n5: 1}), n3: (6, {}), n4: (5, {}),
             n5: (1, {n0: 4, n3: 3, n4: 2})})
        self.g2 = WeightedUndirectedGraph(
            {n0: (7, {n2: 2}), n1: (6, {n0: 1, n2: -4, n3: -6}), n2: (2, {}), n3: (4, {n2: 1, n4: 2}), n4: (3, {}),
             n5: (5, {n0: 3, n4: 4, n6: 5}), n6: (4, {})})
        self.g3 = WeightedUndirectedGraph(
            {n0: (7, {}), n1: (4, {n0: 2, n3: 4, n4: 3, n5: -1}), n2: (3, {n0: 1, n6: 5, n7: 3}),
             n3: (5, {n8: 6, n9: 2}), n4: (6, {}), n5: (2, {n10: 4, n11: 1}), n6: (2, {}), n7: (1, {}), n8: (6, {}),
             n9: (4, {}), n10: (5, {}), n11: (8, {})})

    def test_init(self):
        g = WeightedUndirectedGraph({0: (2, {1: 0}), 2: (3, {1: 3})})
        self.assertDictEqual(g.node_weights(), {n0: 2, n1: 0, n2: 3})
        self.assertDictEqual(g.link_weights(), {Link(0, 1): 0, Link(1, 2): 3})
        self.assertDictEqual(g.neighbors(), {n0: {n1}, n1: {n0, n2}, n2: {n1}})

    def test_total_weight(self):
        self.assertEqual((self.g0.total_weight, self.g1.total_weight, self.g2.total_weight, self.g3.total_weight),
                         (108, 44, 39, 83))

    def test_add_node(self):
        g0 = self.g0.copy().add((-1, 3), {1: 2, 3: 4})
        node_weights = self.g0.node_weights()
        node_weights[Node(-1)] = 3
        self.assertDictEqual(g0.node_weights(), node_weights)
        link_weights = self.g0.link_weights()
        link_weights[Link(-1, 1)] = 2
        link_weights[Link(-1, 3)] = 4
        self.assertDictEqual(g0.link_weights(), link_weights)
        neighbors = self.g0.neighbors()
        neighbors[Node(-1)] = {n1, n3}
        neighbors[n1].add(Node(-1)), neighbors[n3].add(Node(-1))
        self.assertDictEqual(g0.neighbors(), neighbors)

    def test_add_already_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add((3, 5), {1: 2}))

    def test_remove(self):
        g0 = self.g0.copy().remove(10, 14)
        node_weights = self.g0.node_weights()
        node_weights.pop(n10), node_weights.pop(n14)
        self.assertDictEqual(g0.node_weights(), node_weights)
        link_weights = self.g0.link_weights()
        link_weights.pop(Link(n10, n11))
        self.assertDictEqual(g0.link_weights(), link_weights)
        neighbors = self.g0.neighbors()
        neighbors.pop(n10), neighbors.pop(n14), neighbors[n11].remove(n10)
        self.assertDictEqual(g0.neighbors(), neighbors)

    def test_remove_missing_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-1, -2))

    def test_set_node_weight(self):
        g0 = self.g0.copy().set_weight(2, 3)
        node_weights = self.g0.node_weights()
        node_weights[n2] = 3
        self.assertDictEqual(g0.node_weights(), node_weights)

    def test_set_node_weight_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight(Node(-1), 4))

    def test_set_node_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().set_weight(2, [3])

    def test_set_link_weight(self):
        g0 = self.g0.copy().set_weight(Link(2, 3), 3)
        link_weights = self.g0.link_weights()
        link_weights[Link(2, 3)] = 3
        self.assertDictEqual(g0.link_weights(), link_weights)

    def test_set_link_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight(Link(1, 3), 4))

    def test_set_link_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.set_weight(Link(2, 3), [3])

    def test_increase_node_weight(self):
        g0 = self.g0.copy().increase_weight(2, 3)
        node_weights = self.g0.node_weights()
        node_weights[n2] += 3
        self.assertDictEqual(g0.node_weights(), node_weights)

    def test_increase_node_weight_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().increase_weight(Node(-1), 4))

    def test_increase_node_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().increase_weight(2, [3])

    def test_increase_link_weight(self):
        g0 = self.g0.copy().increase_weight(Link(2, 3), 3)
        link_weights = self.g0.link_weights()
        link_weights[Link(2, 3)] += 3
        self.assertDictEqual(g0.link_weights(), link_weights)

    def test_increase_link_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().increase_weight(Link(1, 3), 4))

    def test_increase_link_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().increase_weight(Link(2, 3), [3])

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_component(self):
        r0 = WeightedUndirectedGraph(
            {n0: (7, {n1: 3, n2: 1}), 1: (3, {}), n2: (5, {n1: -4, n3: 6, n4: 2, n5: 4}), n3: (2, {n6: 3}),
             n4: (8, {n5: 3, n6: 7}), n5: (4, {n8: 5}), n6: (6, {n8: 4}), n7: (2, {n6: 2, n8: 1, n9: 3}), n9: (5, {})})
        r1 = WeightedUndirectedGraph(
            {n10: (4, {}), n11: (2, {n10: 2, n12: 3, n13: 4}), n12: (1, {13: 1}), 13: (3, {})})
        self.assertEqual(self.g0.component(0), r0)
        self.assertEqual(self.g0.component(10), r1)
        self.assertEqual(self.g0.component(14), WeightedUndirectedGraph({14: (6, {})}))
        self.assertEqual(self.g1.component(0), self.g1)
        self.assertEqual(self.g2.component(0), self.g2)
        self.assertEqual(self.g3.component(0), self.g3)

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedUndirectedGraph(
            {n0: (7, {n1: 3, n2: 1}), n1: (3, {}), n2: (5, {n1: -4, n3: 6, n4: 2, n5: 4}), n3: (2, {n6: 3}),
             n4: (8, {n5: 3, n6: 7}), n5: (4, {}), n6: (6, {}), n7: (2, {n6: 2, n8: 1, n9: 3}),
             n8: (0, {n5: 5, n6: 4}), n9: (5, {})}), res)
        self.assertIn(WeightedUndirectedGraph(
            {n10: (4, {}), n11: (2, {n10: 2, n12: 3, n13: 4}), n12: (1, {n13: 1}), n13: (3, {})}), res)
        self.assertIn(WeightedUndirectedGraph({14: (6, {})}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        self.assertListEqual(self.g3.connection_components(), [self.g3])

    def test_subgraph(self):
        self.assertEqual(self.g2.subgraph({n0, n1, n2, n3}), WeightedUndirectedGraph(
            {0: (7, {1: 1, 2: 2}), 1: (6, {2: -4, 3: -6}), 2: (2, {3: 1}), 3: (4, {})}))

    def test_subgraph_missing_nodes(self):
        self.assertEqual(self.g0.subgraph({n3, n4, n5, n6, n8, n11, n12}),
                         self.g0.subgraph({Node(-1), Node(-2), n3, n4, n5, n6, n8, n11, n12}))

    def test_links_graph(self):
        l02 = Node(Link(0, 2))
        l05 = Node(Link(0, 5))
        l12 = Node(Link(1, 2))
        l13 = Node(Link(1, 3))
        l14 = Node(Link(1, 4))
        l25 = Node(Link(2, 5))
        l35 = Node(Link(3, 5))
        l45 = Node(Link(4, 5))
        self.assertEqual(self.g1.links_graph(), WeightedUndirectedGraph(
            {l02: (2, {l05: 3, l25: 4, l12: 4}), l25: (1, {l05: 1, l12: 4, l35: 1, l45: 1}), l12: (5, {l14: 2}),
             l13: (2, {l12: 2, l14: 2, l35: 6}), l35: (3, {l05: 1, l45: 1}), l05: (4, {l45: 1}), l14: (4, {l45: 5}),
             l45: (2, {})}))

    def test_minimal_path(self):
        self.assertListEqual(self.g0.minimal_path(n13, n11), [n13, n11])
        self.assertListEqual(self.g0.minimal_path(n1, n13), [])
        self.assertListEqual(self.g0.minimal_path(n1, n6), [n1, n2, n3, n6])
        self.assertListEqual(self.g1.minimal_path(n2, n4), [n2, n5, n4])
        self.assertListEqual(self.g2.minimal_path(n4, n6), [n4, n5, n6])
        self.assertListEqual(self.g2.minimal_path(n0, n4), [n0, n2, n1, n3, n4])

    def test_minimal_path_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g1.minimal_path(3, 7)

    def test_isomorphic_bijection(self):
        g1 = WeightedUndirectedGraph(
            {n10: (3, {}), n11: (2, {n12: 5, n13: 2, n14: 4}), n12: (4, {n10: 2, n15: 1}), n13: (6, {}), n14: (5, {}),
             n15: (1, {n10: 4, n13: 3, n14: 2})})
        func = self.g1.isomorphic_bijection(g1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        self.assertDictEqual(func, self.g1.isomorphic_bijection(UndirectedGraph.copy(g1)))
        g1 = WeightedNodesUndirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n5, n2, n3).disconnect(n1, n4).connect(n3, {n2: 1, n4: 2})
        g2 = self.g2.copy().remove(6)
        self.assertNotEqual(g1.connect(n0, {n1: 1}), g2)
        g1.set_weight(Link(1, 3), -6), g1.set_weight(Link(1, 2), -4), g1.set_weight(Link(0, 5), 3)
        self.assertNotEqual(g1.set_weight(Link(4, 5), 4), g2)
        g1.increase_weight(0, 4), g1.increase_weight(1, 4), g1.set_weight(2, 2)
        g1.set_weight(3, 4), g1.set_weight(4, 3), g1.increase_weight(5, 4)
        self.assertEqual(g1, g2)

    def test_add(self):
        g1 = self.g1.copy()
        g20 = self.g2.copy()
        g21 = UndirectedGraph.copy(self.g2)
        self.assertEqual(g1 + g20, WeightedUndirectedGraph(
            {n0: (10, {n1: 1, n2: 4, n5: 7}), n1: (8, {n2: 1, n3: -4, n4: 4}), n2: (6, {n3: 1, n5: 1}),
             n3: (10, {n4: 2, n5: 3}), n4: (8, {n5: 6}), n5: (6, {n6: 5}), n6: (4, {})}))
        self.assertEqual(g1 + g21, WeightedUndirectedGraph(
            {n0: (3, {n1: 0, n2: 2, n5: 4}), n1: (2, {n2: 5, n3: 2, n4: 4}), n2: (4, {n3: 0}), n3: (6, {n4: 0}),
             n4: (5, {n5: 2}), n5: (1, {n2: 1, n3: 3, n6: 0})}))

    def test_str(self):
        self.assertEqual(str(self.g0), "<{" + ", ".join(
            f"{n} -> {self.g0.node_weights(n)}" for n in self.g0.nodes) + "}, {" + ", ".join(
            f"{l} -> {self.g0.link_weights(l)}" for l in self.g0.links) + "}>")

    def test_repr(self):
        self.assertEqual(repr(self.g0), str(self.g0))


if __name__ == "__main__":
    main()
