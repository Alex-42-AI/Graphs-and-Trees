from unittest import TestCase, main

from Personal.Graphs.src.implementation.undirected_graph import *

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7), Node(8), Node(9), Node(10), Node(11), Node(12), Node(13), Node(14), Node(15)


class TestUndirectedGraph(TestCase):
    def setUp(self):
        self.g0 = UndirectedGraph({n0: [n1, n2], n2: [n1, n3, n4, n5], n3: [n6], n4: [n5, n6], n7: [n6, n8, n9], n8: [n5, n6], n11: [n10, n12, n13], n12: [n13], n14: []})
        self.g1 = UndirectedGraph({n1: [n2, n3, n4], n2: [n0, n5], n5: [n0, n3, n4]})
        self.g2 = UndirectedGraph({n1: [n0, n2, n3], n3: [n2, n4], n5: [n0, n4, n6], n0: [n2]})
        self.g3 = UndirectedGraph({n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11]})

    def test_get_neighboring(self):
        self.assertFalse(self.g0.neighbors(n14))
        self.assertSetEqual(self.g0.neighbors(n2), {n0, n1, n3, n4, n5})
        self.assertSetEqual(self.g0.neighbors(n3), {n2, n6})

    def test_get_degrees(self):
        self.assertEqual(self.g0.degrees(), {n0: 2, n1: 2, n2: 5, n3: 2, n4: 3, n5: 3, n6: 4, n7: 3, n8: 3, n9: 1, n10: 1, n11: 3, n12: 2, n13: 2, n14: 0})

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(UndirectedGraph({n0: [n1, n2], n2: [n1, n3, n4, n5], n3: [n6], n4: [n5, n6], n7: [n6, n8, n9], n8: [n5, n6]}), res)
        self.assertIn(UndirectedGraph({n11: [n10, n12, n13], n12: [n13]}), res)
        self.assertIn(UndirectedGraph({n14: []}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_connected(self):
        self.assertFalse(self.g0.connected())
        self.assertTrue(self.g1.connected())
        self.assertTrue(self.g2.connected())
        self.assertTrue(self.g3.connected())

    def test_is_tree(self):
        self.assertFalse(self.g0.is_tree)
        self.assertTrue(self.g3.is_tree)

    def test_tree(self):
        pass

    def test_reachable(self):
        for u in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
            for v in [n10, n11, n12, n13, n14]:
                self.assertFalse(self.g0.reachable(u, v))
            for v in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
                self.assertTrue(self.g0.reachable(u, v))
        for u in [n10, n11, n12, n13]:
            self.assertFalse(self.g0.reachable(n14, u))

    def test_component(self):
        r0, r1 = UndirectedGraph({n0: [n1, n2], n2: [n1, n3, n4, n5], n3: [n6], n4: [n5, n6], n7: [n6, n8, n9], n8: [n5, n6]}), UndirectedGraph({n11: [n10, n12, n13], n12: [n13]})
        self.assertEqual(self.g0.component(n0), r0)
        self.assertEqual(self.g0.component(n10), r1)
        self.assertEqual(self.g0.component(n14), UndirectedGraph({n14: []}))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), self.g3)

    def test_cut_nodes(self):
        self.assertSetEqual(self.g0.cut_nodes(), {n2, n7, n11})
        self.assertSetEqual(self.g1.cut_nodes(), set())
        self.assertSetEqual(self.g2.cut_nodes(), {n5})
        self.assertSetEqual(self.g3.cut_nodes(), {n0, n1, n2, n3, n5})

    def test_bridge_links(self):
        self.assertSetEqual(self.g0.bridge_links(), {Link(n7, n9), Link(n10, n11)})
        self.assertSetEqual(self.g1.bridge_links(), set())
        self.assertSetEqual(self.g2.bridge_links(), {Link(n5, n6)})
        self.assertSetEqual(self.g3.bridge_links(), self.g3.links)

    def test_path_with_length(self):
        self.assertTrue(self.g0.path_with_length(n0, n5, 6))
        self.assertTrue(self.g0.path_with_length(n0, n5, 7))
        self.assertTrue(self.g0.path_with_length(n10, n12, 3) and self.g0.path_with_length(n10, n13, 3))
        self.assertTrue(self.g0.path_with_length(n4, n8, 3))

    def test_cycle_with_length(self):
        self.assertTrue(self.g0.cycle_with_length(3))
        self.assertTrue(self.g0.cycle_with_length(4))
        self.assertTrue(self.g0.cycle_with_length(5))
        self.assertTrue(self.g0.cycle_with_length(6))
        self.assertTrue(self.g0.cycle_with_length(7))
        self.assertTrue(self.g0.cycle_with_length(8))

    def test_get_shortest_path(self):
        self.assertListEqual(self.g0.get_shortest_path(n0, n6), [n0, n2, n3, n6])
        self.assertListEqual(self.g0.get_shortest_path(n7, n5), [n7, n8, n5])

    def test_width(self):
        self.assertEqual(self.g0.component(n0).diameter(), 5)
        self.assertEqual(self.g1.diameter(), 2)
        self.assertEqual(self.g2.diameter(), 3)

    def test_links_graph(self):
        pass

    def test_max_cliques(self):
        pass

    def test_cliques_graph(self):
        pass

    def test_interval_sort(self):
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

        tmp = self.g0.copy()
        self.assertFalse(tmp.interval_sort())
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, n7, n8)
        self.assertTrue(consecutive_1s(tmp, tmp.interval_sort()))

    def test_full_k_partite(self):
        pass

    def test_euler_walk_and_tour_exist(self):
        tmp = self.g0.copy().connect(n4, n9).connect(n5, n7).component(n0)
        self.assertTrue(tmp.euler_walk_exists(n2, n8))
        self.assertTrue(tmp.connect(n2, n8).euler_tour_exists())

    def test_hamilton_walk_and_tour_exist(self):
        tmp = self.g0.component(n0)
        self.assertFalse(tmp.hamilton_walk_exists(n2, n9))
        self.assertFalse(tmp.hamilton_tour_exists())
        self.assertTrue(tmp.hamilton_walk_exists(n0, n9))
        self.assertTrue(tmp.hamilton_walk_exists(n1, n9))
        tmp.connect(n0, n9), self.assertTrue(tmp.hamilton_tour_exists())

    def test_vertex_cover(self):
        res = self.g0.vertex_cover()
        self.assertEqual(len(res), 7)
        for v_c in res:
            for l in self.g0.links:
                self.assertTrue(l.u in v_c or l.v in v_c)
        res = self.g1.vertex_cover()
        self.assertEqual(len(res), 3)
        for v_c in res:
            for l in self.g1.links:
                self.assertTrue(l.u in v_c or l.v in v_c)
        res = self.g2.vertex_cover()
        self.assertEqual(len(res), 4)
        for v_c in res:
            for l in self.g2.links:
                self.assertTrue(l.u in v_c or l.v in v_c)

    def test_dominating_set(self):
        self.assertSetEqual(self.g0.dominating_set(), {n2, n7, n11, n14})
        res = self.g1.dominating_set()
        self.assertEqual(len(res), 2)
        for u in self.g1.nodes:
            self.assertTrue(u in res or any(v in res for v in self.g1.neighbors(u)))
        res = self.g2.dominating_set()
        self.assertEqual(len(res), 2)
        for u in self.g2.nodes:
            self.assertTrue(u in res or any(v in res for v in self.g2.neighbors(u)))

    def test_independent_set(self):
        res = self.g0.independent_set()
        self.assertListEqual(list(map(len, res)), [8] * 4)
        for i_s in res:
            for i, u in enumerate(i_s):
                for v in i_s[i + 1:]:
                    self.assertFalse(u in self.g0.neighbors(v))
        res = self.g1.independent_set()
        self.assertListEqual(list(map(len, res)), [3] * 2)
        for i_s in res:
            for i, u in enumerate(i_s):
                for v in i_s[i + 1:]:
                    self.assertFalse(u in self.g1.neighbors(v))
        res = self.g2.independent_set()
        self.assertListEqual(list(map(len, res)), [3] * 4)
        for i_s in res:
            for i, u in enumerate(i_s):
                for v in i_s[i + 1:]:
                    self.assertFalse(u in self.g2.neighbors(v))

    def test_euler_walk_and_tour(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n4, n5), tmp.disconnect(n7, n8)
        res = tmp.euler_walk(n2, n9)
        n = len(res)
        self.assertEqual(n, len(tmp.links) + 1)
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighbors(res[i]))
            u = res[i]
        tmp.connect(n2, n9)
        res1 = tmp.euler_tour()
        self.assertEqual(len(res1), n)
        u = res1[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighbors(res1[i]))
            u = res1[i]
        self.assertIn(u, tmp.neighbors(res1[0]))

    def test_hamilton_walk_and_tour(self):
        tmp = self.g0.component(n0)
        res = tmp.hamilton_walk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertIn(n9, (res[0], res[-1]))
        tmp.disconnect(n4, n5)
        self.assertFalse(tmp.hamilton_walk())
        tmp.connect(n4, n5), tmp.connect(n0, n9)
        res1 = tmp.hamilton_tour()
        self.assertIn(res1[0], tmp.neighbors(res1[-1]))
        for i in range(n - 1):
            self.assertIn(res[i], tmp.neighbors(res[i + 1]))
            self.assertIn(res1[i], tmp.neighbors(res1[i + 1]))

    def test_clique(self):
        self.assertTrue(self.g0.clique(n0, n1, n2))
        self.assertTrue(self.g0.clique(n7, n9))
        self.assertFalse(self.g0.clique(n10, n11, n12, n13))
        self.assertTrue(self.g0.clique(n6, n7, n8))

    def test_cliques(self):
        res = self.g0.cliques(3)
        self.assertIn([n0, n1, n2], res)
        self.assertIn([n2, n4, n5], res)
        self.assertIn([n6, n7, n8], res)
        self.assertIn([n11, n12, n13], res)
        self.assertEqual(len(res), 4)
        res = self.g0.cliques(2)
        self.assertEqual(len(res), len(self.g0.links))

    def test_chromatic_number_nodes(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, n7, n8)
        tmp_copy, g0, g3 = tmp.copy(), self.g0.copy(), self.g3.copy()
        res = tmp.chromatic_nodes_partition()
        self.assertEqual(len(res), 4)
        self.assertEqual(sum(map(len, res)), len(tmp.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, tmp.neighbors(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res:
            tmp_sum += c
        for n in tmp.nodes:
            self.assertIn(n, tmp_sum)
        res = self.g3.chromatic_nodes_partition()
        self.assertEqual(len(res), 2)
        self.assertEqual(sum(map(len, res)), len(self.g3.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g3.neighbors(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res:
            tmp_sum += c
        for n in self.g3.nodes:
            self.assertIn(n, tmp_sum)
        res = self.g0.chromatic_nodes_partition()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), len(self.g0.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g0.neighbors(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res:
            tmp_sum += c
        for n in self.g0.nodes:
            self.assertIn(n, tmp_sum)
        self.assertEqual((tmp_copy, g0, g3), (tmp, self.g0, self.g3))

    def test_chromatic_number_links(self):
        g1 = self.g1.copy()
        res = self.g1.chromatic_links_partition()
        self.assertEqual(len(res), 4)
        self.assertEqual(sum(map(len, res)), len(self.g1.links))
        for c in res:
            for l in c:
                for m in c:
                    self.assertEqual(l.u in m, l == m) and self.assertEqual(l.v in m, l == m)
                for d in res:
                    self.assertEqual(l in d, c == d)
        self.assertEqual(g1, self.g1)

    def test_isomorphic_function(self):
        tmp = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})
        func = self.g1.isomorphic_bijection(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            neighbors = self.g1.neighbors(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighbors(m))
        tmp.disconnect(n11, n13)
        self.assertFalse(self.g1.isomorphic_bijection(tmp))

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1, g2 = self.g1.copy(), self.g2.copy()
        g1.disconnect(n5, n2, n3), g1.disconnect(n1, n4), g1.connect(n3, n2, n4)
        self.assertEqual(g1.connect(n0, n1), g2.remove(n6))

    def test_add(self):
        tmp = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})
        res = self.g3 + tmp
        helper = UndirectedGraph({n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11], n11: [n12, n13, n14], n10: [n12, n15], n15: [n12, n13, n14]})
        self.assertEqual(res, helper)


class TestWeightedNodesUndirectedGraph(TestUndirectedGraph):
    def setUp(self):
        self.g0 = WeightedNodesUndirectedGraph({n0: (7, [n1, n2]), n1: (3, []), n2: (5, [n1, n3, n4, n5]),
        n3: (2, [n6]), n4: (8, [n5, n6]), n5: (4, []), n6: (6, []), n7: (2, [n6, n8, n9]), n8: (0, [n5, n6]),
        n9: (5, []), n10: (4, []), n11: (2, [n10, n12, n13]), n12: (1, [n13]), n13: (3, []), n14: (6, [])})
        self.g1 = WeightedNodesUndirectedGraph({n0: (3, []), n1: (2, [n2, n3, n4]), n2: (4, [n0, n5]), n3: (6, []),
                                                n4: (5, []), n5: (1, [n0, n3, n4])})
        self.g2 = WeightedNodesUndirectedGraph({n0: (7, [n2]), n1: (6, [n0, n2, n3]), n2: (2, []), n3: (4, [n2, n4]),
                                                n4: (3, []), n5: (5, [n0, n4, n6]), n6: (4, [])})
        self.g3 = WeightedNodesUndirectedGraph({n0: (7, []), n1: (4, [n0, n3, n4, n5]), n2: (3, [n0, n6, n7]),
                                                n3: (5, [n8, n9]), n4: (6, []), n5: (2, [n10, n11]), n6: (2, []),
                                                n7: (1, []), n8: (6, []), n9: (4, []), n10: (5, []), n11: (8, [])})

    def test_node_weights(self):
        self.assertEqual(self.g0.node_weights(), {n0: 7, n1: 3, n2: 5, n3: 2, n4: 8, n5: 4, n6: 6, n7: 2, n8: 0, n9: 5, n10: 4, n11: 2, n12: 1, n13: 3, n14: 6})

    def test_total_node_weights(self):
        self.assertEqual(self.g0.total_nodes_weight, 58)

    def test_weighted_tree(self):
        pass

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedNodesUndirectedGraph({n0: (7, [n1, n2]), n1: (3, []), n2: (5, [n1, n3, n4, n5]), n3: (2, [n6]), n4: (8, [n5, n6]), n5: (4, []), n6: (6, []), n7: (2, [n6, n8, n9]), n8: (0, [n5, n6]), n9: (5, [])}), res)
        self.assertIn(WeightedNodesUndirectedGraph({n10: (4, []), n11: (2, [n10, n12, n13]), n12: (1, [n13]), n13: (3, [])}), res)
        self.assertIn(WeightedNodesUndirectedGraph({n14: (6, [])}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_component(self):
        self.assertEqual(WeightedNodesUndirectedGraph({n0: (7, [n1, n2]), n1: (3, []), n2: (5, [n1, n3, n4, n5]), n3: (2, [n6]), n4: (8, [n5, n6]), n5: (4, []), n6: (6, []), n7: (2, [n6, n8, n9]), n8: (0, [n5, n6]), n9: (5, [])}), self.g0.component(n0))

    def test_min_path_nodes(self):
        res = self.g1.minimal_path_nodes(n0, n1)
        self.assertEqual(res[1], 9)
        self.assertListEqual(res[0], [n0, n2, n1])

    def test_links_graph(self):
        pass

    def test_weighted_vertex_cover(self):
        g0 = self.g0.copy()
        res = self.g0.weighted_vertex_cover()
        self.assertEqual(sum(map(g0.node_weights, res[0])), 23)
        for i in range(len(res)):
            self.assertEqual(sum(map(self.g0.node_weights, res[i])), 23)
        self.assertEqual(sum(map(self.g0.node_weights, res[0])), 23)
        self.assertEqual(g0, self.g0)

    def test_weighted_dominating_set(self):
        res = self.g0.weighted_dominating_set()
        for i in range(len(res) - 1):
            self.assertEqual(sum(map(self.g0.node_weights, res[i])), sum(map(self.g0.node_weights, res[i + 1])))
        self.assertEqual(sum(map(self.g0.node_weights, res[0])), 15)

    def test_isomorphic_function(self):
        tmp = WeightedNodesUndirectedGraph({n10: (3, [n12, n15]), n11: (2, [n12, n13, n14]), n12: (4, []), n13: (6, []), n14: (5, []), n15: (1, [n12, n13, n14])})
        func = self.g1.isomorphic_bijection(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            self.assertEqual(self.g1.node_weights(n), tmp.node_weights(m))
            neighbors = self.g1.neighbors(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighbors(m))

    def test_equal(self):
        g1, g2 = self.g1, self.g2
        g2.remove(n6), g1.connect(n0, n1), g1.disconnect(n1, n4), g1.connect(n3, n2, n4)
        self.assertNotEqual(g1.disconnect(n2, n5), g2.connect(n3, n5))
        g2.set_weight(n0, 3), g2.set_weight(n1, 2), g2.set_weight(n2, 4)
        g2.set_weight(n3, 6), g2.set_weight(n4, 5)
        self.assertEqual(g1, g2.set_weight(n5, 1))

    def test_add(self):
        self.assertEqual(self.g3 + WeightedNodesUndirectedGraph({n10: (3, [n12, n15]), n11: (2, [n12, n13, n14]),
    n12: (4, []), n13: (6, []), n14: (5, []), n15: (1, [n12, n13, n14])}), WeightedNodesUndirectedGraph({n0: (7, []),
        n1: (4, [n0, n3, n4, n5]), n2: (3, [n0, n6, n7]), n3: (5, [n8, n9]), n4: (6, []), n5: (2, []), n6: (2, []),
        n7: (1, []), n8: (6, []), n9: (4, []), n10: (8, [n5, n12, n15]), n11: (10, [n5, n12, n13, n14]),
                          n12: (4, []), n13: (6, []), n14: (5, []), n15: (1, [n12, n13, n14])}))
        tmp = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})
        self.assertEqual(self.g0 + tmp, WeightedNodesUndirectedGraph({n0: (7, [n1, n2]), n1: (3, []),
                          n2: (5, [n1, n3, n4, n5]), n3: (2, [n6]), n4: (8, [n5]), n5: (4, []), n6: (6, [n4, n7, n8]),
                          n7: (2, [n9]), n8: (0, [n5, n7]), n9: (5, []), n10: (4, [n11, n12, n15]),
                          n11: (2, [n12, n13, n14]), n12: (1, [n13, n15]), n13: (3, [n15]), n14: (6, [n15])}))
        self.assertEqual(self.g3 + tmp, WeightedNodesUndirectedGraph({n0: (7, []), n1: (4, [n0, n3, n4, n5]),
                                  n2: (3, [n0, n6, n7]), n3: (5, [n8, n9]), n4: (6, []), n5: (2, [n10, n11]),
                                  n6: (2, []), n7: (1, []), n8: (6, []), n9: (4, []), n10: (5, []),
                                  n11: (8, [n12, n13, n14]), n12: (0, [n10, n15]), n15: (0, [n10, n13, n14])}))
        tmp = WeightedLinksUndirectedGraph({n1: {n2: 5, n3: 2, n4: 4}, n2: {n0: 2, n5: 1}, n5: {n0: 4, n3: 3, n4: 2}})
        self.assertEqual(self.g0 + tmp, WeightedUndirectedGraph({n0: (7, {n1: 0, n2: 2, n5: 4}),
        n1: (3, {n2: 5, n3: 2, n4: 4}), n2: (5, {n3: 0, n4: 0, n5: 1}), n3: (2, {n5: 3, n6: 0}),
        n4: (8, {n5: 2, n6: 0}), n5: (4, {n8: 0}), n6: (6, {n7: 0, n8: 0}), n7: (2, {n8: 0, n9: 0}), n9: (5, {}),
        n10: (4, {n11: 0}), n11: (2, {n12: 0, n13: 0}), n12: (1, {n13: 0}), n13: (3, {}), n14: (6, {})}))


class TestWeightedLinksUndirectedGraph(TestUndirectedGraph):
    def setUp(self):
        self.g0 = WeightedLinksUndirectedGraph({n0: {n1: 3, n2: 1}, n2: {n1: -4, n3: 6, n4: 2, n5: 4}, n3: {n6: 3},
                                        n4: {n5: 3, n6: 7}, n7: {n6: 2, n8: 1, n9: 3}, n8: {n5: 5, n6: 4},
                                                n11: {n10: 2, n12: 3, n13: 4}, n12: {n13: 1}, n14: {}})
        self.g1 = WeightedLinksUndirectedGraph({n1: {n2: 5, n3: 2, n4: 4}, n2: {n0: 2, n5: 1}, n5: {n0: 4, n3: 3, n4: 2}})
        self.g2 = WeightedLinksUndirectedGraph({n0: {n2: 2}, n1: {n0: 1, n2: -4, n3: -6},
                                                n3: {n2: 1, n4: 2}, n5: {n0: 3, n4: 4, n6: 5}})
        self.g3 = WeightedLinksUndirectedGraph({n1: {n0: 2, n3: 4, n4: 3, n5: -1}, n2: {n0: 1, n6: 5, n7: 3},
                                                n3: {n8: 6, n9: 2}, n5: {n10: 4, n11: 1}})

    def test_link_weights(self):
        self.assertEqual(self.g0.link_weights(), {Link(n0, n1): 3, Link(n0, n2): 1, Link(n1, n2): -4,
              Link(n3, n2): 6, Link(n4, n2): 2, Link(n5, n2): 4, Link(n3, n6): 3, Link(n4, n6): 7, Link(n4, n5): 3,
              Link(n5, n8): 5, Link(n6, n7): 2, Link(n6, n8): 4, Link(n7, n8): 1, Link(n7, n9): 3, Link(n10, n11): 2,
              Link(n11, n12): 3, Link(n11, n13): 4, Link(n13, n12): 1})
        self.assertEqual(self.g0.link_weights(n0), {n1: 3, n2: 1})
        self.assertEqual(self.g0.link_weights(n1), {n0: 3, n2: -4})
        self.assertEqual(self.g0.link_weights(n2), {n0: 1, n1: -4, n3: 6, n4: 2, n5: 4})
        self.assertEqual(self.g0.link_weights(n3), {n2: 6, n6: 3})
        self.assertEqual(self.g0.link_weights(n4), {n2: 2, n5: 3, n6: 7})
        self.assertEqual(self.g0.link_weights(n5), {n2: 4, n4: 3, n8: 5})
        self.assertEqual(self.g0.link_weights(n6), {n3: 3, n4: 7, n7: 2, n8: 4})
        self.assertEqual(self.g0.link_weights(n7), {n6: 2, n8: 1, n9: 3})
        self.assertEqual(self.g0.link_weights(n8), {n5: 5, n6: 4, n7: 1})
        self.assertEqual(self.g0.link_weights(n9), {n7: 3})
        self.assertEqual(self.g0.link_weights(n10), {n11: 2})
        self.assertEqual(self.g0.link_weights(n11), {n10: 2, n12: 3, n13: 4})
        self.assertEqual(self.g0.link_weights(n12), {n11: 3, n13: 1})
        self.assertEqual(self.g0.link_weights(n13), {n11: 4, n12: 1})
        self.assertFalse(self.g0.link_weights(n14))

    def test_total_link_weights(self):
        self.assertEqual(self.g0.total_links_weight, 50)
        self.assertEqual(self.g1.total_links_weight, 23)
        self.assertEqual(self.g2.total_links_weight, 8)
        self.assertEqual(self.g3.total_links_weight, 30)

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedLinksUndirectedGraph({n0: {n1: 3, n2: 1}, n2: {n1: -4, n3: 6, n4: 2, n5: 4}, n3: {n6: 3}, n4: {n5: 3, n6: 7}, n7: {n6: 2, n8: 1, n9: 3}, n8: {n5: 5, n6: 4}}), res)
        self.assertIn(WeightedLinksUndirectedGraph({n11: {n10: 2, n12: 3, n13: 4}, n12: {n13: 1}}), res)
        self.assertIn(WeightedLinksUndirectedGraph({n14: {}}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedLinksUndirectedGraph({n0: {n1: 3, n2: 1}, n2: {n1: -4, n3: 6, n4: 2, n5: 4}, n3: {n6: 3}, n4: {n5: 3, n6: 7}, n7: {n6: 2, n8: 1, n9: 3}, n8: {n5: 5, n6: 4}}))
        self.assertEqual(self.g0.component(n10),WeightedLinksUndirectedGraph({n11: {n10: 2, n12: 3, n13: 4}, n12: {n13: 1}}))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), self.g3)

    def test_minimal_spanning_tree(self):
        res = self.g0.minimal_spanning_tree()
        self.assertEqual(res[0][1], 16)
        self.assertEqual(res[1][1], 6)
        self.assertEqual(res[2][1], 0)
        for c in res:
            g = UndirectedGraph()
            for l in c[0]:
                if l.u not in g.nodes:
                    if l.v not in g.nodes:
                        g.add(l.v)
                    g.add(l.u, l.v)
                else:
                    if l.v not in g.nodes:
                        g.add(l.v, l.u)
                    else:
                        g.connect(l.u, l.v)
            self.assertTrue(all(_g.is_tree for _g in g.connection_components()))
        res = self.g1.minimal_spanning_tree()
        self.assertEqual(res[1], 10)
        g = UndirectedGraph()
        for l in res[0]:
            if l.u not in g.nodes:
                if l.v not in g.nodes: g.add(l.v)
                g.add(l.u, l.v)
            else:
                if l.v not in g.nodes:
                    g.add(l.v, l.u)
                else:
                    g.connect(l.u, l.v)
        self.assertTrue(g.is_tree)
        res = self.g2.minimal_spanning_tree()
        self.assertEqual(res[1], 1)
        g = UndirectedGraph()
        for l in res[0]:
            if l.u not in g.nodes:
                if l.v not in g.nodes: g.add(l.v)
                g.add(l.u, l.v)
            else:
                if l.v not in g.nodes:
                    g.add(l.v, l.u)
                else:
                    g.connect(l.u, l.v)
        self.assertTrue(g.is_tree)
        res = self.g3.minimal_spanning_tree()
        self.assertEqual(res[1], 30)
        g = UndirectedGraph()
        for l in res[0]:
            if l.u not in g.nodes:
                if l.v not in g.nodes: g.add(l.v)
                g.add(l.u, l.v)
            else:
                if l.v not in g.nodes:
                    g.add(l.v, l.u)
                else:
                    g.connect(l.u, l.v)
        self.assertTrue(g.is_tree)

    def test_links_graph(self):
        pass

    def test_min_path_links(self):
        res = self.g0.minimal_path_links(n0, n8)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g0.neighbors(v))
            u = v
        self.assertEqual(res[1], 8)
        res = self.g1.minimal_path_links(n0, n4)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g1.neighbors(v))
            u = v
        self.assertEqual(res[1], 5)
        res = self.g2.minimal_path_links(n0, n5)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g2.neighbors(v))
            u = v
        self.assertEqual(res[1], -2)
        res = self.g2.minimal_path_links(n0, n1)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g2.neighbors(v))
            u = v
        self.assertEqual(res[1], -8)

    def test_path_with_length(self):
        self.assertTrue(self.g0.path_with_length(n0, n5, 6))
        self.assertTrue(self.g0.path_with_length(n0, n5, 7))
        self.assertTrue(self.g0.path_with_length(n10, n12, 3) and self.g0.path_with_length(n10, n13, 3))
        self.assertTrue(self.g0.path_with_length(n4, n8, 3))

    def test_cycle_with_length(self):
        self.assertTrue(self.g0.cycle_with_length(3))
        self.assertTrue(self.g0.cycle_with_length(4))
        self.assertTrue(self.g0.cycle_with_length(5))
        self.assertTrue(self.g0.cycle_with_length(6))
        self.assertTrue(self.g0.cycle_with_length(7))
        self.assertTrue(self.g0.cycle_with_length(8))

    def test_interval_sort(self):
        tmp = self.g0.copy()
        self.assertFalse(tmp.interval_sort())
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, {n7: 0, n8: 0})
        res, total = tmp.interval_sort(), set()
        total.add(res[0])
        for i in range(len(res)):
            neighbors = tmp.neighbors(res[i])
            for v in total: neighbors.remove(v)
            if i + 1 < len(res) and not neighbors and res[i + 1] not in total:
                total.add(res[i + 1])
            else:
                for v in res[i + 1: i + len(neighbors) + 1]:
                    self.assertIn(v, neighbors), total.add(v)

    def test_chromatic_number_nodes(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, {n7: 0, n8: 0})
        res = tmp.chromatic_nodes_partition()
        self.assertEqual(len(res), 4)
        self.assertEqual(sum(map(len, res)), len(tmp.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, tmp.neighbors(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res:
            tmp_sum += c
        for n in tmp.nodes:
            self.assertIn(n, tmp_sum)
        res = self.g3.chromatic_nodes_partition()
        self.assertEqual(len(res), 2)
        self.assertEqual(sum(map(len, res)), len(self.g3.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g3.neighbors(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res:
            tmp_sum += c
        for n in self.g3.nodes:
            self.assertIn(n, tmp_sum)
        res = self.g0.chromatic_nodes_partition()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), len(self.g0.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g0.neighbors(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res:
            tmp_sum += c
        for n in self.g0.nodes:
            self.assertIn(n, tmp_sum)

    def test_euler_walk_and_tour_exist(self):
        self.g0.connect(n4, {n9: 0}), self.g0.connect(n5, {n7: 0})
        tmp = self.g0.component(n0)
        self.g0.disconnect(n5, n7), self.g0.disconnect(n4, n9)
        self.assertTrue(tmp.euler_walk_exists(n2, n8))
        tmp.connect(n2, {n8: 0}), self.assertTrue(tmp.euler_tour_exists())

    def test_hamilton_walk_and_tour_exist(self):
        tmp = self.g0.component(n0)
        self.assertFalse(tmp.hamilton_walk_exists(n2, n9))
        self.assertFalse(tmp.hamilton_tour_exists())
        self.assertTrue(tmp.hamilton_walk_exists(n0, n9))
        self.assertTrue(tmp.hamilton_walk_exists(n1, n9))
        tmp.connect(n0, {n9: 0}), self.assertTrue(tmp.hamilton_tour_exists())

    def test_euler_walk_and_tour(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n4, n5), tmp.disconnect(n7, n8)
        res = tmp.euler_walk(n2, n9)
        n = len(res)
        self.assertEqual(n, len(tmp.links) + 1)
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighbors(res[i]))
            u = res[i]
        tmp.connect(n2, {n9: 0})
        res1 = tmp.euler_tour()
        self.assertEqual(len(res1), n)
        u = res1[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighbors(res1[i]))
            u = res1[i]
        self.assertIn(u, tmp.neighbors(res1[0]))

    def test_hamilton_walk_and_tour(self):
        tmp = self.g0.component(n0)
        res = tmp.hamilton_walk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertIn(n9, (res[0], res[-1]))
        tmp.disconnect(n4, n5)
        self.assertFalse(tmp.hamilton_walk())
        tmp.connect(n4, {n5: 0}), tmp.connect(n0, {n9: 0})
        res1 = tmp.hamilton_tour()
        self.assertIn(res1[0], tmp.neighbors(res1[-1]))
        for i in range(n - 1):
            self.assertIn(res[i], tmp.neighbors(res[i + 1]))
            self.assertIn(res1[i], tmp.neighbors(res1[i + 1]))
        self.assertIn(res1[-2], tmp.neighbors(res1[-1]))

    def test_isomorphic_function(self):
        tmp = WeightedLinksUndirectedGraph({n11: {n12: 5, n13: 2, n14: 4}, n12: {n10: 2, n15: 1}, n15: {n10: 4, n13: 3, n14: 2}})
        func = self.g1.isomorphic_bijection(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            neighbors = self.g1.neighbors(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighbors(m))
                original = None
                for u in neighbors:
                    if func[u] == v:
                        original = u
                        break
                if original is not None:
                    self.assertEqual(self.g1.link_weights(original, n), tmp.link_weights(v, m))

    def test_equal(self):
        pass

    def test_add(self):
        self.assertEqual(self.g0 + self.g2, WeightedLinksUndirectedGraph({n0: {n1: 4, n2: 3, n5: 3},
        n2: {n1: -8, n3: 7, n4: 2, n5: 4}, n3: {n1: -6, n4: 2, n6: 3}, n4: {n5: 7, n6: 7}, n5: {n6: 5},
        n7: {n6: 2, n8: 1, n9: 3}, n8: {n5: 5, n6: 4}, n11: {n10: 2, n12: 3, n13: 4}, n12: {n13: 1}, n14: {}}))


class TestWeightedUndirectedGraph(TestWeightedNodesUndirectedGraph, TestWeightedLinksUndirectedGraph):
    def setUp(self):
        self.g0 = WeightedUndirectedGraph({n0: (7, {n1: 3, n2: 1}), n1: (3, {}), n2: (5, {n1: -4, n3: 6, n4: 2, n5: 4}),
                n3: (2, {n6: 3}), n4: (8, {n5: 3, n6: 7}), n5: (4, {}), n6: (6, {}), n7: (2, {n6: 2, n8: 1, n9: 3}),
                n8: (0, {n5: 5, n6: 4}), n9: (5, {}), n10: (4, {}), n11: (2, {n10: 2, n12: 3, n13: 4}),
                                           n12: (1, {n13: 1}), n13: (3, {}), n14: (6, {})})
        self.g1 = WeightedUndirectedGraph({n0: (3, {}), n1: (2, {n2: 5, n3: 2, n4: 4}), n2: (4, {n0: 2, n5: 1}),
                                           n3: (6, {}), n4: (5, {}), n5: (1, {n0: 4, n3: 3, n4: 2})})
        self.g2 = WeightedUndirectedGraph({n0: (7, {n2: 2}), n1: (6, {n0: 1, n2: -4, n3: -6}),
           n2: (2, {}), n3: (4, {n2: 1, n4: 2}), n4: (3, {}), n5: (5, {n0: 3, n4: 4, n6: 5}), n6: (4, {})})
        self.g3 = WeightedUndirectedGraph({n0: (7, {}), n1: (4, {n0: 2, n3: 4, n4: 3, n5: -1}),
                       n2: (3, {n0: 1, n6: 5, n7: 3}), n3: (5, {n8: 6, n9: 2}), n4: (6, {}), n5: (2, {n10: 4, n11: 1}),
                                       n6: (2, {}), n7: (1, {}), n8: (6, {}), n9: (4, {}), n10: (5, {}), n11: (8, {})})

    def test_total_weights(self):
        self.assertEqual((self.g0.total_weight, self.g1.total_weight, self.g2.total_weight, self.g3.total_weight), (108, 44, 39, 83))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedUndirectedGraph({n0: (7, {n1: 3, n2: 1}), n1: (3, {}), n2: (5, {n1: -4, n3: 6, n4: 2, n5: 4}), n3: (2, {n6: 3}), n4: (8, {n5: 3, n6: 7}), n5: (4, {}), n6: (6, {}), n7: (2, {n6: 2, n8: 1, n9: 3}), n8: (0, {n5: 5, n6: 4}), n9: (5, {})}), res)
        self.assertIn(WeightedUndirectedGraph({n10: (4, {}), n11: (2, {n10: 2, n12: 3, n13: 4}), n12: (1, {n13: 1}), n13: (3, {})}), res)
        self.assertIn(WeightedUndirectedGraph({n14: (6, {})}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedUndirectedGraph({n0: (7, {n1: 3, n2: 1}), n1: (3, {}), n2: (5, {n1: -4, n3: 6, n4: 2, n5: 4}), n3: (2, {n6: 3}), n4: (8, {n5: 3, n6: 7}), n5: (4, {}), n6: (6, {}), n7: (2, {n6: 2, n8: 1, n9: 3}), n8: (0, {n5: 5, n6: 4}), n9: (5, {})}))
        self.assertEqual(self.g0.component(n10), WeightedUndirectedGraph({n10: (4, {}), n11: (2, {n10: 2, n12: 3, n13: 4}), n12: (1, {n13: 1}), n13: (3, {})}))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), self.g3)

    def test_links_graph(self):
        pass

    def test_minimal_path(self):
        self.assertEqual(self.g0.minimal_path(n13, n11), ([n13, n11], 9))
        self.assertEqual(self.g0.minimal_path(n1, n13), ([], 0))
        self.assertEqual(self.g0.minimal_path(n1, n6), ([n1, n2, n3, n6], 21))
        self.assertEqual(self.g1.minimal_path(n2, n4), ([n2, n5, n4], 13))
        self.assertEqual(self.g2.minimal_path(n4, n6), ([n4, n5, n6], 21))
        self.assertEqual(self.g2.minimal_path(n0, n4), ([n0, n2, n1, n3, n4], 16))

    def test_isomorphic_function(self):
        tmp = WeightedUndirectedGraph({n10: (3, {}), n11: (2, {n12: 5, n13: 2, n14: 4}), n12: (4, {n10: 2, n15: 1}), n13: (6, {}), n14: (5, {}), n15: (1, {n10: 4, n13: 3, n14: 2})})
        func = self.g1.isomorphic_bijection(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            self.assertEqual(self.g1.node_weights(n), tmp.node_weights(m))
            neighbors = self.g1.neighbors(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighbors(m))
                original = None
                for u in neighbors:
                    if func[u] == v:
                        original = u
                        break
                if original is not None:
                    self.assertEqual(self.g1.link_weights(original, n), tmp.link_weights(v, m))

    def test_equal(self):
        pass

    def test_add(self):
        pass


if __name__ == '__main__':
    main()
