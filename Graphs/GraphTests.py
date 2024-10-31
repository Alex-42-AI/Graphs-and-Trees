from unittest import TestCase, main
from Personal.DiscreteMath.Graphs.UndirectedGraph import *
from Personal.DiscreteMath.Graphs.DirectedGraph import *
from Personal.DiscreteMath.Graphs.Tree import *
n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21 = Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7), Node(8), Node(9), Node(10), Node(11), Node(12), Node(13), Node(14), Node(15), Node(16), Node(17), Node(18), Node(19), Node(20), Node(21)
class TestUndirectedGraph(TestCase):
    def setUp(self):
        self.g0 = UndirectedGraph(dict([(n0, [n1, n2]), (n2, [n1, n3, n4, n5]), (n3, [n6]), (n4, [n5, n6]), (n7, [n6, n8, n9]), (n8, [n5, n6]), (n11, [n10, n12, n13]), (n12, [n13]), (n14, [])]))
        self.g1 = UndirectedGraph(dict([(n1, [n2, n3, n4]), (n2, [n0, n5]), (n5, [n0, n3, n4])]))
        self.g2 = UndirectedGraph(dict([(n1, [n0, n2, n3]), (n3, [n2, n4]), (n5, [n0, n4, n6]), (n0, [n2])]))
        self.g3 = UndirectedGraph(dict([(n1, [n0, n3, n4, n5]), (n2, [n0, n6, n7]), (n3, [n8, n9]), (n5, [n10, n11])]))
    def test_get_neighboring(self):
        self.assertFalse(self.g0.neighboring(n14))
        self.assertListEqual(self.g0.neighboring(n2).value, [n0, n1, n3, n4, n5])
        self.assertListEqual(self.g0.neighboring(n3).value, [n2, n6])
    def test_get_degrees(self):
        self.assertEqual(self.g0.degrees(), dict([(n0, 2), (n1, 2), (n2, 5), (n3, 2), (n4, 3), (n5, 3), (n6, 4), (n7, 3), (n8, 3), (n9, 1), (n10, 1), (n11, 3), (n12, 2), (n13, 2), (n14, 0)]))
    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())
    def test_connection_components(self):
        r = self.g0.connection_components()
        for i in range(3):
            self.assertListEqual(list(sorted(r[i])), [[n0, n1, n2, n3, n4, n5, n6, n7, n8, n9], [n10, n11, n12, n13], [n14]][i])
        self.assertListEqual([list(sorted(self.g1.connection_components()[0]))], [self.g1.nodes.value])
        self.assertListEqual([list(sorted(self.g2.connection_components()[0]))], [self.g2.nodes.value])
    def test_connected(self):
        self.assertFalse(self.g0.connected())
        self.assertTrue(self.g1.connected())
    def test_tree(self):
        self.assertFalse(self.g0.tree())
        self.assertTrue(self.g3.tree())
    def test_reachable(self):
        for u in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
            for v in [n10, n11, n12, n13, n14]:
                self.assertFalse(self.g0.reachable(u, v))
            for v in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
                self.assertTrue(self.g0.reachable(u, v))
        for u in [n10, n11, n12, n13]:
            self.assertFalse(self.g0.reachable(n14, u))
    def test_component(self):
        r0, r1 = UndirectedGraph(dict([(n0, [n1, n2]), (n2, [n1, n3, n4, n5]), (n3, [n6]), (n4, [n5, n6]), (n7, [n6, n8, n9]), (n8, [n5, n6])])), UndirectedGraph(dict([(n11, [n10, n12, n13]), (n12, [n13])]))
        self.assertEqual(self.g0.component(n0), r0)
        self.assertEqual(self.g0.component(n10), r1)
        self.assertEqual(self.g0.component(n14), UndirectedGraph(dict([(n14, [])])))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), self.g3)
    def test_cut_nodes(self):
        self.assertListEqual(list(sorted(self.g0.cut_nodes())), [n2, n7, n11])
        self.assertListEqual(self.g1.cut_nodes(), [])
        self.assertListEqual(self.g2.cut_nodes(), [n5])
        self.assertListEqual(list(sorted(self.g3.cut_nodes())), [n0, n1, n2, n3, n5])
    def test_bridge_links(self):
        self.assertListEqual(self.g0.bridge_links(), [Link(n7, n9), Link(n10, n11)])
        self.assertListEqual(self.g1.bridge_links(), [])
        self.assertListEqual(self.g2.bridge_links(), [Link(n5, n6)])
        self.assertEqual(len(self.g3.bridge_links()), len(self.g3.links))
    def test_path_with_length(self):
        self.assertTrue(self.g0.pathWithLength(n0, n5, 6))
        self.assertTrue(self.g0.pathWithLength(n0, n5, 7))
        self.assertTrue(self.g0.pathWithLength(n10, n12, 3) and self.g0.pathWithLength(n10, n13, 3))
        self.assertTrue(self.g0.pathWithLength(n4, n8, 3))
    def test_loop_with_length(self):
        self.assertTrue(self.g0.loopWithLength(3))
        self.assertTrue(self.g0.loopWithLength(4))
        self.assertTrue(self.g0.loopWithLength(5))
        self.assertTrue(self.g0.loopWithLength(6))
        self.assertTrue(self.g0.loopWithLength(7))
        self.assertTrue(self.g0.loopWithLength(8))
    def test_get_shortest_path(self):
        self.assertListEqual(self.g0.get_shortest_path(n0, n6), [n0, n2, n3, n6])
        self.assertListEqual(self.g0.get_shortest_path(n7, n5), [n7, n8, n5])
    def test_width(self):
        self.assertEqual(self.g0.component(n0).width(), 5)
        self.assertEqual(self.g1.width(), 2)
        self.assertEqual(self.g2.width(), 3)
    def test_interval_sort(self):
        tmp = self.g0.copy()
        self.assertFalse(tmp.interval_sort())
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, n7, n8)
        res, total = tmp.interval_sort(), SortedList(f=tmp.f)
        total.insert(res[0])
        for i in range(len(res)):
            neighbors = tmp.neighboring(res[i])
            for v in total: neighbors.remove(v)
            if i + 1 < len(res) and not neighbors and res[i + 1] not in total: total.insert(res[i + 1])
            else:
                for v in res[i + 1: i + len(neighbors) + 1]: self.assertIn(v, neighbors), total.insert(v)
    def test_euler_walk_and_tour_exist(self):
        self.g0.connect(n4, n9), self.g0.connect(n5, n7)
        tmp = self.g0.component(n0)
        self.assertTrue(tmp.euler_walk_exists(n2, n8))
        tmp.connect(n2, n8), self.assertTrue(tmp.euler_tour_exists())
        self.g0.disconnect(n5, n7), self.g0.disconnect(n4, n9)
    def test_hamilton_walk_and_tour_exist(self):
        tmp = self.g0.component(n0)
        self.assertFalse(tmp.hamiltonWalkExists(n2, n9))
        self.assertFalse(tmp.hamiltonTourExists())
        self.assertTrue(tmp.hamiltonWalkExists(n0, n9))
        self.assertTrue(tmp.hamiltonWalkExists(n1, n9))
        tmp.connect(n0, n9), self.assertTrue(tmp.hamiltonTourExists())
    def test_vertex_cover(self):
        g0, g1, g2 = self.g0.copy(), self.g1.copy(), self.g2.copy()
        self.assertListEqual(self.g0.vertexCover(), [[n0, n2, n5, n6, n7, n11, n12], [n0, n2, n5, n6, n7, n11, n13], [n1, n2, n5, n6, n7, n11, n12], [n1, n2, n5, n6, n7, n11, n13]])
        self.assertListEqual(self.g1.vertexCover(), [[n0, n1, n5], [n1, n2, n5]])
        self.assertListEqual(self.g2.vertexCover(), [[n0, n1, n3, n5], [n0, n2, n3, n5], [n1, n2, n3, n5], [n1, n2, n4, n5]])
        self.assertEqual((g0, g1, g2), (self.g0, self.g1, self.g2))
    def test_dominating_set(self):
        self.assertListEqual(list(map(lambda x: list(sorted(x)), self.g0.dominatingSet())), [[n2, n7, n11, n14]])
        self.assertListEqual(list(map(lambda x: list(sorted(x)), self.g1.dominatingSet())), [[n0, n1], [n1, n2], [n1, n5], [n2, n5], [n3, n5], [n4, n5]])
        self.assertListEqual(list(map(lambda x: list(sorted(x)), self.g2.dominatingSet())), [[n1, n5], [n2, n5], [n3, n5]])
    def test_independent_set(self):
        g0, g1, g2 = self.g0.copy(), self.g1.copy(), self.g2.copy()
        self.assertListEqual(list(sorted(self.g0.independentSet())), [[n0, n3, n4, n8, n9, n10, n12, n14], [n0, n3, n4, n8, n9, n10, n13, n14], [n1, n3, n4, n8, n9, n10, n12, n14], [n1, n3, n4, n8, n9, n10, n13, n14]])
        self.assertListEqual(list(sorted(self.g1.independentSet())), [[n0, n3, n4], [n2, n3, n4]])
        self.assertListEqual(list(sorted(self.g2.independentSet())), [[n0, n3, n6], [n0, n4, n6], [n1, n4, n6], [n2, n4, n6]])
        self.assertEqual((g0, g1, g2), (self.g0, self.g1, self.g2))
    def test_euler_walk_and_tour(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n4, n5), tmp.disconnect(n7, n8)
        res = tmp.euler_walk(n2, n9)
        n = len(res)
        self.assertEqual(n, len(tmp.links) + 1)
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighboring(res[i]))
            u = res[i]
        tmp.connect(n2, n9)
        res1 = tmp.euler_tour()
        self.assertEqual(len(res1), n)
        u = res1[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighboring(res1[i]))
            u = res1[i]
        self.assertIn(u, tmp.neighboring(res1[0]))
    def test_hamilton_walk_and_tour(self):
        tmp = self.g0.component(n0)
        res = tmp.hamiltonWalk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertIn(n9, (res[0], res[-1]))
        tmp.disconnect(n4, n5)
        self.assertFalse(tmp.hamiltonWalk())
        tmp.connect(n4, n5), tmp.connect(n0, n9)
        res1 = tmp.hamiltonTour()
        self.assertIn(res1[0], tmp.neighboring(res1[-1]))
        for i in range(n - 1):
            self.assertIn(res[i], tmp.neighboring(res[i + 1]))
            self.assertIn(res1[i], tmp.neighboring(res1[i + 1]))
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
        res = tmp.chromaticNumberNodes()
        self.assertEqual(len(res), 4)
        self.assertEqual(sum(map(len, res)), len(tmp.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, tmp.neighboring(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res: tmp_sum += c
        for n in tmp.nodes: self.assertIn(n, tmp_sum)
        res = self.g3.chromaticNumberNodes()
        self.assertEqual(len(res), 2)
        self.assertEqual(sum(map(len, res)), len(self.g3.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g3.neighboring(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res: tmp_sum += c
        for n in self.g3.nodes: self.assertIn(n, tmp_sum)
        res = self.g0.chromaticNumberNodes()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), len(self.g0.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g0.neighboring(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res: tmp_sum += c
        for n in self.g0.nodes: self.assertIn(n, tmp_sum)
        self.assertEqual((tmp_copy, g0, g3), (tmp, self.g0, self.g3))
    def test_chromatic_number_links(self):
        g1 = self.g1.copy()
        res = self.g1.chromaticNumberLinks()
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
        tmp = UndirectedGraph(dict([(n11, [n12, n13, n14]), (n12, [n10, n15]), (n15, [n10, n13, n14])]))
        func = self.g1.isomorphicFunction(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            neighbors = self.g1.neighboring(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighboring(m))
        tmp.disconnect(n11, n13)
        self.assertFalse(self.g1.isomorphicFunction(tmp))
    def test_equal(self):
        g1, g2 = self.g1, self.g2
        self.assertNotEqual(g1, g2)
        g2.remove(n6), g1.disconnect(n5, n2, n3), g1.disconnect(n1, n4), g1.connect(n3, n2, n4), g1.connect(n0, n1)
        self.assertEqual(g1, g2)
    def test_add(self):
        tmp = UndirectedGraph(dict([(n11, [n12, n13, n14]), (n12, [n10, n15]), (n15, [n10, n13, n14])]))
        res = self.g3 + tmp
        helper = UndirectedGraph(dict([(n1, [n0, n3, n4, n5]), (n2, [n0, n6, n7]), (n3, [n8, n9]), (n5, [n10, n11]), (n11, [n12, n13, n14]), (n10, [n12, n15]), (n15, [n12, n13, n14])]))
        self.assertEqual(res, helper)
class TestWeightedNodesUndirectedGraph(TestUndirectedGraph):
    def setUp(self):
        self.g0 = WeightedNodesUndirectedGraph(dict([(n0, (7, [n1, n2])), (n1, (3, [])), (n2, (5, [n1, n3, n4, n5])), (n3, (2, [n6])), (n4, (8, [n5, n6])), (n5, (4, [])), (n6, (6, [])), (n7, (2, [n6, n8, n9])), (n8, (0, [n5, n6])), (n9, (5, [])), (n10, (4, [])), (n11, (2, [n10, n12, n13])), (n12, (1, [n13])), (n13, (3, [])), (n14, (6, []))]))
        self.g1 = WeightedNodesUndirectedGraph(dict([(n0, (3, [])), (n1, (2, [n2, n3, n4])), (n2, (4, [n0, n5])), (n3, (6, [])), (n4, (5, [])), (n5, (1, [n0, n3, n4]))]))
        self.g2 = WeightedNodesUndirectedGraph(dict([(n0, (7, [n2])), (n1, (6, [n0, n2, n3])), (n2, (2, [])), (n3, (4, [n2, n4])), (n4, (3, [])), (n5, (5, [n0, n4, n6])), (n6, (4, []))]))
        self.g3 = WeightedNodesUndirectedGraph(dict([(n0, (7, [])), (n1, (4, [n0, n3, n4, n5])), (n2, (3, [n0, n6, n7])), (n3, (5, [n8, n9])), (n4, (6, [])), (n5, (2, [n10, n11])), (n6, (2, [])), (n7, (1, [])), (n8, (6, [])), (n9, (4, [])), (n10, (5, [])), (n11, (8, []))]))
    def test_node_weights(self): self.assertEqual(self.g0.node_weights(), dict([(n0, 7), (n1, 3), (n2, 5), (n3, 2), (n4, 8), (n5, 4), (n6, 6), (n7, 2), (n8, 0), (n9, 5), (n10, 4), (n11, 2), (n12, 1), (n13, 3), (n14, 6)]))
    def test_total_node_weights(self): self.assertEqual(self.g0.total_nodes_weight, 58)
    def test_component(self): self.assertEqual(WeightedNodesUndirectedGraph(dict([(n0, (7, [n1, n2])), (n1, (3, [])), (n2, (5, [n1, n3, n4, n5])), (n3, (2, [n6])), (n4, (8, [n5, n6])), (n5, (4, [])), (n6, (6, [])), (n7, (2, [n6, n8, n9])), (n8, (0, [n5, n6])), (n9, (5, []))])), self.g0.component(n0))
    def test_min_path_nodes(self):
        res = self.g1.minimalPathNodes(n0, n1)
        self.assertEqual(res[1], 9)
        self.assertListEqual(res[0], [n0, n2, n1])
    def test_weighted_vertex_cover(self):
        g0 = self.g0.copy()
        res = self.g0.weightedVertexCover()
        self.assertEqual(sum(map(g0.node_weights, res[0])), 23)
        for i in range(len(res)): self.assertEqual(sum(map(self.g0.node_weights, res[i])), 23)
        self.assertEqual(sum(map(self.g0.node_weights, res[0])), 23)
        self.assertEqual(g0, self.g0)
    def test_weighted_dominating_set(self):
        res = self.g0.weightedDominatingSet()
        for i in range(len(res) - 1): self.assertEqual(sum(map(self.g0.node_weights, res[i])), sum(map(self.g0.node_weights, res[i + 1])))
        self.assertEqual(sum(map(self.g0.node_weights, res[0])), 15)
    def test_isomorphic_function(self):
        tmp = WeightedNodesUndirectedGraph(dict([(n10, (3, [n12, n15])), (n11, (2, [n12, n13, n14])), (n12, (4, [])), (n13, (6, [])), (n14, (5, [])), (n15, (1, [n12, n13, n14]))]))
        func = self.g1.isomorphicFunction(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            self.assertEqual(self.g1.node_weights(n), tmp.node_weights(m))
            neighbors = self.g1.neighboring(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighboring(m))
    def test_equal(self):
        g1, g2 = self.g1, self.g2
        g2.remove(n6), g2.connect(n3, n5), g1.connect(n0, n1), g1.disconnect(n1, n4), g1.connect(n3, n2, n4), g1.disconnect(n2, n5)
        self.assertNotEqual(g1, g2)
        g2.set_weight(n0, 3), g2.set_weight(n1, 2), g2.set_weight(n2, 4), g2.set_weight(n3, 6), g2.set_weight(n4, 5), g2.set_weight(n5, 1)
        self.assertEqual(g1, g2)
    def test_add(self):
        self.assertEqual(self.g3 + WeightedNodesUndirectedGraph(dict([(n10, (3, [n12, n15])), (n11, (2, [n12, n13, n14])), (n12, (4, [])), (n13, (6, [])), (n14, (5, [])), (n15, (1, [n12, n13, n14]))])), WeightedNodesUndirectedGraph(dict([(n0, (7, [])), (n1, (4, [n0, n3, n4, n5])), (n2, (3, [n0, n6, n7])), (n3, (5, [n8, n9])), (n4, (6, [])), (n5, (2, [])), (n6, (2, [])), (n7, (1, [])), (n8, (6, [])), (n9, (4, [])), (n10, (8, [n5, n12, n15])), (n11, (10, [n5, n12, n13, n14])), (n12, (4, [])), (n13, (6, [])), (n14, (5, [])), (n15, (1, [n12, n13, n14]))])))
        tmp = UndirectedGraph(dict([(n11, [n12, n13, n14]), (n12, [n10, n15]), (n15, [n10, n13, n14])]))
        self.assertEqual(self.g0 + tmp, WeightedNodesUndirectedGraph(dict([(n0, (7, [n1, n2])), (n1, (3, [])), (n2, (5, [n1, n3, n4, n5])), (n3, (2, [n6])), (n4, (8, [n5])), (n5, (4, [])), (n6, (6, [n4, n7, n8])), (n7, (2, [n9])), (n8, (0, [n5, n7])), (n9, (5, [])), (n10, (4, [n11, n12, n15])), (n11, (2, [n12, n13, n14])), (n12, (1, [n13, n15])), (n13, (3, [n15])), (n14, (6, [n15]))])))
        self.assertEqual(self.g3 + tmp, WeightedNodesUndirectedGraph(dict([(n0, (7, [])), (n1, (4, [n0, n3, n4, n5])), (n2, (3, [n0, n6, n7])), (n3, (5, [n8, n9])), (n4, (6, [])), (n5, (2, [n10, n11])), (n6, (2, [])), (n7, (1, [])), (n8, (6, [])), (n9, (4, [])), (n10, (5, [])), (n11, (8, [n12, n13, n14])), (n12, (0, [n10, n15])), (n15, (0, [n10, n13, n14]))])))
        tmp = WeightedLinksUndirectedGraph(dict([(n1, dict([(n2, 5), (n3, 2), (n4, 4)])), (n2, dict([(n0, 2), (n5, 1)])), (n5, dict([(n0, 4), (n3, 3), (n4, 2)]))]))
        self.assertEqual(self.g0 + tmp, WeightedUndirectedGraph(dict([(n0, (7, dict([(n1, 0), (n2, 2), (n5, 4)]))), (n1, (3, dict([(n2, 5), (n3, 2), (n4, 4)]))), (n2, (5, dict([(n3, 0), (n4, 0), (n5, 1)]))), (n3, (2, dict([(n5, 3), (n6, 0)]))), (n4, (8, dict([(n5, 2), (n6, 0)]))), (n5, (4, dict([(n8, 0)]))), (n6, (6, dict([(n7, 0), (n8, 0)]))), (n7, (2, dict([(n8, 0), (n9, 0)]))), (n9, (5, dict())), (n10, (4, dict([(n11, 0)]))), (n11, (2, dict([(n12, 0), (n13, 0)]))), (n12, (1, dict([(n13, 0)]))), (n13, (3, dict())), (n14, (6, dict()))])))
class TestWeightedLinksUndirectedGraph(TestUndirectedGraph):
    def setUp(self):
        self.g0 = WeightedLinksUndirectedGraph(dict([(n0, dict([(n1, 3), (n2, 1)])), (n2, dict([(n1, -4), (n3, 6), (n4, 2), (n5, 4)])), (n3, dict([(n6, 3)])), (n4, dict([(n5, 3), (n6, 7)])), (n7, dict([(n6, 2), (n8, 1), (n9, 3)])), (n8, dict([(n5, 5), (n6, 4)])), (n11, dict([(n10, 2), (n12, 3), (n13, 4)])), (n12, dict([(n13, 1)])), (n14, dict())]))
        self.g1 = WeightedLinksUndirectedGraph(dict([(n1, dict([(n2, 5), (n3, 2), (n4, 4)])), (n2, dict([(n0, 2), (n5, 1)])), (n5, dict([(n0, 4), (n3, 3), (n4, 2)]))]))
        self.g2 = WeightedLinksUndirectedGraph(dict([(n0, dict([(n2, 2)])), (n1, dict([(n0, 1), (n2, -4), (n3, -6)])), (n3, dict([(n2, 1), (n4, 2)])), (n5, dict([(n0, 3), (n4, 4), (n6, 5)]))]))
        self.g3 = WeightedLinksUndirectedGraph(dict([(n1, dict([(n0, 2), (n3, 4), (n4, 3), (n5, -1)])), (n2, dict([(n0, 1), (n6, 5), (n7, 3)])), (n3, dict([(n8, 6), (n9, 2)])), (n5, dict([(n10, 4), (n11, 1)]))]))
    def test_link_weights(self):
        self.assertEqual(self.g0.link_weights(), dict([(Link(n0, n1), 3), (Link(n0, n2), 1), (Link(n1, n2), -4), (Link(n3, n2), 6), (Link(n4, n2), 2), (Link(n5, n2), 4), (Link(n3, n6), 3), (Link(n4, n6), 7), (Link(n4, n5), 3), (Link(n5, n8), 5), (Link(n6, n7), 2), (Link(n6, n8), 4), (Link(n7, n8), 1), (Link(n7, n9), 3), (Link(n10, n11), 2), (Link(n11, n12), 3), (Link(n11, n13), 4), (Link(n13, n12), 1)]))
        self.assertEqual(self.g0.link_weights(n0), dict([(n1, 3), (n2, 1)]))
        self.assertEqual(self.g0.link_weights(n1), dict([(n0, 3), (n2, -4)]))
        self.assertEqual(self.g0.link_weights(n2), dict([(n0, 1), (n1, -4), (n3, 6), (n4, 2), (n5, 4)]))
        self.assertEqual(self.g0.link_weights(n3), dict([(n2, 6), (n6, 3)]))
        self.assertEqual(self.g0.link_weights(n4), dict([(n2, 2), (n5, 3), (n6, 7)]))
        self.assertEqual(self.g0.link_weights(n5), dict([(n2, 4), (n4, 3), (n8, 5)]))
        self.assertEqual(self.g0.link_weights(n6), dict([(n3, 3), (n4, 7), (n7, 2), (n8, 4)]))
        self.assertEqual(self.g0.link_weights(n7), dict([(n6, 2), (n8, 1), (n9, 3)]))
        self.assertEqual(self.g0.link_weights(n8), dict([(n5, 5), (n6, 4), (n7, 1)]))
        self.assertEqual(self.g0.link_weights(n9), dict([(n7, 3)]))
        self.assertEqual(self.g0.link_weights(n10), dict([(n11, 2)]))
        self.assertEqual(self.g0.link_weights(n11), dict([(n10, 2), (n12, 3), (n13, 4)]))
        self.assertEqual(self.g0.link_weights(n12), dict([(n11, 3), (n13, 1)]))
        self.assertEqual(self.g0.link_weights(n13), dict([(n11, 4), (n12, 1)]))
        self.assertFalse(self.g0.link_weights(n14))
    def test_total_link_weights(self):
        self.assertEqual(self.g0.total_links_weight, 50)
        self.assertEqual(self.g1.total_links_weight, 23)
        self.assertEqual(self.g2.total_links_weight, 8)
        self.assertEqual(self.g3.total_links_weight, 30)
    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedLinksUndirectedGraph(dict([(n0, dict([(n1, 3), (n2, 1)])), (n2, dict([(n1, -4), (n3, 6), (n4, 2), (n5, 4)])), (n3, dict([(n6, 3)])), (n4, dict([(n5, 3), (n6, 7)])), (n7, dict([(n6, 2), (n8, 1), (n9, 3)])), (n8, dict([(n5, 5), (n6, 4)]))])))
        self.assertEqual(self.g0.component(n10), WeightedLinksUndirectedGraph(dict([(n11, dict([(n10, 2), (n12, 3), (n13, 4)])), (n12, dict([(n13, 1)]))])))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), self.g3)
    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())
    def test_minimal_spanning_tree(self):
        res = self.g0.minimal_spanning_tree()
        self.assertEqual(res[0][1], 16)
        self.assertEqual(res[1][1], 6)
        self.assertEqual(res[2][1], 0)
        for c in res:
            g = UndirectedGraph(dict())
            for l in c[0]:
                if l.u not in g.nodes:
                    if l.v not in g.nodes: g.add(l.v)
                    g.add(l.u, l.v)
                else:
                    if l.v not in g.nodes: g.add(l.v, l.u)
                    else: g.connect(l.u, l.v)
            self.assertTrue(g.tree())
        res = self.g1.minimal_spanning_tree()
        self.assertEqual(res[1], 10)
        g = UndirectedGraph(dict())
        for l in res[0]:
            if l.u not in g.nodes:
                if l.v not in g.nodes: g.add(l.v)
                g.add(l.u, l.v)
            else:
                if l.v not in g.nodes: g.add(l.v, l.u)
                else: g.connect(l.u, l.v)
        self.assertTrue(g.tree())
        res = self.g2.minimal_spanning_tree()
        self.assertEqual(res[1], 1)
        g = UndirectedGraph(dict())
        for l in res[0]:
            if l.u not in g.nodes:
                if l.v not in g.nodes: g.add(l.v)
                g.add(l.u, l.v)
            else:
                if l.v not in g.nodes: g.add(l.v, l.u)
                else: g.connect(l.u, l.v)
        self.assertTrue(g.tree())
        res = self.g3.minimal_spanning_tree()
        self.assertEqual(res[1], 30)
        g = UndirectedGraph(dict())
        for l in res[0]:
            if l.u not in g.nodes:
                if l.v not in g.nodes: g.add(l.v)
                g.add(l.u, l.v)
            else:
                if l.v not in g.nodes:
                    g.add(l.v, l.u)
                else:
                    g.connect(l.u, l.v)
        self.assertTrue(g.tree())
    def test_min_path_links(self):
        res = self.g0.minimalPathLinks(n0, n8)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g0.neighboring(v))
            u = v
        self.assertEqual(res[1], 8)
        res = self.g1.minimalPathLinks(n0, n4)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g1.neighboring(v))
            u = v
        self.assertEqual(res[1], 5)
        res = self.g2.minimalPathLinks(n0, n5)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g2.neighboring(v))
            u = v
        self.assertEqual(res[1], -2)
        res = self.g2.minimalPathLinks(n0, n1)
        u = res[0][0]
        self.assertEqual(u, n0)
        for v in res[0][1:]:
            self.assertIn(u, self.g2.neighboring(v))
            u = v
        self.assertEqual(res[1], -8)
    def test_path_with_length(self):
        self.assertTrue(self.g0.pathWithLength(n0, n5, 6))
        self.assertTrue(self.g0.pathWithLength(n0, n5, 7))
        self.assertTrue(self.g0.pathWithLength(n10, n12, 3) and self.g0.pathWithLength(n10, n13, 3))
        self.assertTrue(self.g0.pathWithLength(n4, n8, 3))
    def test_loop_with_length(self):
        self.assertTrue(self.g0.loopWithLength(3))
        self.assertTrue(self.g0.loopWithLength(4))
        self.assertTrue(self.g0.loopWithLength(5))
        self.assertTrue(self.g0.loopWithLength(6))
        self.assertTrue(self.g0.loopWithLength(7))
        self.assertTrue(self.g0.loopWithLength(8))
    def test_interval_sort(self):
        tmp = self.g0.copy()
        self.assertFalse(tmp.interval_sort())
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, dict([(n7, 0), (n8, 0)]))
        res, total = tmp.interval_sort(), SortedList(f=tmp.f)
        total.insert(res[0])
        for i in range(len(res)):
            neighbors = tmp.neighboring(res[i])
            for v in total: neighbors.remove(v)
            if i + 1 < len(res) and not neighbors and res[i + 1] not in total: total.insert(res[i + 1])
            else:
                for v in res[i + 1: i + len(neighbors) + 1]: self.assertIn(v, neighbors), total.insert(v)
    def test_chromatic_number_nodes(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n2, n3), tmp.disconnect(n5, n8), tmp.connect(n3, dict([(n7, 0), (n8, 0)]))
        res = tmp.chromaticNumberNodes()
        self.assertEqual(len(res), 4)
        self.assertEqual(sum(map(len, res)), len(tmp.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, tmp.neighboring(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res: tmp_sum += c
        for n in tmp.nodes: self.assertIn(n, tmp_sum)
        res = self.g3.chromaticNumberNodes()
        self.assertEqual(len(res), 2)
        self.assertEqual(sum(map(len, res)), len(self.g3.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g3.neighboring(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res: tmp_sum += c
        for n in self.g3.nodes: self.assertIn(n, tmp_sum)
        res = self.g0.chromaticNumberNodes()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), len(self.g0.nodes))
        for c in res:
            for u in c:
                for v in c:
                    self.assertNotIn(u, self.g0.neighboring(v))
                for d in res:
                    self.assertEqual(u in d, c == d)
        tmp_sum = []
        for c in res: tmp_sum += c
        for n in self.g0.nodes: self.assertIn(n, tmp_sum)
    def test_euler_walk_and_tour_exist(self):
        self.g0.connect(n4, dict([(n9, 0)])), self.g0.connect(n5, dict([(n7, 0)]))
        tmp = self.g0.component(n0)
        self.assertTrue(tmp.euler_walk_exists(n2, n8))
        tmp.connect(n2, dict([(n8, 0)])), self.assertTrue(tmp.euler_tour_exists())
        self.g0.disconnect(n5, n7), self.g0.disconnect(n4, n9)
    def test_hamilton_walk_and_tour_exist(self):
        tmp = self.g0.component(n0)
        self.assertFalse(tmp.hamiltonWalkExists(n2, n9))
        self.assertFalse(tmp.hamiltonTourExists())
        self.assertTrue(tmp.hamiltonWalkExists(n0, n9))
        self.assertTrue(tmp.hamiltonWalkExists(n1, n9))
        tmp.connect(n0, dict([(n9, 0)])), self.assertTrue(tmp.hamiltonTourExists())
    def test_euler_walk_and_tour(self):
        tmp = self.g0.component(n0)
        tmp.disconnect(n4, n5), tmp.disconnect(n7, n8)
        res = tmp.euler_walk(n2, n9)
        n = len(res)
        self.assertEqual(n, len(tmp.links) + 1)
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighboring(res[i]))
            u = res[i]
        tmp.connect(n2, dict([(n9, 0)]))
        res1 = tmp.euler_tour()
        self.assertEqual(len(res1), n)
        u = res1[0]
        for i in range(1, n):
            self.assertIn(u, tmp.neighboring(res1[i]))
            u = res1[i]
        self.assertIn(u, tmp.neighboring(res1[0]))
    def test_hamilton_walk_and_tour(self):
        tmp = self.g0.component(n0)
        res = tmp.hamiltonWalk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertIn(n9, (res[0], res[-1]))
        tmp.disconnect(n4, n5)
        self.assertFalse(tmp.hamiltonWalk())
        tmp.connect(n4, dict([(n5, 0)])), tmp.connect(n0, dict([(n9, 0)]))
        res1 = tmp.hamiltonTour()
        self.assertIn(res1[0], tmp.neighboring(res1[-1]))
        for i in range(n - 1):
            self.assertIn(res[i], tmp.neighboring(res[i + 1]))
            self.assertIn(res1[i], tmp.neighboring(res1[i + 1]))
        self.assertIn(res1[-2], tmp.neighboring(res1[-1]))
    def test_isomorphic_function(self):
        tmp = WeightedLinksUndirectedGraph(dict([(n11, dict([(n12, 5), (n13, 2), (n14, 4)])), (n12, dict([(n10, 2), (n15, 1)])), (n15, dict([(n10, 4), (n13, 3), (n14, 2)]))]))
        func = self.g1.isomorphicFunction(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            neighbors = self.g1.neighboring(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighboring(m))
                original = None
                for u in neighbors:
                    if func[u] == v:
                        original = u
                        break
                if original is not None: self.assertEqual(self.g1.link_weights(original, n), tmp.link_weights(v, m))
    def test_equal(self):
        pass
    def test_add(self):
        self.assertEqual(self.g0 + self.g2, WeightedLinksUndirectedGraph(dict([(n0, dict([(n1, 4), (n2, 3), (n5, 3)])), (n2, dict([(n1, -8), (n3, 7), (n4, 2), (n5, 4)])), (n3, dict([(n1, -6), (n4, 2), (n6, 3)])), (n4, dict([(n5, 7), (n6, 7)])), (n5, dict([(n6, 5)])), (n7, dict([(n6, 2), (n8, 1), (n9, 3)])), (n8, dict([(n5, 5), (n6, 4)])), (n11, dict([(n10, 2), (n12, 3), (n13, 4)])), (n12, dict([(n13, 1)])), (n14, dict())])))

class TestWeightedUndirectedGraph(TestWeightedNodesUndirectedGraph, TestWeightedLinksUndirectedGraph):
    def setUp(self):
        self.g0 = WeightedUndirectedGraph(dict([(n0, (7, dict([(n1, 3), (n2, 1)]))), (n1, (3, dict())), (n2, (5, dict([(n1, -4), (n3, 6), (n4, 2), (n5, 4)]))), (n3, (2, dict([(n6, 3)]))), (n4, (8, dict([(n5, 3), (n6, 7)]))), (n5, (4, dict())), (n6, (6, dict())), (n7, (2, dict([(n6, 2), (n8, 1), (n9, 3)]))), (n8, (0, dict([(n5, 5), (n6, 4)]))), (n9, (5, dict())), (n10, (4, dict())), (n11, (2, dict([(n10, 2), (n12, 3), (n13, 4)]))), (n12, (1, dict([(n13, 1)]))), (n13, (3, dict())), (n14, (6, dict()))]))
        self.g1 = WeightedUndirectedGraph(dict([(n0, (3, dict())), (n1, (2, dict([(n2, 5), (n3, 2), (n4, 4)]))), (n2, (4, dict([(n0, 2), (n5, 1)]))), (n3, (6, dict())), (n4, (5, dict())), (n5, (1, dict([(n0, 4), (n3, 3), (n4, 2)])))]))
        self.g2 = WeightedUndirectedGraph(dict([(n0, (7, dict([(n2, 2)]))), (n1, (6, dict([(n0, 1), (n2, -4), (n3, -6)]))), (n2, (2, dict())), (n3, (4, dict([(n2, 1), (n4, 2)]))), (n4, (3, dict())), (n5, (5, dict([(n0, 3), (n4, 4), (n6, 5)]))), (n6, (4, dict()))]))
        self.g3 = WeightedUndirectedGraph(dict([(n0, (7, dict())), (n1, (4, dict([(n0, 2), (n3, 4), (n4, 3), (n5, -1)]))), (n2, (3, dict([(n0, 1), (n6, 5), (n7, 3)]))), (n3, (5, dict([(n8, 6), (n9, 2)]))), (n4, (6, dict())), (n5, (2, dict([(n10, 4), (n11, 1)]))), (n6, (2, dict())), (n7, (1, dict())), (n8, (6, dict())), (n9, (4, dict())), (n10, (5, dict())), (n11, (8, dict()))]))
    def test_total_weights(self): self.assertEqual((self.g0.total_weight, self.g1.total_weight, self.g2.total_weight, self.g3.total_weight), (108, 44, 39, 83))
    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedUndirectedGraph(dict([(n0, (7, dict([(n1, 3), (n2, 1)]))), (n1, (3, dict())), (n2, (5, dict([(n1, -4), (n3, 6), (n4, 2), (n5, 4)]))), (n3, (2, dict([(n6, 3)]))), (n4, (8, dict([(n5, 3), (n6, 7)]))), (n5, (4, dict())), (n6, (6, dict())), (n7, (2, dict([(n6, 2), (n8, 1), (n9, 3)]))), (n8, (0, dict([(n5, 5), (n6, 4)]))), (n9, (5, dict()))])))
        self.assertEqual(self.g0.component(n10), WeightedUndirectedGraph(dict([(n10, (4, dict())), (n11, (2, dict([(n10, 2), (n12, 3), (n13, 4)]))), (n12, (1, dict([(n13, 1)]))), (n13, (3, dict()))])))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), self.g3)
    def test_minimal_path(self):
        self.assertEqual(self.g0.minimalPath(n13, n11), ([n13, n11], 9))
        self.assertEqual(self.g0.minimalPath(n1, n13), ([], 0))
        self.assertEqual(self.g0.minimalPath(n1, n6), ([n1, n2, n3, n6], 21))
        self.assertEqual(self.g1.minimalPath(n2, n4), ([n2, n5, n4], 13))
        self.assertEqual(self.g2.minimalPath(n4, n6), ([n4, n5, n6], 21))
        self.assertEqual(self.g2.minimalPath(n0, n4), ([n0, n2, n1, n3, n4], 16))
    def test_isomorphic_function(self):
        tmp = WeightedUndirectedGraph(dict([(n10, (3, dict())), (n11, (2, dict([(n12, 5), (n13, 2), (n14, 4)]))), (n12, (4, dict([(n10, 2), (n15, 1)]))), (n13, (6, dict())), (n14, (5, dict())), (n15, (1, dict([(n10, 4), (n13, 3), (n14, 2)])))]))
        func = self.g1.isomorphicFunction(tmp)
        self.assertTrue(func)
        for n, m in func.items():
            self.assertEqual(self.g1.node_weights(n), tmp.node_weights(m))
            neighbors = self.g1.neighboring(n)
            for v in tmp.nodes:
                self.assertEqual(v in map(func.__getitem__, neighbors), v in tmp.neighboring(m))
                original = None
                for u in neighbors:
                    if func[u] == v:
                        original = u
                        break
                if original is not None: self.assertEqual(self.g1.link_weights(original, n), tmp.link_weights(v, m))
    def test_equal(self):
        pass
    def test_add(self):
        pass
class TestDirectedGraph(TestCase):
    def setUp(self):
        self.g0 = DirectedGraph(dict([(n1, ([n0, n8], [n2])), (n2, ([n0], [n3, n4])), (n3, ([n6], [n7, n8])), (n4, ([n6], [n5])), (n5, ([n7], [n6])), (n7, ([n8], [n9])), (n11, ([n10], [n12, n13])), (n12, ([], [n13])), (n14, ([], []))]))
        self.g1 = DirectedGraph(dict([(n1, ([n3], [n0, n2, n4, n5])), (n2, ([n0, n1], [n3])), (n5, ([n3], [n4]))]))
        self.g2 = DirectedGraph(dict([(n0, ([n2, n5], [n1])), (n2, ([n5], [n1, n3])), (n3, ([n1, n4], [])), (n5, ([n4, n6], []))]))
        self.g3 = DirectedGraph(dict([(n1, ([n0, n2], [n5])), (n2, ([n0, n3], [n4])), (n4, ([], [n5])), (n6, ([], [n7, n8])), (n9, ([n7, n8, n10], [])), (n10, ([], [n11]))]))
    def test_sources(self): self.assertEqual((self.g0.sources, self.g1.sources, self.g2.sources, self.g3.sources), ([n0, n10, n14], [], [n4, n6], [n0, n3, n6, n10]))
    def test_sinks(self): self.assertEqual((self.g0.sinks, self.g1.sinks, self.g2.sinks, self.g3.sinks), ([n9, n13, n14], [n4], [n3], [n5, n9, n11]))
    def test_get_degrees(self): self.assertEqual(self.g0.degrees(), dict([(n0, [0, 2]), (n1, [2, 1]), (n2, [2, 2]), (n3, [2, 2]), (n4, [2, 1]), (n5, [2, 1]), (n6, [1, 2]), (n7, [2, 2]), (n8, [1, 2]), (n9, [1, 0]), (n10, [0, 1]), (n11, [1, 2]), (n12, [1, 1]), (n13, [2, 0]), (n14, [0, 0])]))
    def test_copy(self): self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))
    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), DirectedGraph(dict([(u, (self.g0.next(u), self.g0.prev(u))) for u in self.g0.nodes]), self.g0.f))
        self.assertEqual(self.g1.transposed(), DirectedGraph(dict([(u, (self.g1.next(u), self.g1.prev(u))) for u in self.g1.nodes]), self.g1.f))
        self.assertEqual(self.g2.transposed(), DirectedGraph(dict([(u, (self.g2.next(u), self.g2.prev(u))) for u in self.g2.nodes]), self.g2.f))
        self.assertEqual(self.g3.transposed(), DirectedGraph(dict([(u, (self.g3.next(u), self.g3.prev(u))) for u in self.g3.nodes]), self.g3.f))
    def test_connection_components(self):
        r = self.g0.connection_components()
        for i in range(3): self.assertListEqual(list(sorted(r[i])), [[n0, n1, n2, n3, n4, n5, n6, n7, n8, n9], [n10, n11, n12, n13], [n14]][i])
        self.assertListEqual([list(sorted(self.g1.connection_components()[0]))], [self.g1.nodes])
        self.assertListEqual([list(sorted(self.g2.connection_components()[0]))], [self.g2.nodes])
    def test_connected(self):
        self.assertFalse(self.g0.connected())
        self.assertTrue(self.g1.connected())
        self.assertTrue(self.g2.connected())
        self.assertFalse(self.g3.connected())
    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n1), DirectedGraph(dict([(n1, ([n8], [n2])), (n2, ([], [n3, n4])), (n3, ([n6], [n7, n8])), (n4, ([n6], [n5])), (n5, ([n7], [n6])), (n7, ([n8], [n9]))])))
        self.assertEqual(self.g1.subgraph(n0), self.g1), self.assertEqual(self.g1.subgraph(n4), DirectedGraph(dict([(n4, ([], []))])))
        self.assertEqual(self.g2.subgraph(n2), DirectedGraph(dict([(n1, ([n0, n2], [n3])), (n2, ([], [n0, n1, n3]))])))
        self.assertEqual(self.g3.subgraph(n6), DirectedGraph(dict([(n6, ([], [n7, n8])), (n9, ([n7, n8], []))])))
    def test_component(self):
        self.assertEqual(self.g0.component(n9), DirectedGraph(dict([(n1, ([n0, n8], [n2])), (n2, ([n0], [n3, n4])), (n3, ([n6], [n7, n8])), (n4, ([n6], [n5])), (n5, ([n7], [n6])), (n7, ([n8], [n9]))])))
        self.assertEqual(self.g0.component(n13), DirectedGraph(dict([(n11, ([n10], [n12, n13])), (n12, ([], [n13]))])))
        self.assertEqual(self.g1.component(n4), self.g1)
        self.assertEqual(self.g2.component(n1), self.g2)
        self.assertEqual(self.g3.component(n4), DirectedGraph(dict([(n1, ([n0, n2], [n5])), (n2, ([n0, n3], [n4])), (n4, ([], [n5]))])))
        self.assertEqual(self.g3.component(n10), DirectedGraph(dict([(n6, ([], [n7, n8])), (n9, ([n7, n8, n10], [])), (n10, ([], [n11]))])))
    def test_SCC(self):
        for s in self.g0.stronglyConnectedComponents():
            for u in s:
                for v in s:
                    self.assertTrue(self.g0.reachable(u, v))
        for s in self.g1.stronglyConnectedComponents():
            for u in s:
                for v in s:
                    self.assertTrue(self.g1.reachable(u, v))
        for s in self.g2.stronglyConnectedComponents():
            for u in s:
                for v in s:
                    self.assertTrue(self.g2.reachable(u, v))
        for s in self.g3.stronglyConnectedComponents():
            for u in s:
                for v in s:
                    self.assertTrue(self.g3.reachable(u, v))
    def test_has_loop(self):
        self.assertTrue(self.g0.has_loop())
        self.assertTrue(self.g1.has_loop())
        self.assertFalse(self.g2.has_loop())
        self.assertFalse(self.g3.has_loop())
    def test_toposort(self):
        self.assertFalse(self.g0.toposort())
        self.assertFalse(self.g1.toposort())
        res = self.g2.toposort()
        self.assertEqual(len(self.g2.nodes), len(res))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                self.assertNotEqual(res[i], res[j])
                self.assertFalse(self.g2.reachable(res[j], res[i]))
        res = self.g3.toposort()
        self.assertEqual(len(self.g3.nodes), len(res))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                self.assertNotEqual(res[i], res[j])
                self.assertFalse(self.g3.reachable(res[j], res[i]))
    def test_reachable(self):
        for u in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
            self.assertTrue(self.g0.reachable(n0, u))
            for v in [n10, n11, n12, n13, n14]:
                self.assertFalse(self.g0.reachable(u, v))
            self.assertTrue(self.g0.reachable(u, n9))
        for u in [n10, n11, n12, n13]:
            self.assertTrue(self.g0.reachable(n10, u))
            self.assertFalse(self.g0.reachable(n14, u))
    def test_cut_nodes(self):
        self.assertListEqual(list(sorted(self.g0.cut_nodes())), [n7, n11])
        self.assertListEqual(self.g1.cut_nodes(), [])
        self.assertListEqual(self.g2.cut_nodes(), [n5])
        self.assertListEqual(list(sorted(self.g3.cut_nodes())), [n2, n9, n10])
    def test_path_with_length(self):
        self.assertTrue(self.g0.pathWithLength(n0, n1, 9))
        self.assertFalse(self.g0.pathWithLength(n0, n1, 10))
        self.assertTrue(self.g0.pathWithLength(n0, n5, 6))
        self.assertTrue(self.g0.pathWithLength(n10, n13, 3))
        self.assertTrue(self.g0.pathWithLength(n4, n8, 4))
    def test_loop_with_length(self):
        self.assertTrue(self.g0.loopWithLength(3))
        self.assertTrue(self.g0.loopWithLength(4))
        self.assertFalse(self.g0.loopWithLength(6))
        self.assertFalse(self.g0.loopWithLength(9))
        self.assertTrue(self.g1.loopWithLength(3))
        self.assertTrue(self.g1.loopWithLength(4))
    def test_get_shortest_path(self):
        self.assertListEqual(self.g0.get_shortest_path(n0, n6), [n0, n2, n4, n5, n6])
        self.assertListEqual(self.g0.get_shortest_path(n7, n8), [n7, n5, n6, n3, n8])
        self.assertListEqual(self.g0.get_shortest_path(n10, n13), [n10, n11, n13])
        self.assertListEqual(self.g1.get_shortest_path(n0, n5), [n0, n2, n3, n5])
        self.assertListEqual(self.g2.get_shortest_path(n4, n3), [n4, n3])
    def test_euler_tour_and_walk_exist(self):
        tmp = DirectedGraph.copy(self.g0)
        tmp.connect(n11, [n13]), tmp.connect(n0, [n4]), tmp.connect(n8, [n1]), tmp.connect(n7, [n5], [n6])
        self.assertTrue(tmp.component(n0).euler_walk_exists(n0, n9))
        self.assertTrue(tmp.component(n10).euler_walk_exists(n10, n13))
        self.assertFalse(tmp.euler_walk_exists(n0, n9))
        self.assertFalse(self.g1.euler_tour_exists())
        tmp = DirectedGraph.copy(self.g1)
        tmp.connect(n1, [n2]), tmp.connect(n4, points_to=[n1, n3]), tmp.disconnect(n2, [n1]), tmp.connect(n2, [n5])
        self.assertTrue(tmp.euler_tour_exists())
    def test_hamilton_walk_and_tour_exist(self):
        tmp = DirectedGraph.subgraph(self.g0, n0)
        self.assertFalse(self.g0.hamiltonTourExists())
        self.assertTrue(tmp.hamiltonWalkExists(n0, n9))
        self.assertTrue(self.g0.component(n10).hamiltonWalkExists(n10, n13))
        tmp.connect(n0, [n9]), self.assertTrue(tmp.hamiltonTourExists())
        tmp = DirectedGraph.copy(self.g1)
        self.assertFalse(tmp.hamiltonTourExists())
        self.assertTrue(tmp.hamiltonWalkExists(n1, n4))
        tmp.connect(n1, [n4])
        self.assertTrue(tmp.hamiltonTourExists())
        for n in [n0, n1, n2, n3, n4, n5]:
            self.assertFalse(self.g2.hamiltonWalkExists(n6, n))
        tmp = DirectedGraph.copy(self.g2)
        tmp.connect(n4, [n5]), tmp.connect(n2, [n3])
        self.assertTrue(tmp.hamiltonWalkExists(n6, n1))
    def test_euler_walk_and_tour(self):
        tmp = DirectedGraph.subgraph(self.g0, n0)
        tmp.connect(n0, [n7]), tmp.connect(n5, [n1], [n8]), tmp.disconnect(n4, [n6]), tmp.disconnect(n5, [n7])
        res = tmp.euler_walk(n0, n9)
        n = len(res)
        self.assertEqual(n, len(tmp.links) + 1)
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, tmp.prev(res[i]))
            u = res[i]
        tmp.connect(n0, [n9])
        res1 = tmp.euler_tour()
        self.assertEqual(len(res1), n)
        u = res1[0]
        for i in range(1, n):
            self.assertIn(u, tmp.prev(res1[i]))
            u = res1[i]
        self.assertIn(u, tmp.prev(res1[0]))
    def test_hamilton_walk_and_tour(self):
        tmp = DirectedGraph.subgraph(self.g0, n0)
        res = tmp.hamiltonWalk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertEqual(n9, res[-1])
        tmp.disconnect(n5, [n4])
        self.assertFalse(tmp.hamiltonWalk())
        tmp.connect(n5, [n4]), tmp.connect(n0, [n9])
        res1 = tmp.hamiltonTour()
        self.assertIn(res1[0], tmp.next(res1[-1]))
        for i in range(n - 1):
            self.assertIn(res[i], tmp.prev(res[i + 1]))
            self.assertIn(res1[i], tmp.prev(res1[i + 1]))
        self.assertIn(res1[-2], tmp.prev(res1[-1]))
    def test_isomorphic_function(self):
        pass
        # tmp = DirectedGraph(n10, n11, n12, n13, n14, n15)
        # # tmp.connect(n15, n10, n12, n13, n14), tmp.connect(n11, n12, n13, n14), tmp.connect(n10, n12)
        # self.assertTrue(self.g1.isomorphicFunction(tmp)), tmp.disconnect(n11, n13)
        # self.assertFalse(self.g1.isomorphicFunction(tmp))
    def test_equal(self):
        pass
        # self.assertNotEqual(self.g1, self.g2)
        # self.g2.remove(n6), self.g1.disconnect(n5, n2, n3), self.g1.disconnect(n1, n4)
        # # self.g1.connect(n3, n2, n4), self.g1.connect(n0, n1)
        # self.assertEqual(self.g1, self.g2)
        # # self.g2.add(n6, n5), self.g1.connect(n5, n2, n3), self.g1.connect(n1, n4)
        # self.g1.disconnect(n3, n2, n4), self.g1.disconnect(n0, n1)
    def test_add(self):
        pass
        # tmp = DirectedGraph(n10, n11, n12, n13, n14, n15)
        # # tmp.connect(n15, n10, n12, n13, n14), tmp.connect(n11, n12, n13, n14), tmp.connect(n10, n12)
        # res = self.g3 + tmp
        # helper = DirectedGraph(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15)
        # # helper.connect(n1, n0, n3, n4, n5), helper.connect(n2, n0, n6, n7), helper.connect(n3, n8, n9), helper.connect(n5, n10, n11), helper.connect(n11, n12, n13, n14), helper.connect(n10, n12, n15), helper.connect(n15, n12, n13, n14)
        # self.assertEqual(res, helper)
class TestWeightedNodesDirectedGraph(TestDirectedGraph):
    def setUp(self):
        self.g0 = WeightedNodesDirectedGraph(dict([(n0, (7, ([], []))), (n1, (3, ([n0, n8], [n2]))), (n2, (5, ([n0], [n3, n4]))), (n3, (2, ([n6], [n7, n8]))), (n4, (8, ([n6], [n5]))), (n5, (4, ([n7], [n6]))), (n6, (6, ([], []))), (n7, (6, ([n8], [n9]))), (n8, (2, ([], []))), (n9, (5, ([], []))), (n10, (4, ([], []))), (n11, (2, ([n10], [n12, n13]))), (n12, (1, ([], [n13]))), (n13, (3, ([], []))), (n14, (6, ([], [])))]))
        self.g1 = WeightedNodesDirectedGraph(dict([(n0, (3, ([], []))), (n1, (2, ([n3], [n0, n2, n4, n5]))), (n2, (4, ([n0, n1], [n3]))), (n3, (6, ([], []))), (n4, (5, ([], []))), (n5, (1, ([n1, n3], [n4])))]))
        self.g2 = WeightedNodesDirectedGraph(dict([(n0, (7, ([n2, n5], [n1]))), (n1, (6, ([], []))), (n2, (2, ([n5], [n1, n3]))), (n3, (4, ([n1, n4], []))), (n4, (3, ([], []))), (n5, (5, ([n4, n6], []))), (n6, (4, ([], [])))]))
        self.g3 = WeightedNodesDirectedGraph(dict([(n0, (7, ([], []))), (n1, (4, ([n0, n2], [n5]))), (n2, (3, ([n0, n3], [n4]))), (n3, (5, ([], []))), (n4, (6, ([], [n5]))), (n5, (2, ([], []))), (n6, (2, ([], [n7, n8]))), (n7, (1, ([], []))), (n8, (6, ([], []))), (n9, (4, ([n7, n8, n10], []))), (n10, (5, ([], [n11]))), (n11, (8, ([], [])))]))
    def test_copy(self): self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))
    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedNodesDirectedGraph(dict([(u, (self.g0.node_weights(u), (self.g0.next(u), self.g0.prev(u)))) for u in self.g0.nodes]), self.g0.f))
        self.assertEqual(self.g1.transposed(), WeightedNodesDirectedGraph(dict([(u, (self.g1.node_weights(u), (self.g1.next(u), self.g1.prev(u)))) for u in self.g1.nodes]), self.g1.f))
        self.assertEqual(self.g2.transposed(), WeightedNodesDirectedGraph(dict([(u, (self.g2.node_weights(u), (self.g2.next(u), self.g2.prev(u)))) for u in self.g2.nodes]), self.g2.f))
        self.assertEqual(self.g3.transposed(), WeightedNodesDirectedGraph(dict([(u, (self.g3.node_weights(u), (self.g3.next(u), self.g3.prev(u)))) for u in self.g3.nodes]), self.g3.f))
    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedNodesDirectedGraph(dict([(n0, (7, ([], []))), (n1, (3, ([n0, n8], [n2]))), (n2, (5, ([n0], [n3, n4]))), (n3, (2, ([n6], [n7, n8]))), (n4, (8, ([n6], [n5]))), (n5, (4, ([n7], [n6]))), (n6, (6, ([], []))), (n7, (6, ([n8], [n9]))), (n8, (2, ([], []))), (n9, (5, ([], [])))])))
        self.assertEqual(self.g0.component(n10), WeightedNodesDirectedGraph(dict([(n10, (4, ([], []))), (n11, (2, ([n10], [n12, n13]))), (n12, (1, ([], [n13]))), (n13, (3, ([], [])))])))
        self.assertEqual(self.g0.component(n14), WeightedNodesDirectedGraph(dict([(n14, (6, ([], [])))])))
        self.assertEqual((self.g1.component(n0), self.g2.component(n0)), (self.g1, self.g2))
        self.assertEqual(self.g3.component(n0), WeightedNodesDirectedGraph(dict([(n0, (7, ([], []))), (n1, (4, ([n0, n2], [n5]))), (n2, (3, ([n0, n3], [n4]))), (n3, (5, ([], []))), (n4, (6, ([], [n5]))), (n5, (2, ([], [])))])))
        self.assertEqual(self.g3.component(n7), WeightedNodesDirectedGraph(dict([(n6, (2, ([], [n7, n8]))), (n7, (1, ([], []))), (n8, (6, ([], []))), (n9, (4, ([n7, n8, n10], []))), (n10, (5, ([], [n11]))), (n11, (8, ([], [])))])))
    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n0), self.g0.component(n0))
        self.assertEqual(self.g0.subgraph(n2), WeightedNodesDirectedGraph(dict([(n1, (3, ([n8], [n2]))), (n2, (5, ([], [n3, n4]))), (n3, (2, ([n6], [n7, n8]))), (n4, (8, ([n6], [n5]))), (n5, (4, ([n7], [n6]))), (n6, (6, ([], []))), (n7, (6, ([n8], [n9]))), (n8, (2, ([], []))), (n9, (5, ([], [])))])))
        self.assertEqual(self.g0.subgraph(n10), self.g0.component(n10))
        self.assertEqual(self.g0.subgraph(n11), WeightedNodesDirectedGraph(dict([(n11, (2, ([], [n12, n13]))), (n12, (1, ([], [n13]))), (n13, (3, ([], [])))])))
        self.assertEqual(self.g1.subgraph(n0), self.g1.subgraph(n1))
        self.assertEqual(self.g1.subgraph(n1), self.g1.component(n0))
        self.assertEqual(self.g1.subgraph(n1), self.g1.subgraph(n2))
        self.assertEqual(self.g1.subgraph(n2), self.g1.subgraph(n3))
        self.assertEqual(self.g1.subgraph(n5), WeightedNodesDirectedGraph(dict([(n4, (5, ([n5], []))), (n5, (1, ([], [])))])))
        self.assertEqual(self.g2.subgraph(n4), self.g2.remove(n6)), self.g2.add((n6, 4), [], [n5])
        self.assertEqual(self.g2.subgraph(n6), self.g2.remove(n4)), self.g2.add((n4, 3), [], [n3, n5])
        self.assertEqual(self.g3.subgraph(n0), self.g3.remove(n3).component(n5)), self.g3.add((n3, 5), [], [n2])
        self.assertEqual(self.g3.subgraph(n3), self.g3.remove(n0).component(n5)), self.g3.add((n0, 7), [], [n1, n2])
    def test_minimal_path_nodes(self):
        pass
    def test_isomorphic_function(self):
        pass
    def test_equal(self):
        pass
    def test_add(self):
        pass
class TestWeightedLinksDirectedGraph(TestDirectedGraph):
    def setUp(self):
        self.g0 = WeightedLinksDirectedGraph(dict([(n1, (dict([(n0, 2), (n8, -1)]), dict([(n2, 3)]))), (n2, (dict([(n0, 4)]), dict([(n3, -6), (n4, 5)]))), (n3, (dict([(n6, 3)]), dict([(n7, -3), (n8, 5)]))), (n4, (dict([(n6, 2)]), dict([(n5, 0)]))), (n5, (dict([(n7, 1)]), dict([(n6, 5)]))), (n7, (dict([(n8, 4)]), dict([(n9, 3)]))), (n11, (dict([(n10, 2)]), dict([(n12, 6), (n13, 10)]))), (n13, (dict([(n12, 3)]), dict())), (n14, (dict(), dict()))]))
        self.g1 = WeightedLinksDirectedGraph(dict([(n1, (dict([(n3, 3)]), dict([(n0, 1), (n2, 4), (n4, 9), (n5, 3)]))), (n2, (dict([(n0, 2), (n1, 4)]), dict([(n3, -6)]))), (n5, (dict([(n1, 3), (n3, 2)]), dict([(n4, 5)])))]))
        self.g2 = WeightedLinksDirectedGraph(dict([(n0, (dict([(n2, 2), (n5, 4)]), dict([(n1, 2)]))), (n2, (dict([(n5, 3)]), dict([(n1, 6), (n3, -1)]))), (n3, (dict([(n1, 1), (n4, 4)]), dict())), (n5, (dict([(n4, 1), (n6, 2)]), dict()))]))
        self.g3 = WeightedLinksDirectedGraph(dict([(n1, (dict([(n0, 3), (n2, 4)]), dict([(n5, 5)]))), (n2, (dict([(n0, 6), (n3, 1)]), dict([(n4, 0)]))), (n4, (dict(), dict([(n5, 1)]))), (n6, (dict(), dict([(n7, 2), (n8, 4)]))), (n9, (dict([(n7, 3), (n8, 0), (n10, 4)]), dict())), (n11, (dict([(n10, 1)]), dict()))]))
    def test_copy(self): self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))
    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedLinksDirectedGraph(dict([(u, (self.g0.link_weights(u), dict([(v, self.g0.link_weights(v, u)) for v in self.g0.prev(u)]))) for u in self.g0.nodes]), self.g0.f))
        self.assertEqual(self.g1.transposed(), WeightedLinksDirectedGraph(dict([(u, (self.g1.link_weights(u), dict([(v, self.g1.link_weights(v, u)) for v in self.g1.prev(u)]))) for u in self.g1.nodes]), self.g1.f))
        self.assertEqual(self.g2.transposed(), WeightedLinksDirectedGraph(dict([(u, (self.g2.link_weights(u), dict([(v, self.g2.link_weights(v, u)) for v in self.g2.prev(u)]))) for u in self.g2.nodes]), self.g2.f))
        self.assertEqual(self.g3.transposed(), WeightedLinksDirectedGraph(dict([(u, (self.g3.link_weights(u), dict([(v, self.g3.link_weights(v, u)) for v in self.g3.prev(u)]))) for u in self.g3.nodes]), self.g3.f))
    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedLinksDirectedGraph(dict([(n1, (dict([(n0, 2), (n8, -1)]), dict([(n2, 3)]))), (n2, (dict([(n0, 4)]), dict([(n3, -6), (n4, 5)]))), (n3, (dict([(n6, 3)]), dict([(n7, -3), (n8, 5)]))), (n4, (dict([(n6, 2)]), dict([(n5, 0)]))), (n5, (dict([(n7, 1)]), dict([(n6, 5)]))), (n7, (dict([(n8, 4)]), dict([(n9, 3)])))])))
        self.assertEqual(self.g0.component(n10), WeightedLinksDirectedGraph(dict([(n11, (dict([(n10, 2)]), dict([(n12, 6), (n13, 10)]))), (n13, (dict([(n12, 3)]), dict()))])))
        self.assertEqual(self.g0.component(n14), WeightedLinksDirectedGraph(dict([(n14, (dict(), dict()))])))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), WeightedLinksDirectedGraph(dict([(n1, (dict([(n0, 3), (n2, 4)]), dict([(n5, 5)]))), (n2, (dict([(n0, 6), (n3, 1)]), dict([(n4, 0)]))), (n4, (dict(), dict([(n5, 1)])))])))
        self.assertEqual(self.g3.component(n6), WeightedLinksDirectedGraph(dict([(n6, (dict(), dict([(n7, 2), (n8, 4)]))), (n9, (dict([(n7, 3), (n8, 0), (n10, 4)]), dict())), (n11, (dict([(n10, 1)]), dict()))])))
    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n0), self.g0.component(n0))
        self.assertEqual(self.g0.subgraph(n10), self.g0.component(n10))
        self.assertEqual(self.g0.subgraph(n11), WeightedLinksDirectedGraph(dict([(n11, (dict(), dict([(n12, 6), (n13, 10)]))), (n13, (dict([(n12, 3)]), dict()))])))
        self.assertEqual(self.g1.subgraph(n0), self.g1)
        self.assertEqual(self.g1.subgraph(n1), self.g1)
        tmp = self.g2.copy()
        self.assertEqual(self.g2.subgraph(n6), tmp.remove(n4)), tmp.add(n4, points_to_weights=dict([(n3, 4), (n5, 1)]))
        self.assertEqual(self.g2.subgraph(n4), tmp.remove(n6))
        tmp = self.g3.copy()
        self.assertEqual(self.g3.subgraph(n0), tmp.remove(n3).component(n0)), tmp.add(n3, points_to_weights=dict([(n2, 1)]))
        self.assertEqual(self.g3.subgraph(n3), tmp.remove(n0).component(n1))
        self.assertEqual(self.g3.subgraph(n6), WeightedLinksDirectedGraph(dict([(n6, (dict(), dict([(n7, 2), (n8, 4)]))), (n9, (dict([(n7, 3), (n8, 0)]), dict()))])))
        self.assertEqual(self.g3.subgraph(n10), WeightedLinksDirectedGraph(dict([(n10, (dict(), dict([(n9, 4), (n11, 1)])))])))
    def test_minimal_path_links(self):
        pass
    def test_isomorphic_function(self):
        pass
    def test_equal(self):
        pass
    def test_add(self):
        pass
class TestWeightedDirectedGraph(TestWeightedNodesDirectedGraph, TestWeightedLinksDirectedGraph):
    def setUp(self):
        self.g0 = WeightedDirectedGraph(dict([(n0, (7, (dict(), dict()))), (n1, (3, (dict([(n0, 2), (n8, -1)]), dict([(n2, 3)])))), (n2, (5, (dict([(n0, 4)]), dict([(n3, -6), (n4, 5)])))), (n3, (2, (dict([(n6, 3)]), dict([(n7, -3), (n8, 5)])))), (n4, (8, (dict([(n6, 2)]), dict([(n5, 0)])))), (n5, (4, (dict([(n7, 1)]), dict([(n6, 5)])))), (n6, (6, (dict(), dict()))), (n7, (6, (dict([(n8, 4)]), dict([(n9, 3)])))), (n8, (2, (dict(), dict()))), (n9, (5, (dict(), dict()))), (n10, (4, (dict(), dict()))), (n11, (2, (dict([(n10, 2)]), dict([(n12, 6), (n13, 10)])))), (n12, (1, (dict(), dict()))), (n13, (3, (dict([(n12, 3)]), dict()))), (n14, (6, (dict(), dict())))]))
        self.g1 = WeightedDirectedGraph(dict([(n0, (3, (dict(), dict()))), (n1, (2, (dict([(n3, 3)]), dict([(n0, 1), (n2, 4), (n4, 9), (n5, 3)])))), (n2, (4, (dict([(n0, 2), (n1, 4)]), dict([(n3, -6)])))), (n3, (6, (dict(), dict()))), (n4, (5, (dict(), dict()))), (n5, (1, (dict([(n1, 3), (n3, 2)]), dict([(n4, 5)]))))]))
        self.g2 = WeightedDirectedGraph(dict([(n0, (7, (dict([(n2, 2), (n5, 4)]), dict([(n1, 2)])))), (n1, (6, (dict(), dict()))), (n2, (2, (dict([(n5, 3)]), dict([(n1, 6), (n3, -1)])))), (n3, (4, (dict([(n1, 1), (n4, 4)]), dict()))), (n4, (3, (dict(), dict()))), (n5, (5, (dict([(n4, 1), (n6, 2)]), dict()))), (n6, (4, (dict(), dict())))]))
        self.g3 = WeightedDirectedGraph(dict([(n0, (7, (dict(), dict()))), (n1, (4, (dict([(n0, 3), (n2, 4)]), dict([(n5, 5)])))), (n2, (2, (dict([(n0, 6), (n3, 1)]), dict([(n4, 0)])))), (n3, (5, (dict(), dict()))), (n4, (3, (dict(), dict([(n5, 1)])))), (n5, (2, (dict(), dict()))), (n6, (2, (dict(), dict([(n7, 2), (n8, 4)])))), (n7, (1, (dict(), dict()))),  (n8, (6, (dict(), dict()))), (n9, (4, (dict([(n7, 3), (n8, 0), (n10, 4)]), dict()))), (n10, (5, (dict(), dict()))), (n11, (8, (dict([(n10, 1)]), dict())))]))
    def test_copy(self): self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))
    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedDirectedGraph(dict([(u, (self.g0.node_weights(u), (self.g0.link_weights(u), dict([(v, self.g0.link_weights(v, u)) for v in self.g0.prev(u)])))) for u in self.g0.nodes])), self.g0.f)
        self.assertEqual(self.g1.transposed(), WeightedDirectedGraph(dict([(u, (self.g1.node_weights(u), (self.g1.link_weights(u), dict([(v, self.g1.link_weights(v, u)) for v in self.g1.prev(u)])))) for u in self.g1.nodes])), self.g1.f)
        self.assertEqual(self.g2.transposed(), WeightedDirectedGraph(dict([(u, (self.g2.node_weights(u), (self.g2.link_weights(u), dict([(v, self.g2.link_weights(v, u)) for v in self.g2.prev(u)])))) for u in self.g2.nodes])), self.g2.f)
        self.assertEqual(self.g3.transposed(), WeightedDirectedGraph(dict([(u, (self.g3.node_weights(u), (self.g3.link_weights(u), dict([(v, self.g3.link_weights(v, u)) for v in self.g3.prev(u)])))) for u in self.g3.nodes])), self.g3.f)
    def test_component(self):
        self.assertEqual(self.g0.component(n9), WeightedDirectedGraph(dict([(n0, (7, (dict(), dict()))), (n1, (3, (dict([(n0, 2), (n8, -1)]), dict([(n2, 3)])))), (n2, (5, (dict([(n0, 4)]), dict([(n3, -6), (n4, 5)])))), (n3, (2, (dict([(n6, 3)]), dict([(n7, -3), (n8, 5)])))), (n4, (8, (dict([(n6, 2)]), dict([(n5, 0)])))), (n5, (4, (dict([(n7, 1)]), dict([(n6, 5)])))), (n6, (6, (dict(), dict()))), (n7, (6, (dict([(n8, 4)]), dict([(n9, 3)])))), (n8, (2, (dict(), dict()))), (n9, (5, (dict(), dict())))])))
        self.assertEqual(self.g0.component(n12), WeightedDirectedGraph(dict([(n10, (4, (dict(), dict()))), (n11, (2, (dict([(n10, 2)]), dict([(n12, 6), (n13, 10)])))), (n12, (1, (dict(), dict()))), (n13, (3, (dict([(n12, 3)]), dict())))])))
        self.assertEqual(self.g0.component(n14), WeightedDirectedGraph(dict([(n14, (6, (dict(), dict())))])))
        self.assertEqual(self.g1.component(n5), self.g1)
        self.assertEqual(self.g2.component(n6), self.g2)
        self.assertEqual(self.g3.component(n2), WeightedDirectedGraph(dict([(n0, (7, (dict(), dict()))), (n1, (4, (dict([(n0, 3), (n2, 4)]), dict([(n5, 5)])))), (n2, (2, (dict([(n0, 6), (n3, 1)]), dict([(n4, 0)])))), (n3, (5, (dict(), dict()))), (n4, (3, (dict(), dict([(n5, 1)])))), (n5, (2, (dict(), dict())))])))
        self.assertEqual(self.g3.component(n9), WeightedDirectedGraph(dict([(n6, (2, (dict(), dict([(n7, 2), (n8, 4)])))), (n7, (1, (dict(), dict()))),  (n8, (6, (dict(), dict()))), (n9, (4, (dict([(n7, 3), (n8, 0), (n10, 4)]), dict()))), (n10, (5, (dict(), dict()))), (n11, (8, (dict([(n10, 1)]), dict())))])))
    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n0), self.g0.component(n0))
        tmp = self.g0.component(n0)
        self.assertEqual(self.g0.subgraph(n1), tmp.remove(n0))
        self.assertEqual(self.g0.subgraph(n10), self.g0.component(n10))
        tmp = self.g0.component(n10)
        self.assertEqual(self.g0.subgraph(n11), tmp.remove(n10))
        self.assertEqual(self.g1.subgraph(n0), self.g1)
        self.assertEqual(self.g1.subgraph(n1), self.g1)
        self.assertEqual(self.g1.subgraph(n2), self.g1)
        self.assertEqual(self.g1.subgraph(n3), self.g1)
        tmp = self.g2.copy()
        self.assertEqual(self.g2.subgraph(n6), tmp.remove(n4))
        tmp.add((n4, 3), points_to_weights=dict([(n5, 1), (n3, 4)]))
        self.assertEqual(self.g2.subgraph(n4), tmp.remove(n6))
        tmp = self.g3.component(n0)
        self.assertEqual(self.g3.subgraph(n0), tmp.remove(n3))
        tmp.add((n3, 5), points_to_weights=dict([(n2, 1)]))
        self.assertEqual(self.g3.subgraph(n3), tmp.remove(n0))
        tmp = self.g3.component(n6)
        self.assertEqual(self.g3.subgraph(n6), tmp.remove(n10, n11))
        self.assertEqual(self.g3.subgraph(n10), WeightedDirectedGraph(dict([(n9, (4, (dict(), dict()))), (n10, (5, (dict(), dict([(n9, 4), (n11, 1)])))), (n11, (8, (dict(), dict())))])))
    def test_minimal_path(self):
        res, s = self.g0.minimalPath(n2, n5), self.g0.node_weights(n2)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g0.next(res[0][i]))
            s += self.g0.node_weights(res[0][i + 1]) + self.g0.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 9)
        res, s = self.g1.minimalPath(n0, n4), self.g1.node_weights(n0)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g1.next(res[0][i]))
            s += self.g1.node_weights(res[0][i + 1]) + self.g1.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 22)
        res, s = self.g2.minimalPath(n5, n1), self.g2.node_weights(n5)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g2.next(res[0][i]))
            s += self.g2.node_weights(res[0][i + 1]) + self.g2.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 22)
        res, s = self.g3.minimalPath(n6, n9), self.g3.node_weights(n6)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g3.next(res[0][i]))
            s += self.g3.node_weights(res[0][i + 1]) + self.g3.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 12)
    def test_isomorphic_function(self):
        pass
    def test_equal(self):
        pass
    def test_add(self):
        pass
class TestTree(TestCase):
    pass
class TestWeightedTree(TestTree):
    pass
if __name__ == '__main__': main()