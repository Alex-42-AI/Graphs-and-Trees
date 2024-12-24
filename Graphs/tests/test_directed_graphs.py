from unittest import TestCase, main

from Personal.Graphs.src.implementation.directed_graph import *

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7), Node(8), Node(9), Node(10), Node(11), Node(12), Node(13), Node(14), Node(15)


class TestDirectedGraph(TestCase):
    def setUp(self):
        self.g0 = DirectedGraph({n1: ([n0, n8], [n2]), n2: ([n0], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]), n7: ([n8], [n9]), n11: ([n10], [n12, n13]), n12: ([], [n13]), n14: ([], [])})
        self.g1 = DirectedGraph({n1: ([n3], [n0, n2, n4, n5]), n2: ([n0, n1], [n3]), n5: ([n3], [n4])})
        self.g2 = DirectedGraph({n0: ([n2, n5], [n1]), n2: ([n5], [n1, n3]), n3: ([n1, n4], []), n5: ([n4, n6], [])})
        self.g3 = DirectedGraph({n1: ([n0, n2], [n5]), n2: ([n0, n3], [n4]), n4: ([], [n5]), n6: ([], [n7, n8]), n9: ([n7, n8, n10], []), n10: ([], [n11])})

    def test_sources(self):
        self.assertEqual((self.g0.sources, self.g1.sources, self.g2.sources, self.g3.sources), ([n0, n10, n14], [], [n4, n6], [n0, n3, n6, n10]))

    def test_sinks(self):
        self.assertEqual((self.g0.sinks, self.g1.sinks, self.g2.sinks, self.g3.sinks), ([n9, n13, n14], [n4], [n3], [n5, n9, n11]))

    def test_get_degrees(self):
        self.assertEqual(self.g0.degrees(), {n0: [0, 2], n1: [2, 1], n2: [2, 2], n3: [2, 2], n4: [2, 1], n5: [2, 1], n6: [1, 2], n7: [2, 2], n8: [1, 2], n9: [1, 0], n10: [0, 1], n11: [1, 2], n12: [1, 1], n13: [2, 0], n14: [0, 0]})

    def test_copy(self):
        self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), DirectedGraph({u: (self.g0.next(u), self.g0.prev(u)) for u in self.g0.nodes}))
        self.assertEqual(self.g1.transposed(), DirectedGraph({u: (self.g1.next(u), self.g1.prev(u)) for u in self.g1.nodes}))
        self.assertEqual(self.g2.transposed(), DirectedGraph({u: (self.g2.next(u), self.g2.prev(u)) for u in self.g2.nodes}))
        self.assertEqual(self.g3.transposed(), DirectedGraph({u: (self.g3.next(u), self.g3.prev(u)) for u in self.g3.nodes}))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(DirectedGraph({n1: ([n0, n8], [n2]), n2: ([n0], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]), n7: ([n8], [n9])}), res)
        self.assertIn(DirectedGraph({n11: ([n10], [n12, n13]), n12: ([], [n13])}), res)
        self.assertIn(DirectedGraph({n14: ([], [])}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_connected(self):
        self.assertFalse(self.g0.connected())
        self.assertTrue(self.g1.connected())
        self.assertTrue(self.g2.connected())
        self.assertFalse(self.g3.connected())

    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n1), DirectedGraph({n1: ([n8], [n2]), n2: ([], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]), n7: ([n8], [n9])}))
        self.assertEqual(self.g1.subgraph(n0), self.g1), self.assertEqual(self.g1.subgraph(n4), DirectedGraph({n4: ([], [])}))
        self.assertEqual(self.g2.subgraph(n2), DirectedGraph({n1: ([n0, n2], [n3]), n2: ([], [n0, n1, n3])}))
        self.assertEqual(self.g3.subgraph(n6), DirectedGraph({n6: ([], [n7, n8]), n9: ([n7, n8], [])}))

    def test_component(self):
        self.assertEqual(self.g0.component(n9), DirectedGraph({n1: ([n0, n8], [n2]), n2: ([n0], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]), n7: ([n8], [n9])}))
        self.assertEqual(self.g0.component(n13), DirectedGraph({n11: ([n10], [n12, n13]), n12: ([], [n13])}))
        self.assertEqual(self.g1.component(n4), self.g1)
        self.assertEqual(self.g2.component(n1), self.g2)
        self.assertEqual(self.g3.component(n4), DirectedGraph({n1: ([n0, n2], [n5]), n2: ([n0, n3], [n4]), n4: ([], [n5])}))
        self.assertEqual(self.g3.component(n10), DirectedGraph({n6: ([], [n7, n8]), n9: ([n7, n8, n10], []), n10: ([], [n11])}))

    def test_strongly_connected_components(self):
        res = self.g0.strongly_connected_components()
        self.assertEqual(len(res), 8)
        self.assertEqual(sum(map(len, res)), len(self.g0.nodes))
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g0.reachable(u, v))
        res = self.g1.strongly_connected_components()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), len(self.g1.nodes))
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g1.reachable(u, v))
        res = self.g2.strongly_connected_components()
        self.assertEqual(len(res), 7)
        self.assertEqual(sum(map(len, res)), len(self.g2.nodes))
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g2.reachable(u, v))
        res = self.g3.strongly_connected_components()
        self.assertEqual(len(res), 12)
        self.assertEqual(sum(map(len, res)), len(self.g3.nodes))
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g3.reachable(u, v))

    def test_scc_dag(self):
        pass

    def test_has_cycle(self):
        self.assertTrue(self.g0.has_cycle())
        self.assertTrue(self.g1.has_cycle())
        self.assertFalse(self.g2.has_cycle())
        self.assertFalse(self.g3.has_cycle())

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

    def test_path_with_length(self):
        self.assertTrue(self.g0.path_with_length(n0, n1, 9))
        self.assertFalse(self.g0.path_with_length(n0, n1, 10))
        self.assertTrue(self.g0.path_with_length(n0, n5, 6))
        self.assertTrue(self.g0.path_with_length(n10, n13, 3))
        self.assertTrue(self.g0.path_with_length(n4, n8, 4))

    def test_cycle_with_length(self):
        self.assertTrue(self.g0.cycle_with_length(3))
        self.assertTrue(self.g0.cycle_with_length(4))
        self.assertFalse(self.g0.cycle_with_length(6))
        self.assertFalse(self.g0.cycle_with_length(9))
        self.assertTrue(self.g1.cycle_with_length(3))
        self.assertTrue(self.g1.cycle_with_length(4))

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
        self.assertFalse(self.g0.hamilton_tour_exists())
        self.assertTrue(tmp.hamilton_walk_exists(n0, n9))
        self.assertTrue(self.g0.component(n10).hamilton_walk_exists(n10, n13))
        tmp.connect(n0, [n9]), self.assertTrue(tmp.hamilton_tour_exists())
        tmp = DirectedGraph.copy(self.g1)
        self.assertFalse(tmp.hamilton_tour_exists())
        self.assertTrue(tmp.hamilton_walk_exists(n1, n4))
        tmp.connect(n1, [n4])
        self.assertTrue(tmp.hamilton_tour_exists())
        for n in [n0, n1, n2, n3, n4, n5]:
            self.assertFalse(self.g2.hamilton_walk_exists(n6, n))
        tmp = DirectedGraph.copy(self.g2)
        tmp.connect(n4, [n5]), tmp.connect(n2, [n3])
        self.assertTrue(tmp.hamilton_walk_exists(n6, n1))

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
        res = tmp.hamilton_walk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertEqual(n9, res[-1])
        tmp.disconnect(n5, [n4])
        self.assertFalse(tmp.hamilton_walk())
        tmp.connect(n5, [n4]), tmp.connect(n0, [n9])
        res1 = tmp.hamilton_tour()
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
        self.g0 = WeightedNodesDirectedGraph({n0: (7, ([], [])), n1: (3, ([n0, n8], [n2])), n2: (5, ([n0], [n3, n4])),
              n3: (2, ([n6], [n7, n8])), n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])),
              n7: (6, ([n8], [n9])), n8: (2, ([], [])), n9: (5, ([], [])), n10: (4, ([], [])),
              n11: (2, ([n10], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], [])), n14: (6, ([], []))})
        self.g1 = WeightedNodesDirectedGraph({n0: (3, ([], [])), n1: (2, ([n3], [n0, n2, n4, n5])),
                          n2: (4, ([n0, n1], [n3])), n3: (6, ([], [])), n4: (5, ([], [])), n5: (1, ([n1, n3], [n4]))})
        self.g2 = WeightedNodesDirectedGraph({n0: (7, ([n2, n5], [n1])), n1: (6, ([], [])), n2: (2, ([n5], [n1, n3])),
                              n3: (4, ([n1, n4], [])), n4: (3, ([], [])), n5: (5, ([n4, n6], [])), n6: (4, ([], []))})
        self.g3 = WeightedNodesDirectedGraph({n0: (7, ([], [])), n1: (4, ([n0, n2], [n5])), n2: (3, ([n0, n3], [n4])),
          n3: (5, ([], [])), n4: (6, ([], [n5])), n5: (2, ([], [])), n6: (2, ([], [n7, n8])), n7: (1, ([], [])),
                  n8: (6, ([], [])), n9: (4, ([n7, n8, n10], [])), n10: (5, ([], [n11])), n11: (8, ([], []))})

    def test_copy(self):
        self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedNodesDirectedGraph({u: (self.g0.node_weights(u), (self.g0.next(u), self.g0.prev(u))) for u in self.g0.nodes}, self.g0.f))
        self.assertEqual(self.g1.transposed(), WeightedNodesDirectedGraph({u: (self.g1.node_weights(u), (self.g1.next(u), self.g1.prev(u))) for u in self.g1.nodes}, self.g1.f))
        self.assertEqual(self.g2.transposed(), WeightedNodesDirectedGraph({u: (self.g2.node_weights(u), (self.g2.next(u), self.g2.prev(u))) for u in self.g2.nodes}, self.g2.f))
        self.assertEqual(self.g3.transposed(), WeightedNodesDirectedGraph({u: (self.g3.node_weights(u), (self.g3.next(u), self.g3.prev(u))) for u in self.g3.nodes}, self.g3.f))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedNodesDirectedGraph({n0: (7, ([], [])), n1: (3, ([n0, n8], [n2])), n2: (5, ([n0], [n3, n4])), n3: (2, ([n6], [n7, n8])), n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])), n9: (5, ([], []))}), res)
        self.assertIn(WeightedNodesDirectedGraph({n10: (4, ([], [])), n11: (2, ([n10], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], []))}), res)
        self.assertIn(WeightedNodesDirectedGraph({n14: (6, ([], []))}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedNodesDirectedGraph({n0: (7, ([], [])), n1: (3, ([n0, n8], [n2])), n2: (5, ([n0], [n3, n4])), n3: (2, ([n6], [n7, n8])), n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])), n9: (5, ([], []))}))
        self.assertEqual(self.g0.component(n10), WeightedNodesDirectedGraph({n10: (4, ([], [])), n11: (2, ([n10], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], []))}))
        self.assertEqual(self.g0.component(n14), WeightedNodesDirectedGraph({n14: (6, ([], []))}))
        self.assertEqual((self.g1.component(n0), self.g2.component(n0)), (self.g1, self.g2))
        self.assertEqual(self.g3.component(n0), WeightedNodesDirectedGraph({n0: (7, ([], [])), n5: (2, ([], [])), n1: (4, ([n0, n2], [n5])), n2: (3, ([n0, n3], [n4])), n3: (5, ([], [])), n4: (6, ([], [n5]))}))
        self.assertEqual(self.g3.component(n7), WeightedNodesDirectedGraph({n6: (2, ([], [n7, n8])), n7: (1, ([], [])), n8: (6, ([], [])), n9: (4, ([n7, n8, n10], [])), n10: (5, ([], [n11])), n11: (8, ([], []))}))

    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n0), self.g0.component(n0))
        self.assertEqual(self.g0.subgraph(n2), WeightedNodesDirectedGraph({n1: (3, ([n8], [n2])), n2: (5, ([], [n3, n4])), n3: (2, ([n6], [n7, n8])), n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])), n9: (5, ([], []))}))
        self.assertEqual(self.g0.subgraph(n10), self.g0.component(n10))
        self.assertEqual(self.g0.subgraph(n11), WeightedNodesDirectedGraph({n11: (2, ([], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], []))}))
        self.assertEqual(self.g1.subgraph(n0), self.g1.subgraph(n1))
        self.assertEqual(self.g1.subgraph(n1), self.g1.component(n0))
        self.assertEqual(self.g1.subgraph(n1), self.g1.subgraph(n2))
        self.assertEqual(self.g1.subgraph(n2), self.g1.subgraph(n3))
        self.assertEqual(self.g1.subgraph(n5), WeightedNodesDirectedGraph({n4: (5, ([n5], [])), n5: (1, ([], []))}))
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
        self.g0 = WeightedLinksDirectedGraph({n1: ({n0: 2, n8: -1}, {n2: 3}), n2: ({n0: 4}, {n3: -6, n4: 5}),
                              n3: ({n6: 3}, {n7: -3, n8: 5}), n4: ({n6: 2}, {n5: 0}), n5: ({n7: 1}, {n6: 5}),
                      n7: ({n8: 4}, {n9: 3}), n11: ({n10: 2}, {n12: 6, n13: 10}), n13: ({n12: 3}, {}), n14: ({}, {})})
        self.g1 = WeightedLinksDirectedGraph({n1: ({n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3}),
                                              n2: ({n0: 2, n1: 4}, {n3: -6}), n5: ({n1: 3, n3: 2}, {n4: 5})})
        self.g2 = WeightedLinksDirectedGraph({n0: ({n2: 2, n5: 4}, {n1: 2}), n2: ({n5: 3}, {n1: 6, n3: -1}),
                                              n3: ({n1: 1, n4: 4}, {}), n5: ({n4: 1, n6: 2}, {})})
        self.g3 = WeightedLinksDirectedGraph({n1: ({n0: 3, n2: 4}, {n5: 5}), n2: ({n0: 6, n3: 1}, {n4: 0}),
                  n4: ({}, {n5: 1}), n6: ({}, {n7: 2, n8: 4}), n9: ({n7: 3, n8: 0, n10: 4}, {}), n11: ({n10: 1}, {})})

    def test_copy(self):
        self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedLinksDirectedGraph({u: (self.g0.link_weights(u), {v: self.g0.link_weights(v, u) for v in self.g0.prev(u)}) for u in self.g0.nodes}, self.g0.f))
        self.assertEqual(self.g1.transposed(), WeightedLinksDirectedGraph({u: (self.g1.link_weights(u), {v: self.g1.link_weights(v, u) for v in self.g1.prev(u)}) for u in self.g1.nodes}, self.g1.f))
        self.assertEqual(self.g2.transposed(), WeightedLinksDirectedGraph({u: (self.g2.link_weights(u), {v: self.g2.link_weights(v, u) for v in self.g2.prev(u)}) for u in self.g2.nodes}, self.g2.f))
        self.assertEqual(self.g3.transposed(), WeightedLinksDirectedGraph({u: (self.g3.link_weights(u), {v: self.g3.link_weights(v, u) for v in self.g3.prev(u)}) for u in self.g3.nodes}, self.g3.f))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedLinksDirectedGraph({n1: ({n0: 2, n8: -1}, {n2: 3}), n2: ({n0: 4}, {n3: -6, n4: 5}), n3: ({n6: 3}, {n7: -3, n8: 5}), n4: ({n6: 2}, {n5: 0}), n5: ({n7: 1}, {n6: 5}), n7: ({n8: 4}, {n9: 3})}), res)
        self.assertIn(WeightedLinksDirectedGraph({n11: ({n10: 2}, {n12: 6, n13: 10}), n13: ({n12: 3}, {})}), res)
        self.assertIn(WeightedLinksDirectedGraph({n14: ({}, {})}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_component(self):
        self.assertEqual(self.g0.component(n0), WeightedLinksDirectedGraph({n1: ({n0: 2, n8: -1}, {n2: 3}), n2: ({n0: 4}, {n3: -6, n4: 5}), n3: ({n6: 3}, {n7: -3, n8: 5}), n4: ({n6: 2}, {n5: 0}), n5: ({n7: 1}, {n6: 5}), n7: ({n8: 4}, {n9: 3})}))
        self.assertEqual(self.g0.component(n10), WeightedLinksDirectedGraph({n11: ({n10: 2}, {n12: 6, n13: 10}), n13: ({n12: 3}, {})}))
        self.assertEqual(self.g0.component(n14), WeightedLinksDirectedGraph({n14: ({}, {})}))
        self.assertEqual(self.g1.component(n0), self.g1)
        self.assertEqual(self.g2.component(n0), self.g2)
        self.assertEqual(self.g3.component(n0), WeightedLinksDirectedGraph({n1: ({n0: 3, n2: 4}, {n5: 5}), n2: ({n0: 6, n3: 1}, {n4: 0}), n4: ({}, {n5: 1})}))
        self.assertEqual(self.g3.component(n6), WeightedLinksDirectedGraph({n6: ({}, {n7: 2, n8: 4}), n9: ({n7: 3, n8: 0, n10: 4}, {}), n11: ({n10: 1}, {})}))

    def test_subgraph(self):
        self.assertEqual(self.g0.subgraph(n0), self.g0.component(n0))
        self.assertEqual(self.g0.subgraph(n10), self.g0.component(n10))
        self.assertEqual(self.g0.subgraph(n11), WeightedLinksDirectedGraph({n11: ({}, {n12: 6, n13: 10}), n13: ({n12: 3}, {})}))
        self.assertEqual(self.g1.subgraph(n0), self.g1)
        self.assertEqual(self.g1.subgraph(n1), self.g1)
        tmp = self.g2.copy()
        self.assertEqual(self.g2.subgraph(n6), tmp.remove(n4)), tmp.add(n4, points_to_weights={n3: 4, n5: 1})
        self.assertEqual(self.g2.subgraph(n4), tmp.remove(n6))
        tmp = self.g3.copy()
        self.assertEqual(self.g3.subgraph(n0), tmp.remove(n3).component(n0)), tmp.add(n3, points_to_weights={n2: 1})
        self.assertEqual(self.g3.subgraph(n3), tmp.remove(n0).component(n1))
        self.assertEqual(self.g3.subgraph(n6), WeightedLinksDirectedGraph({n6: ({}, {n7: 2, n8: 4}), n9: ({n7: 3, n8: 0}, {})}))
        self.assertEqual(self.g3.subgraph(n10), WeightedLinksDirectedGraph({n10: ({}, {n9: 4, n11: 1})}))

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
        self.g0 = WeightedDirectedGraph({n0: (7, ({}, {})), n1: (3, ({n0: 2, n8: -1}, {n2: 3})),
                         n2: (5, ({n0: 4}, {n3: -6, n4: 5})), n3: (2, ({n6: 3}, {n7: -3, n8: 5})),
                         n4: (8, ({n6: 2}, {n5: 0})), n5: (4, ({n7: 1}, {n6: 5})), n6: (6, ({}, {})),
                         n7: (6, ({n8: 4}, {n9: 3})), n8: (2, ({}, {})), n9: (5, ({}, {})), n10: (4, ({}, {})),
             n11: (2, ({n10: 2}, {n12: 6, n13: 10})), n12: (1, ({}, {})), n13: (3, ({n12: 3}, {})), n14: (6, ({}, {}))})
        self.g1 = WeightedDirectedGraph({n0: (3, ({}, {})), n1: (2, ({n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3})),
         n2: (4, ({n0: 2, n1: 4}, {n3: -6})), n3: (6, ({}, {})), n4: (5, ({}, {})), n5: (1, ({n1: 3, n3: 2}, {n4: 5}))})
        self.g2 = WeightedDirectedGraph({n0: (7, ({n2: 2, n5: 4}, {n1: 2})), n1: (6, ({}, {})),
                                         n2: (2, ({n5: 3}, {n1: 6, n3: -1})), n3: (4, ({n1: 1, n4: 4}, {})),
                                         n4: (3, ({}, {})), n5: (5, ({n4: 1, n6: 2}, {})), n6: (4, ({}, {}))})
        self.g3 = WeightedDirectedGraph({n0: (7, ({}, {})), n1: (4, ({n0: 3, n2: 4}, {n5: 5})),
                     n2: (2, ({n0: 6, n3: 1}, {n4: 0})), n3: (5, ({}, {})), n4: (3, ({}, {n5: 1})), n5: (2, ({}, {})),
                     n6: (2, ({}, {n7: 2, n8: 4})), n7: (1, ({}, {})), n8: (6, ({}, {})),
                     n9: (4, ({n7: 3, n8: 0, n10: 4}, {})), n10: (5, ({}, {})), n11: (8, ({n10: 1}, {}))})

    def test_copy(self):
        self.assertEqual((self.g0, self.g1, self.g2, self.g3), (self.g0.copy(), self.g1.copy(), self.g2.copy(), self.g3.copy()))

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedDirectedGraph({u: (self.g0.node_weights(u), (self.g0.link_weights(u), {v: self.g0.link_weights(v, u) for v in self.g0.prev(u)})) for u in self.g0.nodes}), self.g0.f)
        self.assertEqual(self.g1.transposed(), WeightedDirectedGraph({u: (self.g1.node_weights(u), (self.g1.link_weights(u), {v: self.g1.link_weights(v, u) for v in self.g1.prev(u)})) for u in self.g1.nodes}), self.g1.f)
        self.assertEqual(self.g2.transposed(), WeightedDirectedGraph({u: (self.g2.node_weights(u), (self.g2.link_weights(u), {v: self.g2.link_weights(v, u) for v in self.g2.prev(u)})) for u in self.g2.nodes}), self.g2.f)
        self.assertEqual(self.g3.transposed(), WeightedDirectedGraph({u: (self.g3.node_weights(u), (self.g3.link_weights(u), {v: self.g3.link_weights(v, u) for v in self.g3.prev(u)})) for u in self.g3.nodes}), self.g3.f)

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedDirectedGraph({n0: (7, ({}, {})), n1: (3, ({n0: 2, n8: -1}, {n2: 3})), n2: (5, ({n0: 4}, {n3: -6, n4: 5})), n3: (2, ({n6: 3}, {n7: -3, n8: 5})), n4: (8, ({n6: 2}, {n5: 0})), n5: (4, ({n7: 1}, {n6: 5})), n6: (6, ({}, {})), n7: (6, ({n8: 4}, {n9: 3})), n8: (2, ({}, {})), n9: (5, ({}, {}))}), res)
        self.assertIn(WeightedDirectedGraph({n10: (4, ({}, {})), n11: (2, ({n10: 2}, {n12: 6, n13: 10})), n12: (1, ({}, {})), n13: (3, ({n12: 3}, {}))}), res)
        self.assertIn(WeightedDirectedGraph({n14: (6, ({}, {}))}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])

    def test_component(self):
        self.assertEqual(self.g0.component(n9), WeightedDirectedGraph({n0: (7, ({}, {})),
                   n1: (3, ({n0: 2, n8: -1}, {n2: 3})), n2: (5, ({n0: 4}, {n3: -6, n4: 5})),
                   n3: (2, ({n6: 3}, {n7: -3, n8: 5})), n4: (8, ({n6: 2}, {n5: 0})), n5: (4, ({n7: 1}, {n6: 5})),
                   n6: (6, ({}, {})), n7: (6, ({n8: 4}, {n9: 3})), n8: (2, ({}, {})), n9: (5, ({}, {}))}))
        self.assertEqual(self.g0.component(n12), WeightedDirectedGraph({n10: (4, ({}, {})),
        n11: (2, ({n10: 2}, {n12: 6, n13: 10})), n12: (1, ({}, {})), n13: (3, ({n12: 3}, {}))}))
        self.assertEqual(self.g0.component(n14), WeightedDirectedGraph({n14: (6, ({}, {}))}))
        self.assertEqual(self.g1.component(n5), self.g1)
        self.assertEqual(self.g2.component(n6), self.g2)
        self.assertEqual(self.g3.component(n2), WeightedDirectedGraph({n0: (7, ({}, {})),
                                   n1: (4, ({n0: 3, n2: 4}, {n5: 5})), n2: (2, ({n0: 6, n3: 1}, {n4: 0})),
                                   n3: (5, ({}, {})), n4: (3, ({}, {n5: 1})), n5: (2, ({}, {}))}))
        self.assertEqual(self.g3.component(n9), WeightedDirectedGraph({n6: (2, ({}, {n7: 2, n8: 4})), n7: (1, ({}, {})), n8: (6, ({}, {})), n9: (4, ({n7: 3, n8: 0, n10: 4}, {})), n10: (5, ({}, {})), n11: (8, ({n10: 1}, {}))}))

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
        tmp.add((n4, 3), points_to_weights={n5: 1, n3: 4})
        self.assertEqual(self.g2.subgraph(n4), tmp.remove(n6))
        tmp = self.g3.component(n0)
        self.assertEqual(self.g3.subgraph(n0), tmp.remove(n3))
        tmp.add((n3, 5), points_to_weights={n2: 1})
        self.assertEqual(self.g3.subgraph(n3), tmp.remove(n0))
        tmp = self.g3.component(n6)
        self.assertEqual(self.g3.subgraph(n6), tmp.remove(n10, n11))
        self.assertEqual(self.g3.subgraph(n10), WeightedDirectedGraph({n9: (4, ({}, {})), n10: (5, ({}, {n9: 4, n11: 1})), n11: (8, ({}, {}))}))

    def test_minimal_path(self):
        res, s = self.g0.minimal_path(n2, n5), self.g0.node_weights(n2)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g0.next(res[0][i]))
            s += self.g0.node_weights(res[0][i + 1]) + self.g0.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 9)
        res, s = self.g1.minimal_path(n0, n4), self.g1.node_weights(n0)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g1.next(res[0][i]))
            s += self.g1.node_weights(res[0][i + 1]) + self.g1.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 22)
        res, s = self.g2.minimal_path(n5, n1), self.g2.node_weights(n5)
        for i in range(len(res[0]) - 1):
            self.assertIn(res[0][i + 1], self.g2.next(res[0][i]))
            s += self.g2.node_weights(res[0][i + 1]) + self.g2.link_weights(res[0][i], res[0][i + 1])
        self.assertEqual(res[1], s), self.assertEqual(s, 22)
        res, s = self.g3.minimal_path(n6, n9), self.g3.node_weights(n6)
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


if __name__ == '__main__':
    main()
