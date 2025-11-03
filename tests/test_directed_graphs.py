from unittest import TestCase, main

from undirected_graph import (Node, UndirectedGraph, WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph,
                              WeightedUndirectedGraph, reduce)

from directed_graph import *

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = map(Node, range(16))


class TestDirectedGraph(TestCase):
    def setUp(self):
        self.g0 = DirectedGraph(
            {n1: ([n0, n8], [n2]), n2: ([n0], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]),
             n7: ([n8], [n9]), n11: ([n10], [n12, n13]), n12: ([], [n13]), n14: ([], [])})
        self.g1 = DirectedGraph({n1: ([n3], [n0, n2, n4, n5]), n2: ([n0, n1], [n3]), n5: ([n3], [n4])})
        self.g2 = DirectedGraph({n0: ([n2, n5], [n1]), n2: ([n5], [n1, n3]), n3: ([n1, n4], []), n5: ([n4, n6], [])})
        self.g3 = DirectedGraph(
            {n1: ([n0, n2], [n5]), n2: ([n0, n3], [n4]), n4: ([], [n5]), n6: ([], [n7, n8]), n9: ([n7, n8, n10], []),
             n10: ([], [n11])})

    def test_init(self):
        g = DirectedGraph({0: ({1, 2}, {2, 3}), 1: (set(), {2, 3})})
        self.assertSetEqual(g.nodes, {n0, n1, n2, n3})
        self.assertSetEqual(g.links, {(n0, n2), (n0, n3), (n1, n0), (n2, n0), (n1, n2), (n1, n3)})
        self.assertDictEqual(g.prev(), {n0: {n1, n2}, n1: set(), n2: {n0, n1}, n3: {n0, n1}})
        self.assertDictEqual(g.next(), {n0: {n2, n3}, n1: {n0, n2, n3}, n2: {n0}, n3: set()})

    def test_get_nodes(self):
        self.assertSetEqual(self.g0.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14})
        self.assertSetEqual(self.g1.nodes, {n0, n1, n2, n3, n4, n5})
        self.assertSetEqual(self.g2.nodes, {n0, n1, n2, n3, n4, n5, n6})
        self.assertSetEqual(self.g3.nodes, {n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11})

    def test_get_links(self):
        self.assertSetEqual(self.g0.links,
                            {(n0, n1), (n0, n2), (n1, n2), (n8, n1), (n2, n3), (n3, n8), (n3, n7), (n7, n9), (n8, n7),
                             (n2, n4), (n4, n5), (n5, n6), (n6, n4), (n7, n5), (n6, n3), (n10, n11), (n11, n12),
                             (n11, n13), (n12, n13)})

    def test_prev(self):
        self.assertSetEqual(self.g0.prev(1), {n0, n8})
        self.assertDictEqual(self.g1.prev(), {n0: {n1}, n1: {n3}, n2: {n0, n1}, n3: {n2}, n4: {n1, n5}, n5: {n1, n3}})

    def test_prev_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.prev(6)

    def test_next(self):
        self.assertSetEqual(self.g0.next(0), {n1, n2})
        self.assertDictEqual(self.g1.next(),
                             {n0: {n2}, n1: {n0, n2, n4, n5}, n2: {n3}, n3: {n1, n5}, n4: set(), n5: {n4}})

    def test_next_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.next(6)

    def test_add_node(self):
        g1 = self.g1.copy().add(6, {2, 3}, {1, 2})
        self.assertSetEqual(g1.nodes, {n0, n1, n2, n3, n4, n5, n6})
        self.assertSetEqual(g1.links, self.g1.links.union({(n2, n6), (n3, n6), (n6, n1), (n6, n2)}))
        self.assertDictEqual(g1.prev(),
                             {n0: {n1}, n1: {n3, n6}, n2: {n0, n1, n6}, n3: {n2}, n4: {n1, n5}, n5: {n1, n3},
                              n6: {n2, n3}})
        self.assertDictEqual(g1.next(),
                             {n0: {n2}, n1: {n0, n2, n4, n5}, n2: {n3, n6}, n3: {n1, n5, n6}, n4: set(), n5: {n4},
                              n6: {n1, n2}})

    def test_add_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add(0, {1, 2}))

    def test_remove(self):
        g1 = self.g1.copy().remove(0, 1, 2)
        self.assertEqual(g1, DirectedGraph({5: ({3}, {4})}))

    def test_remove_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-1, -2))

    def test_connect(self):
        g1 = self.g1.copy().connect(2, {3}, {5})
        self.assertEqual(g1, DirectedGraph({1: ({3}, {0, 2, 4, 5}), 2: ({0, 1, 3}, {3, 5}), 5: ({3}, {4})}))

    def test_connect_connected_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().connect(2, {0}, {3}))

    def test_connect_all(self):
        g1 = self.g1.copy()
        g1.connect_all(0, 1, 2)
        self.assertEqual(g1, DirectedGraph(
            {n1: ([n0, n2, n3], [n0, n2, n4, n5]), n2: ([n0, n1], [n0, n1, n3]), n5: ([n3], [n4])}))

    def test_disconnect(self):
        g1 = self.g1.copy().disconnect(2, {0, 1}, {3})
        self.assertEqual(g1, DirectedGraph({1: ({3}, {0, 4, 5}), 2: ({}, {}), 5: ({3}, {4})}))

    def test_disconnect_disconnected_nodes(self):
        self.assertEqual(self.g1, self.g1.copy().disconnect(2, {3, 5}, {4, 5}))

    def test_disconnect_all(self):
        g1 = self.g1.copy().disconnect_all(0, 1, 2, 3)
        self.assertEqual(g1, DirectedGraph({1: (set(), {4}), 5: ({1, 3}, {4}), 0: (set(), set()), 2: (set(), set())}))

    def test_sources(self):
        self.assertEqual((self.g0.sources, self.g1.sources, self.g2.sources, self.g3.sources),
                         ({n0, n10, n14}, set(), {n4, n6}, {n0, n3, n6, n10}))

    def test_source(self):
        self.assertTrue(self.g0.source(0))
        self.assertFalse(self.g0.source(1))

    def test_source_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.source(6)

    def test_sinks(self):
        self.assertEqual((self.g0.sinks, self.g1.sinks, self.g2.sinks, self.g3.sinks),
                         ({n9, n13, n14}, {n4}, {n3}, {n5, n9, n11}))

    def test_sink(self):
        self.assertTrue(self.g0.sink(9))
        self.assertFalse(self.g0.sink(10))

    def test_sink_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.sink(6)

    def test_get_degrees(self):
        self.assertEqual(self.g0.degree(),
                         {n0: (0, 2), n1: (2, 1), n2: (2, 2), n3: (2, 2), n4: (2, 1), n5: (2, 1), n6: (1, 2),
                          n7: (2, 2), n8: (1, 2), n9: (1, 0), n10: (0, 1), n11: (1, 2), n12: (1, 1), n13: (2, 0),
                          n14: (0, 0)})

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_complementary(self):
        self.assertEqual(self.g1.complementary(), DirectedGraph(
            {0: ({2, 3, 4, 5}, {1, 3, 4, 5}), 1: ({2, 4, 5}, {3}), 2: ({3, 4, 5}, {4, 5}), 3: ({4, 5}, {4}),
             4: (set(), {5})}))

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(),
                         DirectedGraph({u: (self.g0.next(u), self.g0.prev(u)) for u in self.g0.nodes}))
        self.assertEqual(self.g1.transposed(),
                         DirectedGraph({u: (self.g1.next(u), self.g1.prev(u)) for u in self.g1.nodes}))
        self.assertEqual(self.g2.transposed(),
                         DirectedGraph({u: (self.g2.next(u), self.g2.prev(u)) for u in self.g2.nodes}))
        self.assertEqual(self.g3.transposed(),
                         DirectedGraph({u: (self.g3.next(u), self.g3.prev(u)) for u in self.g3.nodes}))

    def test_weighted_nodes_graph(self):
        g1 = self.g1.weighted_nodes_graph()
        self.assertDictEqual(g1.node_weights(), {n0: 0, n1: 0, n2: 0, n3: 0, n4: 0, n5: 0})
        g2 = self.g2.weighted_nodes_graph({n0: 7, n1: 6, n2: 2})
        self.assertDictEqual(g2.node_weights(), {n0: 7, n1: 6, n2: 2, n3: 0, n4: 0, n5: 0, n6: 0})

    def test_weighted_links_graph(self):
        g1 = self.g1.weighted_links_graph()
        self.assertDictEqual(g1.link_weights(), dict(zip(self.g1.links, [0] * 9)))
        g2 = self.g2.weighted_links_graph({(n0, n1): 1, (n2, n1): 2})
        link_weights = dict(zip(self.g2.links, [0] * 10))
        link_weights[(n0, n1)] = 1
        link_weights[(n2, n1)] = 2
        self.assertDictEqual(g2.link_weights(), link_weights)

    def test_weighted_graph(self):
        g1 = self.g1.weighted_graph()
        self.assertDictEqual(g1.node_weights(), {n0: 0, n1: 0, n2: 0, n3: 0, n4: 0, n5: 0})
        self.assertDictEqual(g1.link_weights(), dict(zip(self.g1.links, [0] * 9)))
        g2 = self.g2.weighted_graph({n0: 7, n1: 6, n2: 2}, {(n0, n1): 1, (n2, n1): 2})
        self.assertDictEqual(g2.node_weights(), {n0: 7, n1: 6, n2: 2, n3: 0, n4: 0, n5: 0, n6: 0})
        link_weights = dict(zip(self.g2.links, [0] * 10))
        link_weights[(n0, n1)] = 1
        link_weights[(n2, n1)] = 2
        self.assertDictEqual(g2.link_weights(), link_weights)

    def test_undirected(self):
        self.assertTrue(self.g0.undirected() == UndirectedGraph(
            {n0: {n1, n2}, n1: {n2, n8}, n2: {n3, n4}, n3: {n6, n7, n8}, n4: {n5, n6},
             n5: {n6, n7}, n7: {n8, n9}, n10: {n11}, n11: {n12, n13}, n12: {n13}, n14: []}))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(DirectedGraph(
            {n1: ([n0, n8], [n2]), n2: ([n0], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]),
             n7: ([n8], [n9])}), res)
        self.assertIn(DirectedGraph({n11: ([n10], [n12, n13]), n12: ([], [n13])}), res)
        self.assertIn(DirectedGraph({n14: ([], [])}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        res = self.g3.connection_components()
        self.assertEqual(len(res), 2)
        self.assertIn(DirectedGraph({n1: ([n0, n2], [n5]), n2: ([n0, n3], [n4]), n4: ([], [n5])}), res)
        self.assertIn(DirectedGraph({n6: ([], [n7, n8]), n9: ([n7, n8, n10], []), n10: ([], [n11])}), res)

    def test_connected(self):
        self.assertFalse(self.g0.connected())
        self.assertTrue(self.g1.connected())
        self.assertTrue(self.g2.connected())
        self.assertFalse(self.g3.connected())

    def test_reachable(self):
        for u in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9]:
            self.assertTrue(self.g0.reachable(n0, u))
            for v in [n10, n11, n12, n13, n14]:
                self.assertFalse(self.g0.reachable(u, v))
            self.assertTrue(self.g0.reachable(u, n9))
        for u in [n10, n11, n12, n13]:
            self.assertTrue(self.g0.reachable(n10, u))
            self.assertFalse(self.g0.reachable(n14, u))

    def test_reachable_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g1.reachable(4, 6)

    def test_component(self):
        self.assertEqual(self.g0.component(n9), DirectedGraph(
            {n1: ([n0, n8], [n2]), n2: ([n0], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]),
             n7: ([n8], [n9])}))
        self.assertEqual(self.g0.component(n13), DirectedGraph({n11: ([n10], [n12, n13]), n12: ([], [n13])}))
        self.assertEqual(self.g1.component(n4), self.g1)
        self.assertEqual(self.g2.component(n1), self.g2)
        self.assertEqual(self.g3.component(n4),
                         DirectedGraph({n1: ([n0, n2], [n5]), n2: ([n0, n3], [n4]), n4: ([], [n5])}))
        self.assertEqual(self.g3.component(n10),
                         DirectedGraph({n6: ([], [n7, n8]), n9: ([n7, n8, n10], []), n10: ([], [n11])}))

    def test_component_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.component(n9)

    def test_full(self):
        self.assertFalse(self.g1.full())
        g1 = self.g1.copy().connect_all(*self.g1.nodes)
        self.assertTrue(g1.full())

    def test_node_subgraph(self):
        self.assertEqual(self.g0.subgraph(n1), DirectedGraph(
            {n1: ([n8], [n2]), n2: ([], [n3, n4]), n3: ([n6], [n7, n8]), n4: ([n6], [n5]), n5: ([n7], [n6]),
             n7: ([n8], [n9])}))
        self.assertEqual(self.g1.subgraph(n0), self.g1), self.assertEqual(self.g1.subgraph(n4),
                                                                          DirectedGraph({n4: ([], [])}))
        self.assertEqual(self.g2.subgraph(n2), DirectedGraph({n1: ([n0, n2], [n3]), n2: ([], [n0, n1, n3])}))
        self.assertEqual(self.g3.subgraph(n6), DirectedGraph({n6: ([], [n7, n8]), n9: ([n7, n8], [])}))

    def test_node_subgraph_missing_node(self):
        with self.assertRaises(KeyError):
            self.g0.subgraph(-1)

    def test_nodes_set_subgraph(self):
        self.assertEqual(self.g0.subgraph({n1, n2, n3, n7, n8}),
                         DirectedGraph({1: ({8}, {2}), 3: ({2}, {8}), 7: ({3, 8}, set())}))

    def test_dag(self):
        self.assertFalse(self.g0.dag())
        self.assertFalse(self.g1.dag())
        self.assertTrue(self.g2.dag())
        self.assertTrue(self.g3.dag())

    def test_toposort(self):
        self.assertListEqual(self.g0.toposort(), [])
        self.assertListEqual(self.g1.toposort(), [])
        res = self.g2.toposort()
        self.assertSetEqual(self.g2.nodes, set(res))
        self.assertEqual(len(self.g2.nodes), len(res))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                self.assertFalse(self.g2.reachable(res[j], res[i]))
        res = self.g3.toposort()
        self.assertSetEqual(self.g3.nodes, set(res))
        self.assertEqual(len(self.g3.nodes), len(res))
        for i in range(len(res)):
            for j in range(i + 1, len(res)):
                self.assertFalse(self.g3.reachable(res[j], res[i]))

    def test_get_shortest_path(self):
        self.assertListEqual(self.g0.get_shortest_path(0, 6), [n0, n2, n4, n5, n6])
        self.assertListEqual(self.g0.get_shortest_path(7, 8), [n7, n5, n6, n3, n8])
        self.assertListEqual(self.g0.get_shortest_path(10, 13), [n10, n11, n13])
        self.assertListEqual(self.g1.get_shortest_path(0, 5), [n0, n2, n3, n5])
        self.assertListEqual(self.g2.get_shortest_path(4, 3), [n4, n3])
        self.assertListEqual(self.g2.get_shortest_path(0, 5), [])

    def test_get_shortest_path_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.get_shortest_path(0, -6)

    def test_euler_tour_exists(self):
        g0 = DirectedGraph.copy(self.g0)
        g0.connect(n11, [n13])
        g0.connect(n0, [n4]), g0.connect(n8, [n1]), g0.connect(n7, [n5], [n6])
        self.assertFalse(self.g1.euler_tour_exists())
        g0 = DirectedGraph.copy(self.g1)
        g0.connect(n1, [n2])
        g0.connect(n4, points_to=[n1, n3]), g0.disconnect(n2, [n1]), g0.connect(n2, [n5])
        self.assertTrue(g0.euler_tour_exists())

    def test_euler_walk_exists(self):
        g0 = DirectedGraph.copy(self.g0)
        g0.connect(n11, [n13]), g0.connect(n0, [n4])
        g0.connect(n8, [n1]), g0.connect(n7, [n5], [n6])
        self.assertTrue(g0.component(n0).euler_walk_exists(n0, n9))
        self.assertTrue(g0.component(n10).euler_walk_exists(n10, n13))
        self.assertFalse(g0.euler_walk_exists(n0, n9))

    def test_euler_walk_exists_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.euler_walk_exists(0, -1)

    def test_euler_tour(self):
        g0 = DirectedGraph.component(self.g0, n0)
        g0.connect(n0, [n7]), g0.connect(n5, [n1], [n8])
        g0.disconnect(n4, [n6]), g0.disconnect(n5, [n7])
        g0.connect(n0, [n9])
        res = g0.euler_tour()
        self.assertEqual(len(res), (n := len(g0.links) + 1))
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, g0.prev(res[i]))
            u = res[i]

    def test_euler_walk(self):
        g0 = DirectedGraph.component(self.g0, n0)
        g0.connect(n0, [n7]), g0.connect(n5, [n1], [n8])
        g0.disconnect(n4, [n6]), g0.disconnect(n5, [n7])
        res = g0.euler_walk(n0, n9)
        n = len(res)
        self.assertEqual(n, len(g0.links) + 1)
        u = res[0]
        for i in range(1, n):
            self.assertIn(u, g0.prev(res[i]))
            u = res[i]

    def test_euler_walk_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.euler_walk_exists(0, -1)

    def test_strongly_connected_component(self):
        self.assertSetEqual(self.g0.strongly_connected_component(1), {n1, n2, n3, n4, n5, n6, n7, n8})

    def test_strongly_connected_component_missing_node(self):
        with self.assertRaises(KeyError):
            self.g0.strongly_connected_component(-1)

    def test_strongly_connected_components(self):
        res = self.g0.strongly_connected_components()
        self.assertEqual(len(res), 8)
        self.assertEqual(sum(map(len, res)), len(self.g0.nodes))
        self.assertSetEqual(reduce(lambda x, y: x.union(y), res), self.g0.nodes)
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g0.reachable(u, v))
        res = self.g1.strongly_connected_components()
        self.assertEqual(len(res), 3)
        self.assertEqual(sum(map(len, res)), len(self.g1.nodes))
        self.assertSetEqual(reduce(lambda x, y: x.union(y), res), self.g1.nodes)
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g1.reachable(u, v))
        res = self.g2.strongly_connected_components()
        self.assertEqual(len(res), 7)
        self.assertEqual(sum(map(len, res)), len(self.g2.nodes))
        self.assertSetEqual(reduce(lambda x, y: x.union(y), res), self.g2.nodes)
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g2.reachable(u, v))
        res = self.g3.strongly_connected_components()
        self.assertEqual(len(res), 12)
        self.assertEqual(sum(map(len, res)), len(self.g3.nodes))
        self.assertSetEqual(reduce(lambda x, y: x.union(y), res), self.g3.nodes)
        for s in res:
            for u in s:
                for v in s:
                    self.assertTrue(self.g3.reachable(u, v))

    def test_scc_dag(self):
        g0 = self.g0.component(0).scc_dag()
        self.assertSetEqual(g0.nodes, {Node(frozenset({n0})), Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})),
                                       Node(frozenset({n9}))})
        self.assertSetEqual(g0.links, {(Node(frozenset({n0})), Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8}))),
                                       (Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})), Node(frozenset({n9})))})

    def test_path_with_length(self):
        self.assertTrue(self.g0.path_with_length(10, 13, 3))
        self.assertTrue(self.g0.path_with_length(4, 8, 4))
        self.assertTrue(self.g0.path_with_length(0, 5, 6))
        self.assertTrue(self.g0.path_with_length(0, 1, 9))
        self.assertFalse(self.g0.path_with_length(0, 1, 10))

    def test_path_with_length_bad_length_type(self):
        with self.assertRaises(TypeError):
            self.g0.path_with_length(5, 6, "[2]")

    def test_path_with_length_bad_length_value(self):
        self.assertListEqual(self.g0.path_with_length(0, 3, -1), [])
        self.assertListEqual(self.g0.path_with_length(0, 3, 1), [])

    def test_path_with_length_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.path_with_length(-1, 3, 2)

    def test_cycle_with_length(self):
        self.assertTrue(self.g0.cycle_with_length(3))
        self.assertTrue(self.g0.cycle_with_length(4))
        self.assertFalse(self.g0.cycle_with_length(6))
        self.assertFalse(self.g0.cycle_with_length(9))

    def test_cycle_with_length_bad_length_type(self):
        with self.assertRaises(TypeError):
            self.g0.cycle_with_length([4])

    def test_cycle_with_length_bad_length_value(self):
        self.assertListEqual(self.g0.cycle_with_length(1), [])

    def test_hamilton_tour_exists(self):
        g0 = DirectedGraph.component(self.g0, n0)
        self.assertFalse(self.g0.hamilton_tour_exists())
        g0.connect(n0, [n9]), self.assertTrue(g0.hamilton_tour_exists())
        g1 = DirectedGraph.copy(self.g1)
        self.assertFalse(g1.hamilton_tour_exists())
        g1.connect(n1, [n4])
        self.assertTrue(g1.hamilton_tour_exists())

    def test_hamilton_walk_exists(self):
        g0 = DirectedGraph.subgraph(self.g0, n0)
        self.assertTrue(g0.hamilton_walk_exists(n0, n9))
        self.assertTrue(self.g0.component(n10).hamilton_walk_exists(n10, n13))
        g0.connect(n0, [n9])
        g1 = DirectedGraph.copy(self.g1)
        self.assertTrue(g1.hamilton_walk_exists(n1, n4))
        g1.connect(n1, [n4])
        for n in [n0, n1, n2, n3, n4, n5]:
            self.assertFalse(self.g2.hamilton_walk_exists(n6, n))
        g2 = DirectedGraph.copy(self.g2)
        g2.connect(n4, [n5]), g2.connect(n2, [n3])
        self.assertTrue(g0.hamilton_walk_exists(n6, n1))

    def test_hamilton_walk_exists_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.hamilton_walk_exists(0, -1)

    def test_hamilton_tour(self):
        g0 = DirectedGraph.subgraph(self.g0, n0)
        g0.disconnect(n5, [n4])
        self.assertFalse(g0.hamilton_walk())
        g0.connect(n5, [n4]), g0.connect(n0, [n9])
        res = g0.hamilton_tour()
        self.assertEqual(len(res), len(g0.nodes) + 1)
        self.assertEqual(res[0], res[-1])
        for i in range(len(g0.nodes)):
            self.assertIn(res[i], g0.prev(res[i + 1]))

    def test_hamilton_walk(self):
        tmp = DirectedGraph.subgraph(self.g0, n0)
        res = tmp.hamilton_walk()
        n = len(res)
        self.assertEqual(n, len(tmp.nodes))
        self.assertEqual(n9, res[-1])
        for i in range(n - 1):
            self.assertIn(res[i], tmp.prev(res[i + 1]))
        tmp.disconnect(n5, [n4])
        self.assertFalse(tmp.hamilton_walk())

    def test_hamilton_walk_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g0.hamilton_walk_exists(0, -1)

    def test_isomorphic_bijection(self):
        g1 = DirectedGraph({11: ([13], [10, 12, 14, 15]), 12: ([10, 11], [13]), 15: ([13], [14])})
        func = self.g1.isomorphic_bijection(g1)
        self.assertTrue(func)
        for n, m in func.items():
            next_nodes = self.g1.next(n)
            for v in g1.nodes:
                self.assertEqual(v in map(func.get, next_nodes), v in g1.next(m))
        g1.disconnect(n11, {n13})
        self.assertDictEqual(self.g1.isomorphic_bijection(g1), {})

    def test_bool(self):
        g0 = self.g0.component(14)
        self.assertTrue(g0)
        g0.remove(14)
        self.assertFalse(g0)

    def test_reversed(self):
        self.assertEqual(self.g0.transposed(), self.g0.__reversed__())
        self.assertEqual(self.g1.transposed(), self.g1.__reversed__())
        self.assertEqual(self.g2.transposed(), self.g2.__reversed__())
        self.assertEqual(self.g3.transposed(), self.g3.__reversed__())

    def test_contains(self):
        self.assertIn(14, self.g0)
        self.assertNotIn(15, self.g0)

    def test_add(self):
        g1 = DirectedGraph({11: ([13], [10, 12, 14, 15]), 12: ([10, 11], [13]), 15: ([13], [14])})
        expected = DirectedGraph(
            {1: ([0, 2], [5]), 2: ([0, 3], [4]), 4: ([], [5]), 6: ([], [7, 8]), 9: ([7, 8, 10], []),
             11: ([10, 13], [10, 12, 14, 15]), 12: ([10, 11], [13]), 15: ([13], [14])})
        self.assertEqual(self.g3 + g1, expected)

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n2, {n0, n1}).disconnect(n0, {n1}).disconnect(n4, {n5})
        g1.connect(n2, {n5}, {n0, n1}), g1.connect(n3, {n1, n4}), g1.disconnect(n1, points_to={n4, n5})
        g1.disconnect(n1, {n3}), g1.connect(n0, {n5}, {n1}), g1.disconnect(n5, {n3})
        self.assertEqual(g1.connect(n5, {n4}), self.g2.copy().remove(n6))

    def test_str(self):
        self.assertEqual(str(self.g0), "<" + str(self.g0.nodes) + ", {" + ", ".join(
            f"<{l[0]}, {l[1]}>" for l in self.g0.links) + "}>")

    def test_repr(self):
        self.assertEqual(str(self.g0), repr(self.g0))


class TestWeightedNodesDirectedGraph(TestCase):
    def setUp(self):
        self.g0 = WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (3, ([n0, n8], [n2])), n2: (5, ([n0], [n3, n4])), n3: (2, ([n6], [n7, n8])),
             n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])),
             n9: (5, ([], [])), n10: (4, ([], [])), n11: (2, ([n10], [n12, n13])), n12: (1, ([], [n13])),
             n13: (3, ([], [])), n14: (6, ([], []))})
        self.g1 = WeightedNodesDirectedGraph(
            {n0: (3, ([], [])), n1: (2, ([n3], [n0, n2, n4, n5])), n2: (4, ([n0, n1], [n3])), n3: (6, ([], [])),
             n4: (5, ([], [])), n5: (1, ([n1, n3], [n4]))})
        self.g2 = WeightedNodesDirectedGraph(
            {n0: (7, ([n2, n5], [n1])), n1: (6, ([], [])), n2: (2, ([n5], [n1, n3])), n3: (4, ([n1, n4], [])),
             n4: (3, ([], [])), n5: (5, ([n4, n6], [])), n6: (4, ([], []))})
        self.g3 = WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (4, ([n0, n2], [n5])), n2: (3, ([n0, n3], [n4])), n3: (5, ([], [])),
             n4: (6, ([], [n5])), n5: (2, ([], [])), n6: (2, ([], [n7, n8])), n7: (1, ([], [])), n8: (6, ([], [])),
             n9: (4, ([n7, n8, n10], [])), n10: (5, ([], [n11])), n11: (8, ([], []))})

    def test_init(self):
        g = WeightedNodesDirectedGraph({0: (2, ({1, 2}, {3})), 1: (3, ({2, 3}, []))})
        self.assertDictEqual(g.node_weights(), {n0: 2, n1: 3, n2: 0, n3: 0})
        self.assertSetEqual(g.links, {(n0, n3), (n1, n0), (n2, n0), (n2, n1), (n3, n1)})

    def test_node_weights(self):
        self.assertEqual(self.g0.node_weights(),
                         {n0: 7, n1: 3, n2: 5, n3: 2, n4: 8, n5: 4, n6: 6, n7: 6, n8: 2, n9: 5, n10: 4, n11: 2, n12: 1,
                          n13: 3, n14: 6})

    def test_node_weights_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.node_weights(n9)

    def test_total_node_weights(self):
        self.assertEqual(self.g0.total_nodes_weight, 64)

    def test_add_node(self):
        g1 = self.g1.copy().add((6, 3), {2, 3}, {1, 2})
        self.assertDictEqual(g1.node_weights(), {n0: 3, n1: 2, n2: 4, n3: 6, n4: 5, n5: 1, n6: 3})
        self.assertSetEqual(g1.links, self.g1.links.union({(n2, n6), (n3, n6), (n6, n1), (n6, n2)}))
        self.assertDictEqual(g1.prev(),
                             {n0: {n1}, n1: {n3, n6}, n2: {n0, n1, n6}, n3: {n2}, n4: {n1, n5}, n5: {n1, n3},
                              n6: {n2, n3}})
        self.assertDictEqual(g1.next(),
                             {n0: {n2}, n1: {n0, n2, n4, n5}, n2: {n3, n6}, n3: {n1, n5, n6}, n4: set(), n5: {n4},
                              n6: {n1, n2}})

    def test_add_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add((0, 4), {1, 2}))

    def test_remove(self):
        g1 = self.g1.copy().remove(0, 1, 2)
        self.assertEqual(g1, WeightedNodesDirectedGraph({5: (1, ({3}, {4})), 3: (6, ([], [])), 4: (5, ([], []))}))

    def test_remove_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-1, -2))

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
        self.assertEqual(self.g0, self.g0.copy().increase_weight(-4, 3))

    def test_increase_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.increase_weight(4, "5-")

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_component(self):
        self.assertEqual(self.g0.component(n9), WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (3, ([n0, n8], [n2])), n2: (5, ([n0], [n3, n4])), n3: (2, ([n6], [n7, n8])),
             n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])),
             n9: (5, ([], []))}))
        self.assertEqual(self.g0.component(n13), WeightedNodesDirectedGraph(
            {n10: (4, ([], [])), n11: (2, ([n10], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], []))}))
        self.assertEqual(self.g1.component(n4), self.g1)
        self.assertEqual(self.g2.component(n1), self.g2)
        self.assertEqual(self.g3.component(n4), WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (4, ([n0, n2], [n5])), n2: (3, ([n0, n3], [n4])), n3: (5, ([], [])),
             n4: (6, ([], [n5])), n5: (2, ([], []))}))
        self.assertEqual(self.g3.component(n10), WeightedNodesDirectedGraph(
            {n6: (2, ([], [n7, n8])), n7: (1, ([], [])), n8: (6, ([], [])), n9: (4, ([n7, n8, n10], [])),
             n10: (5, ([], [n11])), n11: (8, ([], []))}))

    def test_complementary(self):
        g1 = self.g1.complementary()
        self.assertEqual(g1, WeightedNodesDirectedGraph(
            {0: (3, ({2, 3, 4, 5}, {1, 3, 4, 5})), 1: (2, ({2, 4, 5}, {3})), 5: (1, ([], [])),
             2: (4, ({3, 4, 5}, {4, 5})), 3: (6, ({4, 5}, {4})), 4: (5, (set(), {5}))}))

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedNodesDirectedGraph(
            {u: (self.g0.node_weights(u), (self.g0.next(u), self.g0.prev(u))) for u in self.g0.nodes}))
        self.assertEqual(self.g1.transposed(), WeightedNodesDirectedGraph(
            {u: (self.g1.node_weights(u), (self.g1.next(u), self.g1.prev(u))) for u in self.g1.nodes}))
        self.assertEqual(self.g2.transposed(), WeightedNodesDirectedGraph(
            {u: (self.g2.node_weights(u), (self.g2.next(u), self.g2.prev(u))) for u in self.g2.nodes}))
        self.assertEqual(self.g3.transposed(), WeightedNodesDirectedGraph(
            {u: (self.g3.node_weights(u), (self.g3.next(u), self.g3.prev(u))) for u in self.g3.nodes}))

    def test_weighted_graph(self):
        g1 = self.g1.weighted_graph()
        self.assertDictEqual(g1.link_weights(), dict(zip(self.g1.links, [0] * 9)))
        g2 = self.g2.weighted_graph({(n0, n1): 1, (n2, n1): 2})
        link_weights = dict(zip(self.g2.links, [0] * 10))
        link_weights[(n0, n1)] = 1
        link_weights[(n2, n1)] = 2
        self.assertDictEqual(g2.link_weights(), link_weights)

    def test_undirected(self):
        self.assertEqual(self.g0.undirected(), WeightedNodesUndirectedGraph(
            {n0: (7, [n1, n2]), n1: (3, [n2, n8]), n2: (5, [n3, n4]), n3: (2, [n6, n7, n8]), n4: (8, [n5, n6]),
             n5: (4, {n6, n7}), n6: (6, {}), n7: (6, [n8, n9]), n8: (2, {}), n9: (5, {}), n10: (4, {n11}),
             n11: (2, [n12, n13]), n12: (1, {n13}), n13: (3, {}), n14: (6, {})}))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (3, ([n0, n8], [n2])), n2: (5, ([n0], [n3, n4])), n3: (2, ([n6], [n7, n8])),
             n4: (8, ([n6], [n5])), n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])),
             n9: (5, ([], []))}), res)
        self.assertIn(WeightedNodesDirectedGraph(
            {n10: (4, ([], [])), n11: (2, ([n10], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], []))}), res)
        self.assertIn(WeightedNodesDirectedGraph({n14: (6, ([], []))}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        res = self.g3.connection_components()
        self.assertEqual(len(res), 2)
        self.assertIn(WeightedNodesDirectedGraph(
            {n0: (7, ([], [])), n1: (4, ([n0, n2], [n5])), n2: (3, ([n0, n3], [n4])), n3: (5, ([], [])),
             n4: (6, ([], [n5])), n5: (2, ([], []))}), res)
        self.assertIn(WeightedNodesDirectedGraph(
            {n6: (2, ([], [n7, n8])), n7: (1, ([], [])), n8: (6, ([], [])), n9: (4, ([n7, n8, n10], [])),
             n10: (5, ([], [n11])), n11: (8, ([], []))}), res)

    def test_node_subgraph(self):
        self.assertEqual(self.g0.subgraph(n0), self.g0.component(n0))
        self.assertEqual(self.g0.subgraph(n2), WeightedNodesDirectedGraph(
            {n1: (3, ([n8], [n2])), n2: (5, ([], [n3, n4])), n3: (2, ([n6], [n7, n8])), n4: (8, ([n6], [n5])),
             n5: (4, ([n7], [n6])), n6: (6, ([], [])), n7: (6, ([n8], [n9])), n8: (2, ([], [])), n9: (5, ([], []))}))
        self.assertEqual(self.g0.subgraph(n10), self.g0.component(n10))
        self.assertEqual(self.g0.subgraph(n11), WeightedNodesDirectedGraph(
            {n11: (2, ([], [n12, n13])), n12: (1, ([], [n13])), n13: (3, ([], []))}))

    def test_node_subgraph_missing_node(self):
        with self.assertRaises(KeyError):
            self.g0.subgraph(-1)

    def test_nodes_set_subgraph(self):
        self.assertEqual(self.g1.subgraph({n0, n1, n2, n3}), WeightedNodesDirectedGraph(
            {0: (3, ({1}, {2})), 1: (2, ({3}, {2})), 2: (4, ([], {3})), 3: (6, ([], []))}))

    def test_scc_dag(self):
        g0 = self.g0.component(0).scc_dag()
        self.assertDictEqual(g0.node_weights(),
                             {Node(frozenset({n0})): 7, Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})): 36,
                              Node(frozenset({n9})): 5})
        self.assertSetEqual(g0.links, {(Node(frozenset({n0})), Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8}))),
                                       (Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})), Node(frozenset({n9})))})

    def test_minimal_path_nodes(self):
        self.assertListEqual(self.g3.minimal_path_nodes(6, 9), [n6, n7, n9])

    def test_minimal_path_nodes_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g3.minimal_path_nodes(6, -1)

    def test_isomorphic_bijection(self):
        g1 = WeightedNodesDirectedGraph(
            {11: (2, ([13], [10, 12, 14, 15])), 10: (3, ([], [])), 13: (6, ([], [])), 14: (5, ([], [])),
             12: (4, ([10, 11], [13])), 15: (1, ([13], [14]))})
        func = self.g1.isomorphic_bijection(g1)
        self.assertTrue(func)
        for n, m in func.items():
            next_nodes = self.g1.next(n)
            for v in g1.nodes:
                self.assertEqual(v in map(func.get, next_nodes), v in g1.next(m))
        g1 = DirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))
        g1.disconnect(n11, {n13})
        self.assertDictEqual(self.g1.isomorphic_bijection(g1), {})

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n2, {n0, n1}).disconnect(n0, {n1}).disconnect(n4, {n5})
        g1.connect(n2, {n5}, {n0, n1}), g1.connect(n3, {n1, n4}), g1.disconnect(n1, points_to={n4, n5})
        g1.disconnect(n1, {n3}), g1.connect(n0, {n5}, {n1}), g1.disconnect(n5, {n3})
        g2 = self.g2.copy().remove(n6)
        self.assertNotEqual(g1.connect(n5, {n4}), g2)
        g1.increase_weight(0, 4), g1.increase_weight(1, 4), g1.increase_weight(2, -2)
        g1.set_weight(3, 4), g1.set_weight(4, 3), g1.set_weight(5, 5)
        self.assertEqual(g1, g2)

    def test_add(self):
        g1 = WeightedNodesDirectedGraph(
            {11: (2, ([13], [10, 12, 14, 15])), 10: (3, ([], [])), 13: (6, ([], [])), 14: (5, ([], [])),
             12: (4, ([10, 11], [13])), 15: (1, ([13], [14]))})
        expected = WeightedNodesDirectedGraph(
            {0: (7, ([], [])), 1: (4, ([0, 2], [5])), 2: (3, ([0, 3], [4])), 4: (6, ([], [5])), 5: (2, ([], [])),
             6: (2, ([], [7, 8])), 7: (1, ([], [])), 8: (6, ([], [])), 9: (4, ([7, 8, 10], [])), 10: (8, ([], [])),
             13: (6, ([], [])), 3: (5, ([], [])), 11: (10, ([10, 13], [10, 12, 14, 15])), 12: (4, ([10, 11], [13])),
             14: (5, ([], [])), 15: (1, ([13], [14]))})
        self.assertEqual(self.g3 + g1, expected)

    def test_str(self):
        self.assertEqual(str(self.g0), "<{" + ", ".join(
            f"{n} -> {self.g0.node_weights(n)}" for n in self.g0.nodes) + "}, {" + ", ".join(
            f"<{l[0]}, {l[1]}>" for l in self.g0.links) + "}>")

    def test_repr(self):
        self.assertEqual(str(self.g0), repr(self.g0))


class TestWeightedLinksDirectedGraph(TestCase):
    def setUp(self):
        self.g0 = WeightedLinksDirectedGraph(
            {n1: ({n0: 2, n8: -1}, {n2: 3}), n2: ({n0: 4}, {n3: -6, n4: 5}), n3: ({n6: 3}, {n7: -3, n8: 5}),
             n4: ({n6: 2}, {n5: 0}), n5: ({n7: 1}, {n6: 5}), n7: ({n8: 4}, {n9: 3}),
             n11: ({n10: 2}, {n12: 6, n13: 10}), n13: ({n12: 3}, {}), n14: ({}, {})})
        self.g1 = WeightedLinksDirectedGraph(
            {n1: ({n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3}), n2: ({n0: 2}, {n3: -6}), n5: ({n3: 2}, {n4: 5})})
        self.g2 = WeightedLinksDirectedGraph(
            {n0: ({n2: 2, n5: 4}, {n1: 2}), n2: ({n5: 3}, {n1: 6, n3: -1}), n3: ({n1: 1, n4: 4}, {}),
             n5: ({n4: 1, n6: 2}, {})})
        self.g3 = WeightedLinksDirectedGraph(
            {n1: ({n0: 3, n2: 4}, {n5: 5}), n2: ({n0: 6, n3: 1}, {n4: 0}), n4: ({}, {n5: 1}), n6: ({}, {n7: 2, n8: 4}),
             n9: ({n7: 3, n8: 0, n10: 4}, {}), n11: ({n10: 1}, {})})

    def test_init(self):
        g = WeightedLinksDirectedGraph({0: ({1: 1, 2: 0}, {3: 2}), 1: ({2: 1, 3: 4}, {})})
        self.assertSetEqual(g.nodes, {n0, n1, n2, n3})
        self.assertDictEqual(g.link_weights(), {(n0, n3): 2, (n1, n0): 1, (n2, n0): 0, (n2, n1): 1, (n3, n1): 4})

    def test_link_weights(self):
        self.assertDictEqual(self.g0.link_weights(),
                             {(n0, n1): 2, (n0, n2): 4, (n1, n2): 3, (n2, n3): -6, (n2, n4): 5, (n5, n6): 5,
                              (n6, n3): 3, (n6, n4): 2, (n4, n5): 0, (n3, n8): 5, (n3, n7): -3, (n7, n5): 1,
                              (n8, n7): 4, (n7, n9): 3, (n8, n1): -1, (n10, n11): 2, (n11, n12): 6, (n11, n13): 10,
                              (n12, n13): 3})
        self.assertDictEqual(self.g0.link_weights(n0), {n1: 2, n2: 4})
        self.assertDictEqual(self.g0.link_weights(n1), {n2: 3})
        self.assertDictEqual(self.g0.link_weights(n14), {})
        self.assertEqual(self.g0.link_weights((n2, n4)), 5)

    def test_link_weights_missing_link(self):
        with self.assertRaises(KeyError):
            self.g0.link_weights(1, 0)

    def test_total_link_weights(self):
        self.assertEqual(self.g0.total_links_weight, 48)
        self.assertEqual(self.g1.total_links_weight, 23)
        self.assertEqual(self.g2.total_links_weight, 24)
        self.assertEqual(self.g3.total_links_weight, 34)

    def test_add_node(self):
        g1 = self.g1.copy().add(6, {2: 1, 3: 2}, {1: 2, 2: 3})
        self.assertSetEqual(g1.nodes, {n0, n1, n2, n3, n4, n5, n6})
        link_weights = self.g1.link_weights()
        link_weights[(n2, n6)] = 1
        link_weights[(n3, n6)] = 2
        link_weights[(n6, n1)] = 2
        link_weights[(n6, n2)] = 3
        self.assertDictEqual(g1.link_weights(), link_weights)
        self.assertDictEqual(g1.prev(),
                             {n0: {n1}, n1: {n3, n6}, n2: {n0, n1, n6}, n3: {n2}, n4: {n1, n5}, n5: {n1, n3},
                              n6: {n2, n3}})
        self.assertDictEqual(g1.next(),
                             {n0: {n2}, n1: {n0, n2, n4, n5}, n2: {n3, n6}, n3: {n1, n5, n6}, n4: set(), n5: {n4},
                              n6: {n1, n2}})

    def test_add_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add(0, {1, 2}))

    def test_remove(self):
        g1 = self.g1.copy().remove(0, 1, 2)
        self.assertEqual(g1, WeightedLinksDirectedGraph({5: ({3: 2}, {4: 5})}))

    def test_remove_missing_node(self):
        self.assertEqual(self.g0, self.g0.copy().remove(-1, -2))

    def test_connect(self):
        g1 = self.g1.copy().connect(3, {4: 2}, {2: 1})
        self.assertEqual(g1, WeightedLinksDirectedGraph(
            {n1: ({n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3}), n2: ({n0: 2}, {n3: -6}), n3: ({n4: 2}, {n2: 1}),
             n5: ({n3: 2}, {n4: 5})}))

    def test_connect_connected_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().connect(1, {8: 4}))

    def test_connect_all(self):
        g1 = self.g1.copy()
        g1.connect_all(0, 1, 2)
        self.assertEqual(g1, WeightedLinksDirectedGraph(
            {n1: ({n0: 0, n2: 0, n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3}), n2: ({n0: 2}, {n0: 0, n3: -6}),
             n5: ({n3: 2}, {n4: 5})}))

    def test_disconnect(self):
        g1 = self.g1.copy().disconnect(3, {2}, {5})
        self.assertEqual(g1, WeightedLinksDirectedGraph(
            {n1: ({n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3}), n2: ({n0: 2}, {}), n5: ({}, {n4: 5})}))

    def test_disconnect_disconnected_nodes(self):
        self.assertEqual(self.g0, self.g0.copy().disconnect(1, {3}))

    def test_disconnect_all(self):
        g1 = self.g1.copy().disconnect_all(0, 1, 2, 3)
        self.assertEqual(g1, WeightedLinksDirectedGraph(
            {n0: ({}, {}), n1: ({}, {n4: 9, n5: 3}), n2: ({}, {}), n5: ({n3: 2}, {n4: 5})}))

    def test_set_weight(self):
        g0 = self.g0.copy().set_weight((n2, n4), 3)
        res_link_weights = self.g0.link_weights()
        res_link_weights[(n2, n4)] = 3
        self.assertDictEqual(g0.link_weights(), res_link_weights)

    def test_set_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight((n2, n6), 3))

    def test_set_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.set_weight((n2, n4), [3])

    def test_increase_weight(self):
        g0 = self.g0.copy().increase_weight((n2, n4), 1)
        res_link_weights = self.g0.link_weights()
        res_link_weights[(n2, n4)] = 6
        self.assertDictEqual(g0.link_weights(), res_link_weights)

    def test_increase_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().increase_weight((n2, n6), 1))

    def test_increase_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.increase_weight((n2, n4), [3])

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedLinksDirectedGraph(
            {u: (self.g0.link_weights(u), {v: self.g0.link_weights(v, u) for v in self.g0.prev(u)}) for u in
             self.g0.nodes}))
        self.assertEqual(self.g1.transposed(), WeightedLinksDirectedGraph(
            {u: (self.g1.link_weights(u), {v: self.g1.link_weights(v, u) for v in self.g1.prev(u)}) for u in
             self.g1.nodes}))
        self.assertEqual(self.g2.transposed(), WeightedLinksDirectedGraph(
            {u: (self.g2.link_weights(u), {v: self.g2.link_weights(v, u) for v in self.g2.prev(u)}) for u in
             self.g2.nodes}))
        self.assertEqual(self.g3.transposed(), WeightedLinksDirectedGraph(
            {u: (self.g3.link_weights(u), {v: self.g3.link_weights(v, u) for v in self.g3.prev(u)}) for u in
             self.g3.nodes}))

    def test_weighted_graph(self):
        g1 = self.g1.weighted_graph()
        self.assertDictEqual(g1.node_weights(), {n0: 0, n1: 0, n2: 0, n3: 0, n4: 0, n5: 0})
        g2 = self.g2.weighted_graph({n0: 7, n1: 6, n2: 2})
        self.assertDictEqual(g2.node_weights(), {n0: 7, n1: 6, n2: 2, n3: 0, n4: 0, n5: 0, n6: 0})

    def test_undirected(self):
        self.assertEqual(self.g0.copy().connect(0, {1: 1}).undirected(), WeightedLinksUndirectedGraph(
            {n0: {n1: 3, n2: 4}, n1: {n2: 3, n8: -1}, n2: {n3: -6, n4: 5}, n3: {n6: 3, n7: -3, n8: 5},
             n4: {n5: 0, n6: 2}, n5: {n6: 5, n7: 1}, n7: {n8: 4, n9: 3}, n10: {n11: 2}, n11: {n12: 6, n13: 10},
             n12: {n13: 3}, n14: {}}))

    def test_component(self):
        self.assertEqual(self.g0.component(n9), WeightedLinksDirectedGraph(
            {n1: ({n0: 2, n8: -1}, {n2: 3}), n2: ({n0: 4}, {n3: -6, n4: 5}), n3: ({n6: 3}, {n7: -3, n8: 5}),
             n4: ({n6: 2}, {n5: 0}), n5: ({n7: 1}, {n6: 5}), n7: ({n8: 4}, {n9: 3})}))
        self.assertEqual(self.g0.component(n13),
                         WeightedLinksDirectedGraph({n11: ({n10: 2}, {n12: 6, n13: 10}), n13: ({n12: 3}, {})}))
        self.assertEqual(self.g1.component(n4), self.g1)
        self.assertEqual(self.g2.component(n1), self.g2)
        self.assertEqual(self.g3.component(n4), WeightedLinksDirectedGraph(
            {n1: ({n0: 3, n2: 4}, {n5: 5}), n2: ({n0: 6, n3: 1}, {n4: 0}), n4: ({}, {n5: 1})}))
        self.assertEqual(self.g3.component(n10), WeightedLinksDirectedGraph(
            {n6: ({}, {n7: 2, n8: 4}), n9: ({n7: 3, n8: 0, n10: 4}, {}), n11: ({n10: 1}, {})}))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedLinksDirectedGraph(
            {n1: ({n0: 2, n8: -1}, {n2: 3}), n2: ({n0: 4}, {n3: -6, n4: 5}), n3: ({n6: 3}, {n7: -3, n8: 5}),
             n4: ({n6: 2}, {n5: 0}), n5: ({n7: 1}, {n6: 5}), n7: ({n8: 4}, {n9: 3})}), res)
        self.assertIn(WeightedLinksDirectedGraph({n11: ({n10: 2}, {n12: 6, n13: 10}), n13: ({n12: 3}, {})}), res)
        self.assertIn(WeightedLinksDirectedGraph({n14: ({}, {})}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        res = self.g3.connection_components()
        self.assertEqual(len(res), 2)
        self.assertIn(WeightedLinksDirectedGraph(
            {n1: ({n0: 3, n2: 4}, {n5: 5}), n2: ({n0: 6, n3: 1}, {n4: 0}), n4: ({}, {n5: 1})}), res)
        self.assertIn(WeightedLinksDirectedGraph(
            {n6: ({}, {n7: 2, n8: 4}), n9: ({n7: 3, n8: 0, n10: 4}, {}), n11: ({n10: 1}, {})}), res)

    def test_node_subgraph(self):
        self.assertEqual(self.g0.subgraph(0), self.g0.component(0))
        self.assertEqual(self.g0.subgraph(10), self.g0.component(10))
        self.assertEqual(self.g0.subgraph(11),
                         WeightedLinksDirectedGraph({11: ({}, {12: 6, 13: 10}), 13: ({12: 3}, {})}))
        self.assertEqual(self.g1.subgraph(0), self.g1)
        self.assertEqual(self.g1.subgraph(1), self.g1)

    def test_node_subgraph_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.subgraph(6)

    def test_nodes_set_subgraph(self):
        self.assertEqual(self.g2.subgraph({n0, n1, n2, n3}),
                         WeightedLinksDirectedGraph({1: ({0: 2, 2: 6}, {3: 1}), 2: ({}, {0: 2, 3: -1})}))

    def test_scc_dag(self):
        g0 = self.g0.component(0).scc_dag()
        self.assertSetEqual(g0.nodes, {Node(frozenset({n0})), Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})),
                                       Node(frozenset({n9}))})
        self.assertDictEqual(g0.link_weights(),
                             {(Node(frozenset({n0})), Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8}))): 6,
                              (Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})), Node(frozenset({n9}))): 3})

    def test_minimal_path_links(self):
        self.assertListEqual(self.g0.minimal_path_links(1, 5), [n1, n2, n3, n7, n5])
        self.assertListEqual(self.g0.minimal_path_links(10, 13), [n10, n11, n12, n13])

    def test_minimal_path_links_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g1.minimal_path_links(2, 6)

    def test_isomorphic_bijection(self):
        g1 = WeightedLinksDirectedGraph(
            {n11: ({n13: 3}, {n10: 1, n12: 4, n14: 9, n15: 3}), n12: ({n10: 2}, {n13: -6}), n15: ({n13: 2}, {n14: 5})})
        func = self.g1.isomorphic_bijection(g1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        g1 = DirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n2, {n0, n1}).disconnect(n0, {n1}).disconnect(n4, {n5})
        g1.connect(n2, {n5: 3}, {n0: 2, n1: 6}), g1.connect(n3, {n1: 1, n4: 4})
        g1.disconnect(n1, points_to={n4, n5}), g1.disconnect(n1, {n3}), g1.connect(n0, {n5: 4},
                                                                                   {n1: 2}), g1.disconnect(n5, {n3})
        g2 = self.g2.copy().remove(n6)
        self.assertNotEqual(g1.connect(n5, {n4: 1}), g2)
        g1.increase_weight((n2, n3), 5)
        self.assertEqual(g1, g2)

    def test_add(self):
        expected = WeightedLinksDirectedGraph(
            {n0: ({n5: 4}, {}), n1: ({n0: 2, n2: 6, n3: 3}, {n0: 1, n2: 4, n3: 1, n4: 9, n5: 3}),
             n2: ({n0: 2, n5: 3}, {n0: 2, n3: -7}), n4: ({}, {n3: 4}), n5: ({n3: 2, n4: 1, n6: 2}, {n4: 5})})
        self.assertEqual(self.g1 + self.g2, expected)

    def test_str(self):
        self.assertEqual(str(self.g1), "<" + str(self.g1.nodes) + ", {" + ", ".join(
            f"<{l[0]}, {l[1]}> -> {self.g1.link_weights(l)}" for l in self.g1.links) + "}>")

    def test_repr(self):
        self.assertEqual(str(self.g0), repr(self.g0))


class TestWeightedDirectedGraph(TestCase):
    def setUp(self):
        self.g0 = WeightedDirectedGraph(
            {n0: (7, ({}, {})), n1: (3, ({n0: 2, n8: -1}, {n2: 3})), n2: (5, ({n0: 4}, {n3: -6, n4: 5})),
             n3: (2, ({n6: 3}, {n7: -3, n8: 5})), n4: (8, ({n6: 2}, {n5: 0})), n5: (4, ({n7: 1}, {n6: 5})),
             n6: (6, ({}, {})), n7: (6, ({n8: 4}, {n9: 3})), n8: (2, ({}, {})), n9: (5, ({}, {})), n10: (4, ({}, {})),
             n11: (2, ({n10: 2}, {n12: 6, n13: 10})), n12: (1, ({}, {})), n13: (3, ({n12: 3}, {})),
             n14: (6, ({}, {}))})
        self.g1 = WeightedDirectedGraph(
            {n0: (3, ({}, {})), n1: (2, ({n3: 3}, {n0: 1, n2: 4, n4: 9, n5: 3})), n2: (4, ({n0: 2, n1: 4}, {n3: -6})),
             n3: (6, ({}, {})), n4: (5, ({}, {})), n5: (1, ({n1: 3, n3: 2}, {n4: 5}))})
        self.g2 = WeightedDirectedGraph(
            {n0: (7, ({n2: 2, n5: 4}, {n1: 2})), n1: (6, ({}, {})), n2: (2, ({n5: 3}, {n1: 6, n3: -1})),
             n3: (4, ({n1: 1, n4: 4}, {})), n4: (3, ({}, {})), n5: (5, ({n4: 1, n6: 2}, {})), n6: (4, ({}, {}))})
        self.g3 = WeightedDirectedGraph(
            {n0: (7, ({}, {})), n1: (4, ({n0: 3, n2: 4}, {n5: 5})), n2: (2, ({n0: 6, n3: 1}, {n4: 0})),
             n3: (5, ({}, {})), n4: (3, ({}, {n5: 1})), n5: (2, ({}, {})), n6: (2, ({}, {n7: 2, n8: 4})),
             n7: (1, ({}, {})), n8: (6, ({}, {})), n9: (4, ({n7: 3, n8: 0, n10: 4}, {})), n10: (5, ({}, {})),
             n11: (8, ({n10: 1}, {}))})

    def test_init(self):
        g = WeightedDirectedGraph({0: (2, ({1: 1, 2: 0}, {3: 2})), 1: (3, ({2: 1, 3: 4}, {}))})
        self.assertDictEqual(g.node_weights(), {n0: 2, n1: 3, n2: 0, n3: 0})
        self.assertDictEqual(g.link_weights(), {(n0, n3): 2, (n1, n0): 1, (n2, n0): 0, (n2, n1): 1, (n3, n1): 4})

    def test_total_weight(self):
        self.assertEqual((self.g0.total_weight, self.g1.total_weight, self.g2.total_weight, self.g3.total_weight),
                         (112, 44, 55, 83))

    def test_add_node(self):
        g1 = self.g1.copy().add((6, 3), {2: 1, 3: 2}, {1: 2, 2: 3})
        self.assertDictEqual(g1.node_weights(), {n0: 3, n1: 2, n2: 4, n3: 6, n4: 5, n5: 1, n6: 3})
        link_weights = self.g1.link_weights()
        link_weights[(n2, n6)] = 1
        link_weights[(n3, n6)] = 2
        link_weights[(n6, n1)] = 2
        link_weights[(n6, n2)] = 3
        self.assertDictEqual(g1.link_weights(), link_weights)
        self.assertDictEqual(g1.prev(),
                             {n0: {n1}, n1: {n3, n6}, n2: {n0, n1, n6}, n3: {n2}, n4: {n1, n5}, n5: {n1, n3},
                              n6: {n2, n3}})
        self.assertDictEqual(g1.next(),
                             {n0: {n2}, n1: {n0, n2, n4, n5}, n2: {n3, n6}, n3: {n1, n5, n6}, n4: set(), n5: {n4},
                              n6: {n1, n2}})

    def test_add_present_node(self):
        self.assertEqual(self.g0, self.g0.copy().add((0, 4), {1: 2, 2: 1}))

    def test_remove(self):
        g1 = self.g1.copy().remove(0, 1, 2)
        self.assertEqual(g1, WeightedDirectedGraph({5: (1, ({3: 2}, {4: 5})), 3: (6, ({}, {})), 4: (5, ({}, {}))}))

    def test_remove_missing_node(self):
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
        g0 = self.g0.copy().set_weight((n2, n4), 3)
        res_link_weights = self.g0.link_weights()
        res_link_weights[(n2, n4)] = 3
        self.assertDictEqual(g0.link_weights(), res_link_weights)

    def test_set_link_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().set_weight((n2, n6), 3))

    def test_set_link_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().set_weight((n2, n4), [3])

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
        g0 = self.g0.copy().increase_weight((n2, n4), 1)
        res_link_weights = self.g0.link_weights()
        res_link_weights[(n2, n4)] = 6
        self.assertDictEqual(g0.link_weights(), res_link_weights)

    def test_increase_link_weight_missing_link(self):
        self.assertEqual(self.g0, self.g0.copy().increase_weight((n2, n6), 1))

    def test_increase_link_weight_bad_weight_type(self):
        with self.assertRaises(TypeError):
            self.g0.copy().increase_weight((n2, n4), [3])

    def test_copy(self):
        self.assertEqual(self.g0, self.g0.copy())
        self.assertEqual(self.g1, self.g1.copy())
        self.assertEqual(self.g2, self.g2.copy())
        self.assertEqual(self.g3, self.g3.copy())

    def test_component(self):
        self.assertEqual(self.g0.component(n9), WeightedDirectedGraph(
            {n0: (7, ({}, {})), n1: (3, ({n0: 2, n8: -1}, {n2: 3})), n2: (5, ({n0: 4}, {n3: -6, n4: 5})),
             n3: (2, ({n6: 3}, {n7: -3, n8: 5})), n4: (8, ({n6: 2}, {n5: 0})), n5: (4, ({n7: 1}, {n6: 5})),
             n6: (6, ({}, {})), n7: (6, ({n8: 4}, {n9: 3})), n8: (2, ({}, {})), n9: (5, ({}, {}))}))
        self.assertEqual(self.g0.component(n13), WeightedDirectedGraph(
            {n10: (4, ({}, {})), n11: (2, ({n10: 2}, {n12: 6, n13: 10})), n12: (1, ({}, {})),
             n13: (3, ({n12: 3}, {}))}))
        self.assertEqual(self.g1.component(n4), self.g1)
        self.assertEqual(self.g2.component(n1), self.g2)
        self.assertEqual(self.g3.component(n4), WeightedDirectedGraph(
            {n0: (7, ({}, {})), n1: (4, ({n0: 3, n2: 4}, {n5: 5})), n2: (2, ({n0: 6, n3: 1}, {n4: 0})),
             n3: (5, ({}, {})), n4: (3, ({}, {n5: 1})), n5: (2, ({}, {}))}))
        self.assertEqual(self.g3.component(n10), WeightedDirectedGraph(
            {n6: (2, ({}, {n7: 2, n8: 4})), n7: (1, ({}, {})), n8: (6, ({}, {})),
             n9: (4, ({n7: 3, n8: 0, n10: 4}, {})), n10: (5, ({}, {})), n11: (8, ({n10: 1}, {}))}))

    def test_connection_components(self):
        res = self.g0.connection_components()
        self.assertEqual(len(res), 3)
        self.assertIn(WeightedDirectedGraph(
            {n0: (7, ({}, {})), n1: (3, ({n0: 2, n8: -1}, {n2: 3})), n2: (5, ({n0: 4}, {n3: -6, n4: 5})),
             n3: (2, ({n6: 3}, {n7: -3, n8: 5})), n4: (8, ({n6: 2}, {n5: 0})), n5: (4, ({n7: 1}, {n6: 5})),
             n6: (6, ({}, {})), n7: (6, ({n8: 4}, {n9: 3})), n8: (2, ({}, {})), n9: (5, ({}, {}))}), res)
        self.assertIn(WeightedDirectedGraph(
            {n10: (4, ({}, {})), n11: (2, ({n10: 2}, {n12: 6, n13: 10})), n12: (1, ({}, {})),
             n13: (3, ({n12: 3}, {}))}), res)
        self.assertIn(WeightedDirectedGraph({n14: (6, ({}, {}))}), res)
        self.assertListEqual(self.g1.connection_components(), [self.g1])
        self.assertListEqual(self.g2.connection_components(), [self.g2])
        res = self.g3.connection_components()
        self.assertEqual(len(res), 2)
        self.assertIn(WeightedDirectedGraph(
            {n0: (7, ({}, {})), n1: (4, ({n0: 3, n2: 4}, {n5: 5})), n2: (2, ({n0: 6, n3: 1}, {n4: 0})),
             n3: (5, ({}, {})), n4: (3, ({}, {n5: 1})), n5: (2, ({}, {}))}), res)
        self.assertIn(WeightedDirectedGraph({n6: (2, ({}, {n7: 2, n8: 4})), n7: (1, ({}, {})), n8: (6, ({}, {})),
                                             n9: (4, ({n7: 3, n8: 0, n10: 4}, {})), n10: (5, ({}, {})),
                                             n11: (8, ({n10: 1}, {}))}), res)

    def test_transposed(self):
        self.assertEqual(self.g0.transposed(), WeightedDirectedGraph({u: (
            self.g0.node_weights(u),
            (self.g0.link_weights(u), {v: self.g0.link_weights(v, u) for v in self.g0.prev(u)}))
            for u in self.g0.nodes}))
        self.assertEqual(self.g1.transposed(), WeightedDirectedGraph({u: (
            self.g1.node_weights(u),
            (self.g1.link_weights(u), {v: self.g1.link_weights(v, u) for v in self.g1.prev(u)}))
            for u in self.g1.nodes}))
        self.assertEqual(self.g2.transposed(), WeightedDirectedGraph({u: (
            self.g2.node_weights(u),
            (self.g2.link_weights(u), {v: self.g2.link_weights(v, u) for v in self.g2.prev(u)}))
            for u in self.g2.nodes}))
        self.assertEqual(self.g3.transposed(), WeightedDirectedGraph({u: (
            self.g3.node_weights(u),
            (self.g3.link_weights(u), {v: self.g3.link_weights(v, u) for v in self.g3.prev(u)}))
            for u in self.g3.nodes}))

    def test_undirected(self):
        self.assertEqual(self.g1.undirected(), WeightedUndirectedGraph(
            {n0: (3, {}), n1: (2, {n0: 1, n2: 4, n3: 3, n4: 9, n5: 3}), n2: (4, {n0: 2, n3: -6}), n3: (6, {}),
             n4: (5, {}), n5: (1, {n1: 3, n3: 2, n4: 5})}))

    def test_node_subgraph(self):
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
        self.assertEqual(self.g3.subgraph(n10), WeightedDirectedGraph(
            {n9: (4, ({}, {})), n10: (5, ({}, {n9: 4, n11: 1})), n11: (8, ({}, {}))}))

    def test_node_subgraph_missing_node(self):
        with self.assertRaises(KeyError):
            self.g1.subgraph(6)

    def test_nodes_set_subgraph(self):
        self.assertEqual(self.g2.subgraph({n0, n1, n2, n5}), WeightedDirectedGraph(
            {0: (7, ({2: 2, 5: 4}, {1: 2})), 1: (6, ({2: 6}, {})), 2: (2, ({5: 3}, {})), 5: (5, ({}, {}))}))

    def test_scc_dag(self):
        g0 = self.g0.component(0).scc_dag()
        self.assertDictEqual(g0.node_weights(), {Node(frozenset({n0})): 7, Node(
            frozenset({n1, n2, n3, n4, n5, n6, n7, n8})): 36, Node(frozenset({n9})): 5})
        self.assertDictEqual(g0.link_weights(), {
            (Node(frozenset({n0})), Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8}))): 6,
            (Node(frozenset({n1, n2, n3, n4, n5, n6, n7, n8})), Node(frozenset({n9}))): 3})

    def test_minimal_path(self):
        self.assertListEqual(self.g0.minimal_path(n1, n13), [])
        self.assertListEqual(self.g0.minimal_path(n1, n6), [n1, n2, n3, n7, n5, n6])
        self.assertListEqual(self.g2.minimal_path(n4, n0), [n4, n5, n0])

    def test_minimal_path_missing_nodes(self):
        with self.assertRaises(KeyError):
            self.g1.minimal_path(3, 7)

    def test_isomorphic_bijection(self):
        g1 = WeightedDirectedGraph(
            {n10: (3, ({}, {})), n11: (2, ({n13: 3}, {n10: 1, n12: 4, n14: 9, n15: 3})),
             n12: (4, ({n10: 2}, {n13: -6})), n13: (6, ({}, {})), n14: (5, ({}, {})), n15: (1, ({n13: 2}, {n14: 5}))})
        func = self.g1.isomorphic_bijection(g1)
        self.assertEqual(func, {n0: n10, n1: n11, n2: n12, n3: n13, n4: n14, n5: n15})
        g1 = WeightedNodesDirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))
        g1 = DirectedGraph.copy(g1)
        self.assertDictEqual(func, self.g1.isomorphic_bijection(g1))

    def test_equal(self):
        self.assertNotEqual(self.g1, self.g2)
        g1 = self.g1.copy().disconnect(n2, {n0, n1}).disconnect(n0, {n1}).disconnect(n4, {n5})
        g1.connect(n2, {n5: 3}, {n0: 2, n1: 6}), g1.connect(n3, {n1: 1, n4: 4})
        g1.disconnect(n1, points_to={n4, n5}), g1.disconnect(n1, {n3}), g1.connect(n0, {n5: 4},
                                                                                   {n1: 2}), g1.disconnect(n5, {n3})
        g2 = self.g2.copy().remove(n6)
        self.assertNotEqual(g1.connect(n5, {n4: 1}), g2)
        g1.increase_weight((n2, n3), 5)
        self.assertNotEqual(g1, g2)
        g1.increase_weight(0, 4), g1.increase_weight(1, 4), g1.increase_weight(2, -2)
        g1.set_weight(3, 4), g1.set_weight(4, 3), g1.set_weight(5, 5)
        self.assertEqual(g1, g2)

    def test_add(self):
        expected = WeightedDirectedGraph(
            {n0: (10, ({n5: 4, n2: 2}, {})), n1: (8, ({n0: 2, n2: 6, n3: 3}, {n0: 1, n2: 4, n3: 1, n4: 9, n5: 3})),
             n2: (6, ({n0: 2, n5: 3}, {n3: -7})), n3: (10, ({}, {})), n4: (8, ({}, {n3: 4})),
             n5: (6, ({n3: 2, n4: 1, n6: 2}, {n4: 5})), n6: (4, ({}, {}))})
        self.assertEqual(self.g1 + self.g2, expected)

    def test_str(self):
        self.assertEqual(str(self.g0), "<{" + ", ".join(
            f"{n} -> {self.g0.node_weights(n)}" for n in self.g0.nodes) + "}, {" + ", ".join(
            f"<{l[0]}, {l[1]}> -> {self.g0.link_weights(l)}" for l in self.g0.links) + "}>")

    def test_repr(self):
        self.assertEqual(str(self.g0), repr(self.g0))


if __name__ == "__main__":
    main()
