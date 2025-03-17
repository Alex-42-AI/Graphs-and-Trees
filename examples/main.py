from .. import *

morse_code = BinTree("",
                     BinTree("E",
                             BinTree("I",
                                     BinTree("S",
                                             BinTree("H", "5", "4"),
                                             BinTree("V",
                                                     BinTree(left=BinTree(right="$")), "3")),
                                     BinTree("U", "F",
                                             BinTree(None,
                                                     BinTree(None, "?", "_"), "2"))),
                             BinTree("A",
                                     BinTree("R",
                                             BinTree("L", "&", BinTree(left='"')),
                                             BinTree(left=BinTree("+", right="."))),
                                     BinTree("W",
                                             BinTree("P", BinTree(left="@")),
                                             BinTree("J", right=BinTree("1", "'"))))),
                     BinTree("T",
                             BinTree("N",
                                     BinTree("D",
                                             BinTree("B", BinTree("6", "-"),
                                                     "="), BinTree("X", "/")),
                                     BinTree("K",
                                             BinTree("C", right=BinTree(None, ";",
                                                                        "!")),
                                             BinTree("Y", BinTree("(", right=")")))),
                             BinTree("M",
                                     BinTree("G",
                                             BinTree("Z", BinTree("7", right=BinTree(right=","))),
                                             "Q"),
                                     BinTree("O",
                                             BinTree(left=BinTree("8", ":")),
                                             BinTree(None, "9", "0")))))

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = map(Node, range(16))

k_3_3 = UndirectedGraph({n0: [n3, n4, n5], n1: [n3, n4, n5], n2: [n3, n4, n5]})

k_5 = UndirectedGraph({n0: [n1, n2, n3, n4], n1: [n2, n3, n4], n2: [n3, n4], n3: [n4]})

petersen_graph = UndirectedGraph(
    {n0: [n1, n4, n5], n3: [n2, n4, n8], n9: [n4, n6, n7], n5: [n7, n8], n2: [n1, n7], n6: [n1, n8]})

ug = UndirectedGraph({-1: {0, 1, 2, 3, 4}, 0: {1, 2, 3, 4, 5, 6, 8}, 1: {2, 3, 4, 5}, 2: {3, 8}, 4: {3, 5}, 6: {5, 7}})

ug0 = UndirectedGraph({n1: [n2, n3, n4], n2: [n0, n5], n5: [n0, n3, n4]})

ug1 = UndirectedGraph({n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11]})

ug2 = UndirectedGraph({n1: [n0, n2, n3], n2: [n0, n3], n4: [n3, n5, n6, n7, n8, n9, n10], n5: [n6], n8: [n7, n9],
                       n10: [n9, n11, n12, n13, n14], n13: [n14, n15]})

ug3 = UndirectedGraph({n0: [n1, n2], n2: [n1, n3, n4], n4: [n3, n5], n5: [n6, n7, n8], n6: [n7, n8], n7: [n8, n9]})

ug4 = UndirectedGraph({n0: [n1, n2, n3, n4, n5]})

ug5 = UndirectedGraph({n1: [n0, n2, n3, n4, n5, n6, n8], n2: [n0, n3, n4, n5, n6, n7, n8], n8: [n6, n7, n9, n10]})

ug6 = UndirectedGraph({n1: [n0, n2, n3, n4, n5, n7], n2: [n0, n3, n4, n5, n6, n7], n7: [n6, n8, n9]})

wug = WeightedUndirectedGraph({n0: (7, {n2: 2}), n1: (6, {n0: 1, n2: -4, n3: -6}), n2: (2, {}),
                               n3: (4, {n2: 1, n4: 2}), n4: (3, {}), n5: (5, {n0: 3, n4: 4, n6: 5}), n6: (4, {})})

tmp = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})

dg0 = DirectedGraph({n1: ([n3], [n0, n2, n4, n5]), n2: ([n0], [n3]), n5: ([n3], [n4])})

dg1 = DirectedGraph({n1: ([n0, n2], [n3]), n2: ([n5], [n0, n3]), n3: ([n4], []), n5: ([n4, n6], [n0])})

tree = Tree(n0, {n1: [n3, n4, n5], n2: [n6, n7], n3: [n8, n9], n5: [n10, n11]})

if __name__ == "__main__":
    with open("weighted_undirected_graph.txt", "w") as file:
        file.write(f"{wug}\n")
        file.write("Minimal spanning tree:\n")
        file.write(f"{wug.minimal_spanning_tree()}\n")
        file.write("Weighted vertex cover:\n")
        file.write(f"{wug.weighted_vertex_cover()}\n")
        file.write("Weighted dominating set:\n")
        file.write(f"{wug.weighted_dominating_set()}\n")
        file.write(f"Minimal path from 2 to 4: {wug.minimal_path(n2, n4)}\n")

    with open("morse_code.txt", "w", encoding="utf-8") as file:
        file.write(f"{morse_code}\n")

        for traversal in ("preorder", "in-order", "post-order"):
            file.write(f"Traversal type {traversal}: {morse_code.traverse(traversal)}\n")

        file.write(f"Morse code of '4': {morse_code.code_in_morse(Node("4"))}\n")
        file.write(f"Total nodes: {morse_code.count_nodes()}\n")
        file.write(f"All leaves: {morse_code.leaves}\n")
        file.write(f"Tree height: {morse_code.height()}\n")
        file.write(f"Nodes on level 6: {morse_code.nodes_on_level(6)}\n")
        file.write(f"Tree width: {morse_code.width()}\n")
        file.write(f"Encoding message 'Testing encode.':\n{morse_code.encode("Testing encode.".upper())}\n")

    with open("k_3_3.txt", "w") as file:
        file.write(f"{k_3_3}\n")
        file.write(f"Is full bipartite: {k_3_3.is_full_k_partite(2)}\n")
        file.write(f"Chromatic nodes partition: {k_3_3.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic links partition: {k_3_3.chromatic_links_partition()}\n")
        file.write(f"Hamilton tour: {k_3_3.hamilton_tour()}\n")
        file.write(f"Interval sort: {k_3_3.interval_sort()}\n")
        file.write(f"Dominating set: {k_3_3.dominating_set()}\n")
        file.write(f"Vertex cover: {k_3_3.vertex_cover()} \n")
        file.write(f"Independent set: {k_3_3.independent_set()}\n")

    with open("k_5.txt", "w") as file:
        file.write(f"{k_5}\n")
        file.write(f"Is full: {k_5.full()}\n")
        file.write(f"3-cliques: {k_5.cliques(3)}\n")
        file.write(f"Chromatic links partition: {k_5.chromatic_links_partition()}\n")
        file.write(f"Euler tour: {k_5.euler_tour()}\n")
        file.write(f"Interval sort: {k_5.interval_sort()}\n")
        file.write(f"Dominating set: {k_5.dominating_set()}\n")
        file.write(f"Vertex cover: {k_5.vertex_cover()}\n")
        file.write(f"Independent set: {k_5.independent_set()}\n")

    with open("petersen_graph.txt", "w") as file:
        file.write(f"{petersen_graph}\n")
        file.write(f"Is full k-partite: {petersen_graph.is_full_k_partite()}\n")
        file.write(f"Chromatic nodes partition: {petersen_graph.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic links partition: {petersen_graph.chromatic_links_partition()}\n")
        file.write(f"Hamilton walk: {petersen_graph.hamilton_walk()}\n")
        file.write(f"Interval sort: {petersen_graph.interval_sort()}\n")
        file.write(f"Dominating set: {petersen_graph.dominating_set()}\n")
        file.write(f"Vertex cover: {petersen_graph.vertex_cover()}\n")
        file.write(f"Independent set: {petersen_graph.independent_set()}\n")
        file.write(f"Shortest path from 0 to 7: {petersen_graph.get_shortest_path(n0, n7)}\n")

    with open("undirected_graphs.txt", "w", encoding="utf-8") as file:
        file.write(f"Graph 1: {ug0}\nGraph 2: {ug1}\nGraph 3: {ug2}\nGraph 4: {ug3}\n")
        file.write(f"Graph 5: {ug4}\nGraph 6: {ug5}\nGraph 7: {ug6}\nGraph 8: {ug}\n")
        file.write(f"Graph 2 diameter: {ug1.diameter()}\nGraph 7 diameter: {ug6.diameter()}\n")
        file.write(f"Is graph 1 a tree: {ug0.is_tree()}\nIs graph 2 a tree: {ug1.is_tree()}\n")
        file.write(f"Is graph 3 a tree: {ug2.is_tree()}\nIs graph 4 a tree: {ug3.is_tree()}\n")
        file.write(f"Is graph 5 a tree: {ug4.is_tree()}\nIs graph 6 a tree: {ug5.is_tree()}\n")
        file.write(f"Is graph 7 a tree: {ug6.is_tree()}\n")
        file.write(f"Graph 1 tree with root 2:\n{ug0.tree(n2)}\n")
        file.write(f"Graph 2 tree with root 0:\n{ug1.tree(n0)}\n")
        file.write(f"Graph 5 tree with root 1:\n{ug4.tree(n1)}\n")
        file.write(f"Graph 2 cut nodes: {ug1.cut_nodes()}\nGraph 2 bridge links: {ug1.bridge_links()}\n")
        file.write(f"Graph 3 cut nodes: {ug2.cut_nodes()}\nGraph 3 bridge links: {ug2.bridge_links()}\n")
        file.write(f"Graph 4 cut nodes: {ug3.cut_nodes()}\nGraph 4 bridge links: {ug3.bridge_links()}\n")
        file.write(f"Graph 6 cut nodes: {ug5.cut_nodes()}\nGraph 6 bridge links: {ug5.bridge_links()}\n")
        file.write(f"Graph 7 cut nodes: {ug6.cut_nodes()}\nGraph 7 bridge links: {ug6.bridge_links()}\n")
        file.write(f"Euler walk from 2 to 1 in graph 1: {ug0.euler_walk(n2, n1)}\n")
        file.write(f"Shortest path from 10 to 3 in graph 2: {ug1.get_shortest_path(n10, n3)}\n")
        file.write(f"Shortest path from 3 to 9 in graph 6: {ug5.get_shortest_path(n3, n9)}\n")
        file.write(f"3-cliques in graph 1: {ug0.cliques(3)}\n3-cliques in graph 3: {ug2.cliques(3)}\n")
        file.write(f"3-cliques in graph 4: {ug3.cliques(3)}\n4-cliques in graph 4: {ug3.cliques(4)}\n")
        file.write(f"3-cliques in graph 6: {ug5.cliques(3)}\n4-cliques in graph 6: {ug5.cliques(4)}\n")
        file.write(f"3-cliques in graph 7: {ug6.cliques(3)}\n")
        file.write(f"Graph 3 interval sort from 0: {ug2.interval_sort(n0)}\n")
        file.write(f"Graph 4 interval sort from 0: {ug3.interval_sort(n0)}\n")
        file.write(f"Graph 5 interval sort from 0: {ug4.interval_sort(n0)}\n")
        file.write(f"Graph 6 interval sort from 0: {ug5.interval_sort(n0)}\n")
        file.write(f"Graph 7 interval sort from 0: {ug6.interval_sort(n0)}\n")
        file.write(f"Graph 8 interval sort from 8: {ug.interval_sort(n8)}\n")
        file.write(f"Chromatic nodes partition of graph 1: {ug0.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic links partition of graph 1: {ug0.chromatic_links_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 2: {ug1.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic links partition of graph 2: {ug1.chromatic_links_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 3: {ug2.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 4: {ug3.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 5: {ug4.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 6: {ug5.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 7: {ug6.chromatic_nodes_partition()}\n")
        file.write(f"Chromatic nodes partition of graph 8: {ug.chromatic_nodes_partition()}\n")
        file.write(f"Path with a length of 4 in graph 1 between 4 and 5: {ug0.path_with_length(n4, n5, 4)}\n")
        file.write(f"Loop with a length of 5 in graph 1: {ug0.cycle_with_length(5)}\n")
        file.write(f"Graph 1 optimal vertex cover: {ug0.vertex_cover()}\n")
        file.write(f"Graph 1 optimal dominating set: {ug0.dominating_set()}\n")
        file.write(f"Graph 1 optimal independent set: {ug0.independent_set()}\n")
        file.write(f"Graph 2 optimal vertex cover: {ug1.vertex_cover()}\n")
        file.write(f"Graph 2 optimal dominating set: {ug1.dominating_set()}\n")
        file.write(f"Graph 2 optimal independent set: {ug1.independent_set()}\n")
        file.write(f"Graph 3 optimal vertex cover: {ug2.vertex_cover()}\n")
        file.write(f"Graph 3 optimal dominating set: {ug2.dominating_set()}\n")
        file.write(f"Graph 3 optimal independent set: {ug2.independent_set()}\n")
        file.write(f"Graph 4 optimal vertex cover: {ug3.vertex_cover()}\n")
        file.write(f"Graph 4 optimal dominating set: {ug3.dominating_set()}\n")
        file.write(f"Graph 4 optimal independent set: {ug3.independent_set()}\n")
        file.write(f"Graph 5 optimal vertex cover: {ug4.vertex_cover()}\n")
        file.write(f"Graph 5 optimal dominating set: {ug4.dominating_set()}\n")
        file.write(f"Graph 5 optimal independent set: {ug4.independent_set()}\n")
        file.write(f"Graph 6 optimal vertex cover: {ug5.vertex_cover()}\n")
        file.write(f"Graph 6 optimal dominating set: {ug5.dominating_set()}\n")
        file.write(f"Graph 6 optimal independent set: {ug5.independent_set()}\n")
        file.write(f"Graph 7 optimal vertex cover: {ug6.vertex_cover()}\n")
        file.write(f"Graph 7 optimal dominating set: {ug6.dominating_set()}\n")
        file.write(f"Graph 7 optimal independent set: {ug6.independent_set()}\n")
        file.write(f"Graph 8 optimal vertex cover: {ug.vertex_cover()}\n")
        file.write(f"Graph 8 optimal dominating set: {ug.dominating_set()}\n")
        file.write(f"Graph 8 optimal independent set: {ug.independent_set()}\n")
        file.write(f"Graph 1 Hamilton walk: {ug0.hamilton_walk()}\n")
        file.write(f"Graph 4 Hamilton walk: {ug3.hamilton_walk()}\n")
        file.write(f"Helper: {tmp}\n")
        file.write(f"Isomorphic function between graph 1 and helper: {ug0.isomorphic_bijection(tmp)}\n")

    with open("tree.txt", "w", encoding="utf-8") as file:
        file.write(f"{tree}\n")
        file.write(f"Height: {tree.height()}\n")
        file.write(f"Descendants of 2: {tree.descendants(n2)}\n")
        file.write(f"Subtree from 1:\n{tree.subtree(n1)}\n")
        file.write(f"Depth of 5: {tree.node_depth(n5)}\n")
        file.write(f"Depth of 9: {tree.node_depth(n9)}\n")
        file.write(f"Path to 11: {tree.path_to(n11)}\n")
        file.write(f"Vertex cover: {tree.vertex_cover()}\n")
        file.write(f"Dominating set: {tree.dominating_set()}\n")
        file.write(f"Independent set: {tree.independent_set()}\n")

    with open("directed_graphs.txt", "w") as file:
        file.write(f"Graph 1: {dg0}\nGraph 2: {dg1}\n")
        file.write(f"Graph 1:\nsources: {dg0.sources}\nsinks: {dg0.sinks}\n")
        file.write(f"Graph 2:\nsources: {dg1.sources}\nsinks: {dg1.sinks}\n")
        file.write(f"Graph 1 strongly-connected components partition: {dg0.strongly_connected_components()}\n")
        file.write(f"Graph 2 subgraph from 5: {dg1.subgraph(n5)}\n")
        file.write(f"Is graph 1 a dag: {dg0.dag()}\n")
        file.write(f"Is graph 2 a dag: {dg1.dag()}\n")
        file.write(f"Graph 2 toposort: {dg1.toposort()}\n")
        file.write(f"Graph 1 hamilton walk: {dg0.hamilton_walk()}\n")
