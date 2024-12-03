from Graphs.UndirectedGraph import *

from Graphs.DirectedGraph import *

from Graphs.Tree import *


def clique_to_SAT(cnf: list[list[tuple[str, bool]]]):
    def compatible(var1, var2):
        return var1[0] != var2[0] or var1[1] == var2[1]

    def independent_set(x):
        for i_s in independent_sets:
            if x in i_s:
                return i_s

    i = 0
    while i < len(cnf):
        j, clause, contradiction = 0, cnf[i], False
        while j < len(clause):
            for var0 in clause[j + 1:]:
                if not compatible(var := clause[j], var0):
                    contradiction = True
                    break
                if var == var0:
                    clause.pop(j)
                    j -= 1
                    break
                j += 1
            if contradiction:
                break
        if contradiction:
            i -= 1
            cnf.pop(i)
        i += 1
    if not cnf:
        return [[]]
    graph, node_vars, i, independent_sets = UndirectedGraph(), {}, 0, []
    for clause in cnf:
        j = i
        for var in clause:
            node_vars[Node(i)] = var
            graph.add(Node(i))
            i += 1
        i += 1
        independent_sets.append({*map(Node, range(j, i))})
    for u in graph.nodes:
        for v in graph.nodes:
            if u != v and compatible(node_vars[u], node_vars[v]) and v.value not in independent_set(u.value):
                graph.connect(u, v)
    result, n = [], len(cnf)
    for u in min(independent_sets, key=len):
        if len((curr := graph.maxCliquesNode(u))[0]) == n:
            result += [set(map(node_vars.__getitem__, clique)) for clique in curr]
    return result


def make_undirected_from_directed(graph: DirectedGraph):
    if isinstance(graph, WeightedDirectedGraph):
        res = WeightedUndirectedGraph()
        for u in graph.nodes:
            if u not in res:
                res.add((u, graph.node_weights(u)))
            for v in graph.next(u):
                if v not in res:
                    res.add((v, graph.node_weights(v)))
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + graph.link_weights(u, v))
                else:
                    res.connect(u, {v: graph.link_weights(u, v)})
            for v in graph.prev(u):
                if v not in res:
                    res.add((v, graph.node_weights(v)))
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + graph.link_weights(v, u))
                else:
                    res.connect(u, {v: graph.link_weights(v, u)})
    elif isinstance(graph, WeightedLinksDirectedGraph):
        res = WeightedLinksUndirectedGraph()
        for u in graph.nodes:
            if u not in res:
                res.add(u)
            for v in graph.next(u):
                if v not in res:
                    res.add(v)
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + graph.link_weights(u, v))
                else:
                    res.connect(u, {v: graph.link_weights(u, v)})
            for v in graph.prev(u):
                if v not in res:
                    res.add(v)
                if v in res.neighboring(u):
                    res.set_weight(Link(u, v), res.link_weights(u, v) + graph.link_weights(v, u))
                else:
                    res.connect(u, {v: graph.link_weights(v, u)})
    elif isinstance(graph, WeightedNodesDirectedGraph):
        res = WeightedNodesUndirectedGraph()
        for u in graph.nodes:
            if u not in res:
                res.add((u, graph.node_weights(u)))
            for v in graph.next(u):
                if v not in res:
                    res.add((v, graph.node_weights(v)))
                if v not in res.neighboring(u):
                    res.connect(u, v)
            for v in graph.prev(u):
                if v not in res:
                    res.add((v, graph.node_weights(v)))
                if v not in res.neighboring(u):
                    res.connect(u, v)
    res = UndirectedGraph()
    for u in graph.nodes:
        if u not in res:
            res.add(u)
        for v in graph.next(u):
            if v not in res:
                res.add(v)
            if v not in res.neighboring(u):
                res.connect(u, v)
        for v in graph.prev(u):
            if v not in res:
                res.add(v)
            if v not in res.neighboring(u):
                res.connect(u, v)
    return res


def build_heap(ll: list[int], h: int = 0):
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

    if not h:
        h = len(ll)
    for i in range(h // 2, 0, -1):
        heapify(0, h, i)


def binary_heap(l: list):
    build_heap(l, len(l))

    def helper(curr_root, rest, i=1):
        left = helper(rest[0], rest[(2 ** i):], i + 1) if rest else None
        right = helper(rest[1], rest[2 * 2 ** i:], i + 1) if rest[1:] else None
        res = BinTree(curr_root, left, right)
        return res

    return BinTree(helper(l[0], l[1:]))


def print_zig_zag(b_t: BinTree):
    def bfs(from_left: bool, *trees: BinTree):
        new = []
        if from_left:
            for t in trees:
                if t.left and (t.left.left is not None or t.left.right is not None):
                    new.insert(0, t.left), print(t.left.root, end=' ')
                if t.right and (t.right.left is not None or t.right.right is not None):
                    new.insert(0, t.right), print(t.right.root, end=' ')
        else:
            for t in trees:
                if t.right and (t.right.left is not None or t.right.right is not None):
                    new.insert(0, t.right), print(t.right.root, end=' ')
                if t.left and (t.left.left is not None or t.left.right is not None):
                    new.insert(0, t.left), print(t.left.root, end=' ')
        if not new:
            return
        print(), bfs(not from_left, *new)

    print(b_t.root), bfs(True, b_t)


morse_code = BinTree('',
                     BinTree('E',
                             BinTree('I',
                                     BinTree('S',
                                             BinTree('H', '5', '4'),
                                             BinTree('V',
                                                     BinTree(left=BinTree(right='$')), '3')),
                                     BinTree('U', 'F',
                                             BinTree(None,
                                                     BinTree(None, '?', '_'), '2'))),
                             BinTree('A',
                                     BinTree('R',
                                             BinTree('L', '&', BinTree(left='"')),
                                             BinTree(left=BinTree('+', right='.'))),
                                     BinTree('W',
                                             BinTree('P', BinTree(left='@')),
                                             BinTree('J', right=BinTree('1', '\''))))),
                     BinTree('T',
                             BinTree('N',
                                     BinTree('D',
                                             BinTree('B', BinTree('6', '-'), '='),
                                             BinTree('X', '/')),
                                     BinTree('K',
                                             BinTree('C', right=BinTree(None, ';', '!')),
                                             BinTree('Y', BinTree('(', right=')')))),
                             BinTree('M',
                                     BinTree('G',
                                             BinTree('Z', BinTree('7', right=BinTree(right=','))), 'Q'),
                                     BinTree('O',
                                             BinTree(left=BinTree('8', ':')),
                                             BinTree(None, '9', '0')))))
with open("Morse code.txt", "w", encoding="utf-8") as file:
    file.write(f"{morse_code}\n")
    for traversal in ('preorder', 'in-order', 'post-order'):
        file.write(f"Traversal type {traversal}: {morse_code.traverse(traversal)}\n")
    file.write(f"Morse code of '4': {morse_code.code_in_morse('4')}\n")
    file.write(f"Total nodes: {morse_code.count_nodes()}\n")
    file.write(f"All leaves: {morse_code.leaves()}\n")
    file.write(f"Tree height: {morse_code.height()}\n")
    file.write(f"Nodes on level 6: {morse_code.nodes_on_level(6)}\n")
    file.write(f"Tree width: {morse_code.width()}\n")
    file.write(f"Encoding message 'Testing encode.':\n{morse_code.encode('Testing encode.'.upper())}\n")
print_zig_zag(morse_code)

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7), Node(8), Node(9), Node(10), Node(11), Node(12), Node(13), Node(14), Node(15)
k_3_3 = UndirectedGraph({n0: [n3, n4, n5], n1: [n3, n4, n5], n2: [n3, n4, n5]})
k_5 = UndirectedGraph({n0: [n1, n2, n3, n4], n1: [n2, n3, n4], n2: [n3, n4], n3: [n4]})
petersen_graph = UndirectedGraph({n0: [n1, n4, n5], n3: [n2, n4, n8], n9: [n4, n6, n7], n5: [n7, n8], n2: [n1, n7], n6: [n1, n8]})
ug0 = UndirectedGraph({n1: [n2, n3, n4], n2: [n0, n5], n5: [n0, n3, n4]})
ug1 = UndirectedGraph({n1: [n0, n3, n4, n5], n2: [n0, n6, n7], n3: [n8, n9], n5: [n10, n11]})
tmp = UndirectedGraph({n11: [n12, n13, n14], n12: [n10, n15], n15: [n10, n13, n14]})
dg0 = DirectedGraph({n1: ([n3], [n0, n2, n4, n5]), n2: ([n0], [n3]), n5: ([n3], [n4])})
dg1 = DirectedGraph({n1: ([n0, n2], [n3]), n2: ([n5], [n0, n3]), n3: ([n4], []), n5: ([n4, n6], [n0])})
tree = Tree(n0, {n1: [n3, n4, n5], n2: [n6, n7], n3: [n8, n9], n5: [n10, n11]})
with open("K_3_3.txt", "w") as file:
    file.write(f"{k_3_3}\n")
    file.write(f"Is full bipartite: {k_3_3.is_full_k_partite(2)}\n")
    file.write(f"Chromatic nodes partition: {k_3_3.chromaticNodesPartition()}\n")
    file.write(f"Chromatic links partition: {k_3_3.chromaticLinksPartition()}\n")
    file.write(f"Hamilton tour: {k_3_3.hamiltonTour()}\n")
    file.write(f"Interval sort: {k_3_3.interval_sort()}\n")
    file.write(f"Dominating set: {k_3_3.dominatingSet()}\n")
    file.write(f"Vertex cover: {k_3_3.vertexCover()} \n")
    file.write(f"Independent set: {k_3_3.independentSet()}\n")
with open("K_5.txt", "w") as file:
    file.write(f"{k_5}\n")
    file.write(f"Is full: {k_5.full()}\n")
    file.write(f"3-cliques: {k_5.cliques(3)}\n")
    file.write(f"Chromatic links partition: {k_5.chromaticLinksPartition()}\n")
    file.write(f"Euler tour: {k_5.euler_tour()}\n")
    file.write(f"Interval sort: {k_5.interval_sort()}\n")
    file.write(f"Dominating set: {k_5.dominatingSet()}\n")
    file.write(f"Vertex cover: {k_5.vertexCover()}\n")
    file.write(f"Independent set: {k_5.independentSet()}\n")
with open("Petersen graph.txt", "w") as file:
    file.write(f"{petersen_graph}\n")
    file.write(f"Is full k-partite: {petersen_graph.is_full_k_partite()}\n")
    file.write(f"Chromatic nodes partition: {petersen_graph.chromaticNodesPartition()}\n")
    file.write(f"Chromatic links partition: {petersen_graph.chromaticLinksPartition()}\n")
    file.write(f"Hamilton walk: {petersen_graph.hamiltonWalk()}\n")
    file.write(f"Interval sort: {petersen_graph.interval_sort()}\n")
    file.write(f"Dominating set: {petersen_graph.dominatingSet()}\n")
    file.write(f"Vertex cover: {petersen_graph.vertexCover()}\n")
    file.write(f"Independent set: {petersen_graph.independentSet()}\n")
    file.write(f"Shortest path from 0 to 7: {petersen_graph.get_shortest_path(n0, n7)}\n")
with open("Undirected graphs.txt", "w", encoding="utf-8") as file:
    file.write(f"Graph 1: {ug0}\nGraph 2: {ug1}\n")
    file.write(f"Graph 2 width: {ug1.diameter()}\n")
    file.write(f"Is graph 1 a tree: {ug0.is_tree()}\nIs graph 2 a tree: {ug1.is_tree()}\n")
    file.write(f"Graph 1 tree with root 2:\n{ug0.tree(n2)}\n")
    file.write(f"Graph 2 tree with root 0:\n{ug1.tree(n0)}\n")
    file.write(f"Graph 2 cut nodes: {ug1.cut_nodes()}\nGraph 2 bridge links: {ug1.bridge_links()}\n")
    file.write(f"Euler walk from 2 to 1 in graph 1: {ug0.euler_walk(n2, n1)}\n")
    file.write(f"Shortest path from 10 to 3 in graph 2: {ug1.get_shortest_path(n10, n3)}\n")
    file.write(f"3-cliques in graph 1: {ug0.cliques(3)}\n")
    file.write(f"Chromatic nodes partition of graph 1: {ug0.chromaticNodesPartition()}\n")
    file.write(f"Chromatic links partition of graph 1: {ug0.chromaticLinksPartition()}\n")
    file.write(f"Chromatic nodes partition of graph 2: {ug1.chromaticNodesPartition()}\n")
    file.write(f"Chromatic links partition of graph 2: {ug1.chromaticLinksPartition()}\n")
    file.write(f"Path with a length of 4 in graph 1 between 4 and 5: {ug0.pathWithLength(n4, n5, 4)}\n")
    file.write(f"Loop with a length of 5 in graph 1: {ug0.loopWithLength(5)}\n")
    file.write(f"Graph 1 optimal vertex cover: {ug0.vertexCover()}\n")
    file.write(f"Graph 1 optimal dominating set: {ug0.dominatingSet()}\n")
    file.write(f"Graph 1 optimal independent set: {ug0.independentSet()}\n")
    file.write(f"Graph 2 optimal vertex cover: {ug1.vertexCover()}\n")
    file.write(f"Graph 2 optimal dominating set: {ug1.dominatingSet()}\n")
    file.write(f"Graph 2 optimal independent set: {ug1.independentSet()}\n")
    file.write(f"Graph 1 Hamilton walk: {ug0.hamiltonWalk()}\n")
    file.write(f"Helper: {tmp}\n")
    file.write(f"Isomorphic function between graph 1 and helper: {ug0.isomorphicFunction(tmp)}\n")
with open("Tree.txt", "w", encoding="utf-8") as file:
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
with open("Directed graphs.txt", "w") as file:
    file.write(f"Graph 1: {dg0}\nGraph 2: {dg1}\n")
    file.write(f"Graph 1:\nsources: {dg0.sources}\nsinks: {dg0.sinks}\n")
    file.write(f"Graph 2:\nsources: {dg1.sources}\nsinks: {dg1.sinks}\n")
    file.write(f"Graph 1 strongly-connected components partition: {dg0.strongly_connected_components()}\n")
    file.write(f"Graph 2 subgraph from 5: {dg1.subgraph(n5)}\n")
    file.write(f"Is graph 1 a dag: {dg0.dag()}\n")
    file.write(f"Is graph 2 a dag: {dg1.dag()}\n")
    file.write(f"Graph 2 toposort: {dg1.toposort()}\n")
    file.write(f"Graph 1 hamilton walk: {dg0.hamiltonWalk()}\n")
