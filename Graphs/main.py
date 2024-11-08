from Personal.DiscreteMath.Graphs.UndirectedGraph import *
from Personal.DiscreteMath.Graphs.DirectedGraph import *
from Personal.DiscreteMath.Graphs.Tree import *

def make_undirected_from_directed(graph: DirectedGraph):
    if isinstance(graph, WeightedDirectedGraph):
        res = WeightedUndirectedGraph(f=graph.f)
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
        res = WeightedLinksUndirectedGraph(f=graph.f)
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
        res = WeightedNodesUndirectedGraph(f=graph.f)
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
    res = UndirectedGraph(f=graph.f)
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


def heapify(ll: [int], l: int, h: int, i: int, f=max):
    left, right = 2 * i - l, 2 * i - l + 1
    res = i
    if left <= h and (el := ll[i - 1]) != f(ll[left - 1], el):
        res = left
    if right <= h and (el := ll[res - 1]) != f(ll[right - 1], el):
        res = right
    if res != i:
        ll[i - 1], ll[res - 1] = ll[res - 1], ll[i - 1]
        heapify(ll, res - l - 1, h, res, f)


def build_heap(ll: [int], h: int = 0):
    if not h:
        h = len(ll)
    for i in range(h // 2, 0, -1):
        heapify(ll, 0, h, i)


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
                if t.left() and (t.left().left() is not None or t.left().right() is not None):
                    new.insert(0, t.left()), print(t.left().root(), end=' ')
                if t.right() and (t.right().left() is not None or t.right().right() is not None):
                    new.insert(0, t.right()), print(t.right().root(), end=' ')
        else:
            for t in trees:
                if t.right() and (t.right().left() is not None or t.right().right() is not None):
                    new.insert(0, t.right()), print(t.right().root(), end=' ')
                if t.left() and (t.left().left() is not None or t.left().right() is not None):
                    new.insert(0, t.left()), print(t.left().root(), end=' ')
        if not new:
            return
        print(), bfs(not from_left, *new)

    print(b_t.root()), bfs(True, b_t)


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
with open("Morse code.txt", "a") as file:
    file.write(f"{morse_code}\n")
    file.write(f"Code in morse of '4' - {morse_code.code_in_morse(Node('4'))}\n")
    file.write(f"Total nodes - {morse_code.count_nodes()}\n")
    file.write(f"Total leaves - {morse_code.count_leaves()}\n")
    file.write(f"Tree height - {morse_code.height}\n")
    file.write(f"Nodes on level 6 - {morse_code.nodes_on_level(6)}\n")
    file.write(f"Tree width - {morse_code.width}\n")
    file.write(f"Encoding message 'Testing encode.':\n{morse_code.encode('Testing encode.')}\n")
for Type in ('preorder', 'in-order', 'post-order'):
    morse_code.print(Type)

n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, = Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7), Node(8), Node(9)
k_3_3 = UndirectedGraph({n0: [n3, n4, n5], n1: [n3, n4, n5], n2: [n3, n4, n5]})
k_5 = UndirectedGraph({n0: [n1, n2, n3, n4], n1: [n2, n3, n4], n2: [n3, n4], n3: [n4]})
peterson_graph = UndirectedGraph({n0: [n1, n4, n5], n3: [n2, n4, n8], n9: [n4, n6, n7], n5: [n7, n8], n2: [n1, n7], n6: [n1, n8]})
with open("K_3_3.txt", "a") as file:
    file.write(f"{k_3_3}\n")
    file.write(f"Is full bipartite - {k_3_3.is_full_k_partite()}\n")
    file.write(f"Chromatic set destruction - {k_3_3.chromaticNumberNodes()}\n")
    file.write(f"Hamilton tour - {k_3_3.hamiltonTour()}\n")
    file.write(f"Interval sort - {k_3_3.interval_sort()}\n")
    file.write(f"Dominating sets - {k_3_3.dominatingSet()}\n")
    file.write(f"Vertex covers - {k_3_3.vertexCover()} \n")
    file.write(f"Independent sets - {k_3_3.independentSet()}\n")
with open("K_5.txt", "a") as file:
    file.write(f"{k_5}\n")
    file.write(f"Is full - {k_5.full()}\n")
    file.write(f"Euler tour - {k_5.eulerTour()}\n")
    file.write(f"Interval sort - {k_5.interval_sort()}\n")
    file.write(f"Dominating sets - {k_5.dominatingSet()}\n")
    file.write(f"Vertex covers - {k_5.vertexCover()}\n")
    file.write(f"Independent sets - {k_5.independentSet()}\n")
with open("Peterson.txt", "a") as file:
    file.write(f"{peterson_graph}\n")
    file.write(f"Is full k-partite - {peterson_graph.is_full_k_partite()}\n")
    file.write(f"Chromatic set destruction - {peterson_graph.chromaticNumberNodes()}\n")
    file.write(f"Hamilton walk - {peterson_graph.hamiltonWalk()}\n")
    file.write(f"Interval sort - {peterson_graph.interval_sort()}\n")
    file.write(f"Dominating sets - {peterson_graph.dominatingSet()}\n")
    file.write(f"Vertex covers - {peterson_graph.vertexCover()}\n")
    file.write(f"Independent sets - {peterson_graph.independentSet()}\n")
    file.write(f"Shortest path from (0) to (7) - {peterson_graph.get_shortest_path(n0, n7)}\n")
