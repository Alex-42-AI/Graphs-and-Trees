from Personal.DiscreteMath.Graphs.UndirectedGraph import UndirectedGraph, Link, Node
from Personal.DiscreteMath.Graphs.DirectedGraph import DirectedGraph
def make_graph_from_links(links: list):
    if not links:
        return
    if isinstance(links[0], Link):
        res = UndirectedGraph()
        for l in links:
            res.add(l.u()), res.add(l.v(), l.u())
    elif isinstance(links[0], tuple):
        res = DirectedGraph()
        for l in links:
            res.add(l[0]), res.add(l[1], [l[0]])
    else:
        raise TypeError("Invalid input!")
    return res
n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16 = Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7), Node(8), Node(9), Node(10), Node(11), Node(12), Node(13), Node(14), Node(15), Node(16)
k_3_3 = UndirectedGraph({n0: [n3, n4, n5], n1: [n3, n4, n5], n2: [n3, n4, n5]})
k_5 = UndirectedGraph({n0: [n1, n2, n3, n4], n1: [n2, n3, n4], n2: [n3, n4], n3: [n4]})
peterson_graph = UndirectedGraph({n0: [n1, n4, n5], n3: [n2, n4, n8], n9: [n4, n6, n7], n5: [n7, n8], n2: [n1, n7], n6: [n1, n8]})
print(k_3_3.chromaticNumberNodes(), k_5.euler_tour(), peterson_graph.vertexCover(), peterson_graph.dominatingSet(), sep='\n')
