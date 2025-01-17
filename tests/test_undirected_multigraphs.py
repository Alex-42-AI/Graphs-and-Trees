from unittest import TestCase, main

from ..src.undirected_multigraph import *


class TestUndirectedMultiGraph(TestCase):
    ...


class TestWeightedNodesUndirectedMultiGraph(TestUndirectedMultiGraph):
    ...


class TestWeightedLinksUndirectedMultiGraph(TestUndirectedMultiGraph):
    ...


class TestWeightedUndirectedMultiGraph(TestWeightedNodesUndirectedMultiGraph,
                                       TestWeightedLinksUndirectedMultiGraph):
    ...


if __name__ == '__main__':
    main()
