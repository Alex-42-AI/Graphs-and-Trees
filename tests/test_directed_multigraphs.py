from unittest import TestCase, main

from ..src.directed_multigraph import *


class TestDirectedMultiGraph(TestCase):
    ...


class TestWeightedNodesDirectedMultiGraph(TestDirectedMultiGraph):
    ...


class TestWeightedLinksDirectedMultiGraph(TestDirectedMultiGraph):
    ...


class TestWeightedDirectedMultiGraph(TestWeightedNodesDirectedMultiGraph,
                                     TestWeightedLinksDirectedMultiGraph):
    ...


if __name__ == '__main__':
    main()
