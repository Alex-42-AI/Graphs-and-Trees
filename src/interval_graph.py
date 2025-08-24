"""
Module for implementing a set of intervals as an undirected graph
"""

from __future__ import annotations

from base import Hashable

from undirected_graph import UndirectedGraph, WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph, \
    WeightedUndirectedGraph

from bisect import bisect_right

__all__ = ["IntervalGraph", "WeightedNodesIntervalGraph", "WeightedLinksIntervalGraph", "WeightedUndirectedGraph"]

Interval = tuple[float, float]

class IntervalGraph:
    """
    Class for implementing a set of intervals as an unweighted undirected graph
    """

    def __init__(self, *intervals: Interval) -> None:
        """
        Args:
            intervals (Interval): list of intervals
        Creates an IntervalGraph object
        """

        self.__graph = UndirectedGraph()
        self.add(*intervals)

    @property
    def graph(self) -> UndirectedGraph:
        """
        Returns:
            The UndirectedGraph object
        """

        return self.__graph

    @property
    def nodes(self) -> set[Interval]:
        """
        Returns:
            The set of intervals
        """

        return {u.value for u in self.graph.nodes}

    def neighbors(self, interval: Interval = None) -> set[Interval] | dict[Interval, set[Interval]]:
        """
        Args:
            interval (Interval): A given interval or None
        Returns:
            The other intervals in the graph, that intersect with the given one, if it isn't None, otherwise the
            dictionary of all intervals and their neighbors
        """

        if interval is None:
            return {i: self.neighbors(i) for i in self.nodes}

        return {u.value for u in self.graph.neighbors(interval)}

    def add(self, *intervals: Interval) -> IntervalGraph:
        """
        Args:
            intervals (Interval): A given list of intervals
        Adds new intervals to the graph
        """

        intervals = sorted({i for i in intervals if i[0] <= i[1]})

        for i, interval in enumerate(intervals):
            starts = [i[0] for i in intervals]
            j = bisect_right(starts, interval[1])
            self.graph.add(interval, *set(intervals[i:j]))

        return self

    def remove(self, *intervals: Interval) -> IntervalGraph:
        """
        Args:
            intervals (Interval): A given list of intervals
        Removes given intervals from the graph
        """

        self.graph.remove(*intervals)

        return self

    def copy(self) -> IntervalGraph:
        """
        Returns:
            An identical copy of the graph
        """

        return type(self)(*self.nodes)

    def sort(self, end: bool = False) -> list[Interval]:
        """
        Args:
            end (bool): A boolean flag, indicating whether the intervals will be sorted, based on start or end point
        Returns:
            The sorted list of intervals
        """

        return sorted(self.nodes, key=lambda i: i[bool(end)])

    def weighted_nodes_graph(self) -> WeightedNodesIntervalGraph:
        return WeightedNodesIntervalGraph(*self.nodes)

    def weighted_links_graph(self) -> WeightedLinksIntervalGraph:
        return WeightedLinksIntervalGraph(*self.nodes)

    def weighted_graph(self) -> WeightedIntervalGraph:
        return WeightedIntervalGraph(*self.nodes)

    def chromatic_partition(self) -> list[set[Interval]]:
        """
        Returns:
            An optimal chromatic partition of the intervals
        """

        result = []

        for u in self.sort():
            for i, partition in enumerate(result):
                if self.neighbors(u).isdisjoint(partition):
                    result[i].add(u)

                    break
            else:
                result.append({u})

        return result

    def vertex_cover(self) -> set[Interval]:
        """
        Returns:
            An optimal vertex cover of the graph
        """

        return self.nodes - self.independent_set()

    def independent_set(self) -> set[Interval]:
        """
        Returns:
            An optimal independent set in the graph
        """

        result = set()

        for interval in self.sort(True):
            if self.neighbors(interval).isdisjoint(result):
                result.add(interval)

        return result

    def __contains__(self, item: Hashable) -> bool:
        """
        Args:
            item (Hashable): A given object
        Returns:
            Whether item is an interval in the graph
        """

        return item in self.nodes

    def __add__(self, other: IntervalGraph) -> IntervalGraph:
        """
        Args:
            other (IntervalGraph): Another interval graph
        Returns:
            The combination of both graphs
        """

        if isinstance(other, IntervalGraph):
            nodes = isinstance(self, WeightedNodesIntervalGraph) or isinstance(other, WeightedNodesIntervalGraph)
            links = isinstance(self, WeightedLinksIntervalGraph) or isinstance(other, WeightedLinksIntervalGraph)
            res_t = [IntervalGraph, WeightedNodesIntervalGraph, WeightedLinksIntervalGraph, WeightedUndirectedGraph][
                nodes + 2 * links]
            return res_t(*self.nodes.union(other.nodes))

        raise TypeError(f"Can't add type {type(other).__name__} to type IntervalGraph")

    def __eq__(self, other: IntervalGraph) -> bool:
        """
        Args:
            other (IntervalGraph): Another interval graph
        Returns:
            Whether the two interval graphs are equal
        """

        if type(self) != type(other):
            return False

        return self.nodes == other.nodes

    def __str__(self) -> str:
        """
        Returns:
            The string representation of the graph
        """

        return str(self.nodes)

    __repr__: str = __str__


class WeightedNodesIntervalGraph(IntervalGraph):
    """
    Class for implementing a set of intervals as a graph with node weights, where the weights are interval lengths
    """

    def __init__(self, *intervals: Interval) -> None:
        """
        Args:
            intervals (Interval): list of intervals
        Creates a WeightedNodesIntervalGraph object
        """

        super().__init__()
        self.__graph = WeightedNodesUndirectedGraph()
        self.add(*intervals)

    def node_weights(self, i: Interval = None) -> dict[Interval, float] | float:
        """
        Args:
            i: A present interval or None
        Returns:
            The length of interval i or the dictionary with all interval lengths
        """

        return self.graph.node_weights(i)

    def total_node_weights(self) -> float:
        """
        Returns:
            The combined length of all intervals
        """

        return self.graph.total_node_weights()

    def add(self, *intervals: Interval) -> WeightedNodesIntervalGraph:
        intervals = sorted({i for i in intervals if i[0] <= i[1]})

        for i, interval in enumerate(intervals):
            starts = [i[0] for i in intervals]
            j = bisect_right(starts, interval[1])
            self.graph.add((interval, interval[1] - interval[0]), *set(intervals[i:j]))

        return self

    def weighted_graph(self) -> WeightedIntervalGraph:
        return WeightedIntervalGraph(*self.nodes)


class WeightedLinksIntervalGraph(IntervalGraph):
    """
    Class for implementing a set of intervals as a graph with link weights, where the weights are lengths of interval
    intersections
    """

    def __init__(self, *intervals: Interval) -> None:
        """
        Args:
            intervals (Interval): list of intervals
        Creates a WeightedLinksIntervalGraph object
        """

        super().__init__()
        self.__graph = WeightedLinksUndirectedGraph()
        self.add(*intervals)

    def link_weights(self, u: Interval, v: Interval) -> float:
        """
        Args:
            u: Given first interval
            v: Given second interval
        Returns:
            Intervals intersection
        """

        return self.graph.link_weights(u, v)

    def total_link_weights(self) -> float:
        """
        Returns:
            The combined length of all intervals intersections
        """

        return self.graph.total_link_weights()

    def add(self, *intervals: Interval) -> WeightedLinksIntervalGraph:
        intervals = sorted({i for i in intervals if i[0] <= i[1]})

        for i, interval in enumerate(intervals):
            starts = [i[0] for i in intervals]
            j = bisect_right(starts, interval[1])
            self.graph.add(interval, {k: interval[1] - k[0] for k in intervals[i:j]})

        return self

    def weighted_graph(self) -> WeightedIntervalGraph:
        return WeightedIntervalGraph(*self.nodes)


class WeightedIntervalGraph(WeightedNodesIntervalGraph, WeightedLinksIntervalGraph):
    """
    Class for implementing a set of intervals as a graph with node and link weights, where the node weights are
    interval lengths and link weights are lengths of interval intersections
    """

    def __init__(self, *intervals: Interval) -> None:
        """
        Args:
            intervals (Interval): list of intervals
        Creates a WeightedIntervalGraph object
        """

        super().__init__()
        self.__graph = WeightedUndirectedGraph()
        self.add(*intervals)

    def total_weights(self) -> float:
        """
        Returns:
            The combined length of all intervals and all interval intersections
        """

        return self.graph.total_weights()

    def add(self, *intervals: Interval) -> WeightedIntervalGraph:
        intervals = sorted({i for i in intervals if i[0] <= i[1]})

        for i, interval in enumerate(intervals):
            starts = [i[0] for i in intervals]
            j = bisect_right(starts, interval[1])
            self.graph.add((interval, interval[1] - interval[0]), {k: interval[1] - k[0] for k in intervals[i:j]})

        return self
