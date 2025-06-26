"""
Module for implementing a set of intervals as an undirected graph
"""

from __future__ import annotations

from base import Link, Hashable

from undirected_graph import UndirectedGraph, WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph, \
    WeightedUndirectedGraph

from bisect import bisect_right

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

    @property
    def links(self) -> set[Link]:
        """
        Returns:
            The set of links
        """

        return self.graph.links

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
            neighbors = set(intervals[i:j])
            self.graph.add(interval, *neighbors)

        return self

    def remove(self, *intervals: Interval) -> IntervalGraph:
        """
        Args:
            intervals (Interval): A given list of intervals
        Removes given intervals from the graph
        """

        self.graph.remove(*intervals)

        return self

    def sort(self, end: bool = False) -> list[Interval]:
        """
        Args:
            end (bool): A boolean flag, indicating whether the intervals will be sorted, based on start or end point
        Returns:
            The sorted list of intervals
        """

        return sorted(self.nodes, key=lambda i: i[bool(end)])

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
            return IntervalGraph(*self.nodes.union(other.nodes))

        raise TypeError(f"Can't add type {type(other).__name__} to type IntervalGraph")

    def __eq__(self, other: IntervalGraph) -> bool:
        """
        Args:
            other (IntervalGraph): Another interval graph
        Returns:
            Whether the two interval graphs are equal
        """

        if type(other) != IntervalGraph:
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

    pass


class WeightedLinksIntervalGraph(IntervalGraph):
    """
    Class for implementing a set of intervals as a graph with link weights, where the weights are lengths of interval
    intersections
    """

    pass


class WeightedIntervalGraph(WeightedNodesIntervalGraph, WeightedLinksIntervalGraph):
    """
    Class for implementing a set of intervals as a graph with node and link weights, where the node weights are
    interval lengths and link weights are lengths of interval intersections
    """

    pass
