"""
Module for implementing sets of intervals as undirected graphs
"""

from __future__ import annotations

from functools import total_ordering

from base import Hashable

from undirected_graph import UndirectedGraph, WeightedNodesUndirectedGraph, WeightedLinksUndirectedGraph, \
    WeightedUndirectedGraph

__all__ = ["Interval", "IntervalGraph", "WeightedNodesIntervalGraph", "WeightedLinksIntervalGraph",
           "WeightedIntervalGraph"]


@total_ordering
class Interval:
    """
    A class for creating an interval on the real number line
    """

    def __init__(self, a: float, b: float, left_closed: bool = True, right_closed: bool = True) -> None:
        """
        Creates an Interval object
        """

        a, b = float(a), float(b)
        self.__a, self.__b = min(a, b), max(a, b)
        self.__left_closed, self.__right_closed = bool(left_closed), bool(right_closed)

    @property
    def a(self) -> float:
        """
        Returns:
            Interval beginning
        """

        return self.__a

    @property
    def b(self) -> float:
        """
        Returns:
            Interval ending
        """

        return self.__b

    @property
    def length(self) -> float:
        return self.b - self.a

    @property
    def left_closed(self) -> bool:
        """
        Returns:
            Whether the beginning point of the interval is in it
        """

        return self.__left_closed

    @property
    def right_closed(self) -> bool:
        """
        Returns:
            Whether the ending point of the interval is in it
        """

        return self.__right_closed

    def copy(self) -> Interval:
        """
        Returns:
            An identical copy of the Interval object
        """

        return Interval(self.a, self.b, self.left_closed, self.right_closed)

    def intersects(self, other: Interval) -> bool:
        """
        Returns:
            Whether the interval intersects with the other interval
        """

        return bool(self * other)

    def issubset(self, other: Interval) -> bool:
        """
        Args:
            other (Interval): Another Interval object
        Returns:
            Whether the interval is a subset of the other interval
        """

        return (self.a > other.a or self.a == other.a and (not self.left_closed or other.right_closed)) and (
                self.b < other.b or self.b == other.b and (not self.right_closed or other.left_closed))

    def __contains__(self, item: float) -> bool:
        """
        Returns:
            Whether the given number is in the interval
        """

        return self.left_closed and item == self.a or self.a < item < self.b or self.right_closed and item == self.b

    def __hash__(self) -> int:
        """
        Returns:
            The hash value of the Interval object
        """

        return hash((self.a, self.left_closed, self.b, self.right_closed))

    def __bool__(self) -> bool:
        """
        Returns:
            Whether the interval isn't the empty set
        """

        return self.a < self.b or self.left_closed and self.right_closed

    def __add__(self, other: Interval) -> Interval:
        """
        Args:
            other (Interval): Another Interval object
        Returns:
            The union of the two intervals if it's also an interval, otherwise raises a ValueError
        """

        if self.intersects(other):
            return Interval(min(self.a, other.a), max(self.b, other.b), self.left_closed or other.left_closed,
                            self.right_closed or other.right_closed)

        raise ValueError("Resulting set is not an interval")

    def __sub__(self, other: Interval) -> Interval:
        """
        Args:
            other (Interval): Another Interval object
        Returns:
            The set of points on the real number line belonging to self but not to other, if it's an interval, otherwise raises a ValueError
        """

        if not self.intersects(other):
            return self.copy()

        if self.b == other.a:
            return Interval(self.a, self.b, self.left_closed, self.right_closed and not other.left_closed)

        if self.a == other.b:
            return Interval(self.a, self.b, self.left_closed and not other.right_closed, self.right_closed)

        if self.issubset(other):
            return Interval(self.a, self.a, False, False)

        if self.a < other.a or self.a == other.a and (not self.left_closed or other.left_closed):
            if self.b < other.b or self.b == other.b and other.right_closed or not self.right_closed:
                return Interval(self.a, other.a, self.left_closed, not other.left_closed)

            raise ValueError("Resulting set is not an interval")

        if self.a > other.a or self.a == other.a and (not self.left_closed or other.left_closed):
            if self.b > other.b or self.b == other.b and other.right_closed or not self.right_closed:
                return Interval(other.b, self.b, not other.right_closed, self.right_closed)

            raise ValueError("Resulting set is not an interval")

        raise ValueError("Resulting set is not an interval")

    def __mul__(self, other: Interval) -> Interval:
        """
        Args:
            other (Interval): Another Interval object
        Returns:
            The intersection interval of the intervals
        """

        res_a, res_b = max(self.a, other.a), min(self.b, other.b)

        if res_a > res_b:
            return Interval(res_a, res_a, False, False)

        res_left = self.left_closed and other.left_closed if self.a == other.a else (
            self.left_closed if self.a > other.a else other.left_closed)
        res_right = self.right_closed and other.right_closed if self.b == other.b else (
            self.right_closed if self.b < other.b else other.right_closed)

        return Interval(res_a, res_b, res_left, res_right)

    def __lt__(self, other: Interval) -> bool:
        """
        Args:
            other (Interval): Another Interval object
        Returns:
            self < other
        """

        return (self.a, not self.left_closed, self.b, not self.right_closed) < (other.a, not other.left_closed,
                                                                                other.b, not other.right_closed)

    def __eq__(self, other: Interval) -> bool:
        """
        Args:
            other (Interval): Another Interval object
        Returns:
            self == other
        """

        if type(other) is not Interval:
            return False

        return (self.a, self.left_closed, self.b, self.right_closed) == (other.a, other.left_closed, other.b,
                                                                         other.right_closed) or not (self or other)

    def __str__(self) -> str:
        """
        Returns:
            String representation of the interval
        """

        left, right = "(["[self.left_closed], ")]"[self.right_closed]
        return f"{left}{self.a}, {self.b}{right}"

    def __repr__(self) -> str:
        """
        Returns:
            repr(self)
        """

        return f"Interval{(self.a, self.b, self.left_closed, self.right_closed)}"


def find_last(iv: Interval, intervals: list[Interval], i=0, j=-1) -> int:
    if j == -1:
        j = len(intervals)

    if i >= j:
        return i - 1

    if j - i == 1:
        return i + iv.intersects(intervals[i]) - 1

    index = (i + j) // 2

    if iv.intersects(intervals[index]) and (index == j - 1 or not iv.intersects(intervals[index + 1])):
        return index

    return find_last(iv, intervals, index + 1, j) if iv.intersects(intervals[index]) else find_last(iv, intervals, i,
                                                                                                    index - 1)


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

        intervals = sorted(set(intervals) - self.nodes)

        for iv in intervals:
            self.graph.add(iv)

        all_intervals = sorted(self.nodes.union(intervals))

        for i, interval in enumerate(all_intervals):
            if interval in intervals:
                j = find_last(interval, all_intervals, i + 1) + i + 1

                if neighbors := all_intervals[i + 1:j + 1]:
                    self.graph.connect(interval, *neighbors)

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
        """
        Returns:
            The version of the object with node weights
        """

        return WeightedNodesIntervalGraph(*self.nodes)

    def weighted_links_graph(self) -> WeightedLinksIntervalGraph:
        """
        Returns:
            The version of the object with link weights
        """

        return WeightedLinksIntervalGraph(*self.nodes)

    def weighted_graph(self) -> WeightedIntervalGraph:
        """
        Returns:
            The version of the object with node and link weights
        """

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
            res_t = [[IntervalGraph, WeightedNodesIntervalGraph], [WeightedLinksIntervalGraph, WeightedIntervalGraph]][
                links][nodes]
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

        if i is None:
            return {k.value: v for k, v in self.graph.node_weights()}

        return self.graph.node_weights(i)

    def total_node_weights(self) -> float:
        """
        Returns:
            The combined length of all intervals
        """

        return self.graph.total_node_weights()

    def add(self, *intervals: Interval) -> WeightedNodesIntervalGraph:
        intervals = sorted(set(intervals) - self.nodes)

        for iv in intervals:
            self.graph.add((iv, iv.length))

        all_intervals = sorted(self.nodes.union(intervals))

        for i, interval in enumerate(all_intervals):
            if interval in intervals:
                j = find_last(interval, all_intervals, i + 1) + i + 1

                if neighbors := all_intervals[i + 1:j + 1]:
                    self.graph.connect(interval, *neighbors)

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
        intervals = sorted(set(intervals) - self.nodes)

        for iv in intervals:
            self.graph.add(iv)

        all_intervals = sorted(self.nodes.union(intervals))

        for i, interval in enumerate(all_intervals):
            if interval in intervals:
                j = find_last(interval, all_intervals, i + 1) + i + 1
                self.graph.connect(interval, {iv: iv.length for iv in all_intervals[i + 1:j + 1]})

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
        intervals = sorted(set(intervals) - self.nodes)

        for iv in intervals:
            self.graph.add((iv, iv.length))

        all_intervals = sorted(self.nodes.union(intervals))

        for i, interval in enumerate(all_intervals):
            if interval in intervals:
                j = find_last(interval, all_intervals, i + 1) + i + 1
                self.graph.connect(interval, {iv: iv.length for iv in all_intervals[i + 1:j + 1]})

        return self

