"""
--- Day 6: Chronal Coordinates ---

The device on your wrist beeps several times, and once again you feel like
you're falling.

"Situation critical," the device announces. "Destination indeterminate. Chronal
interference detected. Please specify new target coordinates."

The device then produces a list of coordinates (your puzzle input). Are they
places it thinks are safe or dangerous? It recommends you check manual page
729. The Elves did not give you a manual.

If they're dangerous, maybe you can minimize the danger by finding the
coordinate that gives the largest distance from the other points.

Using only the Manhattan distance, determine the area around each coordinate by
counting the number of integer X,Y locations that are closest to that
coordinate (and aren't tied in distance to any other coordinate).

Your goal is to find the size of the largest area that isn't infinite. For
example, consider the following list of coordinates:

1, 1
1, 6
8, 3
3, 4
5, 5
8, 9

If we name these coordinates A through F, we can draw them on a grid, putting
0,0 at the top left:

..........
.A........
..........
........C.
...D......
.....E....
.B........
..........
..........
........F.

This view is partial - the actual grid extends infinitely in all directions.
Using the Manhattan distance, each location's closest coordinate can be
determined, shown here in lowercase:

aaaaa.cccc
aAaaa.cccc
aaaddecccc
aadddeccCc
..dDdeeccc
bb.deEeecc
bBb.eeee..
bbb.eeefff
bbb.eeffff
bbb.ffffFf

Locations shown as . are equally far from two or more coordinates, and so they
don't count as being closest to any.

In this example, the areas of coordinates A, B, C, and F are infinite - while
not shown here, their areas extend forever outside the visible grid. However,
the areas of coordinates D and E are finite: D is closest to 9 locations, and
E is closest to 17 (both including the coordinate's location itself).
Therefore, in this example, the size of the largest area is 17.

What is the size of the largest area that isn't infinite?

--- Part Two ---

On the other hand, if the coordinates are safe, maybe the best you can do is
try to find a region near as many coordinates as possible.

For example, suppose you want the sum of the Manhattan distance to all of the
coordinates to be less than 32. For each location, add up the distances to all
of the given coordinates; if the total of those distances is less than 32, that
location is within the desired region. Using the same coordinates as above, the
resulting region looks like this:

..........
.A........
..........
...###..C.
..#D###...
..###E#...
.B.###....
..........
..........
........F.

In particular, consider the highlighted location 4,3 located at the top middle
of the region. Its calculation is as follows, where abs() is the absolute value
function:

    - Distance to coordinate A: abs(4-1) + abs(3-1) =  5
    - Distance to coordinate B: abs(4-1) + abs(3-6) =  6
    - Distance to coordinate C: abs(4-8) + abs(3-3) =  4
    - Distance to coordinate D: abs(4-3) + abs(3-4) =  2
    - Distance to coordinate E: abs(4-5) + abs(3-5) =  3
    - Distance to coordinate F: abs(4-8) + abs(3-9) = 10
    - Total distance: 5 + 6 + 4 + 2 + 3 + 10 = 30

Because the total distance to all coordinates (30) is less than 32, the
location is within the region.

This region, which also includes coordinates D and E, has a total size of 16.

Your actual region will need to be much larger than this example, though,
instead including all locations with a total distance of less than 10000.

What is the size of the region containing all locations which have a total
distance to all given coordinates of less than 10000?
"""
from itertools import product
import re
import numpy


def construct_distance_map(coordinates: numpy.ndarray) -> numpy.ndarray:
    """
    Construct map with shortest distance using the Manhattan distance to set of
    `coordinates`.

    Note that the map is the minimum enclosing map that captures all locations
    with a single extra edge beyond that.

    Args:
        coordinates:
            Coordinates that group shorted distance to a set of indices. Has
            ``shape == (K, 2)`` where ``K`` is the number of coordinates.

    Returns:
        A 2-dimensional grid where values are grouped by the coordinate they
        are closes to. In other words, all values that are the same, is closet
        to the same coordinate. If multiple coordinates share shortest
        distance, the value -1 is used.

    Examples:
        >>> coordinates = [(1, 1), (1, 6), (8, 3), (3, 4), (5, 5), (8, 9)]
        >>> print(construct_distance_map(coordinates).T)
        [[ 0  0  0  0  0 -1  5  5  5]
         [ 0  0  0  0  0 -1  5  5  5]
         [ 0  0  0  2  2  3  5  5  5]
         [ 0  0  2  2  2  3  5  5  5]
         [-1 -1  2  2  2  3  3  5  5]
         [ 1  1 -1  2  3  3  3  3  5]
         [ 1  1  1 -1  3  3  3  3 -1]
         [ 1  1  1 -1  3  3  3  4  4]
         [ 1  1  1 -1  3  3  4  4  4]
         [ 1  1  1 -1  4  4  4  4  4]]
    """
    # adjust coordinates to first quadrant to simplify grid
    coordinates = numpy.asarray(coordinates, dtype=int)
    east, north = numpy.max(coordinates, axis=0)
    west, south = numpy.min(coordinates, axis=0)
    distance_map = -numpy.ones((east-west+2, north-south+2), dtype=int)
    coordinates -= west-1, south-1

    counts = {}
    for coord in product(range(distance_map.shape[0]),
                         range(distance_map.shape[1])):
        distances = (numpy.abs(coord[0]-coordinates[:, 0])+
                     numpy.abs(coord[1]-coordinates[:, 1]))
        index = numpy.argmin(distances)
        if index not in counts:
            counts[index] = len(counts)
        if numpy.sum(distances == distances[index]) == 1:
            distance_map[coord] = counts[index]
    return distance_map


def part1(distance_map: numpy.ndarray) -> int:
    """Do part 1 of the assignment."""
    no_go_zone = [distance_map[0], distance_map[-1],
                  distance_map[:, 0], distance_map[:, -1], [-1]]
    no_go_zone = numpy.unique(numpy.concatenate(no_go_zone)).tolist()
    return max((element == distance_map).sum()
               for element in range(distance_map.max())
               if element not in no_go_zone)


def part2(coordinates: numpy.ndarray, distance_map: numpy.ndarray) -> int:
    """Do part 2 of the assignment."""
    count = 0
    for x, y in product(range(distance_map.shape[0]), range(distance_map.shape[1])):
        distance = numpy.abs(x-coordinates[0]) + numpy.abs(y-coordinates[1])
        count += numpy.sum(distance) < 10000
    return count


if __name__ == "__main__":
    with open("input") as src:
        COORDINATES = numpy.array([re.findall("\d+", src.read())], dtype=int)
        COORDINATES = COORDINATES.reshape(-1, 2)
    DISTANCE_MAP = construct_distance_map(COORDINATES)
    print("solution part 1:", part1(DISTANCE_MAP))
    # solution part 1: 3358
    print("solution part 2:", part2(COORDINATES, DISTANCE_MAP))
    # solution part 2: 97335
