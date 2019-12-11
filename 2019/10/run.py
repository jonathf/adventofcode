"""
--- Day 10: Monitoring Station ---

You fly into the asteroid belt and reach the Ceres monitoring station. The
Elves here have an emergency: they're having trouble tracking all of the
asteroids and can't be sure they're safe.

The Elves would like to build a new monitoring station in a nearby area of
space; they hand you a map of all of the asteroids in that region (your puzzle
input).

The map indicates whether each position is empty (.) or contains an asteroid
(#). The asteroids are much smaller than they appear on the map, and every
asteroid is exactly in the center of its marked position. The asteroids can be
described with X,Y coordinates where X is the distance from the left edge and
Y is the distance from the top edge (so the top-left corner is 0,0 and the
position immediately to its right is 1,0).

Your job is to figure out which asteroid would be the best place to build a new
monitoring station. A monitoring station can detect any asteroid to which it
has direct line of sight - that is, there cannot be another asteroid exactly
between them. This line of sight can be at any angle, not just lines aligned to
the grid or diagonally. The best location is the asteroid that can detect the
largest number of other asteroids.

For example, consider the following map:

.#..#
.....
#####
....#
...##

The best location for a new monitoring station on this map is the highlighted
asteroid at 3,4 because it can detect 8 asteroids, more than any other
location. (The only asteroid it cannot detect is the one at 1,0; its view of
this asteroid is blocked by the asteroid at 2,2.) All other asteroids are worse
locations; they can detect 7 or fewer other asteroids. Here is the number of
other asteroids a monitoring station on each asteroid could detect:

.7..7
.....
67775
....7
...87

Here is an asteroid (#) and some examples of the ways its line of sight might
be blocked. If there were another asteroid at the location of a capital letter,
the locations marked with the corresponding lowercase letter would be blocked
and could not be detected:

#.........
...A......
...B..a...
.EDCG....a
..F.c.b...
.....c....
..efd.c.gb
.......c..
....f...c.
...e..d..c

Here are some larger examples:

    Best is 5,8 with 33 other asteroids detected:

    ......#.#.
    #..#.#....
    ..#######.
    .#.#.###..
    .#..#.....
    ..#....#.#
    #..#....#.
    .##.#..###
    ##...#..#.
    .#....####

    Best is 1,2 with 35 other asteroids detected:

    #.#...#.#.
    .###....#.
    .#....#...
    ##.#.#.#.#
    ....#.#.#.
    .##..###.#
    ..#...##..
    ..##....##
    ......#...
    .####.###.

    Best is 6,3 with 41 other asteroids detected:

    .#..#..###
    ####.###.#
    ....###.#.
    ..###.##.#
    ##.##.#.#.
    ....###..#
    ..#.#..#.#
    #..#.#.###
    .##...##.#
    .....#.#..

    Best is 11,13 with 210 other asteroids detected:

    .#..##.###...#######
    ##.############..##.
    .#.######.########.#
    .###.#######.####.#.
    #####.##.#.##.###.##
    ..#####..#.#########
    ####################
    #.####....###.#.#.##
    ##.#################
    #####.##.###..####..
    ..######..##.#######
    ####.##.####...##..#
    .#####..#.######.###
    ##...#.##########...
    #.##########.#######
    .####.#.###.###.#.##
    ....##.##.###..#####
    .#.#.###########.###
    #.#.#.#####.####.###
    ###.##.####.##.#..##

Find the best location for a new monitoring station. How many other asteroids
can be detected from that location?

--- Part Two ---

Once you give them the coordinates, the Elves quickly deploy an Instant
Monitoring Station to the location and discover the worst: there are simply too
many asteroids.

The only solution is complete vaporization by giant laser.

Fortunately, in addition to an asteroid scanner, the new monitoring station
also comes equipped with a giant rotating laser perfect for vaporizing
asteroids. The laser starts by pointing up and always rotates clockwise,
vaporizing any asteroid it hits.

If multiple asteroids are exactly in line with the station, the laser only has
enough power to vaporize one of them before continuing its rotation. In other
words, the same asteroids that can be detected can be vaporized, but if
vaporizing one asteroid makes another one detectable, the newly-detected
asteroid won't be vaporized until the laser has returned to the same position
by rotating a full 360 degrees.

For example, consider the following map, where the asteroid with the new
monitoring station (and laser) is marked X:

.#....#####...#..
##...##.#####..##
##...#...#.#####.
..#.....X...###..
..#.#.....#....##

The first nine asteroids to get vaporized, in order, would be:

.#....###24...#..
##...##.13#67..9#
##...#...5.8####.
..#.....X...###..
..#.#.....#....##

Note that some asteroids (the ones behind the asteroids marked 1, 5, and 7)
won't have a chance to be vaporized until the next full rotation. The laser
continues rotating; the next nine to be vaporized are:

.#....###.....#..
##...##...#.....#
##...#......1234.
..#.....X...5##..
..#.9.....8....76

The next nine to be vaporized are then:

.8....###.....#..
56...9#...#.....#
34...7...........
..2.....X....##..
..1..............

Finally, the laser completes its first full rotation (1 through 3), a second
rotation (4 through 8), and vaporizes the last asteroid (9) partway through its
third rotation:

......234.....6..
......1...5.....7
.................
........X....89..
.................

In the large example above (the one with the best monitoring station location
at 11,13):

    - The 1st asteroid to be vaporized is at 11,12.
    - The 2nd asteroid to be vaporized is at 12,1.
    - The 3rd asteroid to be vaporized is at 12,2.
    - The 10th asteroid to be vaporized is at 12,8.
    - The 20th asteroid to be vaporized is at 16,0.
    - The 50th asteroid to be vaporized is at 16,9.
    - The 100th asteroid to be vaporized is at 10,16.
    - The 199th asteroid to be vaporized is at 9,6.
    - The 200th asteroid to be vaporized is at 8,2.
    - The 201st asteroid to be vaporized is at 10,9.
    - The 299th and final asteroid to be vaporized is at 11,1.

The Elves are placing bets on which will be the 200th asteroid to be vaporized.
Win the bet by determining which asteroid that will be; what do you get if you
multiply its X coordinate by 100 and then add its Y coordinate? (For example,
8,2 becomes 802.)
"""
from typing import Any, DefaultDict, Dict, NamedTuple, Sequence, Set, Tuple
from itertools import cycle, product
from collections import defaultdict
from math import gcd, sqrt, atan, pi

import numpy


class Astroid(NamedTuple):
    coord: Tuple[int, int]
    distance: float
    angle: float


Station = Dict[Tuple[int, int], Set[Astroid]]
Stations = Dict[Tuple[int, int], DefaultDict[Tuple[int, int], Set[Station]]]


def get_angle(denominator: int, numerator: int) -> float:
    """
    Args:
        denominator:
            Position along the y-axis.
        numerator:
            Position along the x-axis.

    Returns:
        The angle from the `(0, -1)` unit vector and counter clockwise.

    Examples:
        >>> get_angle(0, -2)
        6.283185307179586
        >>> get_angle(2, -2)
        5.497787143782138
        >>> get_angle(2, 0)
        4.71238898038469
        >>> get_angle(2, 2)
        3.9269908169872414
        >>> get_angle(0, 2)
        3.141592653589793
        >>> get_angle(-2, 2)
        2.356194490192345
        >>> get_angle(-2, 0)
        1.5707963267948966
        >>> get_angle(-2, -2)
        0.7853981633974483
    """
    if numerator == 0:
        return 3*pi/2 if denominator > 0 else pi/2
    out = atan(denominator/numerator)-pi
    if numerator < 0:
        out += pi
    elif denominator <= 0:
        out += 2*pi
    if out <= 0:
        out += 2*pi
    return out


def plot_astroid_field(coords: Sequence[Tuple[int, int]]) -> str:
    r"""
    Create ASCII plot out of coordinates.

    Args:
        coords:
            Coordinate to visualize.

    Returns:
        String representation of grid with `coords` position illustrated with
        '#' symbols. The first 10 symbols have index printed instead.

    Examples:
        >>> coords = [(1, 0), (4, 0), (0, 2), (1, 2), (2, 2),
        ...           (3, 2), (4, 2), (4, 3), (3, 4), (4, 4)]
        >>> print(plot_astroid_field(coords))
        .0..1
        .....
        23456
        ....7
        ...89

    """
    maxval = int(max(x for coord in coords for x in coord))
    data = numpy.zeros((maxval+1, maxval+1), dtype="U1")
    data[:] = "."
    for idx, coord in enumerate(coords):
        if idx < 10:
            data[coord] = str(idx)
        else:
            data[coord] = "#"
    return "\n".join("".join(line) for line in data.T)


def create_outer_field(data: str) -> Stations:
    r"""
    Construct dictionary representation of asteroid field.

    Args:
        Raw data as ASCII picture of asteroid field.

    Returns:
        Dictionary where keys are coordinate to each asteroid. Values are empty 
        default dictionaries with set defaults.

    Examples:
        >>> data = ".#..#\n.....\n#####\n....#\n...##"
        >>> construct_astroid_field(data)  # doctest: +NORMALIZE_WHITESPACE
        {(1, 0): defaultdict(<class 'set'>, {}), (4, 0): defaultdict(<class 'set'>, {}),
         (0, 2): defaultdict(<class 'set'>, {}), (1, 2): defaultdict(<class 'set'>, {}),
         (2, 2): defaultdict(<class 'set'>, {}), (3, 2): defaultdict(<class 'set'>, {}),
         (4, 2): defaultdict(<class 'set'>, {}), (4, 3): defaultdict(<class 'set'>, {}),
         (3, 4): defaultdict(<class 'set'>, {}), (4, 4): defaultdict(<class 'set'>, {})}
    """
    lines = [list(line) for line in data.strip().split("\n")]
    field = {
        (idx, idy): defaultdict(set)
        for idy, line in enumerate(lines)
        for idx, char in enumerate(line)
        if char == "#"
    }
    return field


def construct_astroid_field(data: str) -> Stations:
    stations = create_outer_field(data)
    maxval = max(val for coord in stations for val in coord)

    for a, b in product(range(maxval), range(maxval)):

        if not a and not b:
            continue

        common = gcd(a, b)
        for (idx, idy), station in stations.items():
            for signx, signy in product((-1, 1), (-1, 1)):

                reference = (a*signx//common, b*signy//common)
                coord = (idx+a*signx, idy+b*signy)
                if coord in stations and coord not in [x[0] for x in stations[coord][reference]]:
                    distance = sqrt(a**2+b**2)
                    angle = get_angle(a*signx, b*signy)
                    station[reference].add(Astroid(coord, distance, angle))

    # shed defaultdict and sort by distance and angle
    stations = {coord: {ref: sorted(val, key=lambda x: x[1]+x[2]*1000)
                        for ref, val in stations[coord].items() if val}
                for coord in stations}
    return stations


def part1(stations: Stations) -> int:
    """Do part 2 of the assignment."""
    return max(len(value) for value in stations.values())


def part2(stations: Stations) -> int:
    """Do part 2 of the assignment."""
    station = stations[sorted(stations, key=lambda x: len(stations[x]))[-1]]
    idx = 0
    while station:
        keys = reversed(sorted(station, key=lambda x: station[x][0][2]))
        for key in keys:
            idx += 1
            coord = station[key].pop(0).coord
            if idx == 200:
                return coord[0]*100+coord[1]
            if not station[key]:
                del station[key]


if __name__ == "__main__":
    with open("input") as src:
        data = src.read()
    stations = construct_astroid_field(data)
    print(part1(stations))
    # solution part 1: 274
    print(part2(stations))
    # solution part 2: 305
