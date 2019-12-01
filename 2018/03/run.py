"""
--- Day 3: No Matter How You Slice It ---

The Elves managed to locate the chimney-squeeze prototype fabric for Santa's
suit (thanks to someone who helpfully wrote its box IDs on the wall of the
warehouse in the middle of the night). Unfortunately, anomalies are still
affecting them - nobody can even agree on how to cut the fabric.

The whole piece of fabric they're working on is a very large square - at least
1000 inches on each side.

Each Elf has made a claim about which area of fabric would be ideal for Santa's
suit. All claims have an ID and consist of a single rectangle with edges
parallel to the edges of the fabric. Each claim's rectangle is defined as
follows:

    The number of inches between the left edge of the fabric and the left edge of the rectangle.
    The number of inches between the top edge of the fabric and the top edge of the rectangle.
    The width of the rectangle in inches.
    The height of the rectangle in inches.

A claim like #123 @ 3,2: 5x4 means that claim ID 123 specifies a rectangle
3 inches from the left edge, 2 inches from the top edge, 5 inches wide, and
4 inches tall. Visually, it claims the square inches of fabric represented by
# (and ignores the square inches of fabric represented by .) in the diagram
below:

...........
...........
...#####...
...#####...
...#####...
...#####...
...........
...........
...........

The problem is that many of the claims overlap, causing two or more claims to
cover part of the same areas. For example, consider the following claims:

#1 @ 1,3: 4x4
#2 @ 3,1: 4x4
#3 @ 5,5: 2x2

Visually, these claim the following areas:

........
...2222.
...2222.
.11XX22.
.11XX22.
.111133.
.111133.
........

The four square inches marked with X are claimed by both 1 and 2. (Claim 3,
while adjacent to the others, does not overlap either of them.)

If the Elves all proceed with their own plans, none of them will have enough
fabric. How many square inches of fabric are within two or more claims?

--- Part Two ---

Amidst the chaos, you notice that exactly one claim doesn't overlap by even
a single square inch of fabric with any other claim. If you can somehow draw
attention to it, maybe the Elves will be able to make Santa's suit after all!

For example, in the claims above, only claim 3 is intact after all claims are
made.

What is the ID of the only claim that doesn't overlap?
"""
from typing import Iterator, Tuple
import re
import numpy


def rectangles(
    claims: List[Tuple[str, str, str, str, str]],
) -> Iterator[Tuple[int, slice, slice]]:
    claims = numpy.array(claims, dtype=int)
    for index, x_coor, y_coor, width, hight in claims:
        x_slice = slice(x_coor, x_coor+width)
        y_slice = slice(y_coor, y_coor+hight)
        yield index, x_slice, y_slice


def create_fabric_grid(
    claims: List[Tuple[str, str, str, str, str]],
) -> numpy.ndarray:
    fabric_grid = numpy.zeros((1000, 1000), dtype=int)
    for _, *coordinates in rectangles(claims):
        fabric_grid[coordinates] += 1
    return fabric_grid


def part1(fabric_grid: numpy.ndarray) -> int:
    return numpy.sum(fabric_grid > 1)


def part2(
    claims: List[Tuple[str, str, str, str, str]],
    fabric_grid: numpy.ndarray,
) -> int:
    for index, *coordinates in rectangles(claims):
        if numpy.all(fabric_grid[coordinates] == 1):
            return index


if __name__ == "__main__":
    with open("input") as src:
        CLAIMS = re.findall(r"#(\d+) @ (\d+),(\d+): (\d+)x(\d+)", src.read())
    FABRIC_GRID = create_fabric_grid(CLAIMS)
    print("solution part 1:", part1(FABRIC_GRID))
    # solution part 1: 108961
    print("solution part 2:", part2(CLAIMS, FABRIC_GRID))
    # solution part 2: 681
