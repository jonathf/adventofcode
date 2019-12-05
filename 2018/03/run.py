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

* The number of inches between the left edge of the fabric and the left edge of
  the rectangle.
* The number of inches between the top edge of the fabric and the top edge of
  the rectangle.
* The width of the rectangle in inches.
* The height of the rectangle in inches.

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
import pandas


def create_fabric_grid(claims: pandas.DataFrame) -> numpy.ndarray:
    """
    Construct grid indicating how many patches overlap a give grid point by
    index.

    Args:
        claims:
            Frame with the following columns:

            claim_id:
                Identifier for the specific claim.
            x_start:
                Index of the left edge.
            x_stop:
                Index of the right edge.
            y_start:
                Index of the top edge.
            y_stop:
                Index of the bottom edge.

    Returns:
        2-dimensional numpy array where indices represents coordinates, and
        values represents the number of patches overlap a position.

    Examples:
        >>> claims = pandas.DataFrame({"claim_id": [1, 2, 3],
        ...                            "x_start": [1, 3, 5],
        ...                            "x_stop": [5, 7, 7],
        ...                            "y_start": [3, 1, 5],
        ...                            "y_stop": [7, 5, 7]})
        >>> create_fabric_grid(claims)
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 1, 1, 1, 1],
               [0, 1, 1, 2, 2, 1, 1],
               [0, 1, 1, 2, 2, 1, 1],
               [0, 1, 1, 1, 1, 1, 1],
               [0, 1, 1, 1, 1, 1, 1]])

    """
    fabric_grid = numpy.zeros((claims.x_stop.max(), claims.y_stop.max()), dtype=int)
    for _, index in iterate_patches(claims):
        fabric_grid[index] += 1
    return fabric_grid


def iterate_patches(
    claims: pandas.DataFrame,
) -> Iterator[Tuple[int, Tuple[slice, slice]]]:
    """Iterate slices that covers each patch."""
    for _, claim in claims.iterrows():
        index = (slice(claim.x_start, claim.x_stop),
                 slice(claim.y_start, claim.y_stop))
        yield claim.claim_id, index


def part1(claims: pandas.DataFrame) -> int:
    """Do part 1 of the assignment."""
    fabric_grid = create_fabric_grid(claims)
    return numpy.sum(fabric_grid > 1)


def part2(claims: pandas.DataFrame) -> int:
    """Do part 2 of the assignment."""
    fabric_grid = create_fabric_grid(claims)
    for caim_id, index in iterate_patches(claims):
        if numpy.all(fabric_grid[index] == 1):
            return claim_id


if __name__ == "__main__":
    with open("input") as src:
        CLAIMS = re.findall(r"#(\d+) @ (\d+),(\d+): (\d+)x(\d+)", src.read())
    CLAIMS = pandas.DataFrame(
        numpy.array(CLAIMS, dtype=int),
        columns=["claim_id", "x_start", "y_start", "width", "height"],
    )
    CLAIMS["x_stop"] = CLAIMS.x_start + CLAIMS.width
    CLAIMS["y_stop"] = CLAIMS.y_start + CLAIMS.height

    print("solution part 1:", part1(CLAIMS))
    # solution part 1: 108961
    print("solution part 2:", part2(CLAIMS))
    # solution part 2: 681
