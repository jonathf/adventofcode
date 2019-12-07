"""
--- Day 10: The Stars Align ---

It's no use; your navigation system simply isn't capable of providing walking
directions in the arctic circle, and certainly not in 1018.

The Elves suggest an alternative. In times like these, North Pole rescue
operations will arrange points of light in the sky to guide missing Elves back
to base. Unfortunately, the message is easy to miss: the points move slowly
enough that it takes hours to align them, but have so much momentum that they
only stay aligned for a second. If you blink at the wrong time, it might be
hours before another message appears.

You can see these points of light floating in the distance, and record their
position in the sky and their velocity, the relative change in position per
second (your puzzle input). The coordinates are all given from your
perspective; given enough time, those positions and velocities will move the
points into a cohesive message!

Rather than wait, you decide to fast-forward the process and calculate what the
points will eventually spell.

For example, suppose you note the following points:

position=< 9,  1> velocity=< 0,  2>
position=< 7,  0> velocity=<-1,  0>
position=< 3, -2> velocity=<-1,  1>
position=< 6, 10> velocity=<-2, -1>
position=< 2, -4> velocity=< 2,  2>
position=<-6, 10> velocity=< 2, -2>
position=< 1,  8> velocity=< 1, -1>
position=< 1,  7> velocity=< 1,  0>
position=<-3, 11> velocity=< 1, -2>
position=< 7,  6> velocity=<-1, -1>
position=<-2,  3> velocity=< 1,  0>
position=<-4,  3> velocity=< 2,  0>
position=<10, -3> velocity=<-1,  1>
position=< 5, 11> velocity=< 1, -2>
position=< 4,  7> velocity=< 0, -1>
position=< 8, -2> velocity=< 0,  1>
position=<15,  0> velocity=<-2,  0>
position=< 1,  6> velocity=< 1,  0>
position=< 8,  9> velocity=< 0, -1>
position=< 3,  3> velocity=<-1,  1>
position=< 0,  5> velocity=< 0, -1>
position=<-2,  2> velocity=< 2,  0>
position=< 5, -2> velocity=< 1,  2>
position=< 1,  4> velocity=< 2,  1>
position=<-2,  7> velocity=< 2, -2>
position=< 3,  6> velocity=<-1, -1>
position=< 5,  0> velocity=< 1,  0>
position=<-6,  0> velocity=< 2,  0>
position=< 5,  9> velocity=< 1, -2>
position=<14,  7> velocity=<-2,  0>
position=<-3,  6> velocity=< 2, -1>

Each line represents one point. Positions are given as <X, Y> pairs:
X represents how far left (negative) or right (positive) the point appears,
while Y represents how far up (negative) or down (positive) the point appears.

At 0 seconds, each point has the position given. Each second, each point's
velocity is added to its position. So, a point with velocity <1, -2> is moving
to the right, but is moving upward twice as quickly. If this point's initial
position were <3, 9>, after 3 seconds, its position would become <6, 3>.

Over time, the points listed above would move like this:

Initially:
........#.............
................#.....
.........#.#..#.......
......................
#..........#.#.......#
...............#......
....#.................
..#.#....#............
.......#..............
......#...............
...#...#.#...#........
....#..#..#.........#.
.......#..............
...........#..#.......
#...........#.........
...#.......#..........

After 1 second:
......................
......................
..........#....#......
........#.....#.......
..#.........#......#..
......................
......#...............
....##.........#......
......#.#.............
.....##.##..#.........
........#.#...........
........#...#.....#...
..#...........#.......
....#.....#.#.........
......................
......................

After 2 seconds:
......................
......................
......................
..............#.......
....#..#...####..#....
......................
........#....#........
......#.#.............
.......#...#..........
.......#..#..#.#......
....#....#.#..........
.....#...#...##.#.....
........#.............
......................
......................
......................

After 3 seconds:
......................
......................
......................
......................
......#...#..###......
......#...#...#.......
......#...#...#.......
......#####...#.......
......#...#...#.......
......#...#...#.......
......#...#...#.......
......#...#..###......
......................
......................
......................
......................

After 4 seconds:
......................
......................
......................
............#.........
........##...#.#......
......#.....#..#......
.....#..##.##.#.......
.......##.#....#......
...........#....#.....
..............#.......
....#......#...#......
.....#.....##.........
...............#......
...............#......
......................
......................

After 3 seconds, the message appeared briefly: HI. Of course, your message will
be much longer and will take many more seconds to appear.

What message will eventually appear in the sky?

--- Part Two ---

Good thing you didn't have to wait, because that would have taken a long time
- much longer than the 3 seconds in the example above.

Impressed by your sub-hour communication capabilities, the Elves are curious:
exactly how many seconds would they have needed to wait for that message to
appear?
"""
import time
import re
import numpy


def estimate_seconds_used(
        x_coor: numpy.ndarray,
        y_coor: numpy.ndarray,
        x_velocity: numpy.ndarray,
        y_velocity: numpy.ndarray,
) -> int:
    """
    Estimate the number of seconds used using least squares regression.

    Along each axis, when the message is displayed, the samples are likely to
    be as close to each other as possible, and there exist a linear equation
    ``current_position + velocity*seconds = final_position`` that maps the
    samples from this location for all samples. Or reformulated into a least
    square regression problem::

        final_position - velocity*seconds = current_position
        [ones, -velocity] @ [final_position, seconds] = current_position

    Or, more classically:

        A @ x = b

    This we can solve for x.

    This is true both along the x-axis and y-axis, so it is easy to verify that
    the estimate holds.

    Note:
        In practice, the assumption might not hold entirely. It might be a step
        or two off in practice, simply by the unprecises nature of the problem
        formulation.

    Args:
        x_coor:
            The current coordinates along the x-axis.
        y_coor:
            The current coordinates along the y-axis.
        x_velocity:
            The velocities along the x-axis.
        y_velocity:
            The velocities along the y-axis.

    Return:
        The estimated number of seconds until the are right next to each other.

    Examples:
        >>> x_coor = numpy.array([9, 7, 3, 6, 2, -6, 1, 1, -3, 7, -2, -4, 10,
        ...     5, 4, 8, 15, 1, 8, 3, 0, -2, 5, 1, -2, 3, 5, -6, 5, 14, -3])
        >>> y_coor = numpy.array([1, 0, -2, 10, -4, 10, 8, 7, 11, 6, 3, 3, -3,
        ...     11, 7, -2, 0, 6, 9, 3, 5, 2, -2, 4, 7, 6, 0, 0, 9, 7, 6])
        >>> x_velocity = numpy.array([0, -1, -1, -2, 2, 2, 1, 1, 1, -1, 1, 2,
        ...     -1, 1, 0, 0, -2, 1, 0, -1, 0, 2, 1, 2, 2, -1, 1, 2, 1, -2, 2])
        >>> y_velocity = numpy.array([2, 0, 1, -1, 2, -2, -1, 0, -2, -1, 0, 0,
        ...     1, -2, -1, 1, 0, 0, -1, 1, -1, 0, 2, 1, -2, -1, 0, 0, -2, 0, -1])
        >>> estimate_seconds_used(x_coor, y_coor, x_velocity, y_velocity)
        3

    """
    x_velocity_stack = numpy.vstack([numpy.ones(len(x_velocity)), -x_velocity]).T
    _, x_seconds = numpy.linalg.lstsq(x_velocity_stack, x_coor)[0]
    y_velocity_stack = numpy.vstack([numpy.ones(len(y_velocity)), -y_velocity]).T
    _, y_seconds = numpy.linalg.lstsq(y_velocity_stack, y_coor)[0]
    assert int(x_seconds) == int(y_seconds)
    return int(x_seconds)


def plot_coordinates(x_coor: numpy.ndarray, y_coor: numpy.ndarray) -> str:
    """
    Plot coordinate as ASCII graphics.

    Args:
        x_coor:
            The coordinates along the x-axis.
        y_coor:
            The coordinates along the y-axis.

    Returns:
        String representation of the sky and the objects on it represented with
        '#' symbols.

    Examples:
        >>> x_coor = numpy.array([9, 7, 3, 6, 2, -6, 1, 1, -3, 7, -2, -4, 10,
        ...     5, 4, 8, 15, 1, 8, 3, 0, -2, 5, 1, -2, 3, 5, -6, 5, 14, -3])
        >>> y_coor = numpy.array([1, 0, -2, 10, -4, 10, 8, 7, 11, 6, 3, 3, -3,
        ...     11, 7, -2, 0, 6, 9, 3, 5, 2, -2, 4, 7, 6, 0, 0, 9, 7, 6])
        >>> x_velocity = numpy.array([0, -1, -1, -2, 2, 2, 1, 1, 1, -1, 1, 2,
        ...     -1, 1, 0, 0, -2, 1, 0, -1, 0, 2, 1, 2, 2, -1, 1, 2, 1, -2, 2])
        >>> y_velocity = numpy.array([2, 0, 1, -1, 2, -2, -1, 0, -2, -1, 0, 0,
        ...     1, -2, -1, 1, 0, 0, -1, 1, -1, 0, 2, 1, -2, -1, 0, 0, -2, 0, -1])
        >>> print(plot_coordinates(x_coor+3*x_velocity, y_coor+3*y_velocity))
        #   #  ###
        #   #   #
        #   #   #
        #####   #
        #   #   #
        #   #   #
        #   #   #
        #   #  ###
    """
    x_coor = x_coor - numpy.min(x_coor)
    y_coor = y_coor - numpy.min(y_coor)

    grid = numpy.zeros((x_coor.max()+1, y_coor.max()+1), dtype="U1")
    grid[:] = " "
    grid[x_coor, y_coor] = "#"
    ascii_plot = "\n".join("".join(row) for row in grid.T)
    return ascii_plot


def part1(x_coor, y_coor, x_velocity, y_velocity):
    """Do part 1 of the assignment."""
    seconds = estimate_seconds_used(x_coor, y_coor, x_velocity, y_velocity)
    x_coor = x_coor + seconds*x_velocity
    y_coor = y_coor + seconds*y_velocity
    return plot_coordinates(x_coor, y_coor)


if __name__ == "__main__":
    with open("input") as src:
        DATA = numpy.array(re.findall(r"-?\d+", src.read()), dtype=int)
    (X_COOR, Y_COOR, X_VELOCITY, Y_VELOCITY) = DATA.reshape(-1, 4).T
    print("solution part 1:\n", part1(X_COOR, Y_COOR, X_VELOCITY, Y_VELOCITY))
    #        ######    ##    ######  ######  #####   #       ######   ####
    #            #   #  #   #            #  #    #  #            #  #    #
    #            #  #    #  #            #  #    #  #            #  #
    #           #   #    #  #           #   #    #  #           #   #
    #          #    #    #  #####      #    #####   #          #    #
    #         #     ######  #         #     #  #    #         #     #  ###
    #        #      #    #  #        #      #   #   #        #      #    #
    #       #       #    #  #       #       #   #   #       #       #    #
    #       #       #    #  #       #       #    #  #       #       #   ##
    #       ######  #    #  ######  ######  #    #  ######  ######   ### #
    print("solution part 2:", estimate_seconds_used(X_COOR, Y_COOR, X_VELOCITY, Y_VELOCITY))
    # solution part 2: 10105
