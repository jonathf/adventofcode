"""
--- Day 11: Chronal Charge ---

You watch the Elves and their sleigh fade into the distance as they head toward
the North Pole.

Actually, you're the one fading. The falling sensation returns.

The low fuel warning light is illuminated on your wrist-mounted device. Tapping
it once causes it to project a hologram of the situation: a 300x300 grid of
fuel cells and their current power levels, some negative. You're not sure what
negative power means in the context of time travel, but it can't be good.

Each fuel cell has a coordinate ranging from 1 to 300 in both the
X (horizontal) and Y (vertical) direction. In X,Y notation, the top-left cell
is 1,1, and the top-right cell is 300,1.

The interface lets you select any 3x3 square of fuel cells. To increase your
chances of getting to your destination, you decide to choose the 3x3 square
with the largest total power.

The power level in a given fuel cell can be found through the following
process:

    - Find the fuel cell's rack ID, which is its X coordinate plus 10.
    - Begin with a power level of the rack ID times the Y coordinate.
    - Increase the power level by the value of the grid serial number (your
      puzzle input).
    - Set the power level to itself multiplied by the rack ID.
    - Keep only the hundreds digit of the power level (so 12345 becomes 3;
      numbers with no hundreds digit become 0).
    - Subtract 5 from the power level.

For example, to find the power level of the fuel cell at 3,5 in a grid with
serial number 8:

    - The rack ID is 3 + 10 = 13.
    - The power level starts at 13 * 5 = 65.
    - Adding the serial number produces 65 + 8 = 73.
    - Multiplying by the rack ID produces 73 * 13 = 949.
    - The hundreds digit of 949 is 9.
    - Subtracting 5 produces 9 - 5 = 4.

So, the power level of this fuel cell is 4.

Here are some more example power levels:

    - Fuel cell at  122,79, grid serial number 57: power level -5.
    - Fuel cell at 217,196, grid serial number 39: power level  0.
    - Fuel cell at 101,153, grid serial number 71: power level  4.

Your goal is to find the 3x3 square which has the largest total power. The
square must be entirely within the 300x300 grid. Identify this square using the
X,Y coordinate of its top-left fuel cell. For example:

For grid serial number 18, the largest total 3x3 square has a top-left corner
of 33,45 (with a total power of 29); these fuel cells appear in the middle of
this 5x5 region:

-2  -4   4   4   4
-4   4   4   4  -5
 4   3   3   4  -4
 1   1   2   4  -3
-1   0   2  -5  -2

For grid serial number 42, the largest 3x3 square's top-left is 21,61 (with
a total power of 30); they are in the middle of this region:

-3   4   2   2   2
-4   4   3   3   4
-5   3   3   4  -4
 4   3   3   4  -3
 3   3   3  -5  -1

What is the X,Y coordinate of the top-left fuel cell of the 3x3 square with the
largest total power?

--- Part Two ---

You discover a dial on the side of the device; it seems to let you select
a square of any size, not just 3x3. Sizes from 1x1 to 300x300 are supported.

Realizing this, you now must find the square of any size with the largest total
power. Identify this square by including its size as a third parameter after
the top-left coordinate: a 9x9 square with a top-left corner of 3,5 is
identified as 3,5,9.

For example:

    - For grid serial number 18, the largest total square (with a total power
      of 113) is 16x16 and has a top-left corner of 90,269, so its identifier
      is 90,269,16.
    - For grid serial number 42, the largest total square (with a total power
      of 119) is 12x12 and has a top-left corner of 232,251, so its identifier
      is 232,251,12.

What is the X,Y,size identifier of the square with the largest total power?
Your puzzle input was 7689.
"""
from typing import Tuple
from itertools import product
import numpy


def generate_power_level_grid(
        serial_number: int,
        size: int = 300,
) -> numpy.ndarray:
    """
    Generate a grid of power levels.

    Args:
        serial_number:
            User identifier, defining the signature for the grid.
        size:
            The size of the edge of the grid square.

    Returns:
        A 2-dimensional array of integers of shape ``(size, size)`` with power
        levels as values.

    Examples:
        >>> generate_power_level_grid(18, 5)
        array([[-2, -1,  0,  1,  3],
               [-2,  0,  1,  2,  4],
               [-1,  0,  2,  4, -5],
               [-1,  1,  3, -5, -3],
               [-1,  2,  4, -4, -2]])

    """
    x_coor, y_coor = numpy.mgrid[1:size+1, 1:size+1]
    rack_id = x_coor+10
    power_level = ((rack_id*y_coor+serial_number)*rack_id//100)%10-5
    return power_level


def find_max_power_level(
        power_grid: numpy.ndarray,
        size: int,
) -> Tuple[int, int, int]:
    """
    Given a (size, size) square, find the location and the level on
    `power_grid` that is the highest.

    Args:
        power_grid:
            A 2-dimensional grid of power levels as integers to search over.
        size:
            The size of the edge of the square to sum up.

    Returns:
        x_coor:
            The location of the optimal placed square on `power_grid` along the
            x-axis. Using 1-based indexing.
        y_coor:
            The location of the optimal placed square on `power_grid` along the
            y-axis. Using 1-based indexing.
        power_level:
            The sum of power levels over the optimal square on `power_grid`.

    Examples:
        >>> power_grid = numpy.array([[-3, 4, 2, 2, 2],
        ...                           [-4, 4, 3, 3, 4],
        ...                           [-5, 3, 3, 4,-4],
        ...                           [ 4, 3, 3, 4,-3],
        ...                           [ 3, 3, 3,-5,-1]])
        >>> x_coor, y_coor, levels = find_max_power_level(power_grid, 3)
        >>> x_coor, y_coor, levels
        (2, 2, 30)
        >>> power_grid[x_coor-1:x_coor+2, y_coor-1:y_coor+2]
        array([[4, 3, 3],
               [3, 3, 4],
               [3, 3, 4]])

    """
    grid_size = len(power_grid)
    length = grid_size-size+1
    power_levels = numpy.zeros((length, length), dtype=int)
    for idx, idy in product(range(size), range(size)):
        power_levels += power_grid[idx:length+idx, idy:length+idy]
    x_coor = numpy.argmax(power_levels.max(1))
    y_coor = numpy.argmax(power_levels.max(0))
    return x_coor+1, y_coor+1, power_levels[x_coor, y_coor]


def part1(power_grid: numpy.ndarray) -> Tuple[int, int]:
    """Do part 1 of the assignment."""
    return find_max_power_level(power_grid, size=3)[:2]


def part2(power_grid: numpy.ndarray) -> Tuple[int, int, int]:
    """Do part 1 of the assignment."""
    x_coors, y_coors, power_levels = zip(*[
        find_max_power_level(power_grid, size) for size in range(1, 300)])
    index = numpy.argmax(power_levels)
    return x_coors[index], y_coors[index], index+1


if __name__ == "__main__":
    SERIAL_NUMBER = 7689
    POWER_GRID = generate_power_level_grid(SERIAL_NUMBER)
    print("solution part 1:", part1(POWER_GRID))
    # solution part 1: (20, 37)
    print("solution part 2:", part2(POWER_GRID))
    # solution part 2: (90, 169, 15)
