"""
--- Day 11: Space Police ---

On the way to Jupiter, you're pulled over by the Space Police.

"Attention, unmarked spacecraft! You are in violation of Space Law! All
spacecraft must have a clearly visible registration identifier! You have 24
hours to comply or be sent to Space Jail!"

Not wanting to be sent to Space Jail, you radio back to the Elves on Earth for
help. Although it takes almost three hours for their reply signal to reach you,
they send instructions for how to power up the emergency hull painting robot
and even provide a small Intcode program (your puzzle input) that will cause it
to paint your ship appropriately.

There's just one problem: you don't have an emergency hull painting robot.

You'll need to build a new emergency hull painting robot. The robot needs to be
able to move around on the grid of square panels on the side of your ship,
detect the color of its current panel, and paint its current panel black or
white. (All of the panels are currently black.)

The Intcode program will serve as the brain of the robot. The program uses
input instructions to access the robot's camera: provide 0 if the robot is over
a black panel or 1 if the robot is over a white panel. Then, the program will
output two values:

    - First, it will output a value indicating the color to paint the panel the
      robot is over: 0 means to paint the panel black, and 1 means to paint the
      panel white.
    - Second, it will output a value indicating the direction the robot should
      turn: 0 means it should turn left 90 degrees, and 1 means it should turn
      right 90 degrees.

After the robot turns, it should always move forward exactly one panel. The
robot starts facing up.

The robot will continue running for a while like this and halt when it is
finished drawing. Do not restart the Intcode computer inside the robot during
this process.

For example, suppose the robot is about to start running. Drawing black panels
as ., white panels as #, and the robot pointing the direction it is facing (<
^ > v), the initial state and region near the robot looks like this:

.....
.....
..^..
.....
.....

The panel under the robot (not visible here because a ^ is shown instead) is
also black, and so any input instructions at this point should be provided 0.
Suppose the robot eventually outputs 1 (paint white) and then 0 (turn left).
After taking these actions and moving forward one panel, the region now looks
like this:

.....
.....
.<#..
.....
.....

Input instructions should still be provided 0. Next, the robot might output
0 (paint black) and then 0 (turn left):

.....
.....
..#..
.v...
.....

After more outputs (1,0, 1,0):

.....
.....
..^..
.##..
.....

The robot is now back where it started, but because it is now on a white panel,
input instructions should be provided 1. After several more outputs (0,1, 1,0,
1,0), the area looks like this:

.....
..<#.
...#.
.##..
.....

Before you deploy the robot, you should probably have an estimate of the area
it will cover: specifically, you need to know the number of panels it paints at
least once, regardless of color. In the example above, the robot painted
6 panels at least once. (It painted its starting panel twice, but that panel is
still only counted once; it also never painted the panel it ended on.)

Build a new emergency hull painting robot and run the Intcode program on it.
How many panels does it paint at least once?

"""
import logging
from typing import Dict, List, Optional, Union, Tuple
from operator import add, eq, mul, lt
from itertools import permutations
from collections import defaultdict

import numpy

OPERATORS = {1: add, 2: mul, 7: lt, 8: eq}


class IntProgram:
    """
    IntCode interpreter.

    Examples:
        >>> program = [3, 9, 8, 9, 10, 9, 4, 9, 99, -1, 8]
        >>> [IntProgram(program).run(inputs) for inputs in [7, 8, 9]]
        [0, 1, 0]
        >>> program = [3, 9, 7, 9, 10, 9, 4, 9, 99, -1, 8]
        >>> [IntProgram(program).run(inputs) for inputs in [7, 8, 9]]
        [1, 0, 0]
        >>> program = [3, 3, 1108, -1, 8, 3, 4, 3, 99]
        >>> [IntProgram(program).run(inputs) for inputs in [7, 8, 9]]
        [0, 1, 0]
        >>> program = [3, 3, 1107, -1, 8, 3, 4, 3, 99]
        >>> [IntProgram(program).run(inputs) for inputs in [7, 8, 9]]
        [1, 0, 0]
        >>> program = [3, 12, 6, 12, 15, 1, 13, 14, 13, 4, 13, 99, -1, 0, 1, 9]
        >>> [IntProgram(program).run(inputs) for inputs in [-1, 0, 1]]
        [1, 0, 1]
        >>> program = [3, 3, 1105, -1, 9, 1101, 0, 0, 12, 4, 12, 99, 1]
        >>> [IntProgram(program).run(inputs) for inputs in [-1, 0, 1]]
        [1, 0, 1]
        >>> program = [3, 21, 1008, 21, 8, 20, 1005, 20, 22, 107, 8, 21, 20,
        ...            1006, 20, 31, 1106, 0, 36, 98, 0, 0, 1002, 21, 125, 20,
        ...            4, 20, 1105, 1, 46, 104, 999, 1105, 1, 46, 1101, 1000,
        ...            1, 20, 4, 20, 1105, 1, 46, 98, 99]
        >>> IntProgram(program).run(0)
        999

        >>> program = IntProgram([109, 1, 204, -1, 1001, 100, 1, 100,
        ...                       1008, 100, 16, 101, 1006, 101, 0, 99])
        >>> program.run()
        99
        >>> program.outputs  # doctest: +NORMALIZE_WHITESPACE
        [109, 1, 204, -1, 1001, 100, 1, 100,
         1008, 100, 16, 101, 1006, 101, 0, 99]
        >>> program = IntProgram([1102, 34915192, 34915192, 7, 4, 7, 99, 0])
        >>> program.run()
        1219070632396864
        >>> program = IntProgram([104, 1125899906842624, 99])
        >>> program.run()
        1125899906842624

    """

    def __init__(
            self,
            program: List[int],
            inputs: Optional[List[int]] = None,
            outputs: Optional[List[int]] = None,
    ):
        """
        Args:
            program:
                The Intcode program to execute.
            inputs:
                List of initial inputs. Values are extracted from the list by
                reference during program. Share this as another programs
                `outputs` and it becomes effectively a pipe.
            outputs:
                List of outputs. Filled by reference so to be able to pipe
                results.
        """
        self.program = program[:]
        self.inputs = [] if inputs is None else inputs
        self.outputs = [] if outputs is None else outputs
        self.offset = 0
        self.index = 0

    def get_indices(self, modes: str, *indices: int) -> Union[int, List[int]]:
        """
        Get index either by index to reference (position mode), or by index to
        value (immediate mode).

        Args:
            modes:
                String with modes, in opposite ordering as `indices`. "0"
                represents position mode, "1" represents immediate mode and "2"
                represents relative mode.
            indices:
                The index to value in program (in immediate mode) or index to
                reference (in position mode).

        Returns:
            Values from program. Either `index` (position mode),
            `program[index]` (immediate mode), or
            `program[index]+offset` (relative mode). If a single value
            is passed to `indices`, output is also a single value instead of
            a list.
        """
        logger = logging.getLogger(__name__)
        output = []
        for mode, index in zip(reversed(modes), indices):

            logger.warning("Getting value %r: %d", mode, index)
            if mode == "0":
                index = self[index]
                logger.warning("  from position: %d", index)
            elif mode == "1":
                pass
            elif mode == "2":
                index = self[index]+self.offset
                logger.warning("  using relative base %d", self.offset)
                logger.warning("  from position: %d", index)

            output.append(index)
            logger.warning("  referencing value: %d", self[index])

        if len(output) == 1:
            output = output[0]
        return output

    def __getitem__(self, index: int):
        """Get program instruction from memory by index."""
        assert index >= 0, "program index out of bounds"
        if len(self.program) <= index:
            self.program.extend([0]*(index-len(self.program)+1))
        return self.program[index]

    def __setitem__(self, index: int, value: int) -> None:
        """Set program instruction from memory by index."""
        logger = logging.getLogger(__name__)
        assert index >= 0, "program index out of bounds"
        if len(self.program) <= index:
            self.program.extend([0]*(index-len(self.program)+1))
            logger.warning("Extending program memory to %d", len(self.program))
        logger.warning("Storing value %d at index %d", value, index)
        self.program[index] = value

    def run(self, value: Optional[int] = None) -> int:
        """
        Execute Intcode program.

        Args:
            value:
                Optional input passed to the program.

        Returns:
            If program halts because of missing input, None is returned. If the
            program is terminated successfully, the return code located in the
            first registry is returned.
        """
        logger = logging.getLogger(__name__)
        if value is not None:
            self.inputs.append(value)

        while True:
            # cast to string to capture leading zeros
            instruction = f"{self[self.index]:05d}"
            logger.warning("\nInstruction[%d]: %s", self.index, instruction)
            mode, opcode = instruction[:-2], int(instruction[-2:])

            # x[2] = operator(x[0], x[1])
            if opcode in OPERATORS:
                noun, verb, output = self.get_indices(
                    mode, self.index+1, self.index+2, self.index+3)
                operator = OPERATORS[opcode]
                value = int(operator(self[noun], self[verb]))
                logger.warning("Operator: %d %d %s %d",
                               noun, verb, operator.__name__, value)
                self[output] = value
                self.index += 4

            # x[0] = inputs
            elif opcode == 3:
                if not self.inputs:
                    logger.warning("no input; halting")
                    return None
                logger.warning("Reading input: %s", self.inputs[0])
                self[self.get_indices(mode, self.index+1)] = self.inputs.pop(0)
                self.index += 2

            # output = x[0]
            elif opcode == 4:
                output = self[self.get_indices(mode, self.index+1)]
                logger.warning("Writing output: %s", output)
                self.outputs.append(output)
                self.index += 2

            # index = x[1] if x[0] else index
            elif opcode in (5, 6):
                condition, new_index = self.get_indices(mode, self.index+1, self.index+2)
                # exclusive-or operator to capture both opcodes 5 and 6
                if bool(self[condition]) == (opcode == 5):
                    logger.warning("Jump to index: %d", new_index)
                    self.index = self[new_index]
                else:
                    logger.warning("No jump")
                    self.index = self.index+3

            # offset += x[0]
            elif opcode == 9:
                value = self[self.get_indices(mode, self.index+1)]
                logger.warning("Adjusting relative base: %d -> %d",
                               self.offset, self.offset+value)
                self.offset += value
                self.index += 2

            # terminate program
            elif opcode == 99:
                break

            else:
                raise ValueError(f"unrecognized instruction: {instruction}")

        logger.warning("program terminated; return code: %d", self.outputs[-1])
        return self.outputs[-1]


def run_paint_robot(
    code: List[int],
    inputs: int,
) -> Dict[Tuple[int, int], int]:
    """
    Run pain robot.

    Args:
        code:
            The IntProgram to execute.
        inputs:
            The starting code provided to the IntCode as input.

    Returns:
        Dictionary where the keys are coordinates relative to the start, and
        values are the color painted.

    """
    tiles = defaultdict(int)
    coord = (0, 0)
    direction = 0

    program = IntProgram(code)
    return_code = program.run(inputs)
    new_color, turn = program.outputs[-2:]
    while return_code is None:
        tiles[coord] = new_color
        direction = (direction - (2*turn-1)) % 4
        coord = (coord[0] + {0: 0, 1: -1, 2: 0, 3: 1}[direction],
                 coord[1] + {0, -1, 1: 0, 2: 1, 3: 0}[direction])

        return_code = program.run(tiles[coord] > 0)
        new_color, turn = program.outputs[-2:]

    return dict(tiles)


def plot_paint_job(
    tiles: Dict[Tuple[int, int], int],
    black: str = " ",
) -> str:
    """
    Create ASCII plotting figure.

    Args:
        tiles:
            Dictionary where keys are coordinate and values are color. The
            values 0 are black and 1 are white.
        black:
            Single character defining the color black.

    Returns:
        ASCII representation of `tiles`.

    Examples:
        >>> tiles = {(0, 0): 1, (-1, -1): 1, (0, -1): 1, (1, 0): 1, (1, 1): 1}
        >>> print(plot_paint_job(tiles, black="."))
        ##.
        .##
        ..#

    """
    # shift grid to the first quadrant starting at 0
    min_val = numpy.min(list(tiles.keys()), 0)
    tiles = {(x-min_val[0], y-min_val[1]): val for (x, y), val in tiles.items()}

    # create black gird
    max_val = numpy.max(list(tiles.keys()), 0)
    data = numpy.zeros((max_val[0]+1, max_val[1]+1), dtype="U1")
    data[:] = black

    # fill in whites
    for key, val in tiles.items():
        if val > 0:
            data[key] = "#"

    return "\n".join("".join(line) for line in data.T)


def part1(code: List[int]) ->int:
    """Do part 1 of the assignment."""
    return len(run_paint_robot(code, 0))


def part2(code: List[int]) -> str:
    """Do part 2 of the assignment."""
    tiles = run_paint_robot(code, 1)
    return plot_paint_job(tiles)


if __name__ == "__main__":
    CODE = numpy.fromfile("input", sep=",", dtype=int).tolist()
    print("solution part 1:", part1(CODE))
    # solution part 1: 1686
    print("solution part 2:", part2(CODE), sep="\n")
    # solution part 2:
     ##   ##  ###  ###  #  # #### #  # #
    #  # #  # #  # #  # # #     # #  # #
    #    #  # #  # #  # ##     #  #  # #
    # ## #### ###  ###  # #   #   #  # #
    #  # #  # # #  #    # #  #    #  # #
     ### #  # #  # #    #  # ####  ##  ####
