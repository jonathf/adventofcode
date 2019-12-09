"""
--- Day 9: Sensor Boost ---

You've just said goodbye to the rebooted rover and left Mars when you receive
a faint distress signal coming from the asteroid belt. It must be the Ceres
monitoring station!

In order to lock on to the signal, you'll need to boost your sensors. The Elves
send up the latest BOOST program - Basic Operation Of System Test.

While BOOST (your puzzle input) is capable of boosting your sensors, for
tenuous safety reasons, it refuses to do so until the computer it runs on
passes some checks to demonstrate it is a complete Intcode computer.

Your existing Intcode computer is missing one key feature: it needs support for
parameters in relative mode.

Parameters in mode 2, relative mode, behave very similarly to parameters in
position mode: the parameter is interpreted as a position. Like position mode,
parameters in relative mode can be read from or written to.

The important difference is that relative mode parameters don't count from
address 0. Instead, they count from a value called the relative base. The
relative base starts at 0.

The address a relative mode parameter refers to is itself plus the current
relative base. When the relative base is 0, relative mode parameters and
position mode parameters with the same value refer to the same address.

For example, given a relative base of 50, a relative mode parameter of -7
refers to memory address 50 + -7 = 43.

The relative base is modified with the relative base offset instruction:

    - Opcode 9 adjusts the relative base by the value of its only parameter.
      The relative base increases (or decreases, if the value is negative) by
      the value of the parameter.

For example, if the relative base is 2000, then after the instruction 109,19,
the relative base would be 2019. If the next instruction were 204,-34, then the
value at address 1985 would be output.

Your Intcode computer will also need a few other capabilities:

    - The computer's available memory should be much larger than the initial
      program. Memory beyond the initial program starts with the value 0 and
      can be read or written like any other memory. (It is invalid to try to
      access memory at a negative address, though.)
    - The computer should have support for large numbers. Some instructions
      near the beginning of the BOOST program will verify this capability.

Here are some example programs that use these features:

    - 109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99 takes no input
      and produces a copy of itself as output.
    - 1102,34915192,34915192,7,4,7,99,0 should output a 16-digit number.
    - 104,1125899906842624,99 should output the large number in the middle.

The BOOST program will ask for a single input; run it in test mode by providing
it the value 1. It will perform a series of checks on each opcode, output any
opcodes (and the associated parameter modes) that seem to be functioning
incorrectly, and finally output a BOOST keycode.

Once your Intcode computer is fully functional, the BOOST program should report
no malfunctioning opcodes when run in test mode; it should only output a single
value, the BOOST keycode. What BOOST keycode does it produce?
"""
import logging
from typing import List, Optional, Union
from operator import add, eq, mul, lt
from itertools import permutations

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


def part1(program):
    """Do part 1 of the assignment."""
    return IntProgram(program).run(1)


def part2(program):
    """Do part 2 of the assignment."""
    return IntProgram(program).run(2)


if __name__ == "__main__":
    PROGRAM = numpy.fromfile("input", sep=",", dtype=int).tolist()
    print("solution part 1:", part1(PROGRAM))
    # solution part 1: 2714716640
    print("solution part 2:", part2(PROGRAM))
    # solution part 2: 58879
