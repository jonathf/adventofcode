"""
--- Day 7: Amplification Circuit ---

Based on the navigational maps, you're going to need to send more power to your
ship's thrusters to reach Santa in time. To do this, you'll need to configure
a series of amplifiers already installed on the ship.

There are five amplifiers connected in series; each one receives an input
signal and produces an output signal. They are connected such that the first
amplifier's output leads to the second amplifier's input, the second
amplifier's output leads to the third amplifier's input, and so on. The first
amplifier's input value is 0, and the last amplifier's output leads to your
ship's thrusters.

    O-------O  O-------O  O-------O  O-------O  O-------O
0 ->| Amp A |->| Amp B |->| Amp C |->| Amp D |->| Amp E |-> (to thrusters)
    O-------O  O-------O  O-------O  O-------O  O-------O

The Elves have sent you some Amplifier Controller Software (your puzzle input),
a program that should run on your existing Intcode computer. Each amplifier
will need to run a copy of the program.

When a copy of the program starts running on an amplifier, it will first use an
input instruction to ask the amplifier for its current phase setting (an
integer from 0 to 4). Each phase setting is used exactly once, but the Elves
can't remember which amplifier needs which phase setting.

The program will then call another input instruction to get the amplifier's
input signal, compute the correct output signal, and supply it back to the
amplifier with an output instruction. (If the amplifier has not yet received an
input signal, it waits until one arrives.)

Your job is to find the largest output signal that can be sent to the thrusters
by trying every possible combination of phase settings on the amplifiers. Make
sure that memory is not shared or reused between copies of the program.

For example, suppose you want to try the phase setting sequence 3,1,2,4,0,
which would mean setting amplifier A to phase setting 3, amplifier B to setting
1, C to 2, D to 4, and E to 0. Then, you could determine the output signal that
gets sent from amplifier E to the thrusters with the following steps:

    - Start the copy of the amplifier controller software that will run on
      amplifier A. At its first input instruction, provide it the amplifier's
      phase setting, 3. At its second input instruction, provide it the input
      signal, 0. After some calculations, it will use an output instruction to
      indicate the amplifier's output signal.
    - Start the software for amplifier B. Provide it the phase setting (1) and
      then whatever output signal was produced from amplifier A. It will then
      produce a new output signal destined for amplifier C.
    - Start the software for amplifier C, provide the phase setting (2) and the
      value from amplifier B, then collect its output signal.
    - Run amplifier D's software, provide the phase setting (4) and input
      value, and collect its output signal.
    - Run amplifier E's software, provide the phase setting (0) and input
      value, and collect its output signal.

The final output signal from amplifier E would be sent to the thrusters.
However, this phase setting sequence may not have been the best one; another
sequence might have sent a higher signal to the thrusters.

Here are some example programs:

    - Max thruster signal 43210 (from phase setting sequence 4,3,2,1,0):

        3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0

    - Max thruster signal 54321 (from phase setting sequence 0,1,2,3,4):

        3,23,3,24,1002,24,10,24,1002,23,-1,23,
        101,5,23,23,1,24,23,23,4,23,99,0,0

    - Max thruster signal 65210 (from phase setting sequence 1,0,4,3,2):

        3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,
        1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0

Try every combination of phase settings on the amplifiers. What is the highest
signal that can be sent to the thrusters?

--- Part Two ---

It's no good - in this configuration, the amplifiers can't generate a large
enough output signal to produce the thrust you'll need. The Elves quickly talk
you through rewiring the amplifiers into a feedback loop:

      O-------O  O-------O  O-------O  O-------O  O-------O
0 -+->| Amp A |->| Amp B |->| Amp C |->| Amp D |->| Amp E |-.
   |  O-------O  O-------O  O-------O  O-------O  O-------O |
   |                                                        |
   '--------------------------------------------------------+
                                                            |
                                                            v
                                                     (to thrusters)

Most of the amplifiers are connected as they were before; amplifier A's output
is connected to amplifier B's input, and so on. However, the output from
amplifier E is now connected into amplifier A's input. This creates the
feedback loop: the signal will be sent through the amplifiers many times.

In feedback loop mode, the amplifiers need totally different phase settings:
integers from 5 to 9, again each used exactly once. These settings will cause
the Amplifier Controller Software to repeatedly take input and produce output
many times before halting. Provide each amplifier its phase setting at its
first input instruction; all further input/output instructions are for signals.

Don't restart the Amplifier Controller Software on any amplifier during this
process. Each one should continue receiving and sending signals until it halts.

All signals sent or received in this process will be between pairs of
amplifiers except the very first signal and the very last signal. To start the
process, a 0 signal is sent to amplifier A's input exactly once.

Eventually, the software on the amplifiers will halt after they have processed
the final loop. When this happens, the last output signal from amplifier E is
sent to the thrusters. Your job is to find the largest output signal that can
be sent to the thrusters using the new phase settings and feedback loop
arrangement.

Here are some example programs:

    - Max thruster signal 139629729 (from phase setting sequence 9,8,7,6,5):

        3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,
        27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5

    - Max thruster signal 18216 (from phase setting sequence 9,7,8,5,6):

        3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,
        -5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,
        53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10

Try every combination of the new phase settings on the amplifier feedback loop.
What is the highest signal that can be sent to the thrusters?
"""
import logging
from typing import List, Optional, Union
from operator import add, eq, mul, lt
from itertools import permutations

import numpy

OPERATORS = {1: add, 2: mul, 7: lt, 8: eq}


class IntProgram:
    """IntCode interpreter."""

    def __init__(
            self,
            program: List[int],
            inputs: List[int],
            outputs: List[int],
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
        self.inputs = inputs
        self.outputs = outputs
        self.index = 0

    def get_value(self, modes: str, *indices: int) -> Union[int, List[int]]:
        """
        Get values either by index to reference (position mode), or by index to
        value (immediate mode).

        Args:
            modes:
                String with modes, in opposite ordering as `indices`. "0" represents
                position mode, "1" represents immediate mode.
            indices:
                The index to value in program (in immediate mode) or index to
                reference (in position mode).

        Returns:
            Values from program. Either `program[index]` (position mode) or
            `program[program[index]]` (immediate mode). If a single value is passed
            to `indices`, output is also a single value instead of a list.
        """
        output = [(self.program[index] if mode == "1"
                   else self.program[self.program[index]])
                  for mode, index in zip(reversed(modes), indices)]
        if len(output) == 1:
            output = output[0]
        return output

    def run(self, value: Optional[int] = None):
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
            instruction = f"{self.program[self.index]:04d}"
            mode, opcode = instruction[:-2], int(instruction[-2:])

            # x[2] = operator(x[0], x[1])
            if opcode in OPERATORS:
                noun, verb = self.get_value(mode, self.index+1, self.index+2)
                operator = OPERATORS[opcode]
                self.program[self.program[self.index+3]] = int(operator(noun, verb))
                self.index += 4

            # x[0] = inputs
            elif opcode == 3:
                if not self.inputs:
                    logger.info("no input; halting")
                    return None
                logger.info("reading input: %s", self.inputs[0])
                self.program[self.program[self.index+1]] = self.inputs.pop(0)
                self.index += 2

            # output = x[0]
            elif opcode == 4:
                output = self.get_value(mode, self.index+1)
                logger.info("writing output: %s", output)
                self.outputs.append(output)
                self.index += 2

            # index = x[1] if x[0] else index
            elif opcode in (5, 6):
                condition, new_index = self.get_value(mode, self.index+1, self.index+2)
                # exclusive-or operator to capture both opcodes 5 and 6
                if bool(condition) == (opcode == 5):
                    self.index = new_index
                else:
                    self.index = self.index+3

            # terminate program
            elif opcode == 99:
                break

            else:
                raise ValueError(f"unrecognized instruction: {instruction}")

        logger.info("program terminated; return code: %d", self.outputs[-1])
        return self.outputs[-1]


def run_amplifier(program, settings, start_value=0):
    """
    >>> program = [3, 15, 3, 16, 1002, 16, 10, 16, 1, 16, 15, 15, 4, 15, 99, 0, 0]
    >>> run_amplifier(program,  [4, 3, 2, 1, 0])
    43210
    >>> program = [3, 23, 3, 24, 1002, 24, 10, 24, 1002, 23, -1, 23,
    ...            101, 5, 23, 23, 1, 24, 23, 23, 4, 23, 99, 0, 0]
    >>> run_amplifier(program,  [0, 1, 2, 3, 4])
    54321
    """
    # Pipe 5 programs in a circle
    links = [], [], [], [], []
    programs = [IntProgram(program, links[idx-1], links[idx%5])
                for idx in range(1, 6)]

    # Execute initial settings
    for amp, setting in zip(programs, settings):
        amp.run(setting)

    # Run forever until last program terminates
    value = start_value
    while True:
        for amp in programs:
            value = amp.run(value)
        if value is not None:
            break
    return value


def part1(program):
    """Do part 1 of the assignment."""
    max_val = -9999
    for settings in permutations(range(5), 5):
        max_val = max(max_val, run_amplifier(program, settings, 0))
    return max_val

def part2(program):
    """Do part 2 of the assignment."""
    max_val = -9999
    for settings in permutations(range(5, 10), 5):
        max_val = max(max_val, run_amplifier(program, settings, 0))
    return max_val


if __name__ == "__main__":
    PROGRAM = numpy.fromfile("input", sep=",", dtype=int).tolist()
    print("solution part 1:", part1(PROGRAM))
    # solution part 1: 255590
    print("solution part 2:", part2(PROGRAM))
    # solution part 2: 58285150
