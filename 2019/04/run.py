"""
--- Day 4: Secure Container ---

You arrive at the Venus fuel depot only to discover it's protected by
a password. The Elves had written the password on a sticky note, but someone
threw it out.

However, they do remember a few key facts about the password:

    * It is a six-digit number.
    * The value is within the range given in your puzzle input.
    * Two adjacent digits are the same (like 22 in 122345).
    * Going from left to right, the digits never decrease; they only ever
      increase or stay the same (like 111123 or 135679).

Other than the range rule, the following are true:

    * 111111 meets these criteria (double 11, never decreases).
    * 223450 does not meet these criteria (decreasing pair of digits 50).
    * 123789 does not meet these criteria (no double).

How many different passwords within the range given in your puzzle input meet
these criteria?

--- Part Two ---

An Elf just remembered one more important detail: the two adjacent matching
digits are not part of a larger group of matching digits.

Given this additional criterion, but still ignoring the range rule, the
following are now true:

    * 112233 meets these criteria because the digits never decrease and all
      repeated digits are exactly two digits long.
    * 123444 no longer meets the criteria (the repeated 44 is part of a larger
      group of 444).
    * 111122 meets the criteria (even though 1 is repeated more than twice, it
      still contains a double 22).

How many different passwords within the range given in your puzzle input meet
all of the criteria?

Your puzzle input is 357253-892942.
"""
from typing import Iterator


def increasing_numbers(start: int, stop: int) -> Iterator[str]:
    """
    Iterate all numbers in a range that is increasing in digits.

    Args:
        start:
            The first number to check in range.
        stop:
            The last number (inclusive) to check.

    Yields:
        Numbers with increasing digits, represented as strings.

    Examples:
        >>> list(increasing_numbers(60, 80))
        ['66', '67', '68', '69', '77', '78', '79']
        >>> list(increasing_numbers(895, 999))
        ['899', '999']

    """
    for number in range(start, stop+1):
        number = str(number)
        consecutives = zip(number[:-1], number[1:])
        if all(int(digit1) <= int(digit2) for digit1, digit2 in consecutives):
            yield number


def part1(start: int, stop: int) -> int:
    """Do part 1 of the assignment."""
    count = 0
    for number in increasing_numbers(start, stop):
        consecutives = zip(number[:-1], number[1:])
        count += any(digit1 == digit2 for digit1, digit2 in consecutives)
    return count


def part2(start: int, stop: int) -> int:
    """Do part 2 of the assignment."""
    count = 0
    for number in increasing_numbers(start, stop):
        consecutive_digits = zip(number[:-1], number[1:])
        matches = [digit1 == digit2 for digit1, digit2 in consecutive_digits]
        # Pad values to allow for consecutive numbers at the beginning and end of number.
        matches = [False] + matches + [False]
        consecutive_matches = zip(matches[:-2], matches[1:-1], matches[2:])
        # A match surrounded by non-matches.
        count += any(triplet == (False, True, False)
                     for triplet in consecutive_matches)
    return count


if __name__ == "__main__":
    START = 357253
    STOP = 892942
    print("solution part 1:", part1(START, STOP))
    # solution part 1: 530
    print("solution part 2:", part2(START, STOP))
    # solution part 2: 324
