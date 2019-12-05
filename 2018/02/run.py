"""
--- Day 2: Inventory Management System ---

You stop falling through time, catch your breath, and check the screen on the
device. "Destination reached. Current Year: 1518. Current Location: North Pole
Utility Closet 83N10." You made it! Now, to find those anomalies.

Outside the utility closet, you hear footsteps and a voice. "...I'm not sure
either. But now that so many people have chimneys, maybe he could sneak in that
way?" Another voice responds, "Actually, we've been working on a new kind of
suit that would let him fit through tight spaces like that. But, I heard that
a few days ago, they lost the prototype fabric, the design plans, everything!
Nobody on the team can even seem to remember important details of the project!"

"Wouldn't they have had enough fabric to fill several boxes in the warehouse?
They'd be stored together, so the box IDs should be similar. Too bad it would
take forever to search the warehouse for two similar box IDs..." They walk too
far away to hear any more.

Late at night, you sneak to the warehouse - who knows what kinds of paradoxes
you could cause if you were discovered - and use your fancy wrist device to
quickly scan every box and produce a list of the likely candidates (your puzzle
input).

To make sure you didn't miss any, you scan the likely candidate boxes again,
counting the number that have an ID containing exactly two of any letter and
then separately counting those with exactly three of any letter. You can
multiply those two counts together to get a rudimentary checksum and compare it
to what your device predicts.

For example, if you see the following box IDs:

    abcdef contains no letters that appear exactly two or three times.
    bababc contains two a and three b, so it counts for both.
    abbcde contains two b, but no letter appears exactly three times.
    abcccd contains three c, but no letter appears exactly two times.
    aabcdd contains two a and two d, but it only counts once.
    abcdee contains two e.
    ababab contains three a and three b, but it only counts once.

Of these box IDs, four of them contain a letter which appears exactly twice,
and three of them contain a letter which appears exactly three times.
Multiplying these together produces a checksum of 4 * 3 = 12.

What is the checksum for your list of box IDs?

--- Part Two ---

Confident that your list of box IDs is complete, you're ready to find the boxes
full of prototype fabric.

The boxes will have IDs which differ by exactly one character at the same
position in both strings. For example, given the following box IDs:

    abcde
    fghij
    klmno
    pqrst
    fguij
    axcye
    wvxyz

The IDs abcde and axcye are close, but they differ by two characters (the
second and fourth). However, the IDs fghij and fguij differ by exactly one
character, the third (h and u). Those must be the correct boxes.

What letters are common between the two correct box IDs? (In the example above,
this is found by removing the differing character from either ID, producing
fgij.)
"""
from typing import List
from collections import defaultdict
import numpy


def create_match_count_matrix(box_ids: List[str]) -> numpy.ndarray:
    """
    Construct a (N x N) matrix with how many characters in each box-ID matches
    with other box-IDs.

    To ensure that match count selects pairs consisting of the same word twice,
    the diagonal of the matrix is removed.

    Examples:
        >>> create_match_count_matrix(["abb", "bbb", "bbb", "bcc"])
        array([[0, 2, 2, 0],
               [2, 0, 3, 1],
               [2, 3, 0, 1],
               [0, 1, 1, 0]])
    """
    # convert to matrix of characters
    char_matrix = numpy.array(box_ids).view("U1").reshape(len(box_ids), -1)
    # count up characters in the same column that are the same
    matches = numpy.zeros((len(box_ids), len(box_ids)), dtype=int)
    for column in char_matrix.T:
        # all against all compare of character
        matches += column.reshape(1, -1) == column.reshape(-1, 1)
    # remove self matching
    matches -= numpy.diag(numpy.diag(matches))
    return matches


def part1(box_ids: List[str]) -> int:
    """Do part 1 of the assignment."""
    doubles = triplets = 0
    for box_id in box_ids:
        char_counts = defaultdict(int)
        for char in box_id:
            char_counts[char] += 1
        doubles += 2 in char_counts.values()
        triplets += 3 in char_counts.values()
    return doubles*triplets


def part2(box_ids: List[str]) -> str:
    """Do part 2 of the assignment."""
    matches = create_match_count_matrix(box_ids)
    index = numpy.argmax(matches)
    word_index, char_index = index % len(matches), index // len(matches)
    prefix = box_ids[word_index][:char_index]
    suffix = box_ids[word_index][char_index+1:]
    return prefix+suffix


if __name__ == "__main__":
    with open("input") as src:
        BOX_IDS = src.read().split()
    print("solution part 1:", part1(BOX_IDS))
    # solution part 1: 8398
    print("solution part 2:", part2(BOX_IDS))
    # solution part 2: hhvsdkatysmiqjxjunezgwcdpr
