r"""
--- Day 13: Mine Cart Madness ---

A crop of this size requires significant logistics to transport produce, soil,
fertilizer, and so on. The Elves are very busy pushing things around in carts
on some kind of rudimentary system of tracks they've come up with.

Seeing as how cart-and-track systems don't appear in recorded history for
another 1000 years, the Elves seem to be making this up as they go along. They
haven't even figured out how to avoid collisions yet.

You map out the tracks (your puzzle input) and see where you can help.

Tracks consist of straight paths (| and -), curves (/ and \), and intersections
(+). Curves connect exactly two perpendicular pieces of track; for example,
this is a closed loop:

/----\
|    |
|    |
\----/

Intersections occur when two perpendicular paths cross. At an intersection,
a cart is capable of turning left, turning right, or continuing straight. Here
are two loops connected by two intersections:

/-----\
|     |
|  /--+--\
|  |  |  |
\--+--/  |
   |     |
   \-----/

Several carts are also on the tracks. Carts always face either up (^), down
(v), left (<), or right (>). (On your initial map, the track under each cart is
a straight path matching the direction the cart is facing.)

Each time a cart has the option to turn (by arriving at any intersection), it
turns left the first time, goes straight the second time, turns right the third
time, and then repeats those directions starting again with left the fourth
time, straight the fifth time, and so on. This process is independent of the
particular intersection at which the cart has arrived - that is, the cart has
no per-intersection memory.

Carts all move at the same speed; they take turns moving a single step at
a time. They do this based on their current location: carts on the top row move
first (acting from left to right), then carts on the second row move (again
from left to right), then carts on the third row, and so on. Once each cart has
moved one step, the process repeats; each of these loops is called a tick.

For example, suppose there are two carts on a straight track:

|  |  |  |  |
v  |  |  |  |
|  v  v  |  |
|  |  |  v  X
|  |  ^  ^  |
^  ^  |  |  |
|  |  |  |  |

First, the top cart moves. It is facing down (v), so it moves down one square.
Second, the bottom cart moves. It is facing up (^), so it moves up one square.
Because all carts have moved, the first tick ends. Then, the process repeats,
starting with the first cart. The first cart moves down, then the second cart
moves up - right into the first cart, colliding with it! (The location of the
crash is marked with an X.) This ends the second and last tick.

Here is a longer example:

/->-\
|   |  /----\
| /-+--+-\  |
| | |  | v  |
\-+-/  \-+--/
  \------/

/-->\
|   |  /----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \->--/
  \------/

/---v
|   |  /----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \-+>-/
  \------/

/---\
|   v  /----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \-+->/
  \------/

/---\
|   |  /----\
| /->--+-\  |
| | |  | |  |
\-+-/  \-+--^
  \------/

/---\
|   |  /----\
| /-+>-+-\  |
| | |  | |  ^
\-+-/  \-+--/
  \------/

/---\
|   |  /----\
| /-+->+-\  ^
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /----<
| /-+-->-\  |
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /---<\
| /-+--+>\  |
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /--<-\
| /-+--+-v  |
| | |  | |  |
\-+-/  \-+--/
  \------/

/---\
|   |  /-<--\
| /-+--+-\  |
| | |  | v  |
\-+-/  \-+--/
  \------/

/---\
|   |  /<---\
| /-+--+-\  |
| | |  | |  |
\-+-/  \-<--/
  \------/

/---\
|   |  v----\
| /-+--+-\  |
| | |  | |  |
\-+-/  \<+--/
  \------/

/---\
|   |  /----\
| /-+--v-\  |
| | |  | |  |
\-+-/  ^-+--/
  \------/

/---\
|   |  /----\
| /-+--+-\  |
| | |  X |  |
\-+-/  \-+--/
  \------/

After following their respective paths for a while, the carts eventually crash.
To help prevent crashes, you'd like to know the location of the first crash.
Locations are given in X,Y coordinates, where the furthest left column is X=0
and the furthest top row is Y=0:

           111
 0123456789012
0/---\
1|   |  /----\
2| /-+--+-\  |
3| | |  X |  |
4\-+-/  \-+--/
5  \------/

In this example, the location of the first crash is 7,3.

--- Part Two ---

There isn't much you can do to prevent crashes in this ridiculous system.
However, by predicting the crashes, the Elves know where to be in advance and
instantly remove the two crashing carts the moment any crash occurs.

They can proceed like this for a while, but eventually, they're going to run
out of carts. It could be useful to figure out where the last cart that hasn't
crashed will end up.

For example:

/>-<\
|   |
| /<+-\
| | | v
\>+</ |
  |   ^
  \<->/

/---\
|   |
| v-+-\
| | | |
\-+-/ |
  |   |
  ^---^

/---\
|   |
| /-+-\
| v | |
\-+-/ |
  ^   ^
  \---/

/---\
|   |
| /-+-\
| | | |
\-+-/ ^
  |   |
  \---/

After four very expensive crashes, a tick ends with only one cart remaining;
its final location is 6,4.

What is the location of the last cart at the end of the first tick where it is
the only cart left?
"""
from typing import Iterator, List, NamedTuple, Sequence, Tuple
import time
import numpy


class Train(NamedTuple):
    """A representation of a train by its position, direction and state."""
    x_coor: int
    y_coor: int
    direction: int
    state: int = 0


def extract_trains(track_string: str) -> Tuple[numpy.ndarray, List[Train]]:
    r"""
    Extract train from ASCII map of of tracks and trains.

    Args:
        track_string:
            Map as a ASCII string. Tracks uses the symbols '|/-\+' and trains
            symbols '<^>v'.

    Returns:
        tracks:
            Same as `track_string`, but as a 2-dimensional char-array. Trains
            are replaced with appropriate tracks symbols.
        trains:
            The trains extracted out, including coordinates and directions.

    Examples:
        >>> tracks = ("/->-\\        \n|   |  /----\\\n| /-+--+-\\  |\n"
        ...           "| | |  | v  |\n\\-+-/  \\-+--/\n  \\------/   ")
        >>> tracks, trains = extract_trains(tracks)
        >>> print(tracks)
        [['/' '-' '-' '-' '\\' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ']
         ['|' ' ' ' ' ' ' '|' ' ' ' ' '/' '-' '-' '-' '-' '\\']
         ['|' ' ' '/' '-' '+' '-' '-' '+' '-' '\\' ' ' ' ' '|']
         ['|' ' ' '|' ' ' '|' ' ' ' ' '|' ' ' '|' ' ' ' ' '|']
         ['\\' '-' '+' '-' '/' ' ' ' ' '\\' '-' '+' '-' '-' '/']
         [' ' ' ' '\\' '-' '-' '-' '-' '-' '-' '/' ' ' ' ' ' ']]
        >>> trains  # doctest: +NORMALIZE_WHITESPACE
        [Train(x_coor=2, y_coor=0, direction=1, state=0),
         Train(x_coor=9, y_coor=3, direction=2, state=0)]

    """
    tracks = numpy.array(track_string.split("\n"))
    tracks = tracks.view("U1").reshape(len(tracks), -1)
    trains = []
    for y_coor, line in enumerate(tracks):
        for x_coor, char in enumerate(line):
            if char in list("><v^"):
                direction = {"^": 0, ">": 1, "v": 2, "<": 3}[char]
                trains.append(Train(x_coor, y_coor, direction))
    tracks[numpy.isin(tracks, (">","<"))] = "-"
    tracks[numpy.isin(tracks, ("^","v"))] = "|"
    return tracks, trains


def plot_track(
        tracks: numpy.ndarray,
        trains: Sequence[Train] = (),
        crashed: Sequence[Train] = (),
) -> str:
    r"""
    Plots tracks with trains as ASCII art.

    Args:
        tracks:
            Tracks as 2-dimensional char array. Tracks uses the symbols
            '/-\|+' to indicate track properties.
        trains:
            All active trains that has not yet crashed. Will be displayed using
            the characters '<^>v'.
        crashed:
            Trains that have crashed during the current step. Will be displayed
            using the character 'X'

    Returns:
        Track as ASCII art.

    Examples:
        >>> tracks = ("/->-\\        \n|   |  /----\\\n| /-+--+-\\  |\n"
        ...           "| | |  | v  |\n\\-+-/  \\-+--/\n  \\------/   ")
        >>> tracks, trains = extract_trains(tracks)
        >>> print(plot_track(tracks))
        #################
        # /---\         #
        # |   |  /----\ #
        # | /-+--+-\  | #
        # | | |  | |  | #
        # \-+-/  \-+--/ #
        #   \------/    #
        #################
        >>> print(plot_track(tracks, trains=trains))
        #################
        # /->-\         #
        # |   |  /----\ #
        # | /-+--+-\  | #
        # | | |  | v  | #
        # \-+-/  \-+--/ #
        #   \------/    #
        #################
        >>> print(plot_track(tracks, crashed=trains))
        #################
        # /-X-\         #
        # |   |  /----\ #
        # | /-+--+-\  | #
        # | | |  | X  | #
        # \-+-/  \-+--/ #
        #   \------/    #
        #################
    """
    tracks = tracks.copy()
    for x_coor, y_coor, direction, _ in trains:
        direction = {0: "^", 1: ">", 2: "v", 3:"<"}[direction]
        tracks[y_coor, x_coor] = direction
    for x_coor, y_coor, direction, _ in crashed:
        tracks[y_coor, x_coor] = "X"
    out = []
    for line in tracks.view(f"U{len(tracks.T)}"):
        out.append("".join(["# "] + line.tolist() + [" #"]))
    out = ["#"*len(out[0])] + out + ["#"*len(out[0])]
    return "\n".join(out)


def move_trains(
    tracks: numpy.ndarray,
    trains: Sequence[Train],
) -> Iterator[Tuple[Sequence[Train], Sequence[Train]]]:
    r"""
    Make the train move along the train track.

    Args:
        tracks:
            Tracks as 2-dimensional char array. Tracks uses the symbols
            '/-\|+' to indicate track properties.
        trains:
            All active trains that are on the tracks.

    Yields:
        active_trains:
            All trains that has not yet crashed.
        crashed_trains:
            All trains that crashed during the last iteration.

    Examples:
        >>> tracks = ("/->-\\        \n|   |  /----\\\n| /-+--+-\\  |\n"
        ...           "| | |  | v  |\n\\-+-/  \\-+--/\n  \\------/   ")
        >>> tracks, trains = extract_trains(tracks)
        >>> moving_trains = move_trains(tracks, trains)
        >>> print(plot_track(tracks, *next(moving_trains)))
        #################
        # /->-\         #
        # |   |  /----\ #
        # | /-+--+-\  | #
        # | | |  | v  | #
        # \-+-/  \-+--/ #
        #   \------/    #
        #################
        >>> print(plot_track(tracks, *next(moving_trains)))
        #################
        # /-->\         #
        # |   |  /----\ #
        # | /-+--+-\  | #
        # | | |  | |  | #
        # \-+-/  \->--/ #
        #   \------/    #
        #################
        >>> for trains, crashes in moving_trains: pass  # fast forward!
        >>> print(plot_track(tracks, trains, crashes))  # last step
        #################
        # /---\         #
        # |   |  /----\ #
        # | /-+--+-\  | #
        # | | |  X |  | #
        # \-+-/  \-+--/ #
        #   \------/    #
        #################
        >>> trains
        ()
        >>> crashes  # doctest: +NORMALIZE_WHITESPACE
        (Train(x_coor=7, y_coor=3, direction=0, state=0),
         Train(x_coor=7, y_coor=3, direction=2, state=2))
    """
    crashed = []
    trains = trains[:]
    while True:

        # clean up any crash incidences since last step
        if crashed:
            crashed = sorted(crashed)
            for idx, idy in reversed(list(enumerate(crashed))):
                crashed[idx] = trains[idy]
                del trains[idy]

        yield tuple(trains), tuple(crashed)

        if len(trains) <= 1:
            break

        # make the trains move
        crashed = []
        for idx, train in enumerate(trains):

            if idx in crashed:
                continue

            # move train in direction of its nose
            x_coor = train.x_coor + {0: 0, 1: 1, 2: 0, 3: -1}[train.direction]
            y_coor = train.y_coor + {0: -1, 1: 0, 2: 1, 3: 0}[train.direction]
            next_step = tracks[y_coor, x_coor]

            # rotate direction of nose depending on where train lands
            direction = train.direction
            state = train.state
            if next_step == "/":
                direction = {0: 1, 1: 0, 2: 3, 3: 2}[direction]
            elif next_step == "\\":
                direction = {0: 3, 1: 2, 2: 1, 3: 0}[direction]
            elif next_step == "+":
                direction = (direction+state-1) % 4
                state = (state+1) % 3

            # check for crashes
            coords = [(train_.x_coor, train_.y_coor) for train_ in trains]
            if (x_coor, y_coor) in coords:
                crashed.append(idx)
                crashed.append(coords.index((x_coor, y_coor)))

            # update train state
            trains[idx] = Train(x_coor, y_coor, direction, state)


def part1(tracks: numpy.ndarray, trains: Sequence[Train]) -> str:
    """Do part 1 of the assignment."""
    for trains, crashed in move_trains(tracks, trains):
        if crashed:
            return f"{crashed[0].x_coor},{crashed[0].y_coor}"


def part2(tracks: numpy.ndarray, trains: Sequence[Train]) -> str:
    """Do part 2 of the assignment."""
    for trains, crashed in move_trains(tracks, trains):
        pass
    return f"{trains[0].x_coor},{trains[0].y_coor}"


if __name__ == "__main__":
    with open("input") as src:
        TRACKS, TRAINS = extract_trains(src.read().strip())
    print("solution part 1:", part1(TRACKS, TRAINS))
    # solution part 1: 8,9
    print("solution part 2:", part2(TRACKS, TRAINS))
    # solution part 1: 73,33
