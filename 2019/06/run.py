"""
--- Day 6: Universal Orbit Map ---

You've landed at the Universal Orbit Map facility on Mercury. Because
navigation in space often involves transferring between orbits, the orbit maps
here are useful for finding efficient routes between, for example, you and
Santa. You download a map of the local orbits (your puzzle input).

Except for the universal Center of Mass (COM), every object in space is in
orbit around exactly one other object. An orbit looks roughly like this:

                  \
                   \
                    |
                    |
AAA--> o            o <--BBB
                    |
                    |
                   /
                  /

In this diagram, the object BBB is in orbit around AAA. The path that BBB takes
around AAA (drawn with lines) is only partly shown. In the map data, this
orbital relationship is written AAA)BBB, which means "BBB is in orbit around
AAA".

Before you use your map data to plot a course, you need to make sure it wasn't
corrupted during the download. To verify maps, the Universal Orbit Map facility
uses orbit count checksums - the total number of direct orbits (like the one
shown above) and indirect orbits.

Whenever A orbits B and B orbits C, then A indirectly orbits C. This chain can
be any number of objects long: if A orbits B, B orbits C, and C orbits D, then
A indirectly orbits D.

For example, suppose you have the following map:

COM)B
B)C
C)D
D)E
E)F
B)G
G)H
D)I
E)J
J)K
K)L

Visually, the above map of orbits looks like this:

        G - H       J - K - L
       /           /
COM - B - C - D - E - F
               \
                I

In this visual representation, when two objects are connected by a line, the
one on the right directly orbits the one on the left.

Here, we can count the total number of orbits as follows:

    - D directly orbits C and indirectly orbits B and COM, a total of 3 orbits.
    - L directly orbits K and indirectly orbits J, E, D, C, B, and COM, a total
      of 7 orbits.
    - COM orbits nothing.

The total number of direct and indirect orbits in this example is 42.

What is the total number of direct and indirect orbits in your map data?

--- Part Two ---

Now, you just need to figure out how many orbital transfers you (YOU) need to
take to get to Santa (SAN).

You start at the object YOU are orbiting; your destination is the object SAN is
orbiting. An orbital transfer lets you move from any object to an object
orbiting or orbited by that object.

For example, suppose you have the following map:

COM)B
B)C
C)D
D)E
E)F
B)G
G)H
D)I
E)J
J)K
K)L
K)YOU
I)SAN

Visually, the above map of orbits looks like this:

                          YOU
                         /
        G - H       J - K - L
       /           /
COM - B - C - D - E - F
               \
                I - SAN

In this example, YOU are in orbit around K, and SAN is in orbit around I. To
move from K to I, a minimum of 4 orbital transfers are required:

    - K to J
    - J to E
    - E to D
    - D to I

Afterward, the map of orbits looks like this:

        G - H       J - K - L
       /           /
COM - B - C - D - E - F
               \
                I - SAN
                 \
                  YOU

What is the minimum number of orbital transfers required to move from the
object YOU are orbiting to the object SAN is orbiting? (Between the objects
they are orbiting - not between YOU and SAN.)
"""
from typing import Dict, List, Tuple
from collections import defaultdict


def count_orbits(
    name: str,
    graph: Dict[str, List[str]],
    parent_count: int = 0,
) -> int:
    """
    Recursively count the number of direct and indirect orbits.

    Simple sum of itself and sum of its children, where each child is worth one
    more than the parent.

    Args:
        name:
            Name of the current item.
        graph:
            A one-to-many directed graph where keys are name of parents, and
            values are names of all its children.
        parent_count:
            The number of direct and indirect orbits to current item.

    Returns:
        The sum of all direct and indirect orbits for all (connected) items in
        `graph`.

    Examples:
        >>> graph = {"COM": ["B"], "B": ["C", "G"], "G": ["H"], "H": [],
        ...          "C": ["D"], "D": ["E", "I"], "I": ["SAN"], "SAN": [],
        ...          "E": ["F", "J"], "F": [], "J": ["K"],
        ...          "K": ["L", "YOU"], "L": [], "YOU": []}
        >>> count_orbits("COM", graph=graph)
        54
        >>> count_orbits("E", graph=graph)
        10
        >>> count_orbits("I", graph=graph)
        1

    """
    return sum(count_orbits(child, graph, parent_count+1)
               for child in graph[name]) + parent_count


def find_santa(name: str, graph: Dict[str, List[str]]) -> Tuple[str, int]:
    """
    Recursively locate items "YOU" and "SAN" and sum up the number of transfers
    needed between them.

    Args:
        name:
            Name of the current item.
        graph:
            A one-to-many directed graph where keys are name of parents, and
            values are names of all its children.

    Returns:
        status:
            String representing what has been found. Either "" (nothing is
            found), "SAN" (Santa is found), "YOU" (you are found), or "SANYOU"
            (both you and Santa are found).
        count:
            If both Santa and you are found, then return the number of orbital
            transfers needed to get you and santa in the same orbit. If only
            one of them are found, return the number of orbital transfers back
            to start. Else return 0.

    Examples:
        >>> graph = {"COM": ["B"], "B": ["C", "G"], "G": ["H"], "H": [],
        ...          "C": ["D"], "D": ["E", "I"], "I": ["SAN"], "SAN": [],
        ...          "E": ["F", "J"], "F": [], "J": ["K"], "K": ["L", "YOU"],
        ...          "L": [], "YOU": []}
        >>> find_santa("COM", graph)
        ('SANYOU', 4)
        >>> find_santa("E", graph)
        ('YOU', 3)
        >>> find_santa("I", graph)
        ('SAN', 1)
        >>> find_santa("G", graph)
        ('', 0)

    """
    # Leaf handle
    if not graph[name]:
        # only consider YOU and SAN
        status = name if name in ("SAN", "YOU") else ""
        return status, 0

    # gather results from children
    statuses, counts = zip(*[
        find_santa(child, graph) for child in graph[name]])
    status = "".join(sorted(statuses))
    # add to count only if SAN or YOU is found, but not both
    count = sum(counts) + (status in ("SAN", "YOU"))
    return status, count

















if __name__ == "__main__":

    GRAPH = defaultdict(list)
    with open("input") as src:
        for line in src.read().strip().split():
            center, periphery = line.split(")")
            graph[center].append(periphery)

    print("solution part 1:", count_orbits("COM", GRAPH))
    # solution part 1: 117672
    print("solution part 2:", find_santa("COM", GRAPH)[1])
    # solution part 2: 277
