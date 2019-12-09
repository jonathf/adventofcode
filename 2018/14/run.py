"""
--- Day 14: Chocolate Charts ---

You finally have a chance to look at all of the produce moving around.
Chocolate, cinnamon, mint, chili peppers, nutmeg, vanilla... the Elves must be
growing these plants to make hot chocolate! As you realize this, you hear
a conversation in the distance. When you go to investigate, you discover two
Elves in what appears to be a makeshift underground kitchen/laboratory.

The Elves are trying to come up with the ultimate hot chocolate recipe; they're
even maintaining a scoreboard which tracks the quality score (0-9) of each
recipe.

Only two recipes are on the board: the first recipe got a score of 3, the
second, 7. Each of the two Elves has a current recipe: the first Elf starts
with the first recipe, and the second Elf starts with the second recipe.

To create new recipes, the two Elves combine their current recipes. This
creates new recipes from the digits of the sum of the current recipes' scores.
With the current recipes' scores of 3 and 7, their sum is 10, and so two new
recipes would be created: the first with score 1 and the second with score 0.
If the current recipes' scores were 2 and 3, the sum, 5, would only create one
recipe (with a score of 5) with its single digit.

The new recipes are added to the end of the scoreboard in the order they are
created. So, after the first round, the scoreboard is 3, 7, 1, 0.

After all new recipes are added to the scoreboard, each Elf picks a new current
recipe. To do this, the Elf steps forward through the scoreboard a number of
recipes equal to 1 plus the score of their current recipe. So, after the first
round, the first Elf moves forward 1 + 3 = 4 times, while the second Elf moves
forward 1 + 7 = 8 times. If they run out of recipes, they loop back around to
the beginning. After the first round, both Elves happen to loop around until
they land on the same recipe that they had in the beginning; in general, they
will move to different recipes.

Drawing the first Elf as parentheses and the second Elf as square brackets,
they continue this process:

(3)[7]
(3)[7] 1  0
 3  7  1 [0](1) 0
 3  7  1  0 [1] 0 (1)
(3) 7  1  0  1  0 [1] 2
 3  7  1  0 (1) 0  1  2 [4]
 3  7  1 [0] 1  0 (1) 2  4  5
 3  7  1  0 [1] 0  1  2 (4) 5  1
 3 (7) 1  0  1  0 [1] 2  4  5  1  5
 3  7  1  0  1  0  1  2 [4](5) 1  5  8
 3 (7) 1  0  1  0  1  2  4  5  1  5  8 [9]
 3  7  1  0  1  0  1 [2] 4 (5) 1  5  8  9  1  6
 3  7  1  0  1  0  1  2  4  5 [1] 5  8  9  1 (6) 7
 3  7  1  0 (1) 0  1  2  4  5  1  5 [8] 9  1  6  7  7
 3  7 [1] 0  1  0 (1) 2  4  5  1  5  8  9  1  6  7  7  9
 3  7  1  0 [1] 0  1  2 (4) 5  1  5  8  9  1  6  7  7  9  2

The Elves think their skill will improve after making a few recipes (your
puzzle input). However, that could take ages; you can speed this up
considerably by identifying the scores of the ten recipes after that. For
example:

    - If the Elves think their skill will improve after making 9 recipes, the
      scores of the ten recipes after the first nine on the scoreboard would be
      5158916779 (highlighted in the last line of the diagram).
    - After 5 recipes, the scores of the next ten would be 0124515891.
    - After 18 recipes, the scores of the next ten would be 9251071085.
    - After 2018 recipes, the scores of the next ten would be 5941429882.

What are the scores of the ten recipes immediately after the number of recipes
in your puzzle input?

--- Part Two ---

As it turns out, you got the Elves' plan backwards. They actually want to know
how many recipes appear on the scoreboard to the left of the first recipes
whose scores are the digits from your puzzle input.

    - 51589 first appears after 9 recipes.
    - 01245 first appears after 5 recipes.
    - 92510 first appears after 18 recipes.
    - 59414 first appears after 2018 recipes.

How many recipes appear on the scoreboard to the left of the score sequence in
your puzzle input?
"""


def calculate_score(
    inputs: int,
    stop_on_match: bool = False,
    digits: int = 10,
) -> int:
    """
    Calculate scores for the hot chocolate recipe game.

    Args:
        inputs:
            Depending on `stop_on_match`, defines the number of rounds or the
            value which the score has to reach to declare stop.
        stop_on_match:
            If true, stop when a match of `input` is reached. If false, stop
            after `inputs` number of rounds.
        digits:
            The number of digits to use in match.

    Returns:
        If `stop_on_match`, return the number of rounds played. Otherwise,
        return the value after `inputs` rounds have been played.

    Examples:
        >>> calculate_score(9)
        5158916779
        >>> calculate_score(51589, stop_on_match=True, digits=5)
        9
        >>> calculate_score(5)
        124515891
        >>> calculate_score(1245, stop_on_match=True, digits=5)
        5
        >>> calculate_score(18)
        9251071085
        >>> calculate_score(92510, stop_on_match=True, digits=5)
        18
        >>> calculate_score(2018)
        5941429882
        >>> calculate_score(59414, stop_on_match=True, digits=5)
        2018

    """
    scoreboard = [3, 7]
    player1 = 0
    player2 = 1
    play_round = 0
    while True:

        # play the game
        while len(scoreboard) < play_round+10:
            score = scoreboard[player1] + scoreboard[player2]
            if score >= 10:
                scoreboard.append(score//10)
            scoreboard.append(score % 10)

            player1 = (player1+scoreboard[player1]+1) % len(scoreboard)
            player2 = (player2+scoreboard[player2]+1) % len(scoreboard)

        number = sum(count*10**idx for idx, count in enumerate(
            reversed(scoreboard[play_round:play_round+digits])))

        # check if stop criteria has been reached
        if stop_on_match:
            if number == inputs:
                return play_round
        elif play_round == inputs:
            return number

        play_round += 1
    return number


def part1():
    """Do part 1 of the assignment."""
    return calculate_score(157901)
def part2():
    """Do part 2 of the assignment."""
    return calculate_score(157901, stop_on_match=True, digits=6)

if __name__ == "__main__":
    print("solution part 1:", part1())
    # solution part 1: 9411137133
    print("solution part 2:", part2())
    # solution part 2: 20317612

