"""
--- Day 8: Space Image Format ---

The Elves' spirits are lifted when they realize you have an opportunity to
reboot one of their Mars rovers, and so they are curious if you would spend
a brief sojourn on Mars. You land your ship near the rover.

When you reach the rover, you discover that it's already in the process of
rebooting! It's just waiting for someone to enter a BIOS password. The Elf
responsible for the rover takes a picture of the password (your puzzle input)
and sends it to you via the Digital Sending Network.

Unfortunately, images sent via the Digital Sending Network aren't encoded with
any normal encoding; instead, they're encoded in a special Space Image Format.
None of the Elves seem to remember why this is the case. They send you the
instructions to decode it.

Images are sent as a series of digits that each represent the color of a single
pixel. The digits fill each row of the image left-to-right, then move downward
to the next row, filling rows top-to-bottom until every pixel of the image is
filled.

Each image actually consists of a series of identically-sized layers that are
filled in this way. So, the first digit corresponds to the top-left pixel of
the first layer, the second digit corresponds to the pixel to the right of that
on the same layer, and so on until the last digit, which corresponds to the
bottom-right pixel of the last layer.

For example, given an image 3 pixels wide and 2 pixels tall, the image data
123456789012 corresponds to the following image layers:

Layer 1: 123
         456

Layer 2: 789
         012

The image you received is 25 pixels wide and 6 pixels tall.

To make sure the image wasn't corrupted during transmission, the Elves would
like you to find the layer that contains the fewest 0 digits. On that layer,
what is the number of 1 digits multiplied by the number of 2 digits?

--- Part Two ---

Now you're ready to decode the image. The image is rendered by stacking the
layers and aligning the pixels with the same positions in each layer. The
digits indicate the color of the corresponding pixel: 0 is black, 1 is white,
and 2 is transparent.

The layers are rendered with the first layer in front and the last layer in
back. So, if a given position has a transparent pixel in the first and second
layers, a black pixel in the third layer, and a white pixel in the fourth
layer, the final image would have a black pixel at that position.

For example, given an image 2 pixels wide and 2 pixels tall, the image data
0222112222120000 corresponds to the following image layers:

Layer 1: 02
         22

Layer 2: 11
         22

Layer 3: 22
         12

Layer 4: 00
         00

Then, the full image can be found by determining the top visible pixel in each
position:

    - The top-left pixel is black because the top layer is 0.
    - The top-right pixel is white because the top layer is 2 (transparent),
      but the second layer is 1.
    - The bottom-left pixel is white because the top two layers are 2, but the
      third layer is 1.
    - The bottom-right pixel is black because the only visible pixel in that
      position is 0 (from layer 4).

So, the final image looks like this:

01
10

What message is produced after decoding your image?
"""
from typing import Tuple
import numpy


def collapse_layers(picture_layers: numpy.ndarray) -> numpy.ndarray:
    """
    Collapse layers into a single pixel layer.

    Args:
        picture_layers:
            Picture layers shaped as ``(number_of_layers, number_of_pixels)``.

    Returns:
        Boolean array of pixels, with shape ``(number_of_pixels,)``, and where
        True is the color white, and False is black.

    Examples:
        >>> layers = numpy.array(
        ...     list("0222112222120000"), dtype=int).reshape(-1, 4)
        >>> collapse_layers(layers)
        array([False,  True,  True, False])

    """
    pixels = numpy.repeat(2, picture_layers.shape[-1])
    for layer in picture_layers:
        index = pixels == 2
        pixels[index] = layer[index]
    return pixels.astype(bool)


def plot_pixels(
        pixels: numpy.ndarray,
        shape: Tuple[int, int],
        whites: str = "#",
        blacks: str = " ",
) -> str:
    """
    Convert pixel layer into more human readable ASCII pixel art.

    Args:
        pixels:
            A flat boolean array indicating white and black as True and False.
        shape:
            The shape of the picture to reshape the output to.
        whites:
            The symbol used to indicate white color in the art.
        blacks:
            The symbol used to indicate black color in the art.

    Return:
        ASCII pixel art printing out the picture.

    Examples:
        >>> pixels = numpy.array([0, 1, 1, 0], dtype=bool)
        >>> print(plot_pixels(pixels, shape=(2, 2), blacks="."))
        .#
        #.
    """
    picture = numpy.repeat(blacks, len(pixels))
    picture[pixels] = whites
    picture = picture.reshape(shape)
    return "\n".join("".join(row) for row in picture)


def part1(picture_layers: numpy.ndarray) -> int:
    """Do part 1 of the assignment."""
    index = numpy.argmin(numpy.sum(picture_layers == 0, -1))
    ones = numpy.sum(picture_layers[index] == 1)
    twos = numpy.sum(picture_layers[index] == 2)
    return ones*twos


def part2(picture_layers: numpy.ndarray) -> str:
    """Do part 1 of the assignment."""
    pixels = collapse_layers(picture_layers)
    return plot_pixels(pixels, (6, 25))


if __name__ == "__main__":
    with open("input") as src:
        PICTURE_LAYERS = numpy.array(list(src.read().strip()), dtype=int)
    PICTURE_LAYERS = PICTURE_LAYERS.reshape(-1, 150)
    print("solution part 1:", part1(PICTURE_LAYERS))
    # solution part 1: 828
    print("solution part 2:\n", part2(PICTURE_LAYERS))
    #### #    ###    ## ####
       # #    #  #    # #
      #  #    ###     # ###
     #   #    #  #    # #
    #    #    #  # #  # #
    #### #### ###   ##  #
