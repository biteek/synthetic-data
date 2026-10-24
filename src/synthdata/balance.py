"""Meeting 1: the two classical ways to add Egyptian Mau training examples.

Both functions are yours to write; `uv run synth-train --method oversample` or
`--method augment` uses them.
"""

import random

from PIL import Image
from torchvision import transforms


def oversample(
    minority_photos: list[str], budget: int, rng: random.Random
) -> list[str]:
    """Choose which Egyptian Mau training photos to repeat.

    Return a list of exactly `budget` names taken from `minority_photos`; names
    may repeat. Repeat the photos as evenly as possible and use `rng` for every
    random choice, so a run can be reproduced from its seed.
    """
    full_repeats = budget // len(minority_photos)
    remainder = budget % len(minority_photos)
    chosen = list(minority_photos) * full_repeats
    if remainder > 0:
        chosen.extend(rng.sample(minority_photos, remainder))
    rng.shuffle(chosen)
    return chosen


AUGMENT_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ]
)


def augment(image: Image.Image) -> Image.Image:
    """Return a randomly transformed copy of one Egyptian Mau training photo.

    The training script calls this every time an added copy is loaded, so each
    epoch sees a new variant. Typical choices are crops, flips, rotations, and
    colour changes from `torchvision.transforms`. Keep the cat recognisable.
    """
    return AUGMENT_TRANSFORMS(image)
