import random
from collections import Counter
from PIL import Image

from synthdata.balance import oversample, augment


def test_oversample_returns_budget():
    minority = [f"photo_{i}" for i in range(15)]
    budget = 55
    rng = random.Random(42)
    chosen = oversample(minority, budget, rng)
    assert len(chosen) == budget
    assert set(chosen) <= set(minority)
    
    # Check that photos are repeated as evenly as possible (3 or 4 times each)
    counts = Counter(chosen)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert min(counts.values()) == 3
    assert max(counts.values()) == 4


def test_augment_returns_pil_image():
    img = Image.new("RGB", (256, 256), (120, 150, 200))
    res = augment(img)
    assert isinstance(res, Image.Image)
    assert res.size == img.size

