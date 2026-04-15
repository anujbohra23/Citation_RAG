import random
from typing import List


def paired_bootstrap_mean_diff(a: List[float], b: List[float], n_samples: int = 1000, seed: int = 42):
    rng = random.Random(seed)
    diffs = []
    pairs = list(zip(a, b))
    for _ in range(n_samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        diffs.append(sum(x - y for x, y in sample) / len(sample))
    diffs.sort()
    return {
        'mean_diff': sum(diffs) / len(diffs),
        'ci_95': (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]),
    }
