"""Choose an integer code budget using serving storage alone, without measuring recall."""
from dataclasses import dataclass

TARGET_LOW = 0.99
TARGET_HIGH = 0.995
MAX_ATTEMPTS = 16
MIN_CODE_BITS = 1


@dataclass
class Selection:
    code_bits: int
    built: object
    attempts: list
    in_target_band: bool


def initial_budget(cap, anchors, maximum):
    first, second = anchors[:2]
    slope = (second['bytes'] - first['bytes']) / (second['code_bits'] - first['code_bits'])
    estimate = first['code_bits'] + (cap * TARGET_HIGH - first['bytes']) / slope
    return min(maximum, max(MIN_CODE_BITS, round(estimate)))


def select_budget(cap, anchors, build_index, record_attempt, maximum):
    lower, upper = MIN_CODE_BITS, maximum
    budget = initial_budget(cap, anchors, maximum)
    attempts = []
    best = None
    visited = set()
    for _ in range(MAX_ATTEMPTS):
        if budget in visited or lower > upper:
            break
        visited.add(budget)
        built = build_index(budget)
        attempt = dict(code_bits=budget, bytes_per_vector=built.bytes_per_vector)
        attempts.append(attempt)
        record_attempt(attempt)
        if built.bytes_per_vector <= cap and (best is None or built.bytes_per_vector > best[1]):
            best = (budget, built.bytes_per_vector)
        if cap * TARGET_LOW <= built.bytes_per_vector <= cap * TARGET_HIGH:
            return Selection(budget, built, attempts, True)
        if built.bytes_per_vector > cap * TARGET_HIGH:
            upper = budget - 1
        else:
            lower = budget + 1
        del built
        budget = (lower + upper) // 2
    if best is None:
        raise RuntimeError('No measured positive code budget fits the storage cap')
    budget = best[0]
    built = build_index(budget)
    assert built.bytes_per_vector == best[1]
    return Selection(budget, built, attempts, False)
