"""Low-RAM defaults and streaming builds (#52): SQ8 and callable inputs build the same bytes as the decoded float
rows, write_tier from SQ8 matches the float tier, the nested defaults (1-bit prefix, 8-bit fine suffix tier, 40
candidates) hold, and the fast list training and router switch on at their thresholds, overridable and kept by saves."""
import numpy as np
import pytest

from tieredq import TieredQCoarseTraining, TieredQIVFIndex, TieredQNestedIVFIndex, TieredQRouterConfig

DIMENSIONS = 16
FAST_TRAINING = TieredQCoarseTraining(iterations=4, assignment_supers=256, assignment_probes=16, sample_rows_per_list=40)
MANY_LISTS = TieredQRouterConfig.automatic_lists


def sq8_store(rows, seed=3):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-128, 128, size=(rows, DIMENSIONS), dtype=np.int8)
    scales = (0.01 * rng.uniform(0.5, 1.5, DIMENSIONS) * np.exp(-np.arange(DIMENSIONS) / DIMENSIONS)).astype(np.float32)
    offsets = rng.uniform(-0.5, 0.5, DIMENSIONS).astype(np.float32)
    return codes, scales, offsets


def decoded(codes, scales, offsets):
    return codes.astype(np.float32) * scales + offsets


def memmapped(tmp_path, codes):
    path = tmp_path / "base.i8"
    codes.tofile(path)
    return np.memmap(path, dtype=np.int8, mode="r", shape=codes.shape)


def queries(count=8, seed=11):
    return np.random.default_rng(seed).standard_normal((count, DIMENSIONS)).astype(np.float32) * 0.1


def test_ivf_sq8_and_loader_builds_equal_the_float_build(tmp_path):
    codes, scales, offsets = sq8_store(5000)
    rows = decoded(codes, scales, offsets)
    saved = []
    for name, source in [("float", dict(rows=rows)),
                         ("sq8", dict(codes=memmapped(tmp_path, codes), scales=scales, offsets=offsets)),
                         ("loader", dict(loader=lambda ids: rows[ids], count=len(rows)))]:
        index = TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS, lists=64)
        index.build(**source)
        index.save(tmp_path / f"{name}.tq")
        saved.append((tmp_path / f"{name}.tq").read_bytes())
    assert saved[0] == saved[1] == saved[2]


def test_nested_sq8_build_and_write_tier_equal_the_float_ones(tmp_path):
    codes, scales, offsets = sq8_store(6000)
    rows = decoded(codes, scales, offsets)
    base = memmapped(tmp_path, codes)
    from_float = TieredQNestedIVFIndex.build(rows)
    from_sq8 = TieredQNestedIVFIndex.build(codes=base, scales=scales, offsets=offsets)
    from_loader = TieredQNestedIVFIndex.build(loader=lambda ids: rows[ids], count=len(rows), dimensions=DIMENSIONS)
    assert from_float.serialize() == from_sq8.serialize() == from_loader.serialize()

    from_float.write_tier(tmp_path / "float.rows", rows)
    from_sq8.write_tier(tmp_path / "sq8.rows", codes=base, scales=scales, offsets=offsets)
    assert (tmp_path / "float.rows").read_bytes() == (tmp_path / "sq8.rows").read_bytes()
    np.testing.assert_array_equal(from_float.encode_tier(rows[:100]), from_sq8.encode_tier(rows[:100]))


def test_nested_defaults_and_round_trip(tmp_path):
    codes, scales, offsets = sq8_store(6000)
    rows = decoded(codes, scales, offsets)
    index = TieredQNestedIVFIndex.build(rows)
    assert index.tier_rows == "suffix"
    assert DIMENSIONS <= index.prefix_bits < 2 * DIMENSIONS
    assert 7 * DIMENSIONS <= index.total_bits <= 9 * DIMENSIONS
    assert index.lists == 256 and not index.has_router  # 6,000 rows keep one list per codec centroid
    index.write_tier(tmp_path / "tier.rows", codes=codes, scales=scales, offsets=offsets)
    index.attach_tier(tmp_path / "tier.rows")
    probes = queries()
    defaulted = index.search(probes, k=10, nprobe=8)
    given = index.search(probes, k=10, nprobe=8, candidates=40)
    np.testing.assert_array_equal(defaulted[1], given[1])
    np.testing.assert_array_equal(defaulted[0], given[0])

    index.save(tmp_path / "nested.tq")
    loaded = TieredQNestedIVFIndex.load(tmp_path / "nested.tq")
    assert loaded.serialize() == index.serialize() and loaded.tier_rows == "suffix"
    loaded.attach_tier(tmp_path / "tier.rows")
    np.testing.assert_array_equal(loaded.search(probes, k=10, nprobe=8)[1], defaulted[1])

    explicit = TieredQNestedIVFIndex.build(rows, prefix_bits=2 * DIMENSIONS, total_bits=4 * DIMENSIONS, lists=32,
                                           tier_rows="prefix_and_suffix", fine_tiers=False)
    assert explicit.tier_rows == "prefix_and_suffix" and explicit.lists == 32
    assert explicit.prefix_bits >= 2 * DIMENSIONS and explicit.total_bits <= 4 * DIMENSIONS


def test_coarse_training_switches_at_the_threshold():
    assert TieredQCoarseTraining.automatic(TieredQCoarseTraining.fast_lists - 1) == TieredQCoarseTraining()
    assert TieredQCoarseTraining.automatic(TieredQCoarseTraining.fast_lists) == FAST_TRAINING
    assert TieredQIVFIndex(DIMENSIONS, 32, lists=TieredQCoarseTraining.fast_lists).coarse_training == FAST_TRAINING
    chosen = TieredQCoarseTraining(iterations=9)
    assert TieredQIVFIndex(DIMENSIONS, 32, lists=MANY_LISTS, coarse_training=chosen).coarse_training == chosen
    overridden = TieredQIVFIndex(DIMENSIONS, 32, lists=MANY_LISTS)
    overridden.set_coarse_training(iterations=6, sample_rows_per_list=20)
    assert overridden.coarse_training == TieredQCoarseTraining(iterations=6, sample_rows_per_list=20)
    with pytest.raises(RuntimeError):
        TieredQCoarseTraining(sample_rows_per_list=0)


@pytest.fixture(scope="module")
def many_rows():
    return np.random.default_rng(5).standard_normal((MANY_LISTS + 3000, DIMENSIONS)).astype(np.float32)


def test_router_switches_on_at_the_threshold_and_is_overridable(tmp_path, many_rows):
    automatic = TieredQIVFIndex(DIMENSIONS, 32, lists=MANY_LISTS)
    automatic.build(many_rows)
    assert automatic.has_router
    explicit = TieredQIVFIndex(DIMENSIONS, 32, lists=MANY_LISTS, router=TieredQRouterConfig())
    explicit.build(many_rows)
    probes = queries()
    np.testing.assert_array_equal(automatic.search(probes, nprobe=64)[1], explicit.search(probes, nprobe=64)[1])
    automatic.save(tmp_path / "routed.tq")
    assert TieredQIVFIndex.load(tmp_path / "routed.tq").has_router

    unrouted = TieredQIVFIndex(DIMENSIONS, 32, lists=MANY_LISTS, router=False)
    unrouted.build(many_rows)
    assert not unrouted.has_router
    np.testing.assert_array_equal(unrouted.search(probes, nprobe=64)[1],
                                  automatic.search(probes, nprobe=64, routing="exact")[1])
    automatic.remove_router()
    assert not automatic.has_router

    few = TieredQIVFIndex(DIMENSIONS, 32, lists=MANY_LISTS // 2)
    few.build(many_rows)
    assert not few.has_router
    with pytest.raises(ValueError):
        TieredQIVFIndex(DIMENSIONS, 32, lists=0, router=True)


def test_ivf_lists_are_automatic_until_given():
    codes, scales, offsets = sq8_store(6000)
    index = TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS)
    assert index.lists is None
    with pytest.raises(ValueError):
        index.set_coarse_training(iterations=5)
    index.build(codes=codes, scales=scales, offsets=offsets)
    assert index.lists == 256  # 6,000 rows keep one list per codec centroid
    assert TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS, lists=0).lists == 0


def test_nested_many_lists_route_by_default(many_rows):
    routed = TieredQNestedIVFIndex.build(many_rows, lists=MANY_LISTS)
    assert routed.has_router
    unrouted = TieredQNestedIVFIndex.build(many_rows, lists=MANY_LISTS, router=False,
                                           coarse_training=TieredQCoarseTraining(iterations=2))
    assert not unrouted.has_router


def test_build_input_validation():
    codes, scales, offsets = sq8_store(100)
    with pytest.raises(ValueError):
        TieredQNestedIVFIndex.build(decoded(codes, scales, offsets), codes=codes, scales=scales, offsets=offsets)
    with pytest.raises(ValueError):
        TieredQNestedIVFIndex.build(codes=codes, scales=scales)
    with pytest.raises(ValueError):
        TieredQNestedIVFIndex.build(loader=lambda ids: None, count=100)
    with pytest.raises(ValueError):
        TieredQIVFIndex(DIMENSIONS, 32).build(codes=codes, scales=scales[:4], offsets=offsets[:4])
