"""Fast-only construction owns one representation and persists its search/append semantics."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tieredq import TieredQIndex

DIMENSIONS = 32
INITIAL_ROWS = 257
CODE_WORD_BITS = 32
CODE_WORD_BYTES = 4
CODE_BLOCK_ROWS = 16
CODE_BLOCK_WORD_BYTES = CODE_WORD_BYTES * CODE_BLOCK_ROWS
# High-bit tiers plus trailing bits in each row's last packed code word.
PACKED_CODE_BITS = DIMENSIONS * 7 + 4
PACKED_CODE_KIND = 4
FAST_METADATA_BYTES = 2 * np.dtype(np.float16).itemsize + np.dtype(np.uint8).itemsize


def fast_codes_offset(payload, padded_rows):
    import struct

    header = struct.calcsize("<III")
    array_lengths = struct.calcsize("<QQQQ")
    return payload + header + array_lengths + padded_rows * FAST_METADATA_BYTES


def assert_same(actual, expected):
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


# Fast and full training yield the same model only at 8 bits/dim with exact_fast_scores and no fine tiers (full mode
# trains without them): fast targets allocate on
# variance^1.5 and fill nibble padding (#9), so the 1-bit budget compares the fast index with its own fast scan instead.
@pytest.mark.parametrize("code_bits", [DIMENSIONS, DIMENSIONS * 8])
def test_fast_only_search_storage_append_and_roundtrip(code_bits, tmp_path):
    same_model = code_bits == DIMENSIONS * 8
    rows = np.random.default_rng(13).normal(size=(INITIAL_ROWS * 2, DIMENSIONS)).astype(np.float32)
    fast = TieredQIndex(
        dimensions=DIMENSIONS, code_bits=code_bits, mode="fast", exact_fast_scores=same_model, fine_tiers=False
    )
    exact = TieredQIndex(dimensions=DIMENSIONS, code_bits=code_bits, mode="full")
    assert fast.mode == "fast" and exact.mode == "full"
    assert fast.storage_bytes == 0
    with pytest.raises(AttributeError):
        fast.mode = "full"
    fast.add(rows[:INITIAL_ROWS])
    exact.add(rows[:INITIAL_ROWS])
    planes_bytes = exact.storage_bytes
    exact.build_fast_scan()
    if not exact.has_fast_scan:
        pytest.skip("reference cache unavailable on this CPU")
    assert fast.has_fast_scan
    assert not same_model or fast.storage_bytes == exact.storage_bytes - planes_bytes
    before = fast.storage_bytes
    fast.build_fast_scan()
    assert fast.storage_bytes == before
    queries = rows[:17]
    for k in (0, 1, 10, len(fast) + 1):
        assert_same(fast.search(queries, k=k), fast.search_fast(queries, k=k))
        if same_model:
            assert_same(fast.search(queries, k=k), exact.search_fast(queries, k=k))
            assert_same(fast.search(queries[0], k=k), exact.search_fast(queries[0], k=k))
    for name, index in (("exact", exact), ("fast", fast)):
        path = str(tmp_path / f"{name}.tq")
        index.save(path)
        loaded = TieredQIndex.load(path)
        assert loaded.mode == index.mode
        assert_same(loaded.search(queries), index.search(queries))
    path = str(tmp_path / "fast.tq")
    loaded = TieredQIndex.load(path)
    for index in (fast, loaded, exact):
        index.add(rows[INITIAL_ROWS:])
    exact.build_fast_scan()
    assert loaded.has_fast_scan and fast.has_fast_scan
    if same_model:
        assert_same(fast.search(queries), exact.search_fast(queries))
    assert_same(loaded.search(queries), fast.search(queries))
    loaded.save(path)
    assert_same(TieredQIndex.load(path).search(queries), fast.search(queries))


def test_failed_append_empty_shapes_and_concurrent_calls():
    rows = np.random.default_rng(31).normal(size=(INITIAL_ROWS, DIMENSIONS)).astype(np.float32)
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=64, mode="fast")
    index.add(rows)
    expected = index.search(rows[:3])
    storage = index.storage_bytes
    index.add(rows[:0])
    with pytest.raises(RuntimeError):
        index.add(np.full_like(rows[:1], np.nan))
    assert len(index) == INITIAL_ROWS and index.storage_bytes == storage
    assert_same(index.search(rows[:3]), expected)
    assert index.search(rows[:0])[0].shape == (0, 10)
    invalid = rows[:17].copy()
    invalid[-1, 0] = np.inf
    with pytest.raises(RuntimeError, match="nonfinite"):
        index.search(invalid)

    def operation(task):
        if task % 3 == 0:
            index.add(rows[:1])
        else:
            scores, ids = index.search(rows[:3])
            assert scores.shape == ids.shape == (3, 10)
            assert np.isfinite(scores).all()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(operation, range(18)))
    assert len(index) == INITIAL_ROWS + 6 and index.has_fast_scan


# Budgets whose widest learned tier has 5 or 6 bits (wider budgets add 7-bit tiers).
MID_WIDTH_CODE_BITS = [DIMENSIONS * 4 + 8, DIMENSIONS * 5 + 16]


@pytest.mark.parametrize("code_bits", MID_WIDTH_CODE_BITS)
def test_mid_width_models_store_packed_codes(code_bits, tmp_path):
    """Tiers wider than the 4-bit lookup cache scan exact packed code rows on every backend."""
    import struct

    rows = np.random.default_rng(37).normal(size=(INITIAL_ROWS, DIMENSIONS)).astype(np.float32)
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=code_bits, mode="fast", exact_fast_scores=True)
    index.add(rows)
    assert index.bytes_per_vector == -(-code_bits // CODE_WORD_BITS) * CODE_WORD_BYTES + FAST_METADATA_BYTES
    path = tmp_path / "index.tq"
    index.save(str(path))
    data = path.read_bytes()
    payload = struct.unpack_from("<Q", data, 32)[0]
    assert struct.unpack_from("<I", data, payload + 4)[0] == PACKED_CODE_KIND


def test_l2_wide_models_keep_packed_codes():
    rows = np.random.default_rng(29).normal(size=(INITIAL_ROWS, DIMENSIONS)).astype(np.float32)
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=DIMENSIONS * 5, mode="fast", metric="l2")
    index.add(rows)
    assert index.bytes_per_vector == -(-index.code_bits // CODE_WORD_BITS) * CODE_WORD_BYTES + FAST_METADATA_BYTES


def rewrite_payload_checksum(data):
    import struct

    payload_offset = struct.unpack_from("<Q", data, 32)[0]
    value = 14695981039346656037
    for byte in data[payload_offset:]:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    struct.pack_into("<Q", data, 64, value)


@pytest.mark.parametrize("corruption", ["version", "kind", "level_max", "length", "nonfinite", "centroid", "code_padding", "padded_code", "trailing"])
def test_rejects_malformed_fast_only_files(tmp_path, corruption):
    import struct

    rows = np.random.default_rng(41).normal(size=(INITIAL_ROWS, DIMENSIONS)).astype(np.float32)
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=PACKED_CODE_BITS, mode="fast", exact_fast_scores=True)
    index.add(rows)
    path = tmp_path / "index.tq"
    index.save(str(path))
    data = bytearray(path.read_bytes())
    payload = struct.unpack_from("<Q", data, 32)[0]
    padded = struct.unpack_from("<Q", data, payload + 12)[0]
    if corruption == "version":
        struct.pack_into("<I", data, payload, 999)
    elif corruption == "kind":
        struct.pack_into("<I", data, payload + 4, 999)
    elif corruption == "level_max":
        struct.pack_into("<I", data, payload + 8, 255)
    elif corruption == "length":
        struct.pack_into("<Q", data, payload + 12, padded + 1)
    elif corruption == "nonfinite":
        struct.pack_into("<H", data, payload + 20, 0x7E00)  # binary16 NaN
    elif corruption == "centroid":
        data[payload + 36 + padded * 2 * np.dtype(np.float16).itemsize] = 255
    elif corruption == "code_padding":
        last_word = PACKED_CODE_BITS // CODE_WORD_BITS
        data[fast_codes_offset(payload, padded) + last_word * CODE_BLOCK_WORD_BYTES + CODE_WORD_BYTES - 1] = 255
    elif corruption == "padded_code":
        data[-1] = 255
    else:
        data.append(0)
        struct.pack_into("<Q", data, 40, len(data) - payload)
    rewrite_payload_checksum(data)
    path.write_bytes(data)
    with pytest.raises(RuntimeError):
        TieredQIndex.load(str(path))


def foreign_fast_file(tmp_path):
    """Create a valid representation that the local native fast kernel cannot consume."""
    import platform
    import struct

    if platform.machine().lower() in ("arm64", "aarch64"):
        pytest.skip("NEON consumes every persisted fast representation natively")
    dimensions, rows = 8, 32
    vectors = np.random.default_rng(59).normal(size=(rows, dimensions)).astype(np.float32)
    index = TieredQIndex(dimensions=dimensions, code_bits=dimensions * 8, mode="fast", exact_fast_scores=True)
    index.add(vectors)
    path = tmp_path / "foreign.tq"
    index.save(str(path))
    data = bytearray(path.read_bytes())
    payload = struct.unpack_from("<Q", data, 32)[0]
    # 8-bit packed code words have the signed-level payload's length; zero levels are valid
    # inside the wider ARM quantization range, which the x86 kernels cannot consume.
    struct.pack_into("<I", data, payload + 4, 2)
    struct.pack_into("<I", data, payload + 8, 127)
    padded = struct.unpack_from("<Q", data, payload + 12)[0]
    codes = fast_codes_offset(payload, padded)
    data[codes:] = bytes(len(data) - codes)
    rewrite_payload_checksum(data)
    path.write_bytes(data)
    return path, vectors


def test_scalar_warning_once_on_load_and_not_on_search_or_append(tmp_path):
    import warnings

    path, vectors = foreign_fast_file(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = TieredQIndex.load(str(path))
        assert index.uses_scalar_fast_scan
        expected = index.search(vectors[:3])
        assert_same(index.search(vectors[:3]), expected)
        index.add(vectors[:1])
        index.search(vectors[:3])
    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning
    assert "scalar scorer" in str(caught[0].message)
    assert "Rebuild" in str(caught[0].message)


def test_scalar_warning_obeys_warning_as_error(tmp_path):
    import warnings

    path, _ = foreign_fast_file(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="scalar scorer"):
            TieredQIndex.load(str(path))


@pytest.mark.parametrize("mode", ["full", "fast"])
def test_native_and_exact_indexes_do_not_warn(tmp_path, mode):
    import warnings

    vectors = np.random.default_rng(61).normal(size=(32, 8)).astype(np.float32)
    index = TieredQIndex(dimensions=8, code_bits=16, mode=mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index.add(vectors)
        if index.uses_scalar_fast_scan:
            pytest.skip("CPU has no native fast kernel")
        path = str(tmp_path / "native.tq")
        index.save(path)
        loaded = TieredQIndex.load(path)
        assert not loaded.uses_scalar_fast_scan
        loaded.search(vectors[:1])
    assert not caught


def test_each_loaded_scalar_index_can_warn(tmp_path):
    import warnings

    path, _ = foreign_fast_file(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = TieredQIndex.load(str(path))
        second = TieredQIndex.load(str(path))
    assert first.uses_scalar_fast_scan and second.uses_scalar_fast_scan
    assert len(caught) == 2
