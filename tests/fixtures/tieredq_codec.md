# Persisted codec fixture

A model with a 16-dimensional identity basis and pinned trained arrays.
Roundtrip tests require exact serialized bytes. Tests mutate the format
selector to verify explicit rejection of unsupported formats.
