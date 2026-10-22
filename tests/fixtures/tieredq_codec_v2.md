# TieredQ v2 codec fixture

A small persisted recipe-2 model with a 16-dimensional identity basis and pinned
trained arrays. Roundtrip tests require exact serialized bytes. Tests mutate the
recipe selector to ensure older recipes are rejected without a legacy loader.
