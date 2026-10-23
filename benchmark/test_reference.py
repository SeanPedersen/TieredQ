"""Compare adapter IDs with an independent native caller using official defaults."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import codec_native

BUILD = Path(codec_native.__file__).parent


class ReferenceTests(unittest.TestCase):
    def test_official_defaults_match_adapter_for_all_budgets_and_metrics(self):
        rng = np.random.default_rng(23)
        os.environ["BENCH_SEED"] = "0"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for d in (128,200,1536):
                base = rng.normal(size=(1025,d)).astype(np.float32)
                queries = rng.normal(size=(35,d)).astype(np.float32)
                base.tofile(root/"base.f32")
                queries.tofile(root/"query.f32")
                for metric in ("l2","ip"):
                    for bits in (1,2,4,8):
                        subprocess.run([str(BUILD/"codec_reference"),str(root/"base.f32"),str(root/"query.f32"),
                                        str(root/"ids.u32"),str(len(base)),str(d),str(len(queries)),str(bits),metric,"1"],
                                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        expected=np.fromfile(root/"ids.u32",dtype=np.uint32).reshape(len(queries),10)
                        index=codec_native.Index(base,bits,0,False,metric=="l2")
                        actual=np.empty_like(expected)
                        index.search(queries,actual,8)
                        np.testing.assert_array_equal(actual,expected,err_msg=f"{d=} {metric=} {bits=}")
        self.assertTrue(codec_native.rabitq_high_accuracy)
        self.assertTrue(codec_native.rabitq_implicit_ids)
        self.assertEqual(codec_native.rabitq_container,"direct-flat")


if __name__ == "__main__":
    unittest.main()
