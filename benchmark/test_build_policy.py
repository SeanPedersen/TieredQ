"""Prevent wrapper size optimization from silently penalizing a header-only codec."""
import unittest
from build_policy import audit_compile_commands


def commands(extra=()):
    flags = ["c++", "-O3", "-march=native", "-ffp-contract=off"]
    return [{"file": "module.cpp", "arguments": flags + list(extra)},
            {"file": "tieredq.cpp", "arguments": flags}]


class BuildPolicyTests(unittest.TestCase):
    def test_accepts_matching_optimization(self):
        self.assertEqual(len(audit_compile_commands(commands())), 2)

    def test_rejects_late_size_override(self):
        with self.assertRaisesRegex(RuntimeError, "module.cpp: effective optimization"):
            audit_compile_commands(commands(["-Os"]))

    def test_requires_both_codec_translation_units(self):
        with self.assertRaisesRegex(RuntimeError, "both RaBitQ"):
            audit_compile_commands(commands()[:1])


if __name__ == "__main__":
    unittest.main()
