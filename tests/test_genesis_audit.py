"""Standard unittest / pytest entrypoint for Project Genesis-Rigorous 17-Test Master Audit."""

import unittest

from genesis_rigorous.verification import run_master_verification


class TestGenesisRigorousAudit(unittest.TestCase):
    def test_master_17_mathematical_and_gauge_invariance_suite(self):
        summary = run_master_verification(prefer_cuda=True)
        failed = [tid for tid, r in summary["tests"].items() if not r["passed"]]
        self.assertTrue(
            summary["all_passed"],
            msg=f"Failed mathematical audit tests: {failed}",
        )
        self.assertEqual(summary["passed_count"], 17)
        self.assertEqual(summary["total_count"], 17)


if __name__ == "__main__":
    unittest.main()
