"""Pure-Python layers: patent data, calibration, amplification chain, protocol."""

import math
import unittest

import tests.nihiline  # noqa: F401  (sys.path)
from kgirl.nihiline import amplification as amp
from kgirl.nihiline import calibration as cal
from kgirl.nihiline.patent import AFP, CEA, CLAIM_RANGES


class PatentDataTests(unittest.TestCase):
    def test_reported_values_are_transcribed(self):
        self.assertEqual((CEA.conc_curve.slope, CEA.conc_curve.intercept), (1308.9, 3084.5))
        self.assertEqual((AFP.conc_curve.slope, AFP.conc_curve.intercept), (1091.5, 4523.8))
        self.assertEqual((CEA.cell_curve.lod, AFP.cell_curve.lod), (165.0, 1000.0))
        self.assertEqual((CEA.per_cell_fg, AFP.per_cell_fg), (37.0, 0.25))
        self.assertEqual(len(CEA.aptamer), 19)
        self.assertEqual(len(AFP.aptamer), 72)
        self.assertEqual(CLAIM_RANGES["tpa_mM"], (20.0, 70.0, 50.0))

    def test_afp_slope_reading_is_the_self_consistent_one(self):
        # 4473 puts the AFP cell LOD at the same scar as CEA; the misprinted 44703 would not.
        self.assertAlmostEqual(cal.scar(AFP.cell_curve), 168.0, places=6)
        self.assertGreater(44703 * 3 - 13251, 1e5)

    def test_molecule_counts(self):
        self.assertAlmostEqual(CEA.molecules_per_cell, 1.238e5, delta=1e3)
        self.assertAlmostEqual(AFP.molecules_per_cell, 2.18e3, delta=20)


class CalibrationTests(unittest.TestCase):
    def test_round_trip_and_flags(self):
        for m in ("CEA", "AFP"):
            for mode in ("conc", "cells"):
                c = cal.curve(m, mode)
                x = math.sqrt(c.lo * c.hi)
                q = cal.quantify(m, cal.signal(c, x), mode)
                self.assertAlmostEqual(q.value, x, delta=1e-9 * x)
                self.assertEqual(q.flag, "ok")
        self.assertEqual(cal.quantify("CEA", 100.0).flag, "below_lod")
        self.assertEqual(cal.quantify("CEA", cal.signal(CEA.conc_curve, 500.0)).flag, "above_range")

    def test_scar_report_finds_the_afp_conc_outlier(self):
        rows = {(r["marker"], r["mode"]): r for r in cal.scar_report()}
        self.assertTrue(rows[("AFP", "conc")]["outlier"])
        self.assertFalse(any(r["outlier"] for k, r in rows.items() if k != ("AFP", "conc")))
        self.assertAlmostEqual(rows[("CEA", "conc")]["scar_signal"], rows[("CEA", "cells")]["scar_signal"], delta=1.0)

    def test_dilution_sinks_into_the_scar(self):
        series = cal.dilution_series("CEA", 1e4, steps=6, mode="cells")
        flags = [d["flag"] for d in series]
        self.assertEqual(flags[:2], ["ok", "ok"])          # 1e4, 1e3 cells/mL
        self.assertEqual(flags[2], "below_lod")             # 100 < 165 cells/mL (CEA LOD)
        self.assertTrue(all(d["true"] > 0 for d in series))  # the stubbornness of 1

    def test_lod_from_blank_and_per_cell(self):
        self.assertAlmostEqual(cal.lod_from_blank("CEA", 100.0, 29.19, "cells"), 165.0, delta=1.0)
        self.assertAlmostEqual(cal.per_cell_expression("CEA", 1e4, 0.37), 37.0)


class AmplificationTests(unittest.TestCase):
    def test_chain_structure_and_limiting_stage(self):
        c = amp.chain("CEA")
        self.assertEqual(c.limiting, "probe packing on the membrane")
        self.assertEqual(amp.chain("AFP").limiting, "antigen count × binding")
        names = [s.name for s in c.stages]
        self.assertEqual(names[0], "antigen molecules")
        self.assertEqual(names[-1], "detected counts")
        anode = names.index("anode oxidations")
        self.assertEqual(c.stages[anode].gain, 1.0)  # closed-BPE charge balance

    def test_lod_scales_inversely_with_gain(self):
        base = amp.chain("AFP")
        doubled = amp.chain("AFP", amp.ChainParams(electroactive_fraction=0.02))
        self.assertLess(doubled.lod_cells_per_ml, base.lod_cells_per_ml)
        e = dict(amp.sensitivity("AFP"))
        self.assertAlmostEqual(e["electroactive_fraction"], -1.0, delta=0.15)

    def test_required_knob_closes_the_gap(self):
        need = amp.required("AFP")
        closed = amp.chain("AFP", amp.ChainParams(electroactive_fraction=need))
        self.assertAlmostEqual(closed.gap_orders, 0.0, delta=0.05)

    def test_poisson_lod_signal(self):
        s = amp.lod_signal(1e4)
        self.assertAlmostEqual(s / math.sqrt(s + 1e4), 3.0, places=6)


class ProtocolTests(unittest.TestCase):
    def test_nine_levels_deterministic_and_recovering(self):
        from kgirl.nihiline.protocol import run_protocol
        a = run_protocol({"CEA": 1e4, "AFP": 1e4}, f_nihil=0.4)
        b = run_protocol({"CEA": 1e4, "AFP": 1e4}, f_nihil=0.4)
        self.assertEqual([lv.n for lv in a.levels], list(range(10)))
        self.assertEqual(a.reads, b.reads)
        for m in ("CEA", "AFP"):
            r = a.reads[m]
            self.assertEqual(r["flag"], "ok")
            self.assertLess(abs(math.log10(r["recovered_cells_per_ml"] / 1e4)), 0.1)
        self.assertAlmostEqual(a.levels[0].facts["TAPB_mmol"], 0.05, delta=0.001)
        self.assertAlmostEqual(a.levels[1].facts["Au_wt_pct"], 3.0, delta=0.05)

    def test_below_lod_reads_are_flagged_not_negative(self):
        from kgirl.nihiline.protocol import run_protocol
        r = run_protocol({"CEA": 50.0, "AFP": 200.0}, f_nihil=0.4).reads
        self.assertTrue(all(v["flag"] == "below_lod" and v["signal"] >= 0 for v in r.values()))


if __name__ == "__main__":
    unittest.main()


class MCPToolTests(unittest.TestCase):
    def test_ecl_tools_through_the_mcp_server(self):
        import tempfile
        from pathlib import Path

        from kgirl.harness.assistant import Assistant
        from kgirl.harness.hermes import Router
        from kgirl.harness.mcp_server import MCPServer
        with tempfile.TemporaryDirectory() as d:
            a = Assistant(home=Path(d), router=Router({"scout": [], "smith": [], "jev": []}), trace=False)
            try:
                srv = MCPServer(a)
                names = {t["name"] for t in srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
                self.assertTrue({"ecl_quantify", "ecl_protocol", "ecl_gap"} <= names)
                res = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                  "params": {"name": "ecl_quantify", "arguments": {"cea": 3084.5, "afp": 100}}})
                text = res["result"]["content"][0]["text"]
                self.assertIn("CEA: 3084.5 a.u. → 1 ng/mL [ok]", text)
                self.assertIn("[below_lod]", text)
                bad = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                  "params": {"name": "ecl_quantify", "arguments": {}}})
                self.assertTrue(bad["result"]["isError"])
            finally:
                a.close()
