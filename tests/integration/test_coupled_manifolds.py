"""Two coupled manifolds (measured channel) and the Hardy semantic-field ontology (conceptual, L3)."""
import unittest

import numpy as np

import tests.integration  # noqa: F401  (sys.path)


class CoupledManifoldTests(unittest.TestCase):
    def test_port_reproduces_the_original_sweep(self):
        from kgirl.cognitive.coupled_manifolds import sweep
        s = sweep((0.0, 0.10, 0.80))
        self.assertAlmostEqual(s[0.0]["R_sync"], 0.3983, places=4)       # values printed by the original script
        self.assertAlmostEqual(s[0.1]["R_sync"], 0.8309, places=4)
        self.assertAlmostEqual(s[0.8]["intimacy"], 0.0019, places=4)     # amplitude death at strong coupling
        self.assertGreater(s[0.1]["mutual_information_bits"], s[0.0]["mutual_information_bits"])

    def test_information_crosses_only_through_amplitude_and_only_when_coupled(self):
        from kgirl.cognitive.coupled_manifolds import chaos_bits, transmit
        bits = chaos_bits(64)
        self.assertEqual(sum(bits), 32)
        ask = transmit(0.20, bits, "ask")
        control = transmit(0.0, bits, "ask")
        psk = transmit(0.20, bits, "psk")
        self.assertLess(ask.ber, 0.35)                                   # 18/64 errors: p ≈ 3e−4 under chance
        self.assertGreater(ask.mutual_information_bits_per_symbol, 0.1)
        self.assertGreaterEqual(control.ber, 0.45)                        # γ = 0: B cannot depend on the bits
        self.assertGreaterEqual(psk.ber, 0.45)                            # the coupling γ|Ψ_A| is phase-blind


class NonlocalOntologyTests(unittest.TestCase):
    def test_demo_and_rules(self):
        from kgirl.cognitive import nonlocal_consciousness as nc
        r = nc.demo()
        self.assertEqual(r["focus"], nc.FocusState.QST)
        self.assertTrue(r["telepathy"] and r["bio_pk"])
        self.assertEqual(r["sygons"][0].process, nc.PsiProcess.TELEPATHY)
        self.assertEqual(r["sygons"][0].endpoints, ("Alice", "Bob"))
        a, b = nc.SemanticField("a", 2.0), nc.SemanticField("b", 1.0)
        self.assertFalse(nc.Hyperdimension().nonlocal_connect(a, b, nc.PsiProcess.TELEPATHY))   # no link, no sygon
        with self.assertRaises(ValueError):
            a.link_to(b, 1.5)
        iss = nc.InfiniteSpiralStaircase.from_fibonacci(5)
        self.assertTrue(np.allclose(np.diff(np.log(iss.bows)), np.log(nc.PHI)))


if __name__ == "__main__":
    unittest.main()
