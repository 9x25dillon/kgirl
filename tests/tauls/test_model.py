import unittest
from dataclasses import replace

import tests.tauls  # noqa: F401  (sys.path)

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

V = 300


def configs():
    from kgirl.tauls.model import PRESETS
    t = replace(PRESETS["tiny"], vocab_size=V)
    return {
        "tauls": t,
        "tauls-gqa": replace(t, n_kv_heads=2),
        "baseline": replace(PRESETS["tiny-baseline"], vocab_size=V),
        "hrm": replace(PRESETS["tiny-hrm"], vocab_size=V),
        "hrm-tauls": replace(t, h_layers=1, h_cycles=2, l_cycles=3),
        "stablemax": replace(t, loss="stablemax"),
    }


def model(cfg):
    from kgirl.tauls.model import TAULS
    m = TAULS(cfg).eval()
    with torch.no_grad():   # zero-init gains would make the TA-ULS paths trivially pass these tests
        for n, p in m.named_parameters():
            if n.endswith(("kfp.weight", "ctrl.weight", "ctrl.bias", "mix", "recur.weight")):
                p.copy_(torch.linspace(-0.5, 0.5, p.numel()).view_as(p))
    return m


def batch(S=40):
    return torch.tensor([[(7 * i + 3) % V for i in range(S)], [(11 * i + 5) % V for i in range(S)]])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class InvariantTests(unittest.TestCase):
    """The three properties a causal LM must have, for every architecture mode."""

    def test_causal_batch_independent_and_cache_exact(self):
        for name, cfg in configs().items():
            with self.subTest(name), torch.no_grad():
                m = model(cfg)
                idx = batch()
                full = m(idx).logits
                changed = idx.clone()
                changed[0, 25:] = 1
                self.assertEqual((m(changed).logits[0, :25] - full[0, :25]).abs().max().item(), 0.0)
                self.assertLess((m(idx[1:]).logits[0] - full[1]).abs().max().item(), 1e-6)
                cache = m.new_cache()
                parts = [m(idx[:, :10], cache=cache).logits, m(idx[:, 10:23], cache=cache).logits]
                parts += [m(idx[:, t:t + 1], cache=cache).logits for t in range(23, 40)]
                self.assertLess((torch.cat(parts, 1) - full).abs().max().item(), 1e-5)

    def test_chunked_loss_and_grad_checkpoint_match_the_plain_path(self):
        for name in ("tauls", "hrm", "stablemax"):
            with self.subTest(name):
                m = model(configs()[name]).train()
                idx = batch(64)
                y = torch.roll(idx, -1, 1)

                def run(**kw):
                    m.zero_grad()
                    loss = m(idx, y, **kw).loss
                    loss.backward()
                    return loss.detach(), [p.grad.clone() for p in m.parameters()]
                l1, g1 = run()
                l2, g2 = run(logits=False)
                m.grad_checkpoint = True
                l3, g3 = run(logits=False)
                m.grad_checkpoint = False
                self.assertLess((l1 - l2).abs().item(), 1e-5)
                self.assertLess((l1 - l3).abs().item(), 1e-5)
                for a, b, c in zip(g1, g2, g3):
                    self.assertLess((a - b).abs().max().item(), 1e-5)
                    self.assertLess((a - c).abs().max().item(), 1e-5)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class ArchitectureTests(unittest.TestCase):
    def test_tauls_paths_start_as_identity(self):
        from kgirl.tauls.model import TAULS
        m = TAULS(configs()["tauls"])
        b = m.blocks[0]
        x = torch.randn(2, 5, m.cfg.d_model)
        self.assertTrue(torch.equal(b.kfp(x), x))
        self.assertTrue(torch.equal(b.regulator(x, torch.ones(2, 5, 1)), x))
        self.assertEqual(float(torch.sigmoid(b.control.mix).mean()), 0.5)

    def test_init_is_chaos_deterministic_and_order_independent(self):
        from kgirl.tauls.model import TAULS
        a, b = TAULS(configs()["tauls"]), TAULS(configs()["tauls"])
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            self.assertTrue(torch.equal(p, q), n)
        c = TAULS(replace(configs()["tauls"], init_seed=0.2718))
        self.assertFalse(torch.equal(a.embed.weight, c.embed.weight))
        self.assertAlmostEqual(float(a.embed.weight.std()), 0.02, delta=0.002)
        deeper = TAULS(replace(configs()["tauls"], n_layers=3))     # adding a block leaves others' init alone
        self.assertTrue(torch.equal(a.blocks[1].attn.qkv.weight, deeper.blocks[1].attn.qkv.weight))

    def test_param_formula_and_matched_baselines(self):
        from kgirl.tauls.model import PRESETS, TAULS, block_params
        for name in ("tiny", "tiny-baseline", "small", "small-baseline"):
            m = TAULS(PRESETS[name])
            self.assertEqual(sum(p.numel() for p in m.blocks[1].parameters()), block_params(PRESETS[name]), name)
        for t, b in (("small", "small-baseline"), ("desk-8gb", "desk-8gb-baseline")):
            ratio = block_params(PRESETS[b]) / block_params(PRESETS[t])
            self.assertAlmostEqual(ratio, 1.0, delta=0.02)
            self.assertFalse(PRESETS[b].two_level or PRESETS[b].kfp or PRESETS[b].entropy_reg)

    def test_hierarchical_schedule_and_one_step_gradient(self):
        from kgirl.tauls.model import TAULS
        cfg = configs()["hrm"]
        m = TAULS(cfg)
        self.assertEqual([s for s, _ in cfg.schedule()], ["L", "L", "H", "L", "L", "H"])
        self.assertEqual(len(m.new_cache().layers), 6)                 # 4 L applications + 2 H, 1 block each
        seen = []
        hook = m.blocks[0].register_forward_hook(lambda mod, inp, out: seen.append(out[0].requires_grad))
        m.train()
        m(batch(16), torch.roll(batch(16), -1, 1), logits=False).loss.backward()
        hook.remove()
        self.assertEqual(seen, [False, False, False, True])            # only the last L application is graded
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_stablemax_is_a_distribution_and_generation_works(self):
        from kgirl.tauls import chaos
        from kgirl.tauls.model import log_stablemax
        x = torch.tensor([[-50.0, -1.0, 0.0, 2.0, 300.0]])
        self.assertAlmostEqual(float(log_stablemax(x).exp().sum()), 1.0, places=5)
        for name in ("tauls", "hrm", "stablemax"):
            m = model(configs()[name])
            a = m.generate([1, 2, 3], 12, sampler=chaos.Sampler(0.3))
            b = m.generate([1, 2, 3], 12, sampler=chaos.Sampler(0.3))
            self.assertEqual(a, b)
            self.assertEqual(len(a), 12)
            self.assertTrue(all(0 <= t < V for t in a))
            g = m.generate([1, 2, 3], 5, temperature=0)
            self.assertEqual(g[0], int(m(torch.tensor([[1, 2, 3]])).logits[0, -1].argmax()))


if __name__ == "__main__":
    unittest.main()
