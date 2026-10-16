"""IRT geometry: metric, determinant, rank, curvature by three routes, and the Source C comparison."""

import math
import unittest

import tests.iwme  # noqa: F401

import numpy as np

from kgirl.iwme import irt

P = irt.IRTParams()
PTS = [(0.3, 0.7, 0.9), (1.1, 2.5, 1.7), (2.0, 4.0, 0.4), (0.5, 3.0, 3.1)]


def _sphere_s3(a):
    """Round S³ of radius a in R⁴ (hyperspherical coordinates) for testing the Gauss route on a known case."""
    def X(c):
        x1, x2, x3 = c
        return np.array([a * math.cos(x1), a * math.sin(x1) * math.cos(x2),
                         a * math.sin(x1) * math.sin(x2) * math.cos(x3), a * math.sin(x1) * math.sin(x2) * math.sin(x3)])
    return X


class Metric(unittest.TestCase):
    def test_closed_form_equals_jacobian_gram(self):
        for kern in (irt.CONSTANT_KERNEL, irt.DAMPED_KERNEL):
            for t, f, s in PTS:
                np.testing.assert_allclose(irt.metric_from_jacobian(t, f, s, P, kern), irt.metric(f, s, P, kern), atol=1e-13)

    def test_mixed_component_nonzero_and_theta_block_zero(self):
        g = irt.metric(2.5, 1.7, P, irt.DAMPED_KERNEL)
        self.assertGreater(abs(g[1, 2]), 0.1)
        self.assertEqual(g[0, 1], 0.0)
        self.assertEqual(g[0, 2], 0.0)

    def test_determinant_closed_form(self):
        f, s = np.meshgrid(np.linspace(0, 2 * math.pi, 9), np.linspace(0.1, 3, 7), indexing="ij")
        np.testing.assert_allclose(np.linalg.det(irt.metric(f, s, P, irt.DAMPED_KERNEL)),
                                   irt.det_metric(f, s, P, irt.DAMPED_KERNEL), rtol=1e-11, atol=1e-14)

    def test_published_metric_differs_where_the_audit_says(self):
        g, gp = irt.metric(2.5, 1.7, P), irt.published_metric(2.5, 1.7, P)
        self.assertAlmostEqual(g[0, 0], gp[0, 0], places=14)
        self.assertGreater(abs(g[1, 1] - gp[1, 1]), 1e-3)
        self.assertGreater(abs(g[2, 2] - gp[2, 2]), 1e-3)
        self.assertEqual(gp[1, 2], 0.0)


class RankAndLocus(unittest.TestCase):
    def test_generic_rank_three(self):
        for t, f, s in PTS:
            self.assertEqual(irt.jacobian_rank(t, f, s, P, irt.DAMPED_KERNEL), 3)

    def test_constant_kernel_has_only_the_boundary_folds(self):
        loc = irt.singular_locus(P, irt.CONSTANT_KERNEL, 5.0)
        self.assertEqual([L["kind"] for L in loc], ["boundary fold", "boundary fold"])

    def test_rank_two_exactly_on_the_locus(self):
        for L in irt.singular_locus(P, irt.DAMPED_KERNEL, 4.0):
            self.assertEqual(irt.jacobian_rank(1.0, L["phi"], L["psi"], P, irt.DAMPED_KERNEL, rtol=1e-7), 2)
            self.assertLess(float(irt.det_metric(L["phi"], L["psi"], P, irt.DAMPED_KERNEL)), 1e-20)


class Curvature(unittest.TestCase):
    def test_three_routes_agree(self):
        for kern in (irt.CONSTANT_KERNEL, irt.DAMPED_KERNEL):
            for t, f, s in PTS:
                a = float(irt.scalar_curvature_gauss(t, f, s, P, kern))
                b = irt.scalar_curvature_intrinsic((t, f, s), P, kern)
                c = irt.scalar_curvature_warped(f, s, P, kern)
                self.assertAlmostEqual(b / a, 1.0, delta=1e-5)
                self.assertAlmostEqual(c / a, 1.0, delta=1e-5)

    def test_curvature_is_theta_invariant(self):
        vals = [float(irt.scalar_curvature_gauss(t, 1.3, 1.1, P)) for t in np.linspace(0, 6, 7)]
        self.assertLess(max(vals) - min(vals), 1e-10)

    def test_gauss_route_on_known_manifolds(self):
        # round S³ of radius 2: Scal = 6/a² = 1.5, via finite-difference X_ij on a generic embedding
        a, X = 2.0, _sphere_s3(2.0)
        c, h = np.array([0.9, 1.1, 0.4]), 1e-4
        J = np.stack([(X(c + h * e) - X(c - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)
        H2 = np.zeros((3, 3, 4))
        for i in range(3):
            for j in range(3):
                ei, ej = h * np.eye(3)[i], h * np.eye(3)[j]
                H2[i, j] = (X(c + ei + ej) - X(c + ei - ej) - X(c - ei + ej) + X(c - ei - ej)) / (4 * h * h)
        n = irt._normal(J)
        n = n / np.linalg.norm(n)
        S = np.linalg.solve(J.T @ J, np.einsum("ija,a->ij", H2, n))
        self.assertAlmostEqual(np.trace(S) ** 2 - np.trace(S @ S), 6 / a ** 2, delta=1e-5)

    def test_slice_curvature_matches_standard_torus_limit(self):
        # with e^{−αψ} → 1 and sinh βψ → 1 the slice is the standard torus with K = −cos φ /(r (R − r cos φ))
        p = irt.IRTParams(R=3.0, r=1.0, alpha=1e-12, beta=math.asinh(1.0))
        for f in (0.2, 1.0, 2.0, 3.0):
            K = float(irt.slice_gaussian_curvature(f, 1.0, p))
            self.assertAlmostEqual(K, -math.cos(f) / (1.0 * (3.0 - math.cos(f))), places=9)

    def test_kappa_second_derivative_does_not_enter(self):
        vals = [irt.scalar_curvature_intrinsic((0.0, 0.9, 1.2), P, irt.QuadraticKernel(1.0, 0.3, c2, 1.2)) for c2 in (0.0, 3.0, -5.0)]
        self.assertLess(max(vals) - min(vals), 1e-6 * abs(vals[0]))

    def test_sign_indefinite(self):
        s = irt.curvature_survey(P, irt.CONSTANT_KERNEL, n_phi=61, n_psi=41)
        self.assertGreater(s.frac_negative, 0.2)
        self.assertGreater(s.frac_positive, 0.2)


class KernelAndSourceC(unittest.TestCase):
    def test_shadow_coordinate_is_exact_antiderivative(self):
        k, ps, h = irt.DAMPED_KERNEL, np.linspace(0.05, 4, 40), 1e-6
        np.testing.assert_allclose((k.w(ps + h) - k.w(ps - h)) / (2 * h), -k.k(ps), atol=1e-8)
        self.assertEqual(float(k.w(0.0)), 0.0)

    def test_eq16_is_not(self):
        k = irt.DAMPED_KERNEL
        self.assertNotAlmostEqual(float(k.w_source_c_eq16(0.0)), 0.0)

    def test_kernel_derivatives(self):
        k, ps, h = irt.DAMPED_KERNEL, np.linspace(0.1, 3, 20), 1e-5
        np.testing.assert_allclose((k.k(ps + h) - k.k(ps - h)) / (2 * h), k.k1(ps), atol=1e-8)
        np.testing.assert_allclose((k.k1(ps + h) - k.k1(ps - h)) / (2 * h), k.k2(ps), atol=1e-7)

    def test_kernel_zeros(self):
        np.testing.assert_allclose(irt.DAMPED_KERNEL.zeros(4.0), [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
