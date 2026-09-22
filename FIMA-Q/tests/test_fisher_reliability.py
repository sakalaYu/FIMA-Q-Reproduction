"""CPU tests for drift detection; run with unittest from the FIMA-Q directory."""
import unittest
from utils.fisher_reliability import FisherReliability


class ReliabilityTests(unittest.TestCase):
    def test_tracks_improving_kl_without_trigger(self):
        monitor = FisherReliability()
        monitor.reset([1., 2., 3.], [0.1, 0.2, 0.3])
        result = monitor.observe([0.5, 1., 1.5], [0.05, 0.1, 0.15])
        self.assertFalse(result['trigger'])
        self.assertAlmostEqual(result['relative_error'], 0)

    def test_frozen_scale_detects_drift_after_patience(self):
        monitor = FisherReliability()
        monitor.reset([1., 2.], [0.1, 0.2])
        self.assertFalse(monitor.observe([1., 2.], [0.3, 0.6])['trigger'])
        self.assertTrue(monitor.observe([1., 2.], [0.3, 0.6])['trigger'])
        self.assertAlmostEqual(monitor.scale, 0.1)

    def test_opposite_trend_triggers_even_below_error_threshold(self):
        monitor = FisherReliability(error_threshold=100, patience=1)
        monitor.reset([1., 2.], [0.1, 0.2])
        result = monitor.observe([0.8, 1.6], [0.12, 0.24])
        self.assertTrue(result['trend_bad'])
        self.assertTrue(result['trigger'])

    def test_good_probe_and_refresh_reset_patience(self):
        monitor = FisherReliability()
        monitor.reset([1., 2.], [0.1, 0.2])
        monitor.observe([1., 2.], [0.3, 0.6])
        self.assertEqual(monitor.observe([1., 2.], [0.1, 0.2])['bad_count'], 0)
        monitor.reset([1., 2.], [0.3, 0.6])
        self.assertFalse(monitor.observe([1., 2.], [0.3, 0.6])['trigger'])

    def test_zero_and_invalid_inputs(self):
        monitor = FisherReliability()
        monitor.reset([0., 0.], [0., 0.])
        self.assertFalse(monitor.observe([0., 0.], [0., 0.])['trigger'])
        for values in ([float('nan')], [-1.], [float('inf')]):
            with self.assertRaises(ValueError):
                monitor.reset(values, [0.])
        with self.assertRaises(ValueError):
            monitor.reset([], [])


if __name__ == '__main__':
    unittest.main()
