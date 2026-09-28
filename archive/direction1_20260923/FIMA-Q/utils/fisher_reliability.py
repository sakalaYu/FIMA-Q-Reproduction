"""Calibration-only DPLR reliability controller; no inference-time modules.

Fits a positive scale once per Fisher version, then tests subsequent probes
against that frozen mapping. Refitting on every probe would hide drift.
This module uses only the standard library so its decision logic can be tested
on machines without CUDA/PyTorch.
"""
import math


class FisherReliability:
    def __init__(self, error_threshold=0.25, trend_threshold=0.02, patience=2):
        if not math.isfinite(error_threshold) or error_threshold <= 0:
            raise ValueError('error_threshold must be finite and positive')
        if not math.isfinite(trend_threshold) or not 0 < trend_threshold < 1:
            raise ValueError('trend_threshold must be between zero and one')
        if patience < 1:
            raise ValueError('patience must be positive')
        self.error_threshold = error_threshold
        self.trend_threshold = trend_threshold
        self.patience = patience
        self.ready = False
        self.bad_count = 0

    @staticmethod
    def _validate(proxy, kl):
        if not proxy or len(proxy) != len(kl):
            raise ValueError('proxy and KL must be non-empty paired samples')
        if any(not math.isfinite(x) or x < 0 for x in proxy + kl):
            raise ValueError('proxy and KL must be finite and nonnegative')

    def _error(self, proxy, kl):
        numerator = sum((self.scale * x - y) ** 2 for x, y in zip(proxy, kl))
        return math.sqrt(numerator / max(sum(y * y for y in kl), 1e-24))

    def reset(self, proxy, kl):
        proxy, kl = list(proxy), list(kl)
        self._validate(proxy, kl)
        self.scale = max(sum(x * y for x, y in zip(proxy, kl)) /
                         max(sum(x * x for x in proxy), 1e-24), 1e-12)
        self.anchor_error = self._error(proxy, kl)
        self.previous = (sum(proxy) / len(proxy), sum(kl) / len(kl))
        self.bad_count = 0
        self.ready = True

    def observe(self, proxy, kl):
        proxy, kl = list(proxy), list(kl)
        self._validate(proxy, kl)
        if not self.ready:
            raise RuntimeError('reset must be called after a Fisher update')
        proxy_mean, kl_mean = sum(proxy) / len(proxy), sum(kl) / len(kl)
        error = self._error(proxy, kl)
        previous_proxy, previous_kl = self.previous
        trend_bad = (proxy_mean < previous_proxy * (1 - self.trend_threshold)
                     and kl_mean > max(previous_kl, 1e-12) * (1 + self.trend_threshold))
        error_bad = error > self.anchor_error + self.error_threshold
        self.bad_count = self.bad_count + 1 if error_bad or trend_bad else 0
        self.previous = proxy_mean, kl_mean
        return dict(proxy_mean=proxy_mean, kl_mean=kl_mean, scale=self.scale,
                    relative_error=error, anchor_error=self.anchor_error,
                    trend_bad=trend_bad, error_bad=error_bad,
                    bad_count=self.bad_count, trigger=self.bad_count >= self.patience)
