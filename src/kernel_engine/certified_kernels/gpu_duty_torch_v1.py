#!/usr/bin/env python3
"""Duty cycling for a torch step loop (the same contract as the Warp variant in gpu_lbm_luftflode_v1.py).

A saturating GPU workload leaves nothing for the desktop compositor. Duty cycling hands back
(1 - andel) of the wall-clock time.

The only difference from the Warp variant is wp.synchronize() -> torch.cuda.synchronize(). The
synchronise is required: without it the measured block time is queue time, not GPU time, because the
CUDA launch is asynchronous, so the sleep would be scaled against a block time that is already wrong.

andel >= 1.0 makes steg() a no-op; the default path is unchanged.

I/O: GpuDutyTorch(andel, block_ms, synk); call steg() once per iteration; rapport() returns a dict of
requested/achieved duty, block counts, GPU time, actual and requested sleep time, and block-time
median/p95.
"""
from __future__ import annotations

import os
import time


def _env(namn, standard, typ=float):
    v = os.environ.get(namn)
    if v is None or v == "":
        return standard
    try:
        return typ(v)
    except ValueError:
        return standard


class GpuDutyTorch:
    def __init__(self, andel=None, block_ms=None, synk=None):
        if andel is None:
            andel = _env("GPU_ANDEL", 1.0)
        if block_ms is None:
            block_ms = _env("GPU_ANDEL_BLOCK_MS", 12.0)
        self.andel = float(andel)
        self.mal_block_s = max(1e-4, float(block_ms) / 1000.0)
        self.pa = 0.0 < self.andel < 1.0
        if synk is None:
            import torch
            synk = torch.cuda.synchronize
        self._synk = synk
        self.n_block = 1
        self._i = 0
        self._t0 = None
        self.gpu_tid_s = 0.0
        self.sovtid_s = 0.0
        self.begard_somn_s = 0.0
        self.n_block_klara = 0
        self.n_steg = 0
        self.block_tider = []

    def steg(self):
        if not self.pa:
            return
        if self._t0 is None:
            self._t0 = time.time()
        self._i += 1
        self.n_steg += 1
        if self._i < self.n_block:
            return
        self._synk()
        t_block = time.time() - self._t0
        self.gpu_tid_s += t_block
        self.n_block_klara += 1
        n_i = self._i
        somn = t_block * (1.0 - self.andel) / self.andel
        if somn > 0.0:
            _s0 = time.perf_counter()
            time.sleep(somn)   # time.sleep, not a busy wait -- a busy wait moves the load to the CPU
            self.sovtid_s += time.perf_counter() - _s0   # actual sleep, not the requested one
            self.begard_somn_s += somn
        t_steg = t_block / max(1, n_i)
        self.n_block = int(max(1, min(200000, round(self.mal_block_s / max(t_steg, 1e-9)))))
        if len(self.block_tider) < 4000:
            self.block_tider.append(round(t_block * 1000.0, 3))
        self._i = 0
        self._t0 = time.time()

    def rapport(self):
        tot = self.gpu_tid_s + self.sovtid_s
        bt = sorted(self.block_tider)
        return {
            "gpu_andel_bard": self.andel,
            "aktiv": bool(self.pa),
            "mal_block_ms": self.mal_block_s * 1000.0,
            "n_block_slut": self.n_block,
            "n_block_klara": self.n_block_klara,
            "n_steg": self.n_steg,
            "gpu_tid_s": round(self.gpu_tid_s, 3),
            "sovtid_s_faktisk": round(self.sovtid_s, 3),
            "sovtid_s_begard": round(self.begard_somn_s, 3),
            "blocktid_ms_median": bt[len(bt) // 2] if bt else None,
            "blocktid_ms_p95": bt[int(0.95 * (len(bt) - 1))] if bt else None,
            "uppnadd_andel": round(self.gpu_tid_s / tot, 4) if tot > 0 else None,
        }
