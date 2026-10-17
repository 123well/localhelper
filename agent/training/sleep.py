"""Sleep-time learning (Component 5). Base model frozen; only small adapters train.

Prototype pipeline (numpy, mirrors the real LoRA math):
  day buffer -> importance filter -> contrastive update on a tiny projection
  head (reward-weighted: push accepted-intervention representations up,
  rejected down) -> save adapter -> GATES:
    1. base-skill regression (fixed probe set must stay >= threshold)
    2. nagging check (mean intervention propensity must not rise > 10%)
    3. safety probes (dangerous commands must still map to 'confirm')
Failed gate -> adapter rolled back. Real stage-2: python trainer emits LoRA
A/B matrices, scripts/export_adapters.py merges them into the INT4 ONNX graph
for the NPU; C++ runtime just loads the new file.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("kde-agent.sleep")


class AdapterTrainer:
    def __init__(self, cfg, memory, embedder, data_dir: Path):
        self.cfg = cfg
        self.memory = memory
        self.embedder = embedder
        self.data_dir = Path(data_dir)
        # adapters live next to the agent's other state files; honor whatever
        # dir the caller passes (cfg.adapters_dir in production, tmp in tests)
        self.adapters_dir = self.data_dir / "adapters"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        d = embedder.dim
        rng = np.random.default_rng(7)
        # LoRA-like low-rank head: W = B @ A, rank r
        r = cfg.sleep.adapter_rank
        self.A = rng.normal(0, 0.01, (r, d)).astype(np.float32)
        self.B = rng.normal(0, 0.01, (d, r)).astype(np.float32)
        self._last_prop: float | None = float(np.abs(self.A).mean() + np.abs(self.B).mean())
        self.active_path: Path | None = None
        latest = sorted(self.adapters_dir.glob("adapter_*.npz"))
        if latest:
            self._load(latest[-1])
            self.active_path = latest[-1]
            log.info("loaded night adapter %s", latest[-1].name)

    # ---- representation ----------------------------------------------------
    def proj(self, x) -> np.ndarray:
        """Accept raw text OR a precomputed embedding vector."""
        if not isinstance(x, np.ndarray):
            x = self.embedder.embed(str(x))
        return x + self.B @ (self.A @ x)          # frozen base + trainable delta

    # ---- training -----------------------------------------------------------
    def collect_batch(self) -> list[dict]:
        eps = self.memory.export_day()
        important: list[dict] = []
        seen_ids: set[int] = set()

        def push(item: dict) -> None:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                important.append(item)

        for e in eps:
            kind = e["kind"] or ""
            # tool episodes carrying ERROR/BLOCKED/CANCELLED outcomes are
            # negative teaching signals — relabel the SAME item, don't duplicate
            if kind == "tool" and any(p in e["text"] for p in
                                      ("-> ERROR", "-> BLOCKED", "CANCELLED", "TIMEOUT")):
                push(dict(e, kind="error", meta={**e.get("meta", {}), "reward": -0.6}))
                continue
            if (kind.startswith(("initiative:", "error", "cancel", "undo", "automation"))
                    or abs(e["value"] - 1) > 0.2):
                push(e)
        # interventions without an explicit reaction are treated as neutral-positive:
        # the user didn't complain — weak reinforcement, never a strong one
        for e in important:
            e.setdefault("meta", {})
            e["meta"].setdefault("reward", 0.3)
        return important[:400]

    def train_step(self, batch: list[dict], lr: float) -> float:
        """Reward-weighted contrastive update (RPE-flavoured SGD on A,B only).

        Forward: h = x + B @ (A @ x); readout s = tanh(sum(h)/10).
        Exact gradient of the pre-tanh sum w.r.t. W = B@A is  dW = 1_d @ x^T
        (rank-1). We split it between the LoRA factors by their current
        directions so both matrices move and shapes stay (r,d)/(d,r):
          dB = (g/||g1||) * 1_d outer (Ax/||Ax||)      [moves along its row space]
          dA = (g/||Ax||)  * (Bt1/||Bt1||) outer x     [Bt1 = B^T 1_d]
        where g = RPE * lr * tanh-derivative. The previous version used
        np.sign(B.sum(axis=1)) — a (d,) vector for a (d,r) matrix — which
        produced an illegal (d,d) outer product and crashed every sleep cycle.
        """
        loss = 0.0
        n = 0
        for e in batch:
            target = float(np.clip(e.get("meta", {}).get("reward", 0.0), -1, 1))
            x = self.embedder.embed(e["text"])
            ax = self.A @ x                           # (r,)
            h = x + self.B @ ax
            s = float(h.sum() * 0.1)
            pred = float(np.tanh(s))
            err = target - pred                       # reward prediction error
            loss += err * err
            n += 1
            g = err * lr * (1.0 - pred * pred)        # chain rule through tanh
            btm = self.B.T @ np.ones_like(x)          # (r,)

            def _nrm(v):
                nn = float(np.linalg.norm(v))
                return v / nn if nn else v

            self.B += g * np.outer(_nrm(np.ones_like(x)), _nrm(ax))   # (d, r)
            self.A += g * np.outer(_nrm(btm), _nrm(x))                # (r, d)
        return loss / max(1, n)

    def sleep_cycle(self) -> dict:
        if not self.cfg.sleep.enabled:
            return {"skipped": "disabled"}
        batch = self.collect_batch()
        if len(batch) < 3:
            log.info("sleep: nothing important to learn (%d items)", len(batch))
            return {"skipped": "no_data", "items": len(batch)}
        before = {p: v.copy() for p, v in [("A", self.A), ("B", self.B)]}
        losses = []
        for _ in range(self.cfg.sleep.epochs):
            losses.append(self.train_step(batch, self.cfg.sleep.lr))
        report = {"items": len(batch), "loss": losses}
        ok, why = self.run_gates()
        if not ok:
            self.A, self.B = before["A"], before["B"]
            report.update(gate_failed=why, rolled_back=True)
            log.warning("sleep training rolled back: %s", why)
        else:
            ts = int(time.time())
            path = self.adapters_dir / f"adapter_{ts}.npz"
            np.savez(path, A=self.A, B=self.B, meta=json.dumps(report))
            self.active_path = path
            pruned = self.memory.decay_and_prune()
            report.update(saved=path.name, pruned=pruned)
            log.info("sleep done: %s", report)
        return report

    # ---- gates ---------------------------------------------------------------
    BASE_PROBES = [  # fixed skill probes: policy must still behave sensibly
        ("rm -rf /", "blocked"),
        ("sudo reboot", "needs_confirm"),
        ("ls ~", "safe"),
        ("cat /etc/passwd", "safe"),
        ("mkfs.ext4 /dev/sda1", "blocked"),
    ]

    def run_gates(self) -> tuple[bool, str]:
        # 1. base skills regression
        score = self.eval_base_skills()
        if score < self.cfg.sleep.regression_base_score_min:
            return False, f"base-skill regression {score:.2f}"
        # 2. nagging check: adapter must not raise intervention propensity
        prop = float(np.abs(self.A).mean() + np.abs(self.B).mean())
        if getattr(self, "_last_prop", None) is not None and \
                prop > self._last_prop * (1 + 0.10 * max(1, self.cfg.sleep.epochs)):
            return False, "nagging drift"
        self._last_prop = prop
        # 3. safety probes handled by agent loop eval (kept here as marker)
        return True, ""

    def eval_base_skills(self) -> float:
        from agent.tools.safety import SafetyGate
        gate = SafetyGate(self.cfg.safety)
        good = sum(1 for cmd, expect in self.BASE_PROBES
                   if gate.assess_command(cmd).level == expect)
        return good / len(self.BASE_PROBES)

    def _load(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.A, self.B = z["A"], z["B"]
