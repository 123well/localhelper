"""Registry of SSM candidates for the frozen agent core (spec: 7B SSM, INT4, NPU).

Selection logic per user requirement:
  * reasoning/instruction-following quality matters MORE than world knowledge
    (facts come from RAG only — "no syntax/API knowledge in weights" is a
    DESIGN INVARIANT, so a model that already forgets/is low on factual
    trivia is fine; we need planning + JSON discipline);
  * architecture must be SSM (Mamba-2 / RWKV-6 class) for O(1)-state streaming
    inference on XDNA NPU;
  * license must allow local modification/fine-tuning (Apache-2.0 preferred);
  * conversion path to ONNX must exist or be scriptable.

NOTE: as of late 2025/early 2026 there is NO Apache-2.0 Mamba-2 7B base with
strong reasoning. The practical options are:
  A) Falcon-H1-R1-7B-Instruct (TTL license) — Mamba-2 hybrid + RL reasoning;
  B) NeoGRITE-Mamba2-7b-SFT (Apache-2.0) — plain Mamba-2 7B SFT, needs our own
     reasoning distillation at sleep time (LoRA), which the spec ALLOWS
     (model frozen, adapters trainable);
  C) SFR-Mamba-2-RM-DPO-7B (Apache-2.0) — DPO-aligned Mamba-2 7B; better
     instruction following than raw pretrain, weaker multi-step reasoning;
  D) Qwen3-8B GGUF on CPU (transformer fallback) — best reasoning available,
     but violates the NPU-first constraint; kept as emergency fallback only;
  E) Mamba-mimized Llama-3.1-8B (MIT) — transformer distilled INTO Mamba
     recurrent state; interesting research artifact, ONNX export unproven.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    id: str
    repo: str                      # HF org/repo
    family: str                    # mamba2 | rwkv6 | transformer-fallback
    params_b: float
    license: str
    quant_target: str              # int4 | q4_k
    onnx_path: str                 # ready | convertible | research
    reasoning: str                 # known reasoning/instruct quality signal
    pros: tuple = ()
    cons: tuple = ()
    weight_gb_int4: float = 4.5    # rough size on disk after quantization


CANDIDATES: dict[str, Candidate] = {c.id: c for c in [
    Candidate(
        id="falcon-h1-r1-7b",
        repo="tiiuae/Falcon-H1-R1-7B-Instruct",
        family="mamba2", params_b=7.0, license="Falcon-TTL",
        quant_target="int4", onnx_path="convertible",
        reasoning="Mamba-2 hybrid trained with RL (DeepSeek-R1-style): "
                  "top-tier SSM chain-of-thought; explicit long-CoT instruct variant",
        pros=("best reasoning among SSMs", "hybrid keeps attention layers small",
              "official BF16 safetensors -> easy quantization"),
        cons=("TTL license (fine for personal use)",
              "hybrid attention => slightly more complex ONNX graph for XDNA"),
    ),
    Candidate(
        id="neogrite-mamba2-7b",
        repo="griptdn/NeoGRITE-Mamba2-7b-SFT",
        family="mamba2", params_b=7.0, license="apache-2.0",
        quant_target="int4", onnx_path="convertible",
        reasoning="SFT-only Mamba-2 7B; modest reasoning out of the box — "
                  "planned boost via sleep-time LoRA distillation of reasoning traces",
        pros=("pure Apache-2.0", "plain Mamba-2 => simplest ONNX/XDNA export",
              "matches spec's 'distill into adapters, keep base frozen' plan"),
        cons=("needs our reasoning-distillation curriculum before production use",),
    ),
    Candidate(
        id="sfr-mamba2-dpo-7b",
        repo="Salesforce/SFR-Mamba-2-RM-DPO-7B",
        family="mamba2", params_b=7.0, license="apache-2.0",
        quant_target="int4", onnx_path="convertible",
        reasoning="DPO post-trained Mamba-2 7B: solid instruction following and "
                  "JSON formatting discipline; weaker multi-step CoT than Falcon-H1-R1",
        pros=("Apache-2.0", "better format adherence than SFT-only models"),
        cons=("reward-model lineage, not a pure chat model; needs eval"),
    ),
    Candidate(
        id="qwen3-8b-cpu",
        repo="Qwen/Qwen3-8B-GGUF",
        family="transformer-fallback", params_b=8.0, license="apache-2.0",
        quant_target="q4_k", onnx_path="n/a",
        reasoning="strongest reasoning in 8B class overall; transformer KV-cache "
                  "makes it CPU/GPU-side only — violates NPU-first, kept as fallback",
        pros=("drop-in llama.cpp backend already stubbed in agent/core/model.py",),
        cons=("not SSM", "not NPU", "KV cache grows with context"),
    ),
    Candidate(
        id="llama-mimized-mamba",
        repo="seal-03/Mamba2-mimized-Llama-3.1-8B",
        family="mamba2", params_b=8.0, license="mit",
        quant_target="int4", onnx_path="research",
        reasoning="Llama-3.1 reasoning distilled into Mamba recurrent states; "
                  "export tooling immature — track, do not ship yet",
        pros=("MIT", "reasoning quality of teacher in SSM student"),
        cons=("research-grade ONNX path", "conversion scripts unverified"),
    ),
]}

# priority order used when the user does not pick explicitly
_DEFAULT_ORDER = ("falcon-h1-r1-7b", "neogrite-mamba2-7b", "sfr-mamba2-dpo-7b")


def pick_candidate(prefer: str = "") -> Candidate:
    """Return candidate by id, else first default that is licensed for local use."""
    if prefer:
        if prefer not in CANDIDATES:
            raise KeyError(f"unknown model id '{prefer}'; have: {sorted(CANDIDATES)}")
        return CANDIDATES[prefer]
    for cid in _DEFAULT_ORDER:
        return CANDIDATES[cid]
    return CANDIDATES["qwen3-8b-cpu"]


RECIPES: dict[str, dict] = {
    "falcon-h1-r1-7b": {
        "download": "huggingface-cli download tiiuae/Falcon-H1-R1-7B-Instruct --local-dir models/falcon-h1-r1-bf16",
        "quantize": "python scripts/quantize_ssm.py --in models/falcon-h1-r1-bf16 --out models/mamba2-7b-int4.onnx --format int4",
        "tokenizer": "copy tokenizer.json from the BF16 dir next to the ONNX file",
        "note": "TTL allows private modification; keep weights out of git.",
    },
    "neogrite-mamba2-7b": {
        "download": "huggingface-cli download griptdn/NeoGRITE-Mamba2-7b-SFT --local-dir models/neogrite-bf16",
        "quantize": "python scripts/quantize_ssm.py --in models/neogrite-bf16 --out models/mamba2-7b-int4.onnx --format int4",
        "tokenizer": "copy tokenizer.json from the BF16 dir next to the ONNX file",
        "note": "after deploy run sleep-trainer reasoning-distillation curriculum (LoRA).",
    },
    "sfr-mamba2-dpo-7b": {
        "download": "huggingface-cli download Salesforce/SFR-Mamba-2-RM-DPO-7B --local-dir models/sfr-dpo-bf16",
        "quantize": "python scripts/quantize_ssm.py --in models/sfr-dpo-bf16 --out models/mamba2-7b-int4.onnx --format int4",
        "tokenizer": "copy tokenizer.json from the BF16 dir next to the ONNX file",
        "note": "evaluate JSON-contract adherence before switching default.",
    },
    "qwen3-8b-cpu": {
        "download": "huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf --local-dir models/",
        "quantize": "n/a (already GGUF)",
        "tokenizer": "built into GGUF; set model.backend='cpu'",
        "note": "CPU fallback only; expect ~8-15 tok/s on Ryzen AI cores.",
    },
    "llama-mimized-mamba": {
        "download": "see repo README (research release)",
        "quantize": "unsupported yet",
        "tokenizer": "Llama-3.1 tokenizer",
        "note": "watchlist candidate.",
    },
}


def recipe_for(candidate_id: str) -> dict:
    return RECIPES[candidate_id]
