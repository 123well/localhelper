"""Character-level tokenizer JSON for the tiny prototype ONNX model.

WordLevel tokenizers mangle punctuation/paths (JSON contract needs quotes,
braces, slashes intact), so the test graph uses a ByteLevel-style vocab:
every printable ASCII char is its own token. Decode = concat chars.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_tokenizer_json(out_path: Path) -> Path:
    alphabet = [chr(c) for c in range(32, 127)]          # printable ASCII
    specials = {"<unk>": 0, "</s>": len(alphabet) + 1}
    vocab = {ch: i + 1 for i, ch in enumerate(alphabet)}
    vocab.update(specials)
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": specials["</s>"], "content": "</s>",
                          "single_token": True, "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "(?s)."},
                          "behavior": "isolated"},
        "post_processor": None,
        "decoder": {"type": "Bytelevel"},   # placeholder replaced below
        "model": {"type": "ByteLevel", "vocab": vocab, "unk_token": "<unk>",
                  "add_prefix_space": False, "trim_offsets": False,
                  "use_regex": False},
    }
    # ByteLevel model in `tokenizers` expects specific fields; to stay robust
    # across versions we actually use WordLevel semantics with char tokens and
    # a Fuse decoder (chars are already single tokens, spaces must survive):
    tok["model"] = {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    tok["decoder"] = {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": ["\u241F", "\u241E"], "content": ""}]}
    # simplest correct choice for char tokens joined by space during decode:
    tok["decoder"] = {"type": "WordPiece", "separator": "", "continue_subword": ""}
    out_path.write_text(json.dumps(tok))
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(build_tokenizer_json(Path(a.out)))
