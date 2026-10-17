"""Configuration following XDG standards (Arch Linux friendly).

Config:   ~/.config/kde-agent/config.json
Data:     ~/.local/share/kde-agent/
Logs:     ~/.local/state/kde-agent/agent.log
Secrets:  never stored in plaintext -> KDE Wallet (org.kde.kwalletd6)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def xdg_state_home() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))


APP_NAME = "kde-agent"


@dataclass
class ModelConfig:
    backend: str = "auto"          # auto | npu | cpu | llamacpp
    # 7B SSM (Mamba-2 family), INT4 quantized, minimal factual knowledge,
    # max reasoning/tool-use capability. Shipped as ONNX for the Ryzen AI NPU
    # (amdxdna driver + Vitis AI EP of ONNX Runtime) or GGUF for CPU debug.
    npu_model_path: str = ""       # e.g. data_dir/model/mamba2-7b-int4.onnx
    cpu_model_path: str = ""       # e.g. data_dir/model/mamba2-7b-q4.gguf
    max_tokens: int = 512
    temperature: float = 0.3
    context_budget_tokens: int = 2048   # deliberately small: energy + streaming SSM
    device: str = "npu"


@dataclass
class SafetyConfig:
    blacklist: list[str] = field(default_factory=lambda: [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "> /dev/sda",
        "chmod -R 777 /", "userdel", "passwd", "systemctl poweroff",
    ])
    whitelist: list[str] = field(default_factory=lambda: [
        "ls", "cat", "echo", "pwd", "whoami", "date", "df", "du", "uname",
        "git status", "git log", "pacman -Q",
    ])
    require_confirm_prefixes: list[str] = field(default_factory=lambda: [
        "sudo ", "rm ", "mv ", "dd ", "mkfs", "chmod ", "chown ", "kill",
        "pkill", "systemctl", "pacman -R", "pacman -U", "yay -R",
        "git push --force", "git reset --hard", "shutdown", "reboot",
        "curl ", "wget ", "| sh", "| bash",
    ])
    max_files_in_one_op: int = 50
    undo_enabled: bool = True


@dataclass
class InitiativeConfig:
    base_threshold: float = 0.55      # adaptive intervention threshold
    min_threshold: float = 0.35
    max_threshold: float = 0.9
    cooldown_sec: int = 90            # between interventions of same kind
    idle_help_after_sec: int = 300
    unsaved_file_remind_after_sec: int = 900
    repetition_window: int = 12       # last N commands to look for loops
    repetition_min_count: int = 3
    max_interventions_per_hour: int = 6
    typo_use_model: bool = True       # route typo judgement through the SSM
                                      # backend when weights are configured
                                      # (False -> deterministic mock analyzer)


@dataclass
class RagConfig:
    dim: int = 256                    # frozen lightweight embedder output
    top_k: int = 5
    doc_dirs: list[str] = field(default_factory=list)  # extra local docs to index


@dataclass
class KnowledgeConfig:
    enabled: bool = True              # whole subsystem switchable off (spec)
    dedup_high: float = 0.93          # >= : near-duplicate merge band
    dedup_low: float = 0.84           # [low,high) + different summary -> conflict
    max_segments_per_source: int = 40
    study_priority_default: int = 5   # lower = studied first
    sandbox_examples: bool = True     # run doc examples only in sandbox
    mask_secrets: bool = True         # privacy gate before DB writes (never disable in prod)


@dataclass
class CodeAssistConfig:
    enabled: bool = True
    gates: dict = field(default_factory=lambda: {"explain": 0.45, "snippet": 0.6, "edit": 0.75})
    run_tests: bool = True            # pytest step of the verification ladder
    test_timeout_sec: float = 60.0


@dataclass
class SleepConfig:
    enabled: bool = True
    train_at_idle_min: int = 20       # mini-training after this idle
    nightly_start: str = "02:30"
    adapter_rank: int = 4
    lr: float = 3e-4
    epochs: int = 2
    regression_base_score_min: float = 0.8   # keep >=80% of base skills
    nagging_check: bool = True


@dataclass
class EnergyConfig:
    target_bg_watts: float = 2.0
    poll_interval_sec: float = 0.0    # 0 = pure event driven, no polling
    batch_rag_indexing: bool = True   # heavy indexing only during idle/night


@dataclass
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    initiative: InitiativeConfig = field(default_factory=InitiativeConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    code_assist: CodeAssistConfig = field(default_factory=CodeAssistConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    log_level: str = "INFO"

    @property
    def config_dir(self) -> Path:
        return xdg_config_home() / APP_NAME

    @property
    def data_dir(self) -> Path:
        return xdg_data_home() / APP_NAME

    @property
    def state_dir(self) -> Path:
        return xdg_state_home() / APP_NAME

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "agent.log"

    def ensure_dirs(self) -> None:
        for p in (self.config_dir, self.data_dir, self.state_dir, self.adapters_dir):
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path | None = None) -> "AgentConfig":
        cfg = cls()
        path = path or cfg.config_dir / "config.json"
        if path.exists():
            _merge(cfg, json.loads(path.read_text()))
        return cfg

    def save(self, path: Path | None = None) -> Path:
        self.ensure_dirs()
        path = path or self.config_dir / "config.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def _merge(obj, raw: dict) -> None:
    for k, v in raw.items():
        if hasattr(obj, k):
            cur = getattr(obj, k)
            if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
                _merge(cur, v)
            else:
                setattr(obj, k, v)
