"""config/agent.yaml 로딩 (+ 환경변수 덮어쓰기)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROJECT_ROOT

AGENT_CONFIG_PATH = PROJECT_ROOT / "config" / "agent.yaml"


@dataclass
class AgentConfig:
    default_mode: str = "agent"
    backend: str = "responses"
    reasoning: str = "low"
    backends: dict = field(default_factory=dict)
    max_rounds: int = 4
    deadline_sec: float = 45
    llm_timeout_sec: float = 25
    llm_max_retries: int = 2
    tool_timeout_sec: float = 20
    max_output_tokens: int = 2000
    search_k: int = 6
    result_chars: int = 1200
    section_chars: int = 6000
    tool_output_chars: int = 9000
    fallback: str = "rag"
    usage_log: str = "data/llm_usage.jsonl"
    aliases: dict = field(default_factory=dict)
    auto: dict = field(default_factory=dict)      # auto 모드 (라우터·단일 답변) 설정

    @property
    def backend_cfg(self) -> dict:
        return dict(self.backends.get(self.backend) or {})

    @property
    def model(self) -> str:
        return os.environ.get("AGENT_MODEL") or self.backend_cfg.get("model", "gpt-6-luna")

    @property
    def usage_log_path(self) -> Path:
        p = Path(self.usage_log)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_agent_config(path: str | os.PathLike | None = None) -> AgentConfig:
    p = Path(path or AGENT_CONFIG_PATH)
    d: dict = {}
    if p.exists():
        import yaml
        d = yaml.safe_load(p.read_text("utf-8")) or {}
    known = {k: v for k, v in d.items() if k in AgentConfig.__dataclass_fields__}
    cfg = AgentConfig(**known)
    cfg.default_mode = os.environ.get("ASK_MODE", cfg.default_mode)
    cfg.backend = os.environ.get("AGENT_BACKEND", cfg.backend)
    cfg.reasoning = os.environ.get("AGENT_REASONING", cfg.reasoning)
    if os.environ.get("ROUTER"):
        cfg.auto = {**cfg.auto, "router": os.environ["ROUTER"]}
    return cfg
