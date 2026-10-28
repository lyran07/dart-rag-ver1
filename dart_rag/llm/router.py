"""LLM 라우터: 여러 제공자를 순서대로 시도 (fallback chain).

- 모든 제공자를 OpenAI 호환 Chat Completions 로 호출 (Groq, Gemini, Upstage, OpenAI, vLLM, Ollama …)
- 새 제공자 추가 = config/providers.yaml 에 항목 추가. 코드 수정 불필요.
- 컨텍스트 예산이 제공자마다 다르므로 user 메시지를 '예산 → 문자열' 함수로 받는다.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import PROJECT_ROOT


class NoLLMAvailable(RuntimeError):
    pass


@dataclass
class ProviderSpec:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    max_ctx_chars: int = 16000
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: float = 60
    retry_on_429: int = 1
    retry_wait: float = 15
    extra_body: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, d: dict, defaults: dict) -> "ProviderSpec | None":
        """사용 불가(키 없음/비활성)면 None."""
        d = {**defaults, **d}
        if d.get("enabled_env") and not os.environ.get(d["enabled_env"]):
            return None
        base_url = d.get("base_url")
        if d.get("base_url_env"):
            base_url = os.environ.get(d["base_url_env"])
            if not base_url:
                return None
        key = d.get("api_key") or (os.environ.get(d["api_key_env"]) if d.get("api_key_env") else None)
        if not key:
            return None
        model = os.environ.get(d.get("model_env", "") or "_none_") or d["model"]
        return cls(name=name, model=model, base_url=base_url, api_key=key,
                   max_ctx_chars=int(d.get("max_ctx_chars", 16000)), temperature=float(d.get("temperature", 0)),
                   max_tokens=int(d.get("max_tokens", 2000)), timeout=float(d.get("timeout", 60)),
                   retry_on_429=int(d.get("retry_on_429", 1)), retry_wait=float(d.get("retry_wait", 15)),
                   extra_body=d.get("extra_body") or {})


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    attempts: list[dict] = field(default_factory=list)


class LLMRouter:
    def __init__(self, specs: list[ProviderSpec], client_factory: Callable | None = None, sleep=time.sleep):
        self.specs = specs
        self._factory = client_factory or _openai_client
        self._clients: dict[str, object] = {}
        self._sleep = sleep

    @classmethod
    def from_file(cls, path: str | os.PathLike | None = None, **kw) -> "LLMRouter":
        p = Path(path or PROJECT_ROOT / "config" / "providers.yaml")
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        conf: dict = {}
        if p.exists():
            import yaml
            conf = yaml.safe_load(p.read_text("utf-8")) or {}
        order = [s.strip() for s in os.environ.get("LLM_ORDER", "").split(",") if s.strip()] or conf.get("order", [])
        provs, defaults = conf.get("providers", {}), conf.get("defaults", {})
        specs = [s for n in order if n in provs and (s := ProviderSpec.from_config(n, provs[n], defaults))]
        return cls(specs, **kw)

    @property
    def available(self) -> list[str]:
        return [f"{s.name}:{s.model}" for s in self.specs]

    def _client(self, s: ProviderSpec):
        if s.name not in self._clients:
            self._clients[s.name] = self._factory(s)
        return self._clients[s.name]

    def complete(self, system: str, build_user: Callable[[int], str]) -> LLMResult:
        if not self.specs:
            raise NoLLMAvailable("사용 가능한 LLM 제공자가 없습니다 (API 키 확인).")
        attempts: list[dict] = []
        for s in self.specs:
            user = build_user(s.max_ctx_chars)
            for i in range(s.retry_on_429 + 1):
                t = time.perf_counter()
                try:
                    r = self._client(s).chat.completions.create(
                        model=s.model, temperature=s.temperature, max_tokens=s.max_tokens,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        extra_body=s.extra_body or None, timeout=s.timeout)
                    text = r.choices[0].message.content or ""
                    attempts.append({"provider": s.name, "ok": True, "sec": round(time.perf_counter() - t, 2)})
                    return LLMResult(text, s.name, s.model, attempts)
                except Exception as e:  # noqa: BLE001  모든 오류 → 다음 제공자
                    status = getattr(e, "status_code", None)
                    attempts.append({"provider": s.name, "ok": False, "status": status, "error": type(e).__name__})
                    if status == 429 and i < s.retry_on_429:
                        self._sleep(s.retry_wait)
                        continue
                    break
        raise NoLLMAvailable(f"모든 LLM 제공자 실패: {attempts}")


def _openai_client(s: ProviderSpec):
    from openai import OpenAI
    # max_retries=0: SDK 자동 재시도(기본 2회)를 끈다. 켜 두면 타임아웃이 3배로 늘어난 뒤에야
    # 다음 제공자로 넘어간다. 재시도는 complete() 의 retry_on_429 가 관리한다.
    if s.base_url:
        return OpenAI(api_key=s.api_key, base_url=s.base_url, max_retries=0)
    return OpenAI(api_key=s.api_key, max_retries=0)
