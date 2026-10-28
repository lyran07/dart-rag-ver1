"""에이전트용 LLM 백엔드.

두 백엔드는 같은 인터페이스를 가진다 (loop.py 는 어느 쪽인지 모른다):
  start(instructions, user_text) -> 대화 상태
  step(conv, tools, schema, allow_tools=True) -> StepResult   (모델 출력을 conv 에 덧붙임)
  add_tool_results(conv, [(ToolCall, 결과 문자열), ...])
  add_user(conv, text)

ResponsesLLM : OpenAI Responses API (GPT-6 Luna). function calling + JSON 스키마 강제 출력.
               store=False 로 대화를 OpenAI 서버에 저장하지 않고, 추론 내용은 암호화 형태로 받아 다시 넘긴다.
ChatToolsLLM : OpenAI 호환 Chat Completions tool calling (Gemini 의 OpenAI 호환 엔드포인트 등). 실험적.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


class AgentLLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class StepResult:
    tool_calls: list[ToolCall]
    final_text: str | None
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0


def parse_arguments(raw: str | None) -> dict:
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"__parse_error__": "인자가 객체가 아닙니다"}
    except json.JSONDecodeError as e:
        return {"__parse_error__": f"JSON 파싱 실패: {e}"}


def parse_final_json(text: str | None) -> dict:
    """최종 답변 JSON 파싱. ```json 펜스나 앞뒤 잡문이 섞여도 첫 객체를 꺼낸다."""
    if not text:
        raise AgentLLMError("최종 답변이 비어 있습니다.")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                pass
    raise AgentLLMError(f"최종 답변 JSON 파싱 실패: {text[:200]}")


# --------------------------------------------------------------------------- 공통
class UsageLogger:
    """LLM 호출마다 토큰·비용·시간을 JSONL 로 누적. 기록 실패가 답변을 막지 않는다."""

    def __init__(self, path: Path | None, prices: dict | None):
        self.path, self.prices = path, prices or None

    def cost(self, u: dict) -> float | None:
        p = self.prices
        if not p:
            return None
        fresh = max(0, u.get("input_tokens", 0) - u.get("cached_tokens", 0))
        return round((fresh * p.get("input", 0) + u.get("cached_tokens", 0) * p.get("cached_input", p.get("input", 0))
                      + u.get("output_tokens", 0) * p.get("output", 0)) / 1e6, 6)

    def log(self, rec: dict) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass


class QuotaExhausted(AgentLLMError):
    """API 한도·크레딧 소진 (429 insufficient_quota 등). 재시도하지 않는다."""


def _call_with_retry(fn, max_retries: int):
    import openai

    from ..embedding.providers import is_quota_exhausted
    retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable as e:
            if is_quota_exhausted(e):
                raise QuotaExhausted(f"API 한도 소진: {str(e)[:160]}") from e
            if attempt == max_retries:
                raise
            time.sleep(min(8.0, 1.5 * 2 ** attempt))      # 1.5, 3, 6초


def _client(api_key_env: str, base_url: str | None):
    key = os.environ.get(api_key_env)
    if not key:
        raise AgentLLMError(f"환경변수 {api_key_env} 가 없습니다 (.env 확인).")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=base_url, max_retries=0) if base_url else OpenAI(api_key=key, max_retries=0)


# --------------------------------------------------------------------------- Responses API (Luna)
@dataclass
class ResponsesConv:
    instructions: str
    items: list


class ResponsesLLM:
    backend = "responses"

    def __init__(self, model: str, reasoning: str = "low", timeout: float = 25, max_retries: int = 2,
                 max_output_tokens: int = 2000, api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None,
                 cache_key: str = "dart-agent-v1", usage: UsageLogger | None = None, client=None):
        self.model, self.reasoning, self.timeout, self.max_retries = model, reasoning, timeout, max_retries
        self.max_output_tokens, self.cache_key = max_output_tokens, cache_key
        self.usage = usage or UsageLogger(None, None)
        self.client = client or _client(api_key_env, base_url)

    def start(self, instructions: str, user_text: str) -> ResponsesConv:
        return ResponsesConv(instructions, [{"role": "user", "content": user_text}])

    def add_user(self, conv: ResponsesConv, text: str) -> None:
        conv.items.append({"role": "user", "content": text})

    def add_tool_results(self, conv: ResponsesConv, results: list[tuple[ToolCall, str]]) -> None:
        for call, out in results:
            conv.items.append({"type": "function_call_output", "call_id": call.call_id, "output": out})

    def step(self, conv: ResponsesConv, tools: list | None, schema: dict | None, allow_tools: bool = True) -> StepResult:
        kw: dict = dict(model=self.model, instructions=conv.instructions, input=conv.items, store=False,
                        include=["reasoning.encrypted_content"], reasoning={"effort": self.reasoning},
                        max_output_tokens=self.max_output_tokens, prompt_cache_key=self.cache_key,
                        timeout=self.timeout)
        if tools and allow_tools:
            kw.update(tools=tools, parallel_tool_calls=True)
        if schema:
            kw["text"] = {"format": schema}
        t = time.perf_counter()
        r = _call_with_retry(lambda: self.client.responses.create(**kw), self.max_retries)
        sec = time.perf_counter() - t
        conv.items.extend(r.output)          # 추론·tool 호출·메시지 항목을 그대로 다음 입력에 이어 붙인다
        calls = [ToolCall(it.call_id, it.name, parse_arguments(it.arguments), it.arguments or "")
                 for it in r.output if getattr(it, "type", "") == "function_call"]
        u = r.usage
        usage = {"input_tokens": getattr(u, "input_tokens", 0) or 0,
                 "cached_tokens": getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
                 "output_tokens": getattr(u, "output_tokens", 0) or 0,
                 "reasoning_tokens": getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0}
        usage["usd"] = self.usage.cost(usage)
        self.usage.log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": self.backend, "model": self.model,
                        "reasoning": self.reasoning, "seconds": round(sec, 2), "n_tool_calls": len(calls), **usage})
        return StepResult(calls, None if calls else (r.output_text or ""), usage, sec)


# --------------------------------------------------------------------------- Chat Completions (Gemini 등)
@dataclass
class ChatConv:
    messages: list


class ChatToolsLLM:
    """OpenAI 호환 Chat Completions 의 tool calling. 최종 답변 JSON 은 프롬프트로 요구하고,
    tool 없이 강제 마무리할 때만 response_format=json_object 를 건다 (제공자 호환성 때문)."""
    backend = "chat"

    def __init__(self, model: str, timeout: float = 25, max_retries: int = 2, max_output_tokens: int = 2000,
                 api_key_env: str = "GEMINI_API_KEY", base_url: str | None = None,
                 usage: UsageLogger | None = None, client=None, reasoning: str | None = None):
        self.model, self.timeout, self.max_retries, self.max_output_tokens = model, timeout, max_retries, max_output_tokens
        self.reasoning = reasoning
        self.usage = usage or UsageLogger(None, None)
        self.client = client or _client(api_key_env, base_url)

    @staticmethod
    def _tools(tools: list) -> list:
        return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
                for t in tools]

    def start(self, instructions: str, user_text: str) -> ChatConv:
        return ChatConv([{"role": "system", "content": instructions}, {"role": "user", "content": user_text}])

    def add_user(self, conv: ChatConv, text: str) -> None:
        conv.messages.append({"role": "user", "content": text})

    def add_tool_results(self, conv: ChatConv, results: list[tuple[ToolCall, str]]) -> None:
        for call, out in results:
            conv.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": out})

    def step(self, conv: ChatConv, tools: list | None, schema: dict | None, allow_tools: bool = True) -> StepResult:
        kw: dict = dict(model=self.model, messages=conv.messages, max_tokens=self.max_output_tokens,
                        temperature=0, timeout=self.timeout)
        if tools and allow_tools:
            kw.update(tools=self._tools(tools), tool_choice="auto")
        elif schema:
            kw["response_format"] = {"type": "json_object"}
        t = time.perf_counter()
        r = _call_with_retry(lambda: self.client.chat.completions.create(**kw), self.max_retries)
        sec = time.perf_counter() - t
        msg = r.choices[0].message
        tcs = list(getattr(msg, "tool_calls", None) or [])
        conv.messages.append({"role": "assistant", "content": msg.content or "",
                              **({"tool_calls": [{"id": c.id, "type": "function",
                                                  "function": {"name": c.function.name,
                                                               "arguments": c.function.arguments}} for c in tcs]}
                                 if tcs else {})})
        calls = [ToolCall(c.id, c.function.name, parse_arguments(c.function.arguments), c.function.arguments or "")
                 for c in tcs]
        u = r.usage
        usage = {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                 "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
                 "output_tokens": getattr(u, "completion_tokens", 0) or 0, "reasoning_tokens": 0}
        usage["usd"] = self.usage.cost(usage)
        self.usage.log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": self.backend, "model": self.model,
                        "seconds": round(sec, 2), "n_tool_calls": len(calls), **usage})
        return StepResult(calls, None if calls else (msg.content or ""), usage, sec)


def make_llm(cfg, backend: str | None = None, model: str | None = None, reasoning: str | None = None,
             max_output_tokens: int | None = None, timeout: float | None = None) -> ResponsesLLM | ChatToolsLLM:
    """agent.yaml 의 backends 설정으로 LLM 백엔드를 만든다. 인자를 주면 그 값으로 덮어쓴다
    (예: 라우터는 Gemini, 답변은 Luna reasoning=none)."""
    backend = backend or cfg.backend
    b = dict(cfg.backends.get(backend) or {})
    usage = UsageLogger(cfg.usage_log_path, b.get("prices"))
    common = dict(model=model or (cfg.model if backend == cfg.backend else b.get("model")),
                  timeout=timeout or cfg.llm_timeout_sec, max_retries=cfg.llm_max_retries,
                  max_output_tokens=max_output_tokens or cfg.max_output_tokens,
                  api_key_env=b.get("api_key_env", "OPENAI_API_KEY"), base_url=b.get("base_url"), usage=usage,
                  reasoning=reasoning or cfg.reasoning)
    if backend == "responses":
        return ResponsesLLM(**common)
    if backend == "chat":
        return ChatToolsLLM(**common)
    raise AgentLLMError(f"알 수 없는 backend: {backend} (responses | chat)")


def make_agent_llm(cfg) -> ResponsesLLM | ChatToolsLLM:
    return make_llm(cfg)
