"""Tool-calling 에이전트: LLM 이 tool 로 공시를 조회하고, 근거를 인용해 답한다.

구성
  config.py   config/agent.yaml 로딩
  llm.py      LLM 백엔드 (OpenAI Responses API = Luna / OpenAI 호환 Chat Completions = Gemini 등)
  tools.py    LLM 이 호출하는 tool 4개 (회사 확인 · 공시 목록 · 검색 · 섹션 원문)
  prompts.py  시스템 프롬프트 · 답변 JSON 스키마
  loop.py     에이전트 루프 (라운드·시간 제한, 병렬 tool 실행, 인용 검증)
"""
from .config import AgentConfig, load_agent_config
from .loop import AgentError, AgentLoop

__all__ = ["AgentConfig", "AgentError", "AgentLoop", "load_agent_config"]
