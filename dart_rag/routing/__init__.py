"""질문 → 경로 선택 (wiki | hybrid | vector | agent) 과 auto 모드 실행.

  routes.py   config/wiki_routes.yaml (질문 → 위키 섹션 매핑표)
  resolve.py  회사·기간을 질문 문자열에서 찾기 (규칙)
  router.py   RuleRouter (규칙, LLM 없음) · LLMQueryRouter (Gemini / Luna 가 JSON 으로 경로 선택)
  auto.py     AutoAnswerer: 경로 선택 → 근거 수집 → LLM 답변 1회 (근거 부족 시 에이전트로 전환)
"""
from .router import LLMQueryRouter, RouteDecision, RuleRouter
from .routes import RouteTable, load_routes

__all__ = ["LLMQueryRouter", "RouteDecision", "RouteTable", "RuleRouter", "load_routes"]
