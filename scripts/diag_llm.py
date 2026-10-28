"""LLM 제공자 진단 (읽기 전용 · 서비스 코드에 영향 없음).

config/providers.yaml 의 순서대로 각 제공자에 짧은 질문 1개를 보내고
성공 여부 · HTTP 상태 · 오류 메시지 · 응답 시간을 출력한다. API 키 값은 출력하지 않는다.

실행:  python scripts\\diag_llm.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dart_rag.config import load_dotenv  # noqa: E402

load_dotenv()

import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402

from dart_rag.llm.router import LLMRouter, ProviderSpec  # noqa: E402

conf_path = Path(__file__).resolve().parents[1] / "config" / "providers.yaml"
conf = yaml.safe_load(conf_path.read_text("utf-8")) or {}
order = [s.strip() for s in os.environ.get("LLM_ORDER", "").split(",") if s.strip()] or conf.get("order", [])
print(f"시도 순서: {order}" + ("  (LLM_ORDER 환경변수 사용)" if os.environ.get("LLM_ORDER") else ""))

router = LLMRouter.from_file()
active = {s.name for s in router.specs}
for n in order:
    if n not in active:
        print(f"  - {n:12s} 건너뜀 (키 또는 활성화 설정 없음)")

print()
for s in router.specs:
    host = (s.base_url or "https://api.openai.com/v1").split("/")[2]
    print(f"[{s.name}] model={s.model}  host={host}  timeout={s.timeout}s  429재시도={s.retry_on_429}회×{s.retry_wait}s")
    cli = OpenAI(api_key=s.api_key, base_url=s.base_url, max_retries=0) if s.base_url \
        else OpenAI(api_key=s.api_key, max_retries=0)
    t = time.perf_counter()
    try:
        r = cli.chat.completions.create(
            model=s.model, max_tokens=50, temperature=0,
            messages=[{"role": "user", "content": "1+1은? 숫자만 답해."}],
            extra_body=s.extra_body or None, timeout=min(s.timeout, 30))
        txt = (r.choices[0].message.content or "").strip()
        print(f"   OK  {time.perf_counter() - t:.1f}초  응답={txt[:40]!r}")
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", None)
        msg = str(getattr(e, "message", "") or e).replace(s.api_key or "\0", "***")
        print(f"   실패  {time.perf_counter() - t:.1f}초  status={status}  {type(e).__name__}")
        print(f"   메시지: {msg[:400]}")
        if status == 404:   # 모델명 문제일 때 사용 가능한 모델 일부 출력
            try:
                names = [m.id for m in cli.models.list().data]
                key = s.model.split("/")[-1].split("-")[0]
                print("   사용 가능 모델(일부):", [n for n in names if key in n][:15] or names[:15])
            except Exception as e2:  # noqa: BLE001
                print(f"   모델 목록 조회 실패: {type(e2).__name__}")
    print()
