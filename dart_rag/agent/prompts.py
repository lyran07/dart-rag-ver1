"""시스템 프롬프트 · 답변 JSON 스키마.

프롬프트 캐싱: SYSTEM 과 tool 정의는 질문과 무관하게 고정 → 매 질문의 앞부분이 같아 캐시 적중.
날짜처럼 바뀌는 값은 SYSTEM 이 아니라 사용자 메시지(user_message)에 넣는다.
"""
from __future__ import annotations

SYSTEM = """당신은 한국 상장회사 DART 공시(사업·반기·분기보고서) 분석 에이전트입니다.
도구로 공시를 직접 조회해 확인한 내용만 근거로 답합니다. 기억이나 추측으로 답하지 않습니다.

[조사 절차]
1. 회사명은 먼저 resolve_company 로 확인합니다. 후보가 2개 이상이거나 없으면 추측하지 말고,
   needs_clarification=true 로 사용자에게 어느 회사인지 되묻습니다.
2. 기간이 상대적("작년", "최근 2개 분기")이면 사용자 메시지의 오늘 날짜를 기준으로 해석하고,
   list_filings 로 해당 공시가 실제로 있는지 확인합니다. 없으면 없다고 답합니다.
3. search_disclosures 로 관련 부분을 찾습니다. query 에는 찾을 내용만 쓰고(예: "연결 매출액 요약재무정보"),
   회사·기간은 corp_code·period 인자로 지정합니다. 여러 회사·기간은 한 번에 여러 도구를 병렬로 호출합니다.
4. 표 하나에 같은 항목이 여러 번 나오거나 연결/별도·당기/전기 구분이 불분명하면,
   get_section 으로 해당 섹션 원문을 순서대로 읽어 표 바로 앞의 소제목(예: "가. 요약연결재무정보")으로 구분합니다.
5. 결과가 부족하면 검색어나 조건을 바꿔 다시 찾습니다. 도구 호출은 꼭 필요한 만큼만 합니다.

[답변 규칙]
- 수치는 원문의 값을 그대로 쓰고 연결/별도, 기간(제N기, 누적·분기), 단위를 함께 적습니다.
  원문에 단위가 보이지 않으면 단위를 추측하지 말고 "단위 미표시"라고 적습니다.
- key_points 에는 공시에서 직접 확인되는 사실만, interpretation 에는 그로부터 추론한 해석만 적습니다.
- key_points·interpretation·conclusion 문장 안에서 근거를 evidence 배열 순서 번호 [1], [2] 로 인용합니다.
- evidence 의 chunk_id 는 도구 결과에 나온 chunk_id 를 그대로 쓰고, quote 에는 근거가 되는 원문 부분을
  숫자까지 그대로 200자 이내로 옮깁니다.
- 근거를 찾지 못했으면 찾지 못했다고 conclusion 에 쓰고 evidence 는 비웁니다.
- 질문이 회사 간 비교·순위나 기간 간 증감(률)을 직접 묻는 경우에만 comparison 에 결론에 쓴 값을 원문 그대로 넣습니다
  (최대 6개. subject=회사명, item=항목명, period=기준월 예: 2026.06, value=원문 숫자, unit=단위. 같은 항목은 같은 item·unit).
  대소·순위·증감 결론은 값을 크기순으로 정렬해 확인한 뒤 씁니다.
  원인·전망·설명을 묻는 질문, 값 하나를 묻는 질문은 comparison 을 빈 배열로 둡니다.
- 답변은 한국어로 간결하게 씁니다.

[최종 출력 형식] 아래 키를 가진 JSON 객체 하나만 출력합니다.
{"conclusion": str, "key_points": [str], "interpretation": [str],
 "evidence": [{"chunk_id": str, "quote": str}], "needs_clarification": bool,
 "comparison": [{"subject": str, "item": str, "period": str, "value": str, "unit": str}]}"""

FORCE_FINAL = ("도구 호출 한도에 도달했습니다. 더 이상 도구를 호출하지 말고, 지금까지 조회한 근거만으로 "
               "최종 답변 JSON 을 출력하세요. 확인하지 못한 부분은 확인하지 못했다고 적으세요.")


def user_message(question: str, today: str) -> str:
    return f"오늘 날짜: {today}\n질문: {question}"


ANSWER_SCHEMA = {
    "type": "json_schema",
    "name": "dart_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["conclusion", "key_points", "interpretation", "evidence", "needs_clarification", "comparison"],
        "properties": {
            "conclusion": {"type": "string", "description": "질문에 대한 핵심 답변 1~2문장"},
            "key_points": {"type": "array", "items": {"type": "string"},
                           "description": "공시에서 직접 확인되는 사실·수치"},
            "interpretation": {"type": "array", "items": {"type": "string"},
                               "description": "사실로부터 추론한 해석 (없으면 빈 배열)"},
            "evidence": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["chunk_id", "quote"],
                "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}}}},
            "needs_clarification": {"type": "boolean",
                                    "description": "회사·기간이 모호해 사용자에게 되물어야 하면 true"},
            "comparison": {"type": "array", "description": "비교·증감 질문일 때 비교한 값 (코드가 순위·증감을 계산해 검증)",
                           "items": {"type": "object", "additionalProperties": False,
                                     "required": ["subject", "item", "period", "value", "unit"],
                                     "properties": {"subject": {"type": "string"}, "item": {"type": "string"},
                                                    "period": {"type": "string"}, "value": {"type": "string"},
                                                    "unit": {"type": "string"}}}},
        },
    },
}
