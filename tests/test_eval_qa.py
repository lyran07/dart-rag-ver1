"""평가 스크립트 채점 규칙 테스트 (숫자 표기 변환 · 정답 판정)."""
from scripts.eval_qa import answer_score, korean_amounts_mil, value_in_answer


def test_korean_amounts():
    assert korean_amounts_mil("매출 131조 8,950억원") == [131_895_000.0]
    assert 131_900_000.0 in korean_amounts_mil("약 131.9조원")
    assert korean_amounts_mil("영업이익 6,567억원") == [656_700.0]


def test_value_in_answer_units():
    assert value_in_answer("131,895,033", "백만원", "연결 매출액은 131,895,033백만원이다")
    assert value_in_answer("131,895,033", "백만원", "매출은 131조 8,950억원")          # 환산 표기 (1% 이내)
    assert not value_in_answer("131,895,033", "백만원", "매출은 120조원")
    assert value_in_answer("750", "원", "주당 750원")
    assert not value_in_answer("750", "원", "주당 3,000원")


def test_answer_score_groups():
    it = {"answer": [["146,725,209"], ["98,152,891"]], "unit": "백만원", "keywords": [["영업이익"]]}
    assert answer_score(it, "영업이익은 146조 7,252억원과 98.2조원")["correct"]
    assert not answer_score(it, "영업이익은 146조 7,252억원")["correct"]
