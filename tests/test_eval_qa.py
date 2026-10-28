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


def test_conclusion_checks():
    from scripts.eval_qa import conclusion_checks
    it3 = {"corps": ["삼성전자", "SK하이닉스", "현대자동차"], "expect_max": "현대자동차", "expect_min": "삼성전자"}
    wrong = "삼성전자 746원, 현대자동차 5,000원, SK하이닉스 750원입니다. 금액은 SK하이닉스가 가장 크고, 삼성전자가 가장 작습니다."
    right = "현대자동차가 5,000원으로 가장 많고, 삼성전자가 746원으로 가장 적다. SK하이닉스는 삼성전자보다 4원 더 많다."
    assert conclusion_checks(it3, wrong) == {"expect_max": False, "expect_min": True}
    assert conclusion_checks(it3, right) == {"expect_max": True, "expect_min": True}
    it2 = {"corps": ["삼성전자", "SK하이닉스"], "expect_max": "삼성전자"}
    assert conclusion_checks(it2, "매출이 더 큰 곳은 삼성전자입니다.")["expect_max"]
    assert conclusion_checks(it2, "삼성전자가 SK하이닉스보다 더 크다.")["expect_max"]
    assert conclusion_checks(it2, "매출은 삼성전자가 더 큽니다.")["expect_max"]
    assert not conclusion_checks(it2, "SK하이닉스가 삼성전자보다 더 크다.")["expect_max"]
    assert conclusion_checks({"expect_direction": ["증가", "감소"]}, "2024년 증가했다가 2025년 감소했다")["expect_direction"]
    assert not conclusion_checks({"expect_direction": ["증가", "감소"]}, "2024년 감소했다가 2025년 증가했다")["expect_direction"]
    assert conclusion_checks({"expect_calc": {"value": 74.4, "unit": "%", "tol": 0.6}}, "영업이익률은 약 74.4%다")["expect_calc"]
    assert conclusion_checks({"expect_calc": {"value": 3.31, "unit": "배", "tol": 0.06}}, "약 3.3배로 늘었다")["expect_calc"]
