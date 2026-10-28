"""BM25 용 토크나이저.

- RegexTokenizer (기본, 의존성 없음): 한글 어절 + 한글 bigram + 영문 + 숫자(콤마 제거)
- KiwiTokenizer (선택, pip install kiwipiepy): 형태소 기반 (명사/동사 어근/외국어/숫자) + 숫자 토큰
공시 수치 질의("제55기", "258,935,494")는 숫자 토큰 정확 매칭이 중요하므로 두 방식 모두 숫자를 보존한다.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

_WORD = re.compile(r"[가-힣]+|[a-z]+|\d[\d,\.]*")


class Tokenizer(ABC):
    name = "base"

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        ...


class RegexTokenizer(Tokenizer):
    name = "regex-bigram"

    def tokenize(self, text: str) -> list[str]:
        toks: list[str] = []
        for w in _WORD.findall(text.lower()):
            if w[0].isdigit():
                toks.append(w.replace(",", "").rstrip("."))
            elif "가" <= w[0] <= "힣":
                toks.append(w)
                toks += [w[i:i + 2] for i in range(len(w) - 1)]  # 조사 영향 완화
            else:
                toks.append(w)
        return toks


class KiwiTokenizer(Tokenizer):
    name = "kiwi"
    KEEP = ("NN", "VV", "VA", "SL", "SN", "XR", "SH")

    def __init__(self):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()

    def tokenize(self, text: str) -> list[str]:
        toks = []
        for t in self.kiwi.tokenize(text.lower()):
            if t.tag.startswith(self.KEEP):
                toks.append(t.form.replace(",", ""))
        toks += [w.replace(",", "").rstrip(".") for w in re.findall(r"\d[\d,\.]*", text)]
        return toks


def make_tokenizer(kind: str = "regex") -> Tokenizer:
    if kind == "kiwi":
        try:
            return KiwiTokenizer()
        except ImportError:
            print("[tokenizer] kiwipiepy 미설치 → regex 토크나이저 사용")
    return RegexTokenizer()
