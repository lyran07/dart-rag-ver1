"""공시 소스 추상화.

FilingSource 두 가지 책임:
  discover(...)  → 어떤 공시가 있는지 목록 (catalog 에 등록)
  fetch(filing)  → 원문 XML 문자열

DartSource(실데이터) / SampleSource(키 없이 파이프라인 검증) 를 같은 인터페이스로 교체한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..domain import Filing


class FilingSource(ABC):
    name: str = "base"

    @abstractmethod
    def discover(self, corps: list[str], bgn_de: str, end_de: str, skip_amend: bool = False,
                 report_types: tuple[str, ...] | None = None) -> list[Filing]:
        """report_types: DART 공시상세유형 (A001 사업 / A002 반기 / A003 분기). None 이면 전부."""
        ...

    @abstractmethod
    def fetch(self, filing: Filing) -> str:
        ...


class DartSource(FilingSource):
    name = "dart"

    def __init__(self, api_key: str | None = None, corp_cache: Path | None = None):
        from .dart_client import DartClient
        self.client = DartClient(api_key=api_key, corp_cache=corp_cache)

    def discover(self, corps, bgn_de, end_de, skip_amend=False, report_types=None):
        out: list[Filing] = []
        for corp in corps:
            code = self.client.corp_code(corp)
            if report_types:
                fs = self.client.list_filings(code, bgn_de, end_de, types=tuple(report_types))
            else:
                fs = self.client.list_filings(code, bgn_de, end_de)
            if skip_amend:
                fs = [f for f in fs if not f.is_amendment]
            print(f"[discover] {corp} ({code}): {len(fs)}건")
            out += fs
        return out

    def fetch(self, filing: Filing) -> str:
        return self.client.fetch_document_xml(filing.rcept_no)


class SampleSource(FilingSource):
    """sample_data.py 의 합성 XML. corp/기간 인자는 무시하고 전부 반환."""
    name = "sample"

    def __init__(self):
        import importlib
        import sys
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        mod = importlib.import_module("sample_data")
        self._items: dict[str, tuple[Filing, str]] = {}
        for meta, xml in mod.SAMPLES:
            f = Filing(rcept_no=meta["rcept_no"], corp_code=meta.get("corp_code", "00126380"),
                       corp_name=meta["corp_name"], report_nm=meta["report_nm"], rcept_dt=meta["rcept_dt"])
            self._items[f.rcept_no] = (f, xml)

    def discover(self, corps, bgn_de, end_de, skip_amend=False, report_types=None):
        return [f for f, _ in self._items.values()]

    def fetch(self, filing: Filing) -> str:
        return self._items[filing.rcept_no][1]


def make_source(kind: str, **kw) -> FilingSource:
    if kind == "sample":
        return SampleSource()
    if kind == "dart":
        return DartSource(**kw)
    raise ValueError(f"unknown source: {kind}")
