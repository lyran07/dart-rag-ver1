"""OpenDART API 클라이언트 (네트워크 I/O 만 담당).

- 캐시·저장은 하지 않는다 → pipeline.storage.ArtifactStore 가 담당.
- corp_code 목록(corpCode.xml)만 예외적으로 cache_path 에 보관 (하루 1회면 충분).
"""
from __future__ import annotations

import io
import os
import re
import time
import zipfile
from pathlib import Path

import requests

from ..domain import Filing

BASE = "https://opendart.fss.or.kr/api"
REPORT_TYPES = {"A001": "사업보고서", "A002": "반기보고서", "A003": "분기보고서"}


class DartApiError(RuntimeError):
    pass


class DartClient:
    def __init__(self, api_key: str | None = None, corp_cache: Path | None = None,
                 timeout: int = 60, min_interval: float = 0.15):
        self.api_key = api_key or os.environ.get("OPENDART_API_KEY")
        if not self.api_key:
            raise DartApiError("OPENDART_API_KEY 가 필요합니다.")
        self.corp_cache = corp_cache
        self.timeout = timeout
        self.min_interval = min_interval   # 호출 간 최소 간격 (분당 한도 보호)
        self._last = 0.0
        self._corp_index: list[tuple[str, str, str]] | None = None

    # ------------------------------------------------------------------ http
    def _get(self, path: str, retries: int = 3, **params) -> requests.Response:
        params["crtfc_key"] = self.api_key
        for i in range(retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get(f"{BASE}/{path}", params=params, timeout=self.timeout)
                self._last = time.time()
                r.raise_for_status()
                return r
            except requests.RequestException:
                if i == retries - 1:
                    raise
                time.sleep(2 * (i + 1))
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ corp_code
    def _load_corp_index(self) -> list[tuple[str, str, str]]:
        if self._corp_index is not None:
            return self._corp_index
        xml = None
        if self.corp_cache and self.corp_cache.exists():
            xml = self.corp_cache.read_text("utf-8", errors="ignore")
        if xml is None:
            r = self._get("corpCode.xml")
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                xml = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
            if self.corp_cache:
                self.corp_cache.parent.mkdir(parents=True, exist_ok=True)
                self.corp_cache.write_text(xml, "utf-8")
        pat = re.compile(
            r"<list>\s*<corp_code>(\d+)</corp_code>\s*<corp_name>(.*?)</corp_name>\s*"
            r"(?:<corp_eng_name>.*?</corp_eng_name>\s*)?<stock_code>\s*(\S*?)\s*</stock_code>", re.S)
        self._corp_index = [(c, n.strip(), s) for c, n, s in pat.findall(xml)]
        return self._corp_index

    def corp_code(self, name_or_code: str) -> str:
        """법인명(정확히 일치) · 8자리 corp_code · 6자리 종목코드 모두 허용. 상장사 우선."""
        key = name_or_code.strip()
        idx = self._load_corp_index()
        if re.fullmatch(r"\d{8}", key):
            return key
        if re.fullmatch(r"\d{6}", key):
            for c, _, s in idx:
                if s == key:
                    return c
        cands = [(c, n, s) for c, n, s in idx if n == key]
        if not cands:
            raise DartApiError(f"corp_code 를 찾을 수 없음: {name_or_code}")
        cands.sort(key=lambda t: (t[2] == "", t[0]))
        return cands[0][0]

    # ------------------------------------------------------------------ 공시 목록
    def list_filings(self, corp_code: str, bgn_de: str, end_de: str,
                     types: tuple[str, ...] = ("A001", "A002", "A003")) -> list[Filing]:
        out: list[Filing] = []
        for t in types:
            page = 1
            while True:
                j = self._get("list.json", corp_code=corp_code, bgn_de=bgn_de, end_de=end_de,
                              pblntf_detail_ty=t, page_count=100, page_no=page).json()
                status = j.get("status")
                if status == "013":   # 조회 결과 없음
                    break
                if status != "000":
                    raise DartApiError(f"DART list 오류 {status}: {j.get('message')}")
                for it in j["list"]:
                    out.append(Filing(it["rcept_no"], it["corp_code"], it["corp_name"],
                                      it["report_nm"].strip(), it["rcept_dt"]))
                if page >= int(j.get("total_page", 1)):
                    break
                page += 1
        out.sort(key=lambda f: f.rcept_dt, reverse=True)
        return out

    # ------------------------------------------------------------------ 원문
    def fetch_document_xml(self, rcept_no: str) -> str:
        r = self._get("document.xml", rcept_no=rcept_no, retries=3)
        if not r.content.startswith(b"PK"):
            raise DartApiError(f"document.xml 응답이 zip 이 아님: {r.text[:200]}")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # 본문 파일: 보통 '{rcept_no}.xml' (첨부는 _숫자 suffix)
            names = sorted(z.namelist(), key=lambda n: (("_" in Path(n).stem), n))
            raw = z.read(names[0])
        return decode_xml(raw)


def decode_xml(raw: bytes) -> str:
    m = re.match(rb'\s*<\?xml[^>]*encoding="([^"]+)"', raw)
    encs = [m.group(1).decode()] if m else []
    for enc in encs + ["utf-8", "euc-kr", "cp949"]:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="ignore")
