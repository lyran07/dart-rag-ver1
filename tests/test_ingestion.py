from dart_rag.domain import Section
from dart_rag.ingestion.chunker import DartChunker
from dart_rag.ingestion.parser import DartXmlParser
from dart_rag.ingestion.sources import SampleSource


def _sample():
    src = SampleSource()
    f = src.discover([], "", "")[0]
    return f, src.fetch(f)


def test_parse_sections_and_tables():
    f, xml = _sample()
    root = DartXmlParser().parse(xml)
    titles = [s.title for s in root.walk() if s.level == 1]
    assert "I. 회사의 개요" in titles
    tables = [b.table for s in root.walk() for b in s.blocks if b.kind == "table"]
    assert tables and all(t.header for t in tables)


def test_section_roundtrip_serialization():
    f, xml = _sample()
    root = DartXmlParser().parse(xml)
    back = Section.from_dict(root.to_dict())
    assert [s.path for s in back.walk()] == [s.path for s in root.walk()]


def test_chunks_have_breadcrumb_and_kv():
    f, xml = _sample()
    chunks = DartChunker().chunk(DartXmlParser().parse(xml), f)
    assert all(c.text.startswith("[삼성전자 | 사업보고서 2023.12 |") for c in chunks)
    kv = [c for c in chunks if c.kind == "table_kv"]
    assert any("매출액(" in c.raw and "백만원" in c.raw for c in kv)
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunker_version_changes_with_params():
    from dart_rag.config import ChunkingCfg
    assert DartChunker().version != DartChunker(ChunkingCfg(target_chars=500)).version
