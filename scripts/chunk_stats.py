"""청크 통계: 회사·보고서·kind 별 개수와 평균 길이 (임베딩 시간 추정용)."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dart_rag.config import Settings  # noqa: E402
from dart_rag.pipeline import StageContext  # noqa: E402
from dart_rag.pipeline.stages import ChunkStage  # noqa: E402

ctx = StageContext(Settings.load())
kinds, lens, foot = Counter(), defaultdict(list), Counter()
for f in ctx.catalog.filings():
    for c in ctx.store.load_chunks(f, ChunkStage(ctx).version()):
        kinds[(f.corp_name, f.report_type, c.kind)] += 1
        lens[c.kind].append(len(c.text))
        if "주석" in c.meta.get("section_path", ""):
            foot[c.kind] += 1
for k, v in sorted(kinds.items()):
    print(k, v)
for k, v in lens.items():
    print(f"{k:9s} n={len(v):6d} 평균 {sum(v) / len(v):.0f}자, 그중 주석 {foot[k]}")
