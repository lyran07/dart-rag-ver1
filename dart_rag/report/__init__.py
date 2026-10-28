"""보고서 생성 (기업분석 보고서 프로토타입).

  ReportBuilder  템플릿의 섹션마다 위키 근거 수집 → LLM 1회 → 인용 검증 (섹션 병렬, 종합 의견은 마지막)
  to_markdown    완성된 보고서를 마크다운으로
"""
from .builder import ReportBuilder, ReportJob, SectionResult, to_markdown
from .templates import ReportTemplate, SectionSpec, load_templates

__all__ = ["ReportBuilder", "ReportJob", "SectionResult", "to_markdown",
           "ReportTemplate", "SectionSpec", "load_templates"]
