"""DART 공시 RAG 백엔드.

계층 (위가 아래를 사용, 역방향 import 금지)
  cli / api                  진입점
  service                    조립 (설정 → 구현체 선택 → 의존성 주입)
  pipeline                   오프라인 처리: catalog, storage, stages, runner
  generation / retrieval     온라인 질의: 질의분석, 하이브리드 검색, 리랭크, 답변
  llm / embedding / index    교체 가능한 어댑터 (제공자·모델·벡터DB)
  ingestion / wiki           DART 수집·파싱·청킹, 위키 뷰
  domain / config            공통 타입, 설정
"""
__version__ = "0.2.0"
