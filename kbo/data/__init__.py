"""데이터 어댑터 패키지.

모델/튜닝/백테스트는 generator 가 아니라 DataSource 인터페이스에 의존한다.
→ 합성 데이터(SyntheticDataSource)를 (2차) 실데이터 로더로 교체해도 파이프라인 불변.
"""
from .base import DataSource, GameRecord
from .synthetic import SyntheticDataSource

__all__ = ["DataSource", "GameRecord", "SyntheticDataSource"]
