# bayesian_norm
정성+정량 데이터를 결합한 형용사 평가

주의사항
📍이 저장소의 데이터셋은 실제 브랜드 조사 데이터가 아님. 연구/데모 목적을 위해 임의 생성한 샘플 데이터
📍Bayesian_norm.py에는 간단한 로직만 수록되어 있음

분석방법
📍 온라인 리뷰 크롤링 후, Dictionary(기능/형용사)에 맞게 토큰화 및 count한 결과를 사전믿음으로 간주한 후, 향후 기능별 브랜드 이미지 조사 데이터(임의생성)를 likelihood로 정의하고 hierarchical bayesian 모델 학습
📍 Bayesian 결과와 normal disbribution 대조

주 사용 라이브러리
📍PyMC(베이지안 추정), NumPy / Pandas (연산/집계), SciPy(MLE 최적화) / excel VBA(대시보드)

사용 데이터
📍 브랜드별 특정 품목의 리뷰 크롤링 결과 / Dictionary(텍스트마이닝 시 활용) / 임의로 생성한 기능x브랜드 이미지 조사 데이터

주요 분석단계
베이지안 분석단계
📍전처리: CSV 로드 → 충성도 지표에서 Top2/Bot2/이외 기준으로 집단 분류(Low/Mid/High) → 이항 타깃(서열 데이터 분석도 사용할 수 있도록 임계값 설정)
📍인코딩: 세그, 세그-모델(중첩), 충성도 카테고리 코드화 ➡️ 고객 그룹을 큰 그룹(Segment)부터 세부 모델단위 Segment까지 분류해서 확인
📍조건부 Flow 생성: recommend | preference=1 → intent | recommend=1 → purchase | intent=1 ➡️ 실제 구매의 흐름을 반영해 필터링
📍모델 적합(단계별): PyMC 계층 로지스틱 4개(각 단계 독립) ➡️ n수가 적은 경우에도, 상위 계층의 분포를 빌려 와서 신뢰도 있는 예측 가능
📍Posterior 집계(그룹/교차레벨별): 전환율 평균·95% CI, SNR, 리프트(전체 대비), Fail Prob vs 브랜드, 드롭아웃율(1−p), 체인 전환율(pref×rec×intent×buy) ➡️ 예측의 불확실성까지 함께 고려하여 예측
📍스코어링/등급: 가중합(기본: 0.2/0.3/0.0/0.5) → 분위수로 A~D
📍보조 추정: 선호도 서열 로지스틱 MLE 컷포인트 요약

대시보드 구축단계
📍VBA Chart 구축

한계/보완점
📍텍스트마이닝 시, Dictionary finetuning을 사람이 해야 해서 비효율적
📍보완방법 고려: Transformer 등 LLM 학습을 통한 Dictionary 분류 자동화 고려

대시보드 불러오기

예시이미지
