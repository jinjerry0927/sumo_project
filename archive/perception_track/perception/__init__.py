"""실시간 CCTV → YOLO → 차로별 카운트 파이프라인.

모듈 구성:
    stream_source.py   : HLS/RTSP/파일 입력 추상화 + 자동 재연결
    detector.py        : YOLOv8 차량 탐지 (필터 포함)
    lane_aggregator.py : ROI 폴리곤 기반 4방향 카운트
    run_realtime.py    : CLI 진입점

다음 사이클(이번 작업 범위 외):
    state_encoder.py   : 카운트 dict → SUMO state 11차원 벡터
"""
