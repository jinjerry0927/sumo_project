# pres9_v1 — 9차 발표 잔재 (2026-05-20, 보류)

차터 재정립(2026-06-02, Fixed/Webster/SmartSignal **3자 비교**) **이전**에 만들어진
9차 발표 산출물. 당시는 **2자 비교(Fixed vs RL)** 프레임이었고, 이제 폐기된
`results/evaluation_30ep.csv`(30ep)를 참조한다. 현재 본 줄기와 불일치하여 보존만 한다.

| 파일 | 설명 |
|---|---|
| `build.js` | pptxgenjs 9슬라이드 PPT 생성 스크립트 (2자 비교) |
| `package.json` / `package-lock.json` | node 의존성 명세 (`npm install`로 node_modules 재생성) |
| `preview.pptx` | 생성된 발표본 |
| `plot_distribution.py` | 30ep 산점도+박스플롯 (폐기된 evaluation_30ep.csv 참조, 2자) |
| `distribution_30ep.png` | 위 스크립트 출력 |

> 새 발표는 현재 3자 비교 결과(`results/evaluation.csv`)를 기준으로 다시 만든다.
> node_modules는 용량(7.5M) 때문에 제외 — 필요 시 이 폴더에서 `npm install`.
