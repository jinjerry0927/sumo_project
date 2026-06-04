// 9차 발표 PPT 생성 스크립트
// 디자인 시스템 (고정): navy/teal/accent
//   NAVY=0D1B2A, NAVY2=1a2d40, TEAL=00897B, CYAN=00BCD4, GRAY=546E7A
// 레이아웃: 16:9 (10" x 5.625")

const pptxgen = require("pptxgenjs");

const NAVY    = "0D1B2A";
const NAVY2   = "1a2d40";
const TEAL    = "00897B";
const CYAN    = "00BCD4";
const GRAY    = "546E7A";
const WHITE   = "FFFFFF";
const MUTED   = "90A4AE";
const KOR     = "맑은 고딕";          // Malgun Gothic
const ENG     = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";          // 10 x 5.625
pres.title  = "ICT 종합설계 9차 발표";
pres.author = "전형주, 이진욱";

const W = 10.0, H = 5.625;
const TOTAL_SLIDES = 9;

const RESULTS = "../results/";

// ───────── 공통 요소 ─────────
function pageHeader(slide, n, sectionTitle, subTitle) {
  // 좌상단 섹션
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: H, fill: { color: TEAL }, line: { color: TEAL }
  });
  slide.addText(String(n).padStart(2, "0"), {
    x: 0.35, y: 0.2, w: 0.6, h: 0.55,
    fontFace: ENG, fontSize: 28, bold: true, color: CYAN, margin: 0
  });
  slide.addText(sectionTitle, {
    x: 1.0, y: 0.22, w: 6.5, h: 0.45,
    fontFace: KOR, fontSize: 20, bold: true, color: WHITE, margin: 0
  });
  if (subTitle) {
    slide.addText(subTitle, {
      x: 1.0, y: 0.68, w: 7.5, h: 0.3,
      fontFace: KOR, fontSize: 11, color: MUTED, margin: 0
    });
  }
  // 우상단 페이지번호
  slide.addText(`${String(n).padStart(2, "0")} / ${String(TOTAL_SLIDES).padStart(2, "0")}`, {
    x: W - 1.3, y: 0.25, w: 1.1, h: 0.3,
    fontFace: ENG, fontSize: 10, color: MUTED, align: "right", margin: 0
  });
  // 헤더 구분선
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.1, w: W - 0.7, h: 0.015,
    fill: { color: TEAL }, line: { color: TEAL }
  });
}

function statCard(slide, x, y, w, h, label, value, unit, improve, color = TEAL) {
  // 카드 배경
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: NAVY2 }, line: { color, width: 1 }
  });
  // 좌측 액센트 바
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.06, h,
    fill: { color }, line: { color }
  });
  slide.addText(label, {
    x: x + 0.2, y: y + 0.1, w: w - 0.3, h: 0.3,
    fontFace: KOR, fontSize: 10, color: MUTED, margin: 0
  });
  slide.addText([
    { text: value, options: { fontSize: 28, bold: true, color: WHITE } },
    { text: " " + (unit || ""), options: { fontSize: 11, color: MUTED } },
  ], {
    x: x + 0.2, y: y + 0.42, w: w - 0.3, h: 0.55,
    fontFace: ENG, valign: "middle", margin: 0
  });
  if (improve) {
    slide.addText(improve, {
      x: x + 0.2, y: y + h - 0.42, w: w - 0.3, h: 0.32,
      fontFace: ENG, fontSize: 12, bold: true, color: CYAN, margin: 0
    });
  }
}

function footer(slide, n) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.05, w: W, h: 0.05,
    fill: { color: NAVY2 }, line: { color: NAVY2 }
  });
}

// ───────── 1. 표지 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // 좌측 액센트
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.4, h: H, fill: { color: TEAL }, line: { color: TEAL }
  });

  s.addText("ICT 종합설계  |  9차 발표", {
    x: 0.7, y: 0.5, w: 6, h: 0.4,
    fontFace: KOR, fontSize: 13, color: CYAN, margin: 0
  });
  s.addText("강화학습 기반", {
    x: 0.7, y: 0.95, w: 6, h: 0.4,
    fontFace: KOR, fontSize: 18, color: MUTED, margin: 0
  });
  s.addText("스마트 교차로 신호 제어 시스템", {
    x: 0.7, y: 1.4, w: 8.5, h: 0.7,
    fontFace: KOR, fontSize: 32, bold: true, color: WHITE, margin: 0
  });
  s.addText("1000 에피소드 완전 학습 달성 + 정량 성능 평가 + 프로젝트 구조 정리", {
    x: 0.7, y: 2.2, w: 8.5, h: 0.4,
    fontFace: KOR, fontSize: 14, color: CYAN, margin: 0
  });
  s.addText("RL-based Adaptive Traffic Signal Control System", {
    x: 0.7, y: 2.6, w: 8.5, h: 0.3,
    fontFace: ENG, fontSize: 11, italic: true, color: MUTED, margin: 0
  });

  // 4개 키 포인트
  const KP = [
    ["1000ep",    "완전 학습 수렴"],
    ["+98.9%",    "Avg Waiting 감소"],
    ["30ep × 2",  "통계적 평가"],
    ["3 Module",  "구조 재정리"],
  ];
  KP.forEach(([big, lbl], i) => {
    const x = 0.7 + i * 2.15;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.3, w: 1.95, h: 1.1,
      fill: { color: NAVY2 }, line: { color: TEAL, width: 1 }
    });
    s.addText(big, {
      x: x + 0.15, y: 3.4, w: 1.7, h: 0.5,
      fontFace: ENG, fontSize: 22, bold: true, color: CYAN, margin: 0
    });
    s.addText(lbl, {
      x: x + 0.15, y: 3.9, w: 1.7, h: 0.4,
      fontFace: KOR, fontSize: 11, color: MUTED, margin: 0
    });
  });

  // 푸터
  s.addText("과목   ICT 종합설계", {
    x: 0.7, y: 4.7, w: 4, h: 0.3, fontFace: KOR, fontSize: 11, color: MUTED, margin: 0
  });
  s.addText("구성   2인 팀 프로젝트", {
    x: 0.7, y: 5.0, w: 4, h: 0.3, fontFace: KOR, fontSize: 11, color: MUTED, margin: 0
  });
  s.addText("발표자  전형주, 이진욱", {
    x: 5.5, y: 4.85, w: 4, h: 0.3, fontFace: KOR, fontSize: 11, color: MUTED, margin: 0, align: "right"
  });
}

// ───────── 2. 8차 → 9차 변경점 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 2, "8차 → 9차  변경점 요약", "지난 1주간 진행한 핵심 업데이트 4가지");

  // 좌측: 4개 변경 카드 (2x2)
  const changes = [
    {
      lbl: "학습 에피소드",
      from: "405 ep",
      to: "1000 ep",
      note: "ε≈0.131 → ε≈0.05 완전수렴 도달"
    },
    {
      lbl: "평가 방식",
      from: "5 ep 단순 비교",
      to: "30 ep 통계 평가",
      note: "평균 ± 표준편차 산출, 통계적 유의성 확보"
    },
    {
      lbl: "비교 지표",
      from: "Total Reward 1개",
      to: "4개 메트릭",
      note: "Reward · Wait · Queue Avg · Queue Max"
    },
    {
      lbl: "학습 단계",
      from: "탐색 진행 중",
      to: "활용(Exploit) 단계",
      note: "ε_min=0.05 도달, 95% 학습된 정책 사용"
    },
  ];

  changes.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.35 + col * 4.55, y = 1.35 + row * 1.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.4, h: 1.8,
      fill: { color: NAVY2 }, line: { color: TEAL, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.08, h: 1.8, fill: { color: CYAN }, line: { color: CYAN }
    });
    s.addText(c.lbl, {
      x: x + 0.25, y: y + 0.1, w: 4, h: 0.3,
      fontFace: KOR, fontSize: 11, color: MUTED, margin: 0
    });
    // FROM → TO
    s.addText(c.from, {
      x: x + 0.25, y: y + 0.5, w: 1.85, h: 0.45,
      fontFace: KOR, fontSize: 13, color: GRAY, italic: true, margin: 0
    });
    s.addText("→", {
      x: x + 2.1, y: y + 0.5, w: 0.3, h: 0.45,
      fontFace: ENG, fontSize: 18, bold: true, color: CYAN, align: "center", margin: 0
    });
    s.addText(c.to, {
      x: x + 2.4, y: y + 0.5, w: 1.95, h: 0.45,
      fontFace: KOR, fontSize: 13, bold: true, color: WHITE, margin: 0
    });
    // 메모
    s.addText(c.note, {
      x: x + 0.25, y: y + 1.1, w: 4, h: 0.55,
      fontFace: KOR, fontSize: 10.5, color: CYAN, margin: 0
    });
  });

  // 우측 학습 타임라인 (전체 폭 하단 보기로 변경)
  // 핵심 인사이트
  // (제거 - 4 cards 만으로 충분)

  footer(s, 2);
}

// ───────── 3. 1000ep Reward 수렴 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 3, "DQN Reward 수렴 분석  —  1000 에피소드", "탐색→활용 전환 완료, 최종 평균 Reward -7.5 안정화");

  // 좌측: 이미지
  s.addImage({
    path: RESULTS + "reward_convergence_1000.png",
    x: 0.35, y: 1.35, w: 6.3, h: 3.95
  });

  // 우측: 단계별 평균값 마커 카드
  s.addText("학습 단계별 평균 Reward", {
    x: 6.85, y: 1.35, w: 2.8, h: 0.35,
    fontFace: KOR, fontSize: 12, bold: true, color: WHITE, margin: 0
  });
  const stages = [
    ["Initial",  "ep 1~50",     "-27.1", "FF5252"],
    ["ep 200",   "탐색 진행",   "-15.5", "FFA726"],
    ["ep 500",   "수렴 시작",   "-8.7",  "FFEE58"],
    ["Final",    "ep 950~1000", "-7.5",  "69F0AE"],
  ];
  stages.forEach((st, i) => {
    const y = 1.75 + i * 0.55;
    s.addShape(pres.shapes.OVAL, {
      x: 6.85, y: y + 0.05, w: 0.25, h: 0.25,
      fill: { color: st[3] }, line: { color: WHITE, width: 1 }
    });
    s.addText(st[0], {
      x: 7.2, y: y, w: 0.95, h: 0.35,
      fontFace: ENG, fontSize: 12, bold: true, color: WHITE, margin: 0
    });
    s.addText(st[1], {
      x: 7.2, y: y + 0.18, w: 1.1, h: 0.25,
      fontFace: KOR, fontSize: 9, color: MUTED, margin: 0
    });
    s.addText(st[2], {
      x: 8.3, y: y, w: 1.5, h: 0.35,
      fontFace: ENG, fontSize: 16, bold: true, color: CYAN, align: "right", margin: 0
    });
  });

  // 인사이트 박스
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.85, y: 4.2, w: 2.8, h: 1.1,
    fill: { color: NAVY2 }, line: { color: CYAN, width: 1 }
  });
  s.addText("핵심 인사이트", {
    x: 7.0, y: 4.3, w: 2.6, h: 0.3,
    fontFace: KOR, fontSize: 10, bold: true, color: CYAN, margin: 0
  });
  s.addText("• 600 에피소드 이후 ε=0.05 수렴\n• 마지막 50ep 평균 -7.5 (초기 -27.1 대비 +72.3%)\n• ep 850 부근 단발 노이즈 외 모두 안정",  {
    x: 7.0, y: 4.55, w: 2.6, h: 0.75,
    fontFace: KOR, fontSize: 9, color: WHITE, margin: 0
  });

  footer(s, 3);
}

// ───────── 4. 30ep 평가 — Big Stat ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 4, "30 에피소드 정량 평가 결과", "Fixed Signal vs DQN (1000ep) · 60회 시뮬레이션 통계");

  const stats = [
    { label: "Total Reward",      from: "-185.00", to: "-7.04",   imp: "+96.2%", color: TEAL },
    { label: "Avg Waiting Time",  from: "3905.79", to: "44.85",   imp: "+98.9%", color: CYAN },
    { label: "Avg Queue Length",  from: "70.87",   to: "15.01",   imp: "+78.8%", color: TEAL },
    { label: "Max Queue Length",  from: "97.70",   to: "24.53",   imp: "+74.9%", color: TEAL },
  ];

  stats.forEach((st, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.35 + col * 4.55, y = 1.4 + row * 1.95;
    const w = 4.4, h = 1.8;

    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h, fill: { color: NAVY2 }, line: { color: st.color, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.08, h, fill: { color: st.color }, line: { color: st.color }
    });
    s.addText(st.label, {
      x: x + 0.25, y: y + 0.1, w: w - 0.4, h: 0.3,
      fontFace: KOR, fontSize: 11, color: MUTED, margin: 0
    });
    // Fixed 값
    s.addText("Fixed", {
      x: x + 0.25, y: y + 0.5, w: 1.2, h: 0.25,
      fontFace: ENG, fontSize: 10, color: GRAY, margin: 0
    });
    s.addText(st.from, {
      x: x + 0.25, y: y + 0.75, w: 1.5, h: 0.4,
      fontFace: ENG, fontSize: 20, color: GRAY, margin: 0
    });
    // 화살표
    s.addText("→", {
      x: x + 1.75, y: y + 0.65, w: 0.4, h: 0.45,
      fontFace: ENG, fontSize: 22, bold: true, color: CYAN, align: "center", margin: 0
    });
    // RL 값
    s.addText("RL (1000ep)", {
      x: x + 2.15, y: y + 0.5, w: 1.6, h: 0.25,
      fontFace: ENG, fontSize: 10, color: st.color, margin: 0
    });
    s.addText(st.to, {
      x: x + 2.15, y: y + 0.7, w: 2, h: 0.55,
      fontFace: ENG, fontSize: 28, bold: true, color: WHITE, margin: 0
    });
    // 개선율 배지
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.25, y: y + 1.3, w: w - 0.5, h: 0.36,
      fill: { color: NAVY }, line: { color: CYAN, width: 1 }
    });
    s.addText("Improvement  " + st.imp, {
      x: x + 0.25, y: y + 1.3, w: w - 0.5, h: 0.36,
      fontFace: ENG, fontSize: 13, bold: true, color: CYAN, align: "center", valign: "middle", margin: 0
    });
  });

  footer(s, 4);
}

// ───────── 5. 4-panel 비교 그래프 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 5, "Fixed vs RL  —  4개 메트릭 시각화", "평균값 + 표준편차 오차막대로 통계적 신뢰성 표현");

  // 좌측 (전체 폭): 4-panel 이미지
  s.addImage({
    path: RESULTS + "evaluation_chart_1000ep.png",
    x: 0.35, y: 1.3, w: 6.5, h: 4.1
  });

  // 우측: 해석 박스 3개
  const insights = [
    {
      ic: "◆",
      ttl: "통계적 신뢰성",
      desc: "30 에피소드 × 2모드 = 총 60회 시뮬레이션. 표준편차 < 1.5%로 결과 일관됨."
    },
    {
      ic: "◆",
      ttl: "대기시간 핵심 지표",
      desc: "Avg Waiting 3906→45초 (98.9% 감소). 사용자 체감 효과 가장 큰 메트릭."
    },
    {
      ic: "◆",
      ttl: "Queue도 안정화",
      desc: "Max Queue 98→25대로 감소. 피크 혼잡 상황에서도 정체 완화 효과 확인."
    },
  ];
  insights.forEach((ins, i) => {
    const y = 1.4 + i * 1.36;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 7.05, y, w: 2.6, h: 1.2,
      fill: { color: NAVY2 }, line: { color: TEAL, width: 1 }
    });
    s.addText(ins.ic + "  " + ins.ttl, {
      x: 7.2, y: y + 0.08, w: 2.4, h: 0.3,
      fontFace: KOR, fontSize: 11, bold: true, color: CYAN, margin: 0
    });
    s.addText(ins.desc, {
      x: 7.2, y: y + 0.42, w: 2.35, h: 0.75,
      fontFace: KOR, fontSize: 9.5, color: WHITE, margin: 0
    });
  });

  footer(s, 5);
}

// ───────── 6. 30ep 분포 (시연 영상 대체) ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 6, "30 에피소드 전구간 일관성", "한 번의 우연이 아닌 모든 시뮬레이션에서 RL이 우수");

  // 분포 이미지
  s.addImage({
    path: RESULTS + "distribution_30ep.png",
    x: 0.35, y: 1.3, w: 7.6, h: 3.0
  });

  // 우측 해석
  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.1, y: 1.3, w: 1.55, h: 3.0,
    fill: { color: NAVY2 }, line: { color: CYAN, width: 1 }
  });
  s.addText("Why  Boxplot?", {
    x: 8.2, y: 1.4, w: 1.4, h: 0.3,
    fontFace: ENG, fontSize: 11, bold: true, color: CYAN, margin: 0
  });
  s.addText("평균만으로는\n분포의 안정성을\n확인할 수 없음.\n\n산점도+박스플롯으로\n• 분산\n• 이상치\n• 일관성\n동시 확인.",
    {
      x: 8.2, y: 1.7, w: 1.4, h: 2.55,
      fontFace: KOR, fontSize: 9.5, color: WHITE, margin: 0
    });

  // 하단: 핵심 메시지 배너
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 4.5, w: 9.3, h: 0.85,
    fill: { color: NAVY2 }, line: { color: CYAN, width: 1 }
  });
  s.addText([
    { text: "결론  ", options: { fontSize: 13, bold: true, color: CYAN, fontFace: KOR } },
    { text: "60회 시뮬레이션 단 한 번도 Fixed가 RL을 따라잡지 못함.  ", options: { fontSize: 13, color: WHITE, fontFace: KOR } },
    { text: "Gap ≈ 178",                                              options: { fontSize: 13, bold: true, color: CYAN, fontFace: ENG } },
    { text: " (Reward),  ",                                            options: { fontSize: 13, color: WHITE, fontFace: KOR } },
    { text: "Δwait ≈ 3861s",                                            options: { fontSize: 13, bold: true, color: CYAN, fontFace: ENG } },
  ], {
    x: 0.55, y: 4.6, w: 9.0, h: 0.7,
    valign: "middle", margin: 0
  });

  footer(s, 6);
}

// ───────── 7. 프로젝트 3-Module 정리 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 7, "프로젝트 구조 재정리  —  3 Module Architecture",
             "복잡해 보이던 구성요소를 3개 독립 모듈로 명확화");

  const modules = [
    {
      tag: "Module 1",
      title: "AI 신호 정책 학습",
      en: "The Brain",
      stack: "SUMO  +  DQN",
      status: "✓ 완료",
      statusColor: TEAL,
      bullets: [
        "Double DQN + Huber Loss",
        "1000 ep 완전 학습",
        "30 ep 정량 평가 완료",
        "Fixed 대비 +96~99%"
      ]
    },
    {
      tag: "Module 2",
      title: "현실 차량 인식",
      en: "The Eyes",
      stack: "YOLOv10  +  Camera",
      status: "◐ 한계 분석",
      statusColor: "FFA726",
      bullets: [
        "주간 신호등 시점 13대 탐지",
        "야간 CCTV 한계 식별 (8차)",
        "다음: yolov10s/m 교체",
        "다음: ROI 4분할 카운팅"
      ]
    },
    {
      tag: "Module 3",
      title: "하드웨어 시연",
      en: "The Body",
      stack: "RPi 5  +  LED  +  TraCI",
      status: "○ 진행 예정",
      statusColor: GRAY,
      bullets: [
        "라즈베리파이 5 도착 대기",
        "TraCI → GPIO 변환 설계",
        "4방향 LED 신호등 제작",
        "미니어처 교차로 시연"
      ]
    },
  ];

  modules.forEach((m, i) => {
    const x = 0.35 + i * 3.18, y = 1.3;
    const w = 3.0, h = 3.5;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h, fill: { color: NAVY2 }, line: { color: TEAL, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w, h: 0.65,
      fill: { color: TEAL }, line: { color: TEAL }
    });
    s.addText(m.tag, {
      x: x + 0.18, y: y + 0.07, w: w - 0.35, h: 0.25,
      fontFace: ENG, fontSize: 10, bold: true, color: NAVY, margin: 0
    });
    s.addText(m.title, {
      x: x + 0.18, y: y + 0.3, w: w - 0.35, h: 0.32,
      fontFace: KOR, fontSize: 14, bold: true, color: WHITE, margin: 0
    });
    s.addText(m.en, {
      x: x + 0.18, y: y + 0.78, w: w - 0.35, h: 0.25,
      fontFace: ENG, fontSize: 11, italic: true, color: CYAN, margin: 0
    });
    s.addText(m.stack, {
      x: x + 0.18, y: y + 1.05, w: w - 0.35, h: 0.3,
      fontFace: ENG, fontSize: 12, bold: true, color: WHITE, margin: 0
    });
    // 상태 배지
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.18, y: y + 1.4, w: w - 0.35, h: 0.35,
      fill: { color: NAVY }, line: { color: m.statusColor, width: 1 }
    });
    s.addText(m.status, {
      x: x + 0.18, y: y + 1.4, w: w - 0.35, h: 0.35,
      fontFace: KOR, fontSize: 11, bold: true, color: m.statusColor,
      align: "center", valign: "middle", margin: 0
    });
    // 불릿
    s.addText(m.bullets.map((b, idx) => ({
      text: b,
      options: { bullet: { code: "25A0" }, breakLine: idx < m.bullets.length - 1 }
    })), {
      x: x + 0.2, y: y + 1.85, w: w - 0.35, h: 1.55,
      fontFace: KOR, fontSize: 10, color: WHITE, paraSpaceAfter: 4, margin: 0
    });
  });

  // 하단 통합 흐름
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 4.95, w: 9.3, h: 0.42,
    fill: { color: NAVY2 }, line: { color: CYAN, width: 1 }
  });
  s.addText([
    { text: "최종 통합 흐름   ", options: { fontSize: 11, bold: true, color: CYAN, fontFace: KOR } },
    { text: "현실 카메라  →  YOLO 차량 수  →  DQN 신호 결정  →  LED 신호등 출력",
      options: { fontSize: 11, color: WHITE, fontFace: KOR } },
  ], {
    x: 0.55, y: 4.95, w: 9.0, h: 0.42, valign: "middle", margin: 0
  });

  footer(s, 7);
}

// ───────── 8. 남은 작업 + 발표 계획 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 8, "남은 작업 + 발표 로드맵", "Module 2/3 통합과 시연 준비로 마무리");

  // 좌측: 차주별 발표 계획 타임라인
  s.addText("주차별 발표 계획", {
    x: 0.35, y: 1.3, w: 5, h: 0.35,
    fontFace: KOR, fontSize: 13, bold: true, color: CYAN, margin: 0
  });

  const plans = [
    { wk: "10차", title: "YOLOv10 모듈 고도화", desc: "주간 신호등 시점 영상 + yolov10s 교체 + 방향별 ROI 4분할 카운팅" },
    { wk: "11차", title: "라즈베리파이 통신 설계", desc: "TraCI → 시리얼/WiFi → GPIO. 4방향 LED 점등 프로토타입" },
    { wk: "12차", title: "End-to-End 통합", desc: "카메라 영상 → YOLO → DQN → LED 전체 파이프라인 연결" },
    { wk: "최종", title: "미니어처 시연 + 보고서", desc: "폼보드 교차로 + 실제 LED + SUMO 비교 영상" },
  ];

  plans.forEach((p, i) => {
    const y = 1.75 + i * 0.85;
    // 차수 박스
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y, w: 0.85, h: 0.7,
      fill: { color: TEAL }, line: { color: TEAL }
    });
    s.addText(p.wk, {
      x: 0.35, y, w: 0.85, h: 0.7,
      fontFace: ENG, fontSize: 14, bold: true, color: NAVY,
      align: "center", valign: "middle", margin: 0
    });
    // 내용 박스
    s.addShape(pres.shapes.RECTANGLE, {
      x: 1.3, y, w: 4.0, h: 0.7,
      fill: { color: NAVY2 }, line: { color: TEAL, width: 1 }
    });
    s.addText(p.title, {
      x: 1.45, y: y + 0.06, w: 3.8, h: 0.3,
      fontFace: KOR, fontSize: 11, bold: true, color: WHITE, margin: 0
    });
    s.addText(p.desc, {
      x: 1.45, y: y + 0.34, w: 3.8, h: 0.35,
      fontFace: KOR, fontSize: 8.5, color: MUTED, margin: 0
    });
  });

  // 우측: 선택적 작업 + 리스크
  s.addText("선택 작업 (시간 허용 시)", {
    x: 5.6, y: 1.3, w: 4.1, h: 0.35,
    fontFace: KOR, fontSize: 13, bold: true, color: CYAN, margin: 0
  });

  const optionals = [
    { ic: "+", ttl: "v2 환경 (3차선 × 4-phase)", desc: "보호좌회전 포함. 환경 코드 완성, 학습만 12시간+ 소요" },
    { ic: "+", ttl: "긴급차량 우선신호",        desc: "3차 발표에서 제안한 기능. 사운드센서 + 영상 이중 감지" },
    { ic: "+", ttl: "Multi-Intersection",       desc: "단일 교차로 → 2~3개 연동. MARL 적용 검토" },
  ];

  optionals.forEach((o, i) => {
    const y = 1.75 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.6, y, w: 4.1, h: 0.7,
      fill: { color: NAVY2 }, line: { color: GRAY, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.6, y, w: 0.55, h: 0.7,
      fill: { color: NAVY }, line: { color: GRAY, width: 1 }
    });
    s.addText(o.ic, {
      x: 5.6, y, w: 0.55, h: 0.7,
      fontFace: ENG, fontSize: 22, bold: true, color: CYAN,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(o.ttl, {
      x: 6.25, y: y + 0.06, w: 3.4, h: 0.3,
      fontFace: KOR, fontSize: 11, bold: true, color: WHITE, margin: 0
    });
    s.addText(o.desc, {
      x: 6.25, y: y + 0.34, w: 3.4, h: 0.35,
      fontFace: KOR, fontSize: 8.5, color: MUTED, margin: 0
    });
  });

  // 하단 강조 메시지
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 5.15, w: 9.3, h: 0.32,
    fill: { color: NAVY2 }, line: { color: CYAN, width: 1 }
  });
  s.addText("Module 1(AI Brain) 완성  →  Module 2 + 3 통합으로 진입하는 시점", {
    x: 0.55, y: 5.15, w: 9.0, h: 0.32,
    fontFace: KOR, fontSize: 11, bold: true, color: CYAN,
    valign: "middle", margin: 0
  });

  footer(s, 8);
}

// ───────── 9. 최종 시연 컨셉 ─────────
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  pageHeader(s, 9, "최종 시연 컨셉  —  End-to-End 데모",
             "졸업 작품 발표일 기준 시연 시나리오");

  // 흐름도 (5개 노드, 좌→우)
  const flow = [
    { ic: "📹", ko: "카메라 입력",  en: "Real Camera",      c: TEAL },
    { ic: "🔍", ko: "YOLO 탐지",   en: "Vehicle Detect",   c: TEAL },
    { ic: "🔢", ko: "방향별 집계", en: "Direction Count",  c: TEAL },
    { ic: "🧠", ko: "DQN 결정",    en: "Policy Inference", c: CYAN },
    { ic: "🚦", ko: "LED 출력",    en: "GPIO Control",     c: CYAN },
  ];

  const nodeW = 1.7, nodeH = 1.6, gap = 0.15;
  const totalW = nodeW * 5 + gap * 4;
  const startX = (W - totalW) / 2;

  flow.forEach((n, i) => {
    const x = startX + i * (nodeW + gap);
    const y = 1.6;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: nodeW, h: nodeH,
      fill: { color: NAVY2 }, line: { color: n.c, width: 1.5 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: nodeW, h: 0.3,
      fill: { color: n.c }, line: { color: n.c }
    });
    s.addText(`Stage ${i+1}`, {
      x: x + 0.1, y: y + 0.03, w: nodeW - 0.2, h: 0.25,
      fontFace: ENG, fontSize: 9, bold: true, color: NAVY, margin: 0
    });
    s.addText(n.ic, {
      x, y: y + 0.4, w: nodeW, h: 0.5,
      fontFace: KOR, fontSize: 26, align: "center", margin: 0
    });
    s.addText(n.ko, {
      x: x + 0.05, y: y + 0.95, w: nodeW - 0.1, h: 0.3,
      fontFace: KOR, fontSize: 12, bold: true, color: WHITE, align: "center", margin: 0
    });
    s.addText(n.en, {
      x: x + 0.05, y: y + 1.25, w: nodeW - 0.1, h: 0.25,
      fontFace: ENG, fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0
    });
    // 화살표
    if (i < flow.length - 1) {
      s.addText("▶", {
        x: x + nodeW - 0.04, y: y + nodeH/2 - 0.12, w: gap + 0.1, h: 0.3,
        fontFace: ENG, fontSize: 13, color: CYAN, align: "center", valign: "middle", margin: 0
      });
    }
  });

  // 하단: 시연 환경 + 측정 메트릭 두 박스
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 3.55, w: 4.55, h: 1.6,
    fill: { color: NAVY2 }, line: { color: TEAL, width: 1 }
  });
  s.addText("◆  시연 환경", {
    x: 0.55, y: 3.65, w: 4.2, h: 0.35,
    fontFace: KOR, fontSize: 12, bold: true, color: CYAN, margin: 0
  });
  s.addText([
    { text: "폼보드 교차로 (4-way)", options: { bullet: { code: "25A0" }, breakLine: true } },
    { text: "4방향 LED 신호등 + 라즈베리파이 5", options: { bullet: { code: "25A0" }, breakLine: true } },
    { text: "노트북: SUMO + DQN 추론", options: { bullet: { code: "25A0" }, breakLine: true } },
    { text: "전시용 안내 화면 (RL 결정 시각화)", options: { bullet: { code: "25A0" } } },
  ], {
    x: 0.55, y: 3.95, w: 4.2, h: 1.15,
    fontFace: KOR, fontSize: 10, color: WHITE, paraSpaceAfter: 3, margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 3.55, w: 4.55, h: 1.6,
    fill: { color: NAVY2 }, line: { color: CYAN, width: 1 }
  });
  s.addText("◆  Fixed vs RL 라이브 비교", {
    x: 5.3, y: 3.65, w: 4.2, h: 0.35,
    fontFace: KOR, fontSize: 12, bold: true, color: CYAN, margin: 0
  });
  s.addText([
    { text: "동일 차량 흐름으로 두 모드 시연", options: { bullet: { code: "25A0" }, breakLine: true } },
    { text: "Avg Wait 실시간 표시 (3906s vs 45s)", options: { bullet: { code: "25A0" }, breakLine: true } },
    { text: "Queue 길이 누적 그래프 동시 표시", options: { bullet: { code: "25A0" }, breakLine: true } },
    { text: "차량 모형으로 시각적 임팩트 강화", options: { bullet: { code: "25A0" } } },
  ], {
    x: 5.3, y: 3.95, w: 4.2, h: 1.15,
    fontFace: KOR, fontSize: 10, color: WHITE, paraSpaceAfter: 3, margin: 0
  });

  // 마무리 강조
  s.addText("Thank You", {
    x: 0.35, y: 5.2, w: 9.3, h: 0.35,
    fontFace: ENG, fontSize: 12, italic: true, bold: true, color: MUTED,
    align: "center", margin: 0
  });

  footer(s, 9);
}

// ───────── 저장 ─────────
const OUT = "C:/Users/jinuk/Desktop/4학년 1학기/ICT종합설계1/발표자료/ICT종합설계_9차발표전형주이진욱.pptx";
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("[OK] saved:", OUT);
});
