# 경주 모화사거리 전용 데모 실행 wrapper.
# 사용법:
#   $env:ITS_API_KEY = "키"
#   .\scripts\demo_mohwa.ps1                  # 기본: ROI 적용
#   .\scripts\demo_mohwa.ps1 -NoRoi           # ROI 끔 (전체 카운트만)
#   .\scripts\demo_mohwa.ps1 -SaveLog         # JSONL 로그까지

param(
    [switch]$NoRoi,
    [switch]$SaveLog,
    [string]$Roi = "perception/roi_config/mohwa_template.json"
)

if (-not $env:ITS_API_KEY) {
    Write-Host "[ERROR] `$env:ITS_API_KEY 설정 필요" -ForegroundColor Red
    exit 1
}

# 모화사거리 좌표(약 129.331, 35.684) 좁게 잡음 → 응답 빠르고 카메라 1개만
$bbox = @("129.30", "129.36", "35.65", "35.72")

# 모화사거리 카메라(720x480)에 튜닝된 옵션
$opts = @(
    "--source", "its-api",
    "--bbox", $bbox[0], $bbox[1], $bbox[2], $bbox[3],
    "--cctv-name", "모화사거리",
    "--road-type", "its",
    "--display",
    "--model", "yolov8s.pt",
    "--imgsz", "1280",      # 작은 객체 탐지
    "--conf", "0.15",       # 멀리 있는 약한 탐지도 포함
    "--min-w", "12",        # 720x480에 맞춤
    "--min-h", "8",
    "--frame-skip", "3"     # 화면 부드럽게
)

if (-not $NoRoi) {
    $opts += @("--roi", $Roi)
}

if ($SaveLog) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $opts += @("--save-jsonl", "logs/mohwa_$ts.jsonl")
    Write-Host "[log] logs/mohwa_$ts.jsonl"
}

Write-Host "[demo] 모화사거리 데모 시작 (q=종료, s=스크린샷)" -ForegroundColor Cyan
python -m perception.run_realtime @opts
