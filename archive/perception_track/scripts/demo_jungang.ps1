# Busan Gijang Jungang-sageori (Gukdo 14) demo wrapper.
# Pivoted from Yacheok 2026-05-28 — Yacheok was effectively a straight road,
# while Jungang-sageori is a true 4-way intersection with traffic signals
# matching the v2 SUMO model.
# Usage:
#   $env:ITS_API_KEY = "<key>"
#   .\scripts\demo_jungang.ps1                        # default 5min + JSONL (no display)
#   .\scripts\demo_jungang.ps1 -Display               # open cv2 window
#   .\scripts\demo_jungang.ps1 -DurationSec 60        # short run
#   .\scripts\demo_jungang.ps1 -Roi <path.json>       # ROI applied

param(
    [switch]$Display,
    [int]$DurationSec = 300,
    [string]$Roi = "",
    [string]$Model = "rtdetr-l.pt"
)

if (-not $env:ITS_API_KEY) {
    Write-Host "[ERROR] `$env:ITS_API_KEY required" -ForegroundColor Red
    exit 1
}

# Gijang Jungang-sageori coords ~129.216, 35.251
$bbox = @("129.20", "129.23", "35.24", "35.27")

# Korean "jung-ang-sa-geo-ri" UTF-8 byte sequence (avoid PS5.1 cp949 issues)
# 중(0xEC,0xA4,0x91) 앙(0xEC,0x95,0x99) 사(0xEC,0x82,0xAC) 거(0xEA,0xB1,0xB0) 리(0xEB,0xA6,0xAC)
$CctvName = [System.Text.Encoding]::UTF8.GetString(
    [byte[]](0xEC,0xA4,0x91, 0xEC,0x95,0x99, 0xEC,0x82,0xAC, 0xEA,0xB1,0xB0, 0xEB,0xA6,0xAC)
)

$opts = @(
    "--source", "its-api",
    "--bbox", $bbox[0], $bbox[1], $bbox[2], $bbox[3],
    "--cctv-name", $CctvName,
    "--road-type", "its",
    "--model", $Model,
    "--imgsz", "1280",
    "--conf", "0.15",
    "--min-w", "12",
    "--min-h", "8",
    "--frame-skip", "3",
    "--duration-sec", $DurationSec
)

if ($Display) { $opts += "--display" }
if ($Roi)     { $opts += @("--roi", $Roi) }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = "logs/jungang_$ts.jsonl"
$opts += @("--save-jsonl", $logPath)

Write-Host "[demo] Busan Gijang Jungang-sageori ($DurationSec sec, model=$Model)" -ForegroundColor Cyan
Write-Host "[log]  $logPath" -ForegroundColor Gray

python -m perception.run_realtime @opts
