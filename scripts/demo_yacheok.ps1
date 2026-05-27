# Yacheok intersection (Gukdo 7) demo wrapper.
# Mohwa is not in ITS DB -> pivoted to Yacheok on 2026-05-27.
# Usage:
#   $env:ITS_API_KEY = "<key>"
#   .\scripts\demo_yacheok.ps1                        # default 5min live + JSONL (no display)
#   .\scripts\demo_yacheok.ps1 -Display               # open cv2 window
#   .\scripts\demo_yacheok.ps1 -DurationSec 60        # short run
#   .\scripts\demo_yacheok.ps1 -Roi <path.json>       # ROI applied

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

# Yacheok coords ~129.156, 35.860
$bbox = @("129.14", "129.18", "35.84", "35.88")

# Use UTF-8 byte sequence for cctv-name substring to avoid PS encoding issues
# Korean "yacheok-kyochalo" = 0xC57C 0xCC99 0xAD50 0xCC28 0xB85C
$CctvName = [System.Text.Encoding]::UTF8.GetString(
    [byte[]](0xEC,0x95,0xBC, 0xEC,0xB2,0x99, 0xEA,0xB5,0x90, 0xEC,0xB0,0xA8, 0xEB,0xA1,0x9C)
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
$logPath = "logs/yacheok_$ts.jsonl"
$opts += @("--save-jsonl", $logPath)

Write-Host "[demo] Yacheok ($DurationSec sec, model=$Model)" -ForegroundColor Cyan
Write-Host "[log]  $logPath" -ForegroundColor Gray

python -m perception.run_realtime @opts
