param(
    [string]$ImagePath = ".\frames_without_resize_crop\00335\frame_000.jpg",
    [string]$LandmarksPath = ".\npy\00335\frame_000.npy",
    [string]$OutputPath = ".\output_00335_frame_000_landmarks.jpg"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-NpyFloat32Array {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Path))
    $magic = [System.Text.Encoding]::Latin1.GetString($bytes, 0, 6)
    if ($magic -ne "$([char]0x93)NUMPY") {
        throw "Not a .npy file: $Path"
    }

    $major = $bytes[6]
    if ($major -eq 1) {
        $headerLength = [BitConverter]::ToUInt16($bytes, 8)
        $headerOffset = 10
    } else {
        $headerLength = [BitConverter]::ToUInt32($bytes, 8)
        $headerOffset = 12
    }

    $header = [System.Text.Encoding]::Latin1.GetString($bytes, $headerOffset, $headerLength)
    if ($header -notmatch "'descr':\s*'<f4'" -or $header -notmatch "'fortran_order':\s*False") {
        throw "Only little-endian, C-order float32 .npy files are supported."
    }
    if ($header -notmatch "'shape':\s*\((\d+),\s*(\d+)\)") {
        throw "Could not read array shape from .npy header."
    }

    $rows = [int]$Matches[1]
    $cols = [int]$Matches[2]
    $dataOffset = $headerOffset + $headerLength
    $values = New-Object 'float[]' ($rows * $cols)

    for ($i = 0; $i -lt $values.Length; $i++) {
        $values[$i] = [BitConverter]::ToSingle($bytes, $dataOffset + ($i * 4))
    }

    [pscustomobject]@{
        Rows = $rows
        Cols = $cols
        Values = $values
    }
}

function Get-Point {
    param(
        [float[]]$Values,
        [int]$Index,
        [int]$Width,
        [int]$Height
    )

    $x = $Values[$Index * 2]
    $y = $Values[($Index * 2) + 1]
    if (($x -eq 0.0) -and ($y -eq 0.0)) {
        return $null
    }

    [System.Drawing.PointF]::new($x * $Width, $y * $Height)
}

$landmarks = Read-NpyFloat32Array -Path $LandmarksPath
if ($landmarks.Rows -ne 520 -or $landmarks.Cols -ne 2) {
    throw "Expected shape (520, 2), got ($($landmarks.Rows), $($landmarks.Cols))."
}

$image = [System.Drawing.Image]::FromFile((Resolve-Path $ImagePath))
$bitmap = New-Object System.Drawing.Bitmap $image
$image.Dispose()

$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

$handPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(235, 88, 205, 54), 2.0)
$handBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(245, 255, 245, 115))
$faceBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(175, 0, 220, 255))

$handConnections = @(
    @(0, 1), @(1, 2), @(2, 3), @(3, 4),
    @(0, 5), @(5, 6), @(6, 7), @(7, 8),
    @(5, 9), @(9, 10), @(10, 11), @(11, 12),
    @(9, 13), @(13, 14), @(14, 15), @(15, 16),
    @(13, 17), @(0, 17), @(17, 18), @(18, 19), @(19, 20)
)

foreach ($handOffset in @(0, 21)) {
    foreach ($connection in $handConnections) {
        $a = Get-Point -Values $landmarks.Values -Index ($handOffset + $connection[0]) -Width $bitmap.Width -Height $bitmap.Height
        $b = Get-Point -Values $landmarks.Values -Index ($handOffset + $connection[1]) -Width $bitmap.Width -Height $bitmap.Height
        if ($null -ne $a -and $null -ne $b) {
            $graphics.DrawLine($handPen, $a, $b)
        }
    }

    for ($i = 0; $i -lt 21; $i++) {
        $point = Get-Point -Values $landmarks.Values -Index ($handOffset + $i) -Width $bitmap.Width -Height $bitmap.Height
        if ($null -ne $point) {
            $graphics.FillEllipse($handBrush, $point.X - 2.5, $point.Y - 2.5, 5, 5)
        }
    }
}

for ($i = 42; $i -lt 520; $i++) {
    $point = Get-Point -Values $landmarks.Values -Index $i -Width $bitmap.Width -Height $bitmap.Height
    if ($null -ne $point) {
        $graphics.FillEllipse($faceBrush, $point.X - 1.0, $point.Y - 1.0, 2, 2)
    }
}

$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDir = [System.IO.Path]::GetDirectoryName($outputFullPath)
if ($outputDir) {
    [System.IO.Directory]::CreateDirectory($outputDir) | Out-Null
}

$bitmap.Save($outputFullPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)

$graphics.Dispose()
$handPen.Dispose()
$handBrush.Dispose()
$faceBrush.Dispose()
$bitmap.Dispose()

Write-Output "Saved annotated image to $outputFullPath"
