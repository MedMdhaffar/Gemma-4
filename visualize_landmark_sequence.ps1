param(
    [string]$FramesDir = ".\frames_without_resize_crop\00335",
    [string]$LandmarksPath = ".\preprocessed_nslt300\landmarks\00335.npy",
    [string]$MaskPath = "",
    [string]$OutputDir = ".\landmark_preview",
    [ValidateSet("auto", "absolute", "normalized")]
    [string]$Mode = "auto",
    [float]$NormalizedScale = 0.28
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
    if ($header -notmatch "'shape':\s*\(([^\)]*)\)") {
        throw "Could not read array shape from .npy header."
    }

    $shape = @()
    foreach ($part in ($Matches[1] -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed.Length -gt 0) {
            $shape += [int]$trimmed
        }
    }

    $count = 1
    foreach ($dim in $shape) {
        $count *= $dim
    }

    $dataOffset = $headerOffset + $headerLength
    $values = New-Object 'float[]' $count
    for ($i = 0; $i -lt $values.Length; $i++) {
        $values[$i] = [BitConverter]::ToSingle($bytes, $dataOffset + ($i * 4))
    }

    [pscustomobject]@{
        Shape = $shape
        Values = $values
    }
}

function Read-MaskArray {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    $mask = Read-NpyFloat32Array -Path $Path
    if ($mask.Shape.Count -ne 2) {
        throw "Expected mask shape (T, V), got ($($mask.Shape -join ', '))."
    }
    return $mask
}

function Get-SequenceInfo {
    param($Array)

    $shape = $Array.Shape

    if ($shape.Count -eq 4) {
        return [pscustomobject]@{
            Kind = "stgcn"
            T = $shape[1]
            V = $shape[2]
            C = $shape[0]
        }
    }

    if ($shape.Count -eq 3) {
        return [pscustomobject]@{
            Kind = "sequence"
            T = $shape[0]
            V = $shape[1]
            C = $shape[2]
        }
    }

    if ($shape.Count -eq 2) {
        return [pscustomobject]@{
            Kind = "single"
            T = 1
            V = $shape[0]
            C = $shape[1]
        }
    }

    throw "Unsupported landmark shape: ($($shape -join ', '))."
}

function Get-LandmarkValue {
    param(
        $Array,
        $Info,
        [int]$T,
        [int]$V,
        [int]$C
    )

    if ($Info.Kind -eq "stgcn") {
        $m = 0
        $idx = (((($C * $Info.T) + $T) * $Info.V) + $V) + $m
        return $Array.Values[$idx]
    }

    if ($Info.Kind -eq "sequence") {
        $idx = ((($T * $Info.V) + $V) * $Info.C) + $C
        return $Array.Values[$idx]
    }

    $idx = ($V * $Info.C) + $C
    return $Array.Values[$idx]
}

function Get-MaskValue {
    param(
        $Mask,
        [int]$T,
        [int]$V
    )

    if ($null -eq $Mask) {
        return 1.0
    }

    $maskV = $Mask.Shape[1]
    return $Mask.Values[($T * $maskV) + $V]
}

function Get-Point {
    param(
        $Array,
        $Info,
        $Mask,
        [int]$T,
        [int]$V,
        [int]$Width,
        [int]$Height,
        [string]$DrawMode,
        [float]$Scale
    )

    if ($V -ge $Info.V) {
        return $null
    }

    if ((Get-MaskValue -Mask $Mask -T $T -V $V) -le 0.0) {
        return $null
    }

    $x = Get-LandmarkValue -Array $Array -Info $Info -T $T -V $V -C 0
    $y = Get-LandmarkValue -Array $Array -Info $Info -T $T -V $V -C 1

    if (($x -eq 0.0) -and ($y -eq 0.0) -and ($null -eq $Mask)) {
        return $null
    }

    if ($DrawMode -eq "absolute") {
        return [System.Drawing.PointF]::new($x * $Width, $y * $Height)
    }

    $pixelScale = [Math]::Min($Width, $Height) * $Scale
    return [System.Drawing.PointF]::new(($Width / 2.0) + ($x * $pixelScale), ($Height / 2.0) + ($y * $pixelScale))
}

function Draw-Connection {
    param(
        $Graphics,
        $Pen,
        $Array,
        $Info,
        $Mask,
        [int]$T,
        [int]$A,
        [int]$B,
        [int]$Width,
        [int]$Height,
        [string]$DrawMode,
        [float]$Scale
    )

    $pa = Get-Point -Array $Array -Info $Info -Mask $Mask -T $T -V $A -Width $Width -Height $Height -DrawMode $DrawMode -Scale $Scale
    $pb = Get-Point -Array $Array -Info $Info -Mask $Mask -T $T -V $B -Width $Width -Height $Height -DrawMode $DrawMode -Scale $Scale
    if ($null -ne $pa -and $null -ne $pb) {
        $Graphics.DrawLine($Pen, $pa, $pb)
    }
}

$landmarks = Read-NpyFloat32Array -Path $LandmarksPath
$info = Get-SequenceInfo -Array $landmarks
$mask = Read-MaskArray -Path $MaskPath

$drawMode = $Mode
if ($drawMode -eq "auto") {
    if ($info.Kind -eq "stgcn") {
        $drawMode = "normalized"
    } else {
        $drawMode = "absolute"
    }
}

$frames = Get-ChildItem -Path $FramesDir -File |
    Where-Object { $_.Extension -in @(".jpg", ".jpeg", ".png") } |
    Sort-Object Name

if ($frames.Count -eq 0) {
    throw "No frames found in $FramesDir"
}
if ($info.T -ne 1 -and $frames.Count -ne $info.T) {
    throw "Frame count ($($frames.Count)) does not match landmark timesteps ($($info.T))."
}

$outputFullDir = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($outputFullDir) | Out-Null

$handConnections = @(
    @(0, 1), @(1, 2), @(2, 3), @(3, 4),
    @(0, 5), @(5, 6), @(6, 7), @(7, 8),
    @(0, 9), @(9, 10), @(10, 11), @(11, 12),
    @(0, 13), @(13, 14), @(14, 15), @(15, 16),
    @(0, 17), @(17, 18), @(18, 19), @(19, 20)
)
$poseConnections = @(@(42, 43), @(42, 44), @(44, 46), @(43, 45), @(45, 47))

$leftHandPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(235, 80, 220, 80), 2.0)
$rightHandPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(235, 80, 180, 255), 2.0)
$posePen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(235, 255, 170, 80), 2.5)
$leftHandBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(245, 80, 220, 80))
$rightHandBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(245, 80, 180, 255))
$poseBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(245, 255, 170, 80))
$faceBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(220, 255, 80, 220))
$labelBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(190, 15, 15, 15))
$textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
$font = [System.Drawing.Font]::new("Arial", 11)

try {
    $frameCount = if ($info.T -eq 1) { 1 } else { $info.T }
    for ($t = 0; $t -lt $frameCount; $t++) {
        $framePath = if ($info.T -eq 1) { $frames[0].FullName } else { $frames[$t].FullName }
        $image = [System.Drawing.Image]::FromFile($framePath)
        $bitmap = New-Object System.Drawing.Bitmap $image
        $image.Dispose()

        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

        foreach ($connection in $handConnections) {
            Draw-Connection -Graphics $graphics -Pen $leftHandPen -Array $landmarks -Info $info -Mask $mask -T $t -A $connection[0] -B $connection[1] -Width $bitmap.Width -Height $bitmap.Height -DrawMode $drawMode -Scale $NormalizedScale
            Draw-Connection -Graphics $graphics -Pen $rightHandPen -Array $landmarks -Info $info -Mask $mask -T $t -A (21 + $connection[0]) -B (21 + $connection[1]) -Width $bitmap.Width -Height $bitmap.Height -DrawMode $drawMode -Scale $NormalizedScale
        }
        foreach ($connection in $poseConnections) {
            Draw-Connection -Graphics $graphics -Pen $posePen -Array $landmarks -Info $info -Mask $mask -T $t -A $connection[0] -B $connection[1] -Width $bitmap.Width -Height $bitmap.Height -DrawMode $drawMode -Scale $NormalizedScale
        }

        for ($v = 0; $v -lt $info.V; $v++) {
            $point = Get-Point -Array $landmarks -Info $info -Mask $mask -T $t -V $v -Width $bitmap.Width -Height $bitmap.Height -DrawMode $drawMode -Scale $NormalizedScale
            if ($null -eq $point) {
                continue
            }

            if ($v -lt 21) {
                $graphics.FillEllipse($leftHandBrush, $point.X - 3, $point.Y - 3, 6, 6)
            } elseif ($v -lt 42) {
                $graphics.FillEllipse($rightHandBrush, $point.X - 3, $point.Y - 3, 6, 6)
            } elseif ($v -lt 48) {
                $graphics.FillEllipse($poseBrush, $point.X - 4, $point.Y - 4, 8, 8)
            } else {
                $graphics.FillEllipse($faceBrush, $point.X - 2, $point.Y - 2, 4, 4)
            }
        }

        $label = "$(Split-Path $LandmarksPath -Leaf) | frame $("{0:D2}" -f $t) | $drawMode"
        $graphics.FillRectangle($labelBrush, 8, 8, 390, 28)
        $graphics.DrawString($label, $font, $textBrush, 14, 13)

        $outputPath = Join-Path $outputFullDir ("frame_{0:D3}.jpg" -f $t)
        $bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Jpeg)

        $graphics.Dispose()
        $bitmap.Dispose()
    }
} finally {
    $leftHandPen.Dispose()
    $rightHandPen.Dispose()
    $posePen.Dispose()
    $leftHandBrush.Dispose()
    $rightHandBrush.Dispose()
    $poseBrush.Dispose()
    $faceBrush.Dispose()
    $labelBrush.Dispose()
    $textBrush.Dispose()
    $font.Dispose()
}

Write-Output "Saved annotated frames to $outputFullDir"
