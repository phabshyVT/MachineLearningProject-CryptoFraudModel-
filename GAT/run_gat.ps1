# PowerShell script to run GAT training and evaluation
# Make sure you're in the GAT directory

Write-Host "=== GAT Model Training Script ===" -ForegroundColor Cyan

# Check if data directory exists
if (-not (Test-Path "../data")) {
    Write-Host "Creating data directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "../data" | Out-Null
}

# Check if data files exist
$dataFiles = @(
    "../data/elliptic_txs_features.csv",
    "../data/elliptic_txs_classes.csv",
    "../data/elliptic_txs_edgelist.csv"
)

$missingFiles = @()
foreach ($file in $dataFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "ERROR: Missing data files:" -ForegroundColor Red
    foreach ($file in $missingFiles) {
        Write-Host "  - $file" -ForegroundColor Red
    }
    Write-Host "`nPlease copy your CSV files to the ../data/ directory" -ForegroundColor Yellow
    exit 1
}

Write-Host "Data files found!" -ForegroundColor Green

# Create directories for outputs
New-Item -ItemType Directory -Path "models" -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "images" -ErrorAction SilentlyContinue | Out-Null

# Ask user for training parameters
Write-Host "`nTraining Parameters:" -ForegroundColor Cyan
$epochs = Read-Host "Number of epochs (default: 50)"
if ([string]::IsNullOrWhiteSpace($epochs)) { $epochs = 50 }

$lr = Read-Host "Learning rate (default: 0.01)"
if ([string]::IsNullOrWhiteSpace($lr)) { $lr = 0.01 }

$useGPU = Read-Host "Use GPU? (y/n, default: n)"
if ($useGPU -eq "y" -or $useGPU -eq "Y") {
    $device = "cuda"
} else {
    $device = "cpu"
}

# Run training
Write-Host "`n=== Starting Training ===" -ForegroundColor Green
python train.py --data_dir ../data --epochs $epochs --lr $lr --device $device

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Training Completed Successfully! ===" -ForegroundColor Green
    
    # Ask if user wants to evaluate
    $evaluate = Read-Host "`nEvaluate the model? (y/n, default: y)"
    if ($evaluate -ne "n" -and $evaluate -ne "N") {
        Write-Host "`n=== Starting Evaluation ===" -ForegroundColor Green
        python evaluate.py --model_path models/gat_best_model.pth --data_dir ../data --device $device
    }
} else {
    Write-Host "`nTraining failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "`n=== Done ===" -ForegroundColor Cyan

