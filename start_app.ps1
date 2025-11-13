Write-Host "🚀 Starting Parking Management App..." -ForegroundColor Green

# Activate conda environment
conda activate giuxe_new

# Change to project directory  
Set-Location "d:\Documents\IUH_Student\PHAN-MEM-GIU-XE"

# Run the application
python -m phanmemgiuxe.app.main

# Keep window open
Read-Host "Press Enter to exit..."