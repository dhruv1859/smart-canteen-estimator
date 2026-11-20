# install_python.ps1
# Downloads Python 3.12.8 from python.org and installs it silently with PATH set.
# Run this in PowerShell AS ADMINISTRATOR:
#   powershell -ExecutionPolicy Bypass -File install_python.ps1

$version  = "3.12.8"
$url      = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
$installer = "$env:TEMP\python-installer.exe"

Write-Host "Downloading Python $version ..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing

Write-Host "Installing Python (silent, adds to PATH) ..." -ForegroundColor Cyan
Start-Process -FilePath $installer -ArgumentList `
    "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1", "Include_launcher=1" `
    -Wait

Remove-Item $installer -Force

Write-Host ""
Write-Host "Done! Close this window, open a NEW PowerShell, and run:" -ForegroundColor Green
Write-Host "  python --version" -ForegroundColor Yellow
Write-Host "  pip --version"    -ForegroundColor Yellow

