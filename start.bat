@echo off
title Smart Canteen Estimator
color 0A

echo.
echo  =====================================================
echo   Smart Canteen Crowd ^& Queue Estimator
echo   CV Workshop 24B15CS352 ^| 5th Sem ^| 2026-27
echo  =====================================================
echo.

cd /d "C:\Users\dhruv\smart-canteen-estimator"

echo  [INFO] Starting app at http://localhost:8501
echo  [INFO] Press Ctrl+C to stop.
echo.

"C:\Users\dhruv\AppData\Local\Programs\Python\Python314\Scripts\streamlit.exe" run app.py --server.port 8501 --browser.gatherUsageStats false

echo.
echo  [INFO] App stopped.
pause
