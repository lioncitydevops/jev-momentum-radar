@echo off
title S&P 500 Momentum Signal - TypeSafe Jev
echo ========================================================
echo Starting S&P 500 Momentum Signal App...
echo Opening in your browser in a few seconds...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run dashboard.py
pause
