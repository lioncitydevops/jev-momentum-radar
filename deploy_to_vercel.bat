@echo off
title Push to GitHub and Deploy to Vercel
cd /d "%~dp0"

echo ========================================================
echo 1. Pushing code to GitHub (main branch)...
echo ========================================================
git -c credential.helper= -c credential.helper=manager push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [NOTE] If prompted above or in your browser, please approve GitHub login.
    echo.
)

echo.
echo ========================================================
echo 2. Deploying project to Vercel...
echo ========================================================
call npx.cmd vercel --prod

echo.
echo ========================================================
echo Deployment process finished.
echo ========================================================
pause
