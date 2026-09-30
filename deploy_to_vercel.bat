@echo off
title Push to GitHub and Deploy to Vercel
echo ========================================================
echo 1. Pushing code to GitHub (main branch)...
echo ========================================================
git push -u origin main
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Git push failed. Please make sure the repository
    echo 'jev-momentum-radar' exists on https://github.com/lioncitydevops/
    echo and that you have permission to push to it.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo 2. Deploying project to Vercel...
echo ========================================================
call npx.cmd vercel --prod
pause
