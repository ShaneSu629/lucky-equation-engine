@echo off
chcp 65001 >nul

echo =============================================================
echo     Starting Lottery Data Center and Prediction System...
echo =============================================================
echo.

rem Directly create the .streamlit configuration in user home directory to bypass email prompt
if not exist "%USERPROFILE%\.streamlit" (
    mkdir "%USERPROFILE%\.streamlit"
)

(
echo [general]
echo email = ""
) > "%USERPROFILE%\.streamlit\credentials.toml"

(
echo [server]
echo headless = true
echo [browser]
echo gatherUsageStats = false
) > "%USERPROFILE%\.streamlit\config.toml"

rem Also create local config folder as fallback
if not exist ".streamlit" (
    mkdir ".streamlit"
)
copy /y "%USERPROFILE%\.streamlit\config.toml" ".streamlit\config.toml" >nul 2>&1
copy /y "%USERPROFILE%\.streamlit\credentials.toml" ".streamlit\credentials.toml" >nul 2>&1

echo [Info] Please make sure you have run: pip install -r requirements.txt
echo.
streamlit run app.py --server.headless=true
pause
