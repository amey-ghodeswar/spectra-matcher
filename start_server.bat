@echo off
REM ---- Edit these before handing this out to colleagues ----
set RAMAN_APP_USER=labuser
set RAMAN_APP_PASSWORD=yourpassword
set RAMAN_APP_PORT=7860
REM ------------------------------------------------------------

echo Starting Raman/SERS Bacterial Spectra Matcher...
echo (Keep this window open while colleagues are using the tool. Closing it stops the server.)
echo.

python local_app.py

pause
