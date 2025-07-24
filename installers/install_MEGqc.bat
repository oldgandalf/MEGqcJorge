@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo       MEGqc Portable Installer (Windows)
echo ==================================================

:: 1/7) Configuration
set "BASEDIR=%USERPROFILE%\MEGqc"
set "PYDIR=%BASEDIR%\python310"
set "ENVDIR=%BASEDIR%\env"
set "ZIPURL=https://raw.githubusercontent.com/karellopez/BIDS-Manager/main/external/python-embed/python-3.10.11-embed-amd64.zip"

:: Determine Desktop path (Win10 & Win11)
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP=%%D"

:: ────────────────────────────────────────────────────
:: 2/7) Create base folder
:: ────────────────────────────────────────────────────
echo [1/7] Creating base folder at %BASEDIR%...
if not exist "%BASEDIR%" mkdir "%BASEDIR%"
if errorlevel 1 (
  echo ERROR: Could not create base folder
  pause & exit /b 1
)
cd /d "%BASEDIR%"
if errorlevel 1 (
  echo ERROR: Could not change directory
  pause & exit /b 1
)

:: ────────────────────────────────────────────────────
:: 3/7) Download & extract embeddable Python
:: ────────────────────────────────────────────────────
echo [2/7] Downloading embeddable Python...
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%ZIPURL%' -OutFile python310.zip"
if not exist python310.zip (
  echo ERROR: Download failed
  pause & exit /b 1
)

echo [3/7] Extracting Python...
powershell -NoProfile -Command "Expand-Archive -Force python310.zip python310"
if not exist "%PYDIR%\python.exe" (
  echo ERROR: Extraction failed
  pause & exit /b 1
)
del python310.zip

:: ────────────────────────────────────────────────────
:: 4/7) Bootstrap pip
:: ────────────────────────────────────────────────────
echo [4/7] Bootstrapping pip...
powershell -NoProfile -Command ^
  "(Get-Content '%PYDIR%\python310._pth') -replace '^#\s*import site','import site' | Set-Content '%PYDIR%\python310._pth'"
powershell -NoProfile -Command "Invoke-WebRequest https://bootstrap.pypa.io/get-pip.py -OutFile '%PYDIR%\get-pip.py'"
"%PYDIR%\python.exe" "%PYDIR%\get-pip.py"
if errorlevel 1 (
  echo ERROR: pip bootstrap failed
  pause & exit /b 1
)
del "%PYDIR%\get-pip.py"

:: ────────────────────────────────────────────────────
:: 5/7) Create virtual environment
:: ────────────────────────────────────────────────────
echo [5/7] Installing virtualenv and creating venv...
"%PYDIR%\python.exe" -m pip install virtualenv || (
  echo ERROR: virtualenv install failed
  pause & exit /b 1
)
"%PYDIR%\python.exe" -m virtualenv "%ENVDIR%" || (
  echo ERROR: venv creation failed
  pause & exit /b 1
)

:: ────────────────────────────────────────────────────
:: 6/7) Install MEGqc (no error checks here)
:: ────────────────────────────────────────────────────
echo [6/7] Installing MEGqc...
call "%ENVDIR%\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel versioningit
python -m pip install git+https://github.com/ANCPLabOldenburg/MEGqc.git

:: ────────────────────────────────────────────────────
:: 7/7) Create launcher & uninstaller on Desktop
:: ────────────────────────────────────────────────────
echo [7/7] Writing launcher and uninstaller to Desktop...

:: Launcher: calls “megqc” after activation
>"%DESKTOP%\run_MEGqc.bat" (
  echo @echo off
  echo call "%BASEDIR%\env\Scripts\activate.bat"
  echo megqc %%*
)

:: Uninstaller: removes MEGqc folder and these two .bat files
>"%DESKTOP%\uninstall_MEGqc.bat" (
  echo @echo off
  echo echo Removing MEGqc installation...
  echo rmdir /s /q "%BASEDIR%"
  echo del /q "%DESKTOP%\run_MEGqc.bat"
  echo del /q "%DESKTOP%\uninstall_MEGqc.bat"
  echo pause
)

echo.
echo ✅ INSTALLATION COMPLETE!
echo You now have two shortcuts on your Desktop:
echo   • run_MEGqc.bat  
echo   • uninstall_MEGqc.bat  
pause
exit /b 0
