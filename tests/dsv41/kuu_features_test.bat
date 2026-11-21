@echo off
rem kuu_features_test.bat - the usual engine smoke (tests/dsv41/short_prompt_check.py: the pack loaded,
rem three 16-token GENs, fail on any ERR / engine death / illegal-access stderr) run ONCE PER PORTED
rem kuu feature, one feature at a time with every other switch off, then once with all the exact ones
rem together.  PASS/FAIL per run on the console, a full engine log per run in the log folder, and a
rem summary at the end.  Smoke only: it does not measure speed, and the 40-token prompt runs the windows
rem path, so EG_RING and LM_AHEAD build their buffers and idle rather than prefetch - their timing is a
rem model-length run's job (docs/KUU_DSV41.md).
rem
rem usage:  kuu_features_test.bat [exe] [pack]
rem Each run loads the whole pack (minutes, lots of RAM) and wants the VRAM to itself: stop the other
rem servers first.  One run failing does not stop the rest.
setlocal EnableDelayedExpansion

set "EXE=%~1"
if "%EXE%"=="" set "EXE=L:\Temp\opencode\up-cuda-build\strata-dsv41.exe"
set "PACK=%~2"
if "%PACK%"=="" set "PACK=L:\AI_pictures_generate\PolyStrata-data\packs\deepseek-v4.1-1017-ud-iq1-s"
set "SCRIPT=%~dp0short_prompt_check.py"
set "IDS=%~dp0kuu_feature_ids.txt"
set "LOGD=%TEMP%\kuu-feature-logs"
rem the user's usual engine args (from strata-deepseek-v4.1-1017-ud-iq1-s-upstream.json); appended after
rem the script's own defaults, so they win, as in the live server
set "ARGS=--max-context 262144 --eos 1 --dspark auto --cpu-threads 12 --prefill-chunk 1024 --reserve-mib 300 --vram-experts 0.85 --verbose"

if not exist "%LOGD%" mkdir "%LOGD%"
set "SUM=%LOGD%\summary.txt"
echo kuu feature sweep %DATE% %TIME%> "%SUM%"

rem the ids file: 40 token ids, made once (the script reads the first 40 anyway)
if not exist "%IDS%" (
    set "LIST="
    for /l %%i in (100,1,139) do set "LIST=!LIST!,%%i"
    echo !LIST:~1!> "%IDS%"
)

if not exist "%EXE%"  ( echo EXE not found: %EXE%  - build it or pass it as the first argument & exit /b 2 )
if not exist "%PACK%" ( echo PACK not found: %PACK% & exit /b 2 )

echo(
echo Engine: %EXE%
echo Pack:   %PACK%
echo Logs:   %LOGD%
echo(

call :run baseline
call :run router_bf16     "STRATA_DS4_ROUTER_BF16=1"
call :run uniform_store   "DS4_STORE_UNIFORM=1"
call :run mtp_cap         "DS4_MTP_VRAM_GB=0.5"
call :run lazy_dspark     "STRATA_DS4_DSPARK_LAZY=1"
call :run eg_ring         "STRATA_DS4_EG_RING=8"
call :run lm_ahead        "STRATA_DS4_LM_AHEAD=1"
call :run drop_shards     "STRATA_DS4_DROP_SHARDS=1"
call :run defer3_approx   "STRATA_DS4_DEFER=3" "STRATA_DS4_DEFER_W=0.20"
call :run all_exact       "STRATA_DS4_ROUTER_BF16=1" "DS4_MTP_VRAM_GB=0.5" "STRATA_DS4_DSPARK_LAZY=1" "STRATA_DS4_EG_RING=8" "STRATA_DS4_LM_AHEAD=1" "STRATA_DS4_DROP_SHARDS=1"

echo(
echo ===== summary =====
type "%SUM%"
findstr /C:"FAIL" "%SUM%" >nul && exit /b 1
exit /b 0

:run
rem clear every switch the port adds or the skip knobs use, so each run carries only its own
for %%v in (STRATA_DS4_ROUTER_BF16 DS4_MTP_VRAM_GB DS4_STORE_UNIFORM STRATA_DS4_DSPARK_LAZY STRATA_DS4_EG_RING ^
            STRATA_DS4_LM_AHEAD DS4_LM_AHEAD_Q DS4_LM_AHEAD_CHUNKS STRATA_DS4_DROP_SHARDS ^
            STRATA_DS4_DEFER STRATA_DS4_DEFER_W STRATA_DS4_DEFER_K STRATA_DS4_DEFER_COLD STRATA_DS4_DEFER_EARLY STRATA_DS4_DEFER_MAX ^
            STRATA_DS4_RAM_SKIP_W STRATA_DS4_DISK_SKIP_W STRATA_DS4_DISK_SKIP_AUTO STRATA_DS4_PROMPT_SKIP DS4_YOCO) do set "%%v="
set "NAME=%~1"
shift
set "ENVS="
:args
if "%~1"=="" goto args_done
set "%~1"
set "ENVS=!ENVS! %~1"
shift
goto args
:args_done
set "LOGF=%LOGD%\%NAME%.log"
echo(=== %NAME% === !ENVS! (engine load takes minutes; full log: %LOGF%)
set PYTHONIOENCODING=utf-8
rem errorlevel to a known 0: an unset variable above must not decide this run's verdict
ver >nul
python "%SCRIPT%" "%EXE%" "%PACK%" "%IDS%" %ARGS% > "%LOGF%" 2>&1
if errorlevel 1 (
    echo     %NAME%: FAIL - see %LOGF%
    echo %NAME% FAIL>> "%SUM%"
) else (
    echo     %NAME%: PASS
    echo %NAME% PASS>> "%SUM%"
)
goto :eof
