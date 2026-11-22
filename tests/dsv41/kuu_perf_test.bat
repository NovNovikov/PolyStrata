@echo off
rem kuu_perf_test.bat - the per-parameter PERFORMANCE sweep of the ported kuu features. One run per
rem parameter (plus the controls), each in its own engine process with the SAME real load, sized like
rem the user's own benches (the qwen A/B benches: 32,675-token prompt, GEN 512; the mimo bench --gen
rem 1024/2048): a 32,768-token prompt, one warm-up GEN, then three measured GENs of 512 tokens
rem (tests/dsv41/kuu_perf_check.py). Every run reports, on one RESULT line: the engine's cold load, the
rem prefill (prompt landing), decode tok/s per measured GEN - and the activation witness, so a number
rem is never read without knowing whether the switch actually reached its code (PASS ran / SKIP named
rem reason it was idle / FAIL trouble or silence). Compare each run against ITS control, measured GEN
rem against measured GEN - the first GEN of any load is warm-up and never comparable (the short smoke's
rem baseline climbed 10 -> 19 tok/s across its own GENs).
rem
rem   baseline / defer3 / eg_ring / lm_ahead / all_exact   controls: baseline (the usual geometry)
rem   store_base / ring_mmap / ahead_mmap / uniform / drop_shards
rem                                                        controls: store_base (--mmap-experts on +
rem                                                        DS4_STORE_GB=32). The _mmap pair matters:
rem                                                        the ring and the read-ahead work only where
rem                                                        the chunked pass streams experts from the
rem                                                        files - in the copy-into-RAM geometry they
rem                                                        idle, and only the named SKIP would show it.
rem On the IQ1_S pack router_bf16 / mtp_cap / lazy_dspark are provably idle (F32 router, quantized
rem drafter) - all_exact still carries them and their named reasons show in its note.
rem
rem It loads the WHOLE PACK PER RUN (minutes and ~103 GB of RAM each time): wants the GPU and the RAM to
rem itself, every other server stopped. Ten runs plan for well over an hour. Gibberish ids: this is a
rem load, not a quality test - quality is a conversation test, this one counts tokens per second.
rem
rem usage:  kuu_perf_test.bat [exe] [pack] [run ...]   no run names: all eight
rem         kuu_perf_test.bat defer3 eg_ring           run names first: exe and pack take their defaults
rem         (an empty argument would be eaten by cmd before the bat sees it, so never pass "" for a default)
rem Knobs for the load itself (defaults shown): KUUPERF_GENS=3 KUUPERF_WARM=1 KUUPERF_TOKENS=64
rem Logs: %TEMP%\kuu-perf-logs\<name>.log (run + engine stderr teed) and <name>.engine.log (verbatim).
setlocal EnableDelayedExpansion

set "RUNS_FIRST="
for %%r in (baseline defer3 eg_ring lm_ahead all_exact store_base uniform drop_shards) do if /i "%~1"=="%%r" set "RUNS_FIRST=1"
set "EXE=%~1"
set "PACK=%~2"
if defined RUNS_FIRST ( set "EXE=" & set "PACK=" )
if "%EXE%"=="" set "EXE=L:\Temp\opencode\up-cuda-build\strata-dsv41.exe"
if "%PACK%"=="" set "PACK=L:\AI_pictures_generate\PolyStrata-data\packs\deepseek-v4.1-1017-ud-iq1-s"
set "SCRIPT=%~dp0kuu_perf_check.py"
set "IDS=%~dp0kuu_perf_ids.txt"
set "LOGD=%TEMP%\kuu-perf-logs"
rem the user's usual engine args (strata-deepseek-v4.1-1017-ud-iq1-s-upstream.json), appended after the
rem script's own defaults so they win, as in the live server
set "ARGS=--max-context 262144 --eos 1 --dspark auto --cpu-threads 12 --prefill-chunk 1024 --reserve-mib 300 --vram-experts 0.85 --verbose"

rem runs asked for (after exe and pack, or all of them when the first argument is already a run name);
rem empty ONLY = all eight
if defined RUNS_FIRST goto collect
shift
shift
:collect
set "ONLY="
:more
if "%~1"=="" goto more_done
set "ONLY=!ONLY! %~1"
shift & goto more
:more_done

if not exist "%LOGD%" mkdir "%LOGD%"
set "SUM=%LOGD%\summary.txt"
echo kuu per-parameter performance %DATE% %TIME%> "%SUM%"

rem the load: 32,768 token ids, made once (the size the user's benches use; any vocab-valid ids route
rem every expert - it is a load, not a text)
if not exist "%IDS%" (
    set PYTHONIOENCODING=utf-8
    python -c "import sys;sys.stdout.write(','.join(str(100+i) for i in range(32768)))" > "%IDS%"
    if not exist "%IDS%" ( echo could not write %IDS% & exit /b 2 )
)

if not exist "%EXE%"  ( echo EXE not found: %EXE%  - build it or pass it as the first argument & exit /b 2 )
if not exist "%PACK%" ( echo PACK not found: %PACK% & exit /b 2 )
for %%f in ("%IDS%") do echo ids: %%~nxf (%%~zf bytes)

echo(
echo Engine: %EXE%
echo Pack:   %PACK%
echo Logs:   %LOGD%
echo(

set "ANYFAIL="
call :run baseline
call :run defer3      "STRATA_DS4_DEFER=3" "STRATA_DS4_DEFER_W=0.20"
call :run eg_ring     "STRATA_DS4_EG_RING=8"
call :run lm_ahead    "STRATA_DS4_LM_AHEAD=1"
call :run all_exact   "STRATA_DS4_ROUTER_BF16=1" "DS4_MTP_VRAM_GB=0.5" "STRATA_DS4_DSPARK_LAZY=1" "STRATA_DS4_EG_RING=8" "STRATA_DS4_LM_AHEAD=1" "STRATA_DS4_DROP_SHARDS=1"
call :run store_base  "DS4_STORE_GB=32" -- --mmap-experts on
call :run ring_mmap   "DS4_STORE_GB=32" "STRATA_DS4_EG_RING=8" -- --mmap-experts on
call :run ahead_mmap  "DS4_STORE_GB=32" "STRATA_DS4_LM_AHEAD=1" -- --mmap-experts on
call :run uniform     "DS4_STORE_GB=32" "DS4_STORE_UNIFORM=1" -- --mmap-experts on
call :run drop_shards "DS4_STORE_GB=32" "STRATA_DS4_DROP_SHARDS=1" -- --mmap-experts on

echo(
echo ===== summary (compare each run with its control, RESULT line per run) =====
type "%SUM%"
findstr /C:"FAIL" "%SUM%" >nul && exit /b 1
exit /b 0

:run
rem clear every switch the port adds, so each run carries only its own
for %%v in (STRATA_DS4_ROUTER_BF16 DS4_MTP_VRAM_GB DS4_STORE_UNIFORM DS4_STORE_GB STRATA_DS4_DSPARK_LAZY ^
            STRATA_DS4_EG_RING STRATA_DS4_LM_AHEAD DS4_LM_AHEAD_Q DS4_LM_AHEAD_CHUNKS STRATA_DS4_DROP_SHARDS ^
            STRATA_DS4_DEFER STRATA_DS4_DEFER_W STRATA_DS4_DEFER_K STRATA_DS4_DEFER_COLD STRATA_DS4_DEFER_EARLY STRATA_DS4_DEFER_MAX ^
            STRATA_DS4_RAM_SKIP_W STRATA_DS4_DISK_SKIP_W STRATA_DS4_DISK_SKIP_AUTO STRATA_DS4_PROMPT_SKIP DS4_YOCO) do set "%%v="
set "NAME=%~1"
if not defined ONLY goto run_go
echo !ONLY! | findstr /C:" %NAME% " >nul && goto run_go
echo(=== %NAME% === not asked for, skipped
goto :eof
:run_go
shift
set "ENVS="
set "EXTRA="
:args
if "%~1"=="" goto args_done
if "%~1"=="--" ( shift & goto extra )
set "%~1"
set "ENVS=!ENVS! %~1"
shift
goto args
:extra
if "%~1"=="" goto args_done
set "EXTRA=!EXTRA! %~1"
shift
goto extra
:args_done
set "LOGF=%LOGD%\%NAME%.log"
set "KUU_ENGINE_LOG=%LOGD%\%NAME%.engine.log"
echo(=== %NAME% === !ENVS!!EXTRA! (pack load takes minutes; full log: %LOGF%)
set PYTHONIOENCODING=utf-8
ver >nul
python "%SCRIPT%" "%EXE%" "%PACK%" "%NAME%" "%IDS%" %ARGS%%EXTRA% > "%LOGF%" 2>&1
set "RLINE="
if exist "%LOGD%\%NAME%.result" del "%LOGD%\%NAME%.result"
findstr /B /C:"RESULT" "%LOGF%" > "%LOGD%\%NAME%.result" 2>nul
rem findstr's redirect leaves an EMPTY file when nothing matched - size, not existence, is the check
set "RHAS="
for %%F in ("%LOGD%\%NAME%.result") do if not %%~zF==0 set "RHAS=1"
if not defined RHAS (
    echo     %NAME%: FAIL - no RESULT line, see %LOGF%
    echo RESULT %NAME% FAIL load=- ready=- warm=- prefill=- decode=- note=no RESULT line, engine trouble or script crash>> "%SUM%"
    set "ANYFAIL=1"
    goto :eof
)
for /f "usebackq delims=" %%L in ("%LOGD%\%NAME%.result") do set "RLINE=%%L"
set "VERDICT="
for /f "usebackq tokens=3" %%V in ("%LOGD%\%NAME%.result") do set "VERDICT=%%V"
echo     !RLINE!
echo !RLINE!>> "%SUM%"
if "%VERDICT%"=="FAIL" set "ANYFAIL=1"
goto :eof
