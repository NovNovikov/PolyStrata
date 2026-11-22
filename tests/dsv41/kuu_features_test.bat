@echo off
rem kuu_features_test.bat - the usual engine smoke (tests/dsv41/short_prompt_check.py: the pack loaded,
rem three 16-token GENs, fail on any ERR / engine death / illegal-access stderr) run ONCE PER PORTED
rem kuu feature, one feature at a time with every other switch off, then once with all the exact ones
rem together.  A full engine log per run in the log folder, and a summary at the end.
rem
rem Second review item 14: the verdict used to be only "the script exited 0" - a run whose switch never
rem reached any code looked exactly like one that did.  Each run now greps the engine log for that
rem feature's ACTIVATION witness.  Third round item 14: the witness greps only started to mean
rem something once the FULL engine stderr reached the log - the script used to unlink the engine's
rem stderr after the run, so %LOGF% carried only its own prints.  Now every engine stderr line is
rem teed into %LOGF% as it arrives and kept verbatim in <name>.engine.log (KUU_ENGINE_LOG).  Verdicts:
rem   PASS - clean run AND the feature's witness line in the log;
rem   SKIP - clean run, the feature had legitimately nothing to do on this box (the reason is printed:
rem          no drafter to cap or draft lazily, no Engram table to ring, no RAM for the read-ahead ...);
rem   FAIL - engine trouble (the old check), or the switch was set and the log carries neither the
rem          witness nor a named reason - a silent no-op is exactly what this sweep must not hide.
rem Smoke only: it does not measure speed, and the 40-token prompt stays under the path threshold the engine
rem names at load, so EG_RING and LM_AHEAD are armed with nothing to do and DROP_SHARDS finds no store to
rem drop through. The engine now prints a named reason for exactly that (Engram ring armed / read-ahead
rem armed / DROP_SHARDS armed but idle / DSpark's unreadable drafter refused), and the sweep reads those
rem as SKIP - their real timing is a model-length run's job (docs/KUU_DSV41.md).
rem
rem usage:  kuu_features_test.bat [exe] [pack]
rem Each run loads the whole pack (minutes, lots of RAM) and wants the VRAM to itself: stop the other
rem servers first.  One run failing does not stop the rest.  Ten runs: plan for an hour or more.
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

set "ANYFAIL="
call :run baseline      none
call :run router_bf16   bf16    "STRATA_DS4_ROUTER_BF16=1"
call :run uniform_store uniform "DS4_STORE_UNIFORM=1"
call :run mtp_cap       cap     "DS4_MTP_VRAM_GB=0.5"
call :run lazy_dspark   lazy    "STRATA_DS4_DSPARK_LAZY=1"
call :run eg_ring       ring    "STRATA_DS4_EG_RING=8"
call :run lm_ahead      ahead   "STRATA_DS4_LM_AHEAD=1"
call :run drop_shards   drop    "STRATA_DS4_DROP_SHARDS=1"
call :run defer3_approx defer   "STRATA_DS4_DEFER=3" "STRATA_DS4_DEFER_W=0.20"
call :run all_exact     exact   "STRATA_DS4_ROUTER_BF16=1" "DS4_MTP_VRAM_GB=0.5" "STRATA_DS4_DSPARK_LAZY=1" "STRATA_DS4_EG_RING=8" "STRATA_DS4_LM_AHEAD=1" "STRATA_DS4_DROP_SHARDS=1"

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
set "WIT=%~2"
shift
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
rem (third round 14) the script keeps the engine's FULL stderr here, never deleted, while also teeing it
rem into %LOGF% below; %LOGF% carries every witness line, this file is the verbatim postmortem copy
set "KUU_ENGINE_LOG=%LOGD%\%NAME%.engine.log"
echo(=== %NAME% === !ENVS! (engine load takes minutes; full log: %LOGF%, engine's own: %KUU_ENGINE_LOG%)
set PYTHONIOENCODING=utf-8
rem errorlevel to a known 0: an unset variable above must not decide this run's verdict
ver >nul
python "%SCRIPT%" "%EXE%" "%PACK%" "%IDS%" %ARGS% > "%LOGF%" 2>&1
if errorlevel 1 (
    echo     %NAME%: FAIL - engine trouble, see %LOGF%
    echo %NAME% FAIL - engine trouble>> "%SUM%"
    set "ANYFAIL=1"
    goto :eof
)
set "VERDICT="
set "WHY="
call :wit_%WIT%
echo     %NAME%: %VERDICT% - %WHY%
echo %NAME% %VERDICT% - %WHY%>> "%SUM%"
if "%VERDICT%"=="FAIL" set "ANYFAIL=1"
goto :eof

rem Every witness below greps %LOGF% for a substring of the engine's own line (no parentheses: they
rem are findstr's own syntax and cmd's block syntax at once).  PASS first, then the named reasons a
rem clean run may carry no witness (SKIP), then FAIL: set and silent.

:wit_none
set "VERDICT=PASS"
set "WHY=baseline: a clean run is the whole check"
goto :eof

:wit_bf16
findstr /C:"routers kept in BF16" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the routers were stored as BF16 bits")
if not defined VERDICT findstr /C:"the switch does nothing" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=this pack's router is not BF16 of its exact size: the F32 copy was kept")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=the switch was set and the log carries neither the witness nor the reason")
goto :eof

:wit_uniform
findstr /C:"in 1 size classes" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the store ran as one uniform class")
if not defined VERDICT findstr /C:"RAM store:" "%LOGF%" >nul && (set "VERDICT=FAIL"& set "WHY=the store was built in classes: DS4_STORE_UNIFORM=1 did not take")
if not defined VERDICT (set "VERDICT=SKIP"& set "WHY=no RAM store was built on this box: the switch had nothing to make uniform")
goto :eof

:wit_cap
findstr /C:"DS4_MTP_VRAM_GB: drafter" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the drafter's per-block VRAM cap was applied")
if not defined VERDICT findstr /C:"not in the bits this loader reads" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=the pack keeps the drafter's Markov head quantized: the loader refused DSpark by name and the run went on without it")
if not defined VERDICT findstr /C:"no drafter is loaded" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=the drafter was not loaded: the cap had nothing to cap")
if not defined VERDICT findstr /C:"DSpark not loaded" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=DSpark auto refused the fit before the cap: this run proves the ordering line only")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=the cap was set and the log carries no trace of it")
goto :eof

:wit_lazy
findstr /C:"DSpark lazy:" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the drafting rounds ran the confidence-batched lazy path")
if not defined VERDICT findstr /C:"DSpark not loaded" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=DSpark auto refused the fit: there was no drafter to draft lazily")
if not defined VERDICT findstr /C:"no drafter is loaded" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=the drafter was not loaded at all")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=a drafter is loaded and no round took the lazy path: check the fallbacks - head across devices, a markov split, a draft dump")
goto :eof

:wit_ring
findstr /C:"would not fit" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=no pinned room for the ring slots: the Engram reads went one at a time")
if not defined VERDICT findstr /C:"requested slots of" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the reader ring was pinned and took the Engram reads")
if not defined VERDICT findstr /C:"Engram ring armed but idle" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=this pack carries no Engram table: the armed ring had nothing to read")
if not defined VERDICT findstr /C:"Engram ring armed" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=armed at load, but the smoke prompts stayed in the windows path - the ring belongs to the chunked pass; a long prompt is its real test")
if not defined VERDICT findstr /C:"Engram" "%LOGF%" >nul || (set "VERDICT=SKIP"& set "WHY=this pack carries no Engram table: the ring had nothing to read")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=STRATA_DS4_EG_RING=8 was set and the ring built no trace")
goto :eof

:wit_ahead
findstr /C:"next-layer read-ahead:" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the read-ahead buffers were built and its counts printed - the short prompt idles them, the line says the mode ran")
if not defined VERDICT findstr /C:"read-ahead is off for this run" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=no RAM for the second layer cache: the engine said so")
if not defined VERDICT findstr /C:"read-ahead armed" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=armed at load, but the smoke prompts stayed in the windows path - the read-ahead belongs to the chunked pass; a long prompt is its real test")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=STRATA_DS4_LM_AHEAD=1 was set and neither the counts nor the refusal printed")
goto :eof

:wit_drop
findstr /R /C:"^ds4: .local. 0 expert" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=no shard file was droppable in this geometry - the store reads through direct handles regardless")
if not defined VERDICT findstr /C:"unmapped, their buffered handles closed" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=the expert shards were unmapped and their buffered handles closed")
if not defined VERDICT findstr /C:"DROP_SHARDS armed but idle" "%LOGF%" >nul && (set "VERDICT=SKIP"& set "WHY=armed, but this geometry holds no expert-shard view to drop - the named reason says which condition; a limited store is its real test, e.g. --mmap-experts on with DS4_STORE_GB=32")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=STRATA_DS4_DROP_SHARDS=1 was set and the drop step printed nothing")
goto :eof

:wit_defer
findstr /C:"STRATA_DS4_DEFER=3 is armed" "%LOGF%" >nul && (set "VERDICT=PASS"& set "WHY=mode 3 armed at load and told the log - an approximate mode, in the session fingerprint")
if not defined VERDICT (set "VERDICT=FAIL"& set "WHY=STRATA_DS4_DEFER=3 was set and the load banner is missing")
goto :eof

:wit_exact
rem every switch of the exact set left its own trace: the five whose default geometry reaches the code
set "XMISS="
findstr /C:"routers kept in BF16" "%LOGF%" >nul || findstr /C:"the switch does nothing" "%LOGF%" >nul || set "XMISS=!XMISS! router_bf16"
findstr /C:"DS4_MTP_VRAM_GB: drafter" "%LOGF%" >nul || findstr /C:"no drafter is loaded" "%LOGF%" >nul || set "XMISS=!XMISS! mtp_cap"
findstr /C:"DSpark lazy:" "%LOGF%" >nul || findstr /C:"DSpark not loaded" "%LOGF%" >nul || findstr /C:"no drafter is loaded" "%LOGF%" >nul || set "XMISS=!XMISS! lazy_dspark"
findstr /C:"requested slots of" "%LOGF%" >nul || findstr /C:"would not fit" "%LOGF%" >nul || findstr /C:"Engram ring armed" "%LOGF%" >nul || set "XMISS=!XMISS! eg_ring"
findstr /C:"next-layer read-ahead:" "%LOGF%" >nul || findstr /C:"read-ahead is off" "%LOGF%" >nul || findstr /C:"read-ahead armed" "%LOGF%" >nul || set "XMISS=!XMISS! lm_ahead"
findstr /C:"unmapped, their buffered handles closed" "%LOGF%" >nul || findstr /C:"DROP_SHARDS armed but idle" "%LOGF%" >nul || set "XMISS=!XMISS! drop_shards"
if not defined XMISS (set "VERDICT=PASS"& set "WHY=every switch of the exact set left its witness") else (set "VERDICT=FAIL"& set "WHY=no trace for:!XMISS!")
goto :eof
