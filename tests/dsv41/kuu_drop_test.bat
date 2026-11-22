@echo off
rem kuu_drop_test.bat - the one mmap question worth asking on this machine: does DROP_SHARDS actually
rem speed up the streaming geometry, or was its 17:31 sweep run just the warmest cache of the day.
rem A/B/A only: control, feature, control again (store_base / drop_shards / store_base2) - the two
rem identical controls say how much the machine warmed while the feature ran, and DROP_SHARDS only
rem counts if it beats their average, cold prefill and every measured GEN. Three pack loads, about
rem 40 minutes, the machine to itself (stop the other servers).
rem
rem Everything else lives in kuu_perf_test.bat (the full twelve-run matrix) - this wrapper just runs
rem the relevant three and takes the same optional arguments:
rem   kuu_drop_test.bat                                default exe and pack
rem   kuu_drop_test.bat C:\path\strata-dsv41.exe       another exe (default pack)
rem   kuu_drop_test.bat C:\path\exe C:\path\pack       both
setlocal
call "%~dp0kuu_perf_test.bat" %* store_base drop_shards store_base2
set "RC=%ERRORLEVEL%"
echo(
if "%RC%"=="0" echo DROP_SHARDS A/B/A done - compare drop_shards against the AVERAGE of the two
if "%RC%"=="0" echo store_base lines: prefill column against prefill, GEN position against GEN position.
exit /b %RC%
