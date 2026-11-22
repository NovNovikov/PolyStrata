@echo off
rem kuu_defer_test.bat - the one question worth asking on this machine: does DEFER=3 change the speed.
rem A/B/A only: control, feature, control again (baseline / defer3 / baseline2) - the two identical
rem controls say how much the machine warmed while the feature ran, and DEFER only counts if it beats
rem their average. Three pack loads, about half an hour, the machine to itself (stop the other servers).
rem
rem Everything else lives in kuu_perf_test.bat (the full crash-guard and mmap matrix, twelve runs) -
rem this wrapper just runs the relevant three and takes the same optional arguments:
rem   kuu_defer_test.bat                                default exe and pack
rem   kuu_defer_test.bat C:\path\strata-dsv41.exe       another exe (default pack)
rem   kuu_defer_test.bat C:\path\exe C:\path\pack       both
setlocal
call "%~dp0kuu_perf_test.bat" %* baseline defer3 baseline2
set "RC=%ERRORLEVEL%"
echo(
if "%RC%"=="0" echo DEFER A/B/A done - the summary above says it: compare defer3's decode against
if "%RC%"=="0" echo the AVERAGE of the two baseline lines, GEN position against GEN position.
exit /b %RC%
