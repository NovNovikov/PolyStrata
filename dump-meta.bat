@echo off
REM Builds and runs the GGUF metadata dumper (links the repo's reader only).
call "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /I include src\artifact\gguf_meta_dump.cpp /Fe:build-filestream\Release\gguf_meta_dump.exe /Fo:build-filestream\meta_dump.obj /link /LIBPATH:build-filestream\..\..\_deps 2>nul
if errorlevel 1 (
    echo header-only link failed, trying src path
)
build-filestream\Release\gguf_meta_dump.exe %1
