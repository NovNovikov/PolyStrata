@echo off
REM Rebuilds the strata engine from this folder's sources (your fork branch).
REM Run it after pulling or editing src/, include/ or CMakeLists.txt, then
REM restart the server. First run configures (needs internet once
REM for deps); later runs only rebuild what changed.
REM
REM Recipe notes (measured on this box, 2026-10-17):
REM - MSVC 2022 via vcvars64 (cl is NOT on PATH by default).
REM - CUDA 12.4 toolkit. CMAKE_CUDA_STANDARD=17: cudafe 12.4 crashes on C++20
REM   STL with this MSVC; the only C++20 spot (S26 template lambdas in
REM   src/kernels/cuda/iq_kernels.cu) was rewritten C++17-compatible.
REM - CMAKE_CUDA_RUNTIME_LIBRARY=Shared: avoids the static/DLL cudart clash
REM   (upstream #585). CMAKE_CUDA_ARCHITECTURES=89 = RTX 4090.
REM - -j 2: the box is RAM-tight while the model runs; more jobs page it dead.
setlocal EnableExtensions
if not defined STRATA_BUILD_KEEP_OPEN (
    set "STRATA_BUILD_KEEP_OPEN=1"
    cmd /d /k call "%~f0" %*
    exit /b
)

call "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat not found, install VS2022 C++ tools.
    exit /b 1
)
cd /d "L:\AI_pictures_generate\PolyStrata"
cmake -S . -B build-filestream -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
if errorlevel 1 (
    echo ERROR: configure failed.
    exit /b 1
)
cmake --build build-filestream --target strata --config Release -j 2
if errorlevel 1 (
    echo ERROR: build failed.
    exit /b 1
)
echo OK: fresh engine at build-filestream\Release\strata.exe
echo Next: back up engine\strata.exe, copy the new one over it, restart the server.
