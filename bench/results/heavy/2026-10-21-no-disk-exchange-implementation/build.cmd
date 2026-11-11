@echo off
call "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set VSLANG=1049
L:\AI_pictures_generate\PolyStrata\.venv\Scripts\cmake.exe -S L:\AI_pictures_generate\PolyStrata -B L:\AI_pictures_generate\PolyStrata\build-poly-cuda12 -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CXX_COMPILER_LAUNCHER=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/cmake.exe;-E;env;VSLANG=1049" "-DCMAKE_C_COMPILER_LAUNCHER=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/cmake.exe;-E;env;VSLANG=1049" "-DCMAKE_CUDA_COMPILER_LAUNCHER=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/cmake.exe;-E;env;VSLANG=1049"
if errorlevel 1 exit /b 1
L:\AI_pictures_generate\PolyStrata\.venv\Scripts\cmake.exe --build L:\AI_pictures_generate\PolyStrata\build-poly-cuda12 --target strata-poly poly_exchange_test poly_exchange_gpu_test poly_no_disk_test glm_conversations_test -j 4
exit /b %errorlevel%
