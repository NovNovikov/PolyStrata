@echo off
call "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set VSLANG=1049
L:\AI_pictures_generate\PolyStrata\.venv\Scripts\cmake.exe -S L:/AI_pictures_generate/PolyStrata-upstream-glm-20261020 -B L:/AI_pictures_generate/PolyStrata-upstream-glm-20261020/build -G Ninja -DCMAKE_MAKE_PROGRAM=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/ninja.exe -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=L:/AI_pictures_generate/PolyStrata-upstream-glm-20261020/vendor-llama -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CXX_COMPILER_LAUNCHER=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/cmake.exe;-E;env;VSLANG=1049" "-DCMAKE_C_COMPILER_LAUNCHER=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/cmake.exe;-E;env;VSLANG=1049" "-DCMAKE_CUDA_COMPILER_LAUNCHER=L:/AI_pictures_generate/PolyStrata/.venv/Scripts/cmake.exe;-E;env;VSLANG=1049"
if errorlevel 1 exit /b 1
L:\AI_pictures_generate\PolyStrata\.venv\Scripts\cmake.exe --build L:/AI_pictures_generate/PolyStrata-upstream-glm-20261020/build --target strata-poly -j 4
exit /b %errorlevel%
