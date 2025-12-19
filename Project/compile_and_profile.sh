nvcc -lineinfo slow.cu -o slow
nvcc -lineinfo average.cu -o average
nvcc -lineinfo fast.cu -o fast
nvcc -lineinfo fastest.cu -o fastest

nsys profile --stats=true --force-overwrite=true -o profile_reports/slow_profile ./slow 2>&1 | tee profile_reports/slow_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/average_profile ./average 2>&1 | tee profile_reports/average_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/fast_profile ./fast 2>&1 | tee profile_reports/fast_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/fastest_profile ./fastest 2>&1 | tee profile_reports/fastest_full_report.log

nvcc -x cu -std=c++20   -I$ALPAKA_ROOT/include   --expt-relaxed-constexpr   -DALPAKA_ACC_GPU_CUDA_ENABLED   fastest_alpaka.cpp -o fastest_alpaka
nsys profile --stats=true --force-overwrite=true   -o profile_reports/fastest_alpaka_profile   ./fastest_alpaka 2>&1 | tee profile_reports/fastest_alpaka_full_report.log
