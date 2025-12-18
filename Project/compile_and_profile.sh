nvcc -lineinfo slow.cu -o slow
nvcc -lineinfo average.cu -o average
nvcc -lineinfo fast.cu -o fast
nvcc -lineinfo fastest.cu -o fastest
nsys profile --stats=true --force-overwrite=true -o profile_reports/slow_profile ./slow 2>&1 | tee profile_reports/slow_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/average_profile ./average 2>&1 | tee profile_reports/average_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/fast_profile ./fast 2>&1 | tee profile_reports/fast_full_report.log
nsys profile --stats=true --force-overwrite=true -o profile_reports/fastest_profile ./fastest 2>&1 | tee profile_reports/fastest_full_report.log

# nvcc -O2 -lineinfo slow.cu -o slow_O2
# nvcc -O2 -lineinfo average.cu -o average_O2
# nvcc -O2 -lineinfo fast.cu -o fast_O2
# nvcc -O2 -lineinfo fastest.cu -o fastest_O2
# nsys profile --stats=true --force-overwrite=true -o profile_reports/slow_O2_profile ./slow_O2 2>&1 | tee profile_reports/slow_O2_full_report.log
# nsys profile --stats=true --force-overwrite=true -o profile_reports/average_O2_profile ./average_O2 2>&1 | tee profile_reports/average_O2_full_report.log
# nsys profile --stats=true --force-overwrite=true -o profile_reports/fast_O2_profile ./fast_O2 2>&1 | tee profile_reports/fast_O2_full_report.log
# nsys profile --stats=true --force-overwrite=true -o profile_reports/fastest_O2_profile ./fastest_O2 2>&1 | tee profile_reports/fastest_O2_full_report.log