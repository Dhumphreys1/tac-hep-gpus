#!/bin/bash



# Function to extract time from output and calculate average
run_benchmark() {
    local program=$1
    local name=$2
    local sum=0
    local count=0

    echo ""
    echo "Running $name (20 times):"
    echo "----------------------------"

    for i in {1..20}; do
        output=$(./$program)
        echo "$output"
        time=$(echo "$output" | grep -oP '\d+(?= ms)')
        if [ ! -z "$time" ]; then
            sum=$((sum + time))
            count=$((count + 1))
        fi
    done

    if [ $count -gt 0 ]; then
        avg=$((sum / count))
        echo "$program Average Time: $avg ms"
    fi
}
echo "Compiling programs..."
echo "====================="
nvcc slow.cu -o slow
nvcc average.cu -o average
nvcc fast.cu -o fast
nvcc fastest.cu -o fastest
nvcc -x cu -std=c++20   -I$ALPAKA_ROOT/include   --expt-relaxed-constexpr   -DALPAKA_ACC_GPU_CUDA_ENABLED   fastest_alpaka.cpp -o fastest_alpaka

echo "Running benchmarks - 20 iterations each"
echo "========================================"

# run_benchmark "main_cpu" "project_cxx"
run_benchmark "slow" "slow.cu"
run_benchmark "average" "average.cu"
run_benchmark "fast" "fast.cu"
run_benchmark "fastest" "fastest.cu"
run_benchmark "fastest_alpaka" "fastest_alpaka.cpp"

echo ""
echo "Benchmark complete!"
