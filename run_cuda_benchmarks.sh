#!/bin/bash

# =============================================================================
# PRIR Project - CUDA Benchmark Runner
# =============================================================================
# Skrypt automatyzujący testy wydajnościowe dla wersji GPU (CUDA)
# Wyniki są dopisywane do benchmark_results.csv (bez nadpisywania istniejących)
# =============================================================================

set -e  # Zakończ skrypt przy błędzie

# ================================
# KONFIGURACJA
# ================================

BINARY="./build/bin/prir"
LOGFILE="access.log"
OUTPUT_CSV="benchmark_results.csv"
PHRASE="GET"  # Fraza do zliczenia

# Kolory dla outputu
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================================
# FUNKCJE POMOCNICZE
# ================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# (zostawiamy na przyszłość, obecnie nieużywana)
extract_time_ms() {
    local output="$1"

    if echo "$output" | grep -q "Execution time:"; then
        echo "$output" | grep "Execution time:" | sed -n 's/.*Execution time: \([0-9.]*\) ms.*/\1/p' | head -1
    else
        echo "0"
    fi
}

run_test() {
    local test_name="$1"
    local command="$2"
    local threads="$3"
    local mode="$4"

    log_info "Running: $test_name (threads=$threads, mode=$mode)"

    local start_time
    start_time=$(date +%s%3N)

    local output
    output=$($command 2>&1 || echo "ERROR")

    local end_time
    end_time=$(date +%s%3N)
    local duration_ms=$((end_time - start_time))

    # Dopisywanie do CSV
    echo "$threads,$mode,$duration_ms,1,\"$test_name\"" >> "$OUTPUT_CSV"

    log_success "Completed in ${duration_ms}ms"
    return 0
}

# ================================
# WERYFIKACJA ŚRODOWISKA
# ================================

log_info "Starting CUDA benchmark suite..."
echo ""

if [ ! -f "$BINARY" ]; then
    log_error "Binary not found: $BINARY"
    log_info "Please compile the program first: make"
    exit 1
fi
log_success "Binary found: $BINARY"

if [ ! -f "$LOGFILE" ]; then
    log_error "Log file not found: $LOGFILE"
    log_info "Please ensure access.log exists in the project root"
    exit 1
fi

FILESIZE=$(du -h "$LOGFILE" | cut -f1)
log_success "Log file found: $LOGFILE (size: $FILESIZE)"

echo ""
log_info "Configuration:"
echo "  - Threads to test: 1, 2, 4, 8, 16"
echo "  - Phrase: $PHRASE"
echo "  - Output: $OUTPUT_CSV"
echo ""

# ================================
# INICJALIZACJA / DOPISYWANIE DO CSV
# ================================

if [ ! -f "$OUTPUT_CSV" ]; then
    log_info "Creating new output file: $OUTPUT_CSV"
    echo "threads,mode,duration_ms,mpi_procs,test_name" > "$OUTPUT_CSV"
else
    log_info "Appending to existing output file: $OUTPUT_CSV"
fi

# ================================
# TEST 1: GPU (CUDA) - SKALOWANIE WĄTKÓW
# ================================

echo ""
log_info "======================================"
log_info "TEST 1: GPU (CUDA) Thread Scaling"
log_info "======================================"
echo ""

for threads in 1 2 4 8 16; do
    run_test \
        "CUDA GPU threads=$threads" \
        "$BINARY --file $LOGFILE --phrase $PHRASE --threads $threads --use-cuda --no-stats" \
        "$threads" \
        "gpu"
    sleep 1
done

# ================================
# TEST 2: GPU (CUDA) - WIELE FRAZ
# ================================

echo ""
log_info "======================================"
log_info "TEST 2: GPU (CUDA) Multiple Phrases"
log_info "======================================"
echo ""

MULTI_PHRASES="--phrase GET --phrase POST --phrase PUT --phrase DELETE"

run_test \
    "CUDA Multi-phrase GPU" \
    "$BINARY --file $LOGFILE $MULTI_PHRASES --threads 8 --use-cuda --no-stats" \
    "8" \
    "gpu-multi"

# ================================
# PODSUMOWANIE
# ================================

echo ""
log_info "======================================"
log_success "CUDA benchmark suite completed!"
log_info "======================================"
echo ""

TOTAL_TESTS=$(wc -l < "$OUTPUT_CSV")
TOTAL_TESTS=$((TOTAL_TESTS - 1))

log_success "Total rows in CSV (all runs): $TOTAL_TESTS"
log_success "Results saved/updated in: $OUTPUT_CSV"
echo ""

log_info "Sample appended results:"
tail -n 6 "$OUTPUT_CSV" | column -t -s,

echo ""
log_info "Next steps:"
echo "  1. View full results: cat $OUTPUT_CSV"
echo "  2. Generate plots: python3 generate_plots.py"
echo ""

exit 0