#!/bin/bash

# =============================================================================
# PRIR Project - Benchmark Runner
# =============================================================================
# Skrypt automatyzujący testy wydajnościowe dla analizatora logów
# Uruchamia serie testów z różnymi konfiguracjami i zapisuje wyniki do CSV
#
# Użycie:
#   ./run_benchmarks.sh
#
# Wymagania:
#   - Skompilowany program: ./build/bin/prir
#   - Plik testowy: access.log
#   - System: Linux/macOS (dla Windows użyj Git Bash)
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

# Funkcja do wypisywania kolorowych komunikatów
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

# Funkcja do wyciągania czasu z outputu programu
# Program wypisuje czas w formacie: "Execution time: 1234 ms" lub w stdout
extract_time_ms() {
    local output="$1"

    # Próbuj znaleźć czas w różnych formatach
    # Format 1: "Execution time: 1234 ms"
    if echo "$output" | grep -q "Execution time:"; then
        echo "$output" | grep "Execution time:" | sed -n 's/.*Execution time: \([0-9.]*\) ms.*/\1/p' | head -1
    # Format 2: Użyj time command (fallback)
    else
        echo "0"  # Jeśli nie znaleziono, zwróć 0
    fi
}

# Funkcja do uruchamiania testu i mierzenia czasu
run_test() {
    local test_name="$1"
    local command="$2"
    local threads="$3"
    local mode="$4"
    local mpi_procs="${5:-1}"

    log_info "Running: $test_name (threads=$threads, mode=$mode, mpi_procs=$mpi_procs)"

    # Użyj time do mierzenia czasu wykonania
    local start_time=$(date +%s%3N)  # Czas w milisekundach

    # Uruchom komendę i przechwytuj output
    local output
    if [ "$mpi_procs" -gt 1 ]; then
        output=$(mpirun -np "$mpi_procs" $command 2>&1 || echo "ERROR")
    else
        output=$($command 2>&1 || echo "ERROR")
    fi

    local end_time=$(date +%s%3N)
    local duration_ms=$((end_time - start_time))

    # Sprawdź, czy test zakończył się sukcesem
    if echo "$output" | grep -q "ERROR"; then
        log_error "Test failed: $test_name"
        return 1
    fi

    # Zapisz wynik do CSV
    echo "$threads,$mode,$duration_ms,$mpi_procs,\"$test_name\"" >> "$OUTPUT_CSV"

    log_success "Completed in ${duration_ms}ms"

    return 0
}

# ================================
# WERYFIKACJA ŚRODOWISKA
# ================================

log_info "Starting benchmark suite..."
echo ""

# Sprawdź, czy program istnieje
if [ ! -f "$BINARY" ]; then
    log_error "Binary not found: $BINARY"
    log_info "Please compile the program first: make"
    exit 1
fi
log_success "Binary found: $BINARY"

# Sprawdź, czy plik testowy istnieje
if [ ! -f "$LOGFILE" ]; then
    log_error "Log file not found: $LOGFILE"
    log_info "Please ensure access.log exists in the project root"
    exit 1
fi

# Sprawdź rozmiar pliku
FILESIZE=$(du -h "$LOGFILE" | cut -f1)
log_success "Log file found: $LOGFILE (size: $FILESIZE)"

# Sprawdź, czy MPI jest dostępne
if command -v mpirun &> /dev/null; then
    log_success "MPI detected: $(mpirun --version | head -1)"
    HAS_MPI=true
else
    log_warning "MPI not found - MPI tests will be skipped"
    HAS_MPI=false
fi

# Sprawdź, czy CUDA jest dostępna
if $BINARY --help | grep -q "use-cuda"; then
    log_success "CUDA support detected in binary"
    HAS_CUDA=true
else
    log_warning "CUDA not available - GPU tests will be skipped"
    HAS_CUDA=false
fi

echo ""
log_info "Configuration:"
echo "  - Threads to test: 1, 2, 4, 8, 16"
echo "  - MPI processes: 1, 2, 4, 8"
echo "  - Phrase: $PHRASE"
echo "  - Output: $OUTPUT_CSV"
echo ""

# ================================
# INICJALIZACJA PLIKU CSV
# ================================

log_info "Initializing output file: $OUTPUT_CSV"
echo "threads,mode,duration_ms,mpi_procs,test_name" > "$OUTPUT_CSV"

# ================================
# TEST 1: SKALOWANIE OPENMP (CPU)
# ================================

echo ""
log_info "======================================"
log_info "TEST 1: OpenMP Scaling (CPU)"
log_info "======================================"
echo ""

for threads in 1 2 4 8 16; do
    run_test \
        "OpenMP CPU threads=$threads" \
        "$BINARY --file $LOGFILE --phrase $PHRASE --threads $threads --cpu-only --no-stats" \
        "$threads" \
        "cpu" \
        "1"
    sleep 1  # Pauza między testami
done

# ================================
# TEST 2: GPU (CUDA) DLA RÓŻNYCH WĄTKÓW
# ================================

if [ "$HAS_CUDA" = true ]; then
    echo ""
    log_info "======================================"
    log_info "TEST 2: GPU (CUDA) Performance"
    log_info "======================================"
    echo ""

    for threads in 1 2 4 8 16; do
        run_test \
            "CUDA GPU threads=$threads" \
            "$BINARY --file $LOGFILE --phrase $PHRASE --threads $threads --use-cuda --no-stats" \
            "$threads" \
            "gpu" \
            "1"
        sleep 1
    done
else
    log_warning "Skipping GPU tests (CUDA not available)"
fi

# ================================
# TEST 3: SKALOWANIE MPI
# ================================

if [ "$HAS_MPI" = true ]; then
    echo ""
    log_info "======================================"
    log_info "TEST 3: MPI Scaling"
    log_info "======================================"
    echo ""

    for procs in 1 2 4 8; do
        run_test \
            "MPI procs=$procs" \
            "$BINARY --file $LOGFILE --phrase $PHRASE --threads 4 --cpu-only --no-stats" \
            "4" \
            "mpi" \
            "$procs"
        sleep 1
    done
else
    log_warning "Skipping MPI tests (MPI not available)"
fi

# ================================
# TEST 4: KOMBINACJA MPI + OPENMP
# ================================

if [ "$HAS_MPI" = true ]; then
    echo ""
    log_info "======================================"
    log_info "TEST 4: MPI + OpenMP Hybrid"
    log_info "======================================"
    echo ""

    # Test różnych kombinacji MPI procesów × OpenMP wątków
    for procs in 2 4; do
        for threads in 2 4; do
            run_test \
                "Hybrid MPI=$procs OMP=$threads" \
                "$BINARY --file $LOGFILE --phrase $PHRASE --threads $threads --cpu-only --no-stats" \
                "$threads" \
                "hybrid" \
                "$procs"
            sleep 1
        done
    done
else
    log_warning "Skipping hybrid tests (MPI not available)"
fi

# ================================
# TEST 5: WIELE FRAZ (CPU vs GPU)
# ================================

echo ""
log_info "======================================"
log_info "TEST 5: Multiple Phrases"
log_info "======================================"
echo ""

MULTI_PHRASES="--phrase GET --phrase POST --phrase PUT --phrase DELETE"

# CPU
run_test \
    "Multi-phrase CPU" \
    "$BINARY --file $LOGFILE $MULTI_PHRASES --threads 8 --cpu-only --no-stats" \
    "8" \
    "cpu-multi" \
    "1"

# GPU (jeśli dostępne)
if [ "$HAS_CUDA" = true ]; then
    run_test \
        "Multi-phrase GPU" \
        "$BINARY --file $LOGFILE $MULTI_PHRASES --threads 8 --use-cuda --no-stats" \
        "8" \
        "gpu-multi" \
        "1"
fi

# ================================
# PODSUMOWANIE
# ================================

echo ""
log_info "======================================"
log_success "Benchmark suite completed!"
log_info "======================================"
echo ""

# Policz liczbę testów
TOTAL_TESTS=$(wc -l < "$OUTPUT_CSV")
TOTAL_TESTS=$((TOTAL_TESTS - 1))  # Odejmij header

log_success "Total tests completed: $TOTAL_TESTS"
log_success "Results saved to: $OUTPUT_CSV"
echo ""

# Wyświetl próbkę wyników
log_info "Sample results:"
head -n 6 "$OUTPUT_CSV" | column -t -s,

echo ""
log_info "Next steps:"
echo "  1. View full results: cat $OUTPUT_CSV"
echo "  2. Generate plots: python3 generate_plots.py"
echo "  3. Analyze data in spreadsheet software (Excel, LibreOffice)"
echo ""

exit 0
