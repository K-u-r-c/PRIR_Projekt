#!/usr/bin/env python3
"""
PRIR Project - Plot Generator
=============================================================================
Skrypt generujący wykresy wydajności na podstawie wyników benchmarków

Użycie:
    python3 generate_plots.py

Wymagania:
    pip install pandas matplotlib seaborn

Wejście:
    benchmark_results.csv

Wyjście:
    - plot_1_cpu_threads.png        : Czas vs liczba wątków (CPU)
    - plot_2_cpu_vs_gpu.png         : Porównanie CPU vs GPU
    - plot_3_speedup_cpu.png        : Przyspieszenie CPU (speedup)
    - plot_4_mpi_scaling.png        : Skalowanie MPI
    - plot_5_efficiency.png         : Efektywność równoległości
    - plot_6_hybrid_heatmap.png     : Mapa ciepła MPI+OpenMP
    - plot_7_throughput.png         : Przepustowość (GB/s)
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Konfiguracja
CSV_FILE = "benchmark_results.csv"
OUTPUT_DIR = "plots"
FILESIZE_GB = 3.5  # Rozmiar pliku access.log w GB

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Upewnij się, że katalog na wykresy istnieje
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def load_data():
    """Wczytuje dane z CSV i wykonuje podstawowe przetwarzanie"""
    print(f"[INFO] Loading data from {CSV_FILE}...")

    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] File not found: {CSV_FILE}")
        print("[INFO] Please run ./run_benchmarks.sh first")
        exit(1)

    df = pd.read_csv(CSV_FILE)
    print(f"[OK] Loaded {len(df)} test results")

    # Wyświetl podsumowanie
    print("\n[INFO] Data summary:")
    print(df.groupby('mode')['duration_ms'].describe())

    return df

def save_plot(filename):
    """Zapisuje wykres do pliku"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filepath}")
    plt.close()

# =============================================================================
# WYKRES 1: CZAS WYKONANIA VS LICZBA WĄTKÓW (CPU)
# =============================================================================

def plot_1_cpu_threads(df):
    """Wykres liniowy: Czas wykonania vs liczba wątków dla CPU"""
    print("\n[INFO] Generating plot 1: CPU threads scaling...")

    # Filtruj tylko testy CPU (bez MPI)
    cpu_data = df[(df['mode'] == 'cpu') & (df['mpi_procs'] == 1)].copy()

    if cpu_data.empty:
        print("[WARN] No CPU data found, skipping plot 1")
        return

    # Sortuj po liczbie wątków
    cpu_data = cpu_data.sort_values('threads')

    plt.figure(figsize=(10, 6))
    plt.plot(cpu_data['threads'], cpu_data['duration_ms'],
             marker='o', linewidth=2, markersize=8, label='CPU')

    plt.xlabel('Liczba wątków OpenMP', fontsize=12)
    plt.ylabel('Czas wykonania (ms)', fontsize=12)
    plt.title('Skalowanie OpenMP - CPU', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Dodaj wartości na punktach
    for _, row in cpu_data.iterrows():
        plt.annotate(f"{row['duration_ms']:.0f}ms",
                    (row['threads'], row['duration_ms']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=9)

    save_plot('plot_1_cpu_threads.png')

# =============================================================================
# WYKRES 2: CPU VS GPU
# =============================================================================

def plot_2_cpu_vs_gpu(df):
    """Wykres słupkowy porównujący czasy CPU vs GPU"""
    print("\n[INFO] Generating plot 2: CPU vs GPU comparison...")

    # Filtruj testy CPU i GPU (bez MPI)
    cpu_data = df[(df['mode'] == 'cpu') & (df['mpi_procs'] == 1)].copy()
    gpu_data = df[(df['mode'] == 'gpu') & (df['mpi_procs'] == 1)].copy()

    if cpu_data.empty or gpu_data.empty:
        print("[WARN] Missing CPU or GPU data, skipping plot 2")
        return

    # Merge danych
    cpu_data = cpu_data.sort_values('threads')
    gpu_data = gpu_data.sort_values('threads')

    # Przygotuj dane do wykresu
    threads = cpu_data['threads'].values
    cpu_times = cpu_data['duration_ms'].values
    gpu_times = gpu_data['duration_ms'].values

    # Szerokość słupków
    x = np.arange(len(threads))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8)
    bars2 = plt.bar(x + width/2, gpu_times, width, label='GPU (CUDA)', alpha=0.8)

    plt.xlabel('Liczba wątków', fontsize=12)
    plt.ylabel('Czas wykonania (ms)', fontsize=12)
    plt.title('Porównanie wydajności: CPU vs GPU', fontsize=14, fontweight='bold')
    plt.xticks(x, threads)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Dodaj wartości na słupkach
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9)

    save_plot('plot_2_cpu_vs_gpu.png')

# =============================================================================
# WYKRES 3: PRZYSPIESZENIE (SPEEDUP)
# =============================================================================

def plot_3_speedup_cpu(df):
    """Wykres przyspieszenia względem 1 wątku"""
    print("\n[INFO] Generating plot 3: Speedup...")

    cpu_data = df[(df['mode'] == 'cpu') & (df['mpi_procs'] == 1)].copy()

    if cpu_data.empty:
        print("[WARN] No CPU data found, skipping plot 3")
        return

    cpu_data = cpu_data.sort_values('threads')

    # Czas bazowy (1 wątek)
    baseline_time = cpu_data[cpu_data['threads'] == 1]['duration_ms'].values
    if len(baseline_time) == 0:
        print("[WARN] No baseline (1 thread) found, skipping plot 3")
        return

    baseline_time = baseline_time[0]

    # Oblicz speedup
    cpu_data['speedup'] = baseline_time / cpu_data['duration_ms']

    # Linia idealna
    ideal_speedup = cpu_data['threads'].values

    plt.figure(figsize=(10, 6))
    plt.plot(cpu_data['threads'], cpu_data['speedup'],
             marker='o', linewidth=2, markersize=8, label='Rzeczywiste przyspieszenie')
    plt.plot(cpu_data['threads'], ideal_speedup,
             '--', linewidth=2, alpha=0.7, label='Przyspieszenie idealne')

    plt.xlabel('Liczba wątków OpenMP', fontsize=12)
    plt.ylabel('Przyspieszenie (speedup)', fontsize=12)
    plt.title('Przyspieszenie OpenMP vs liczba wątków', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Dodaj wartości
    for _, row in cpu_data.iterrows():
        plt.annotate(f"{row['speedup']:.2f}x",
                    (row['threads'], row['speedup']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=9)

    save_plot('plot_3_speedup_cpu.png')

# =============================================================================
# WYKRES 4: SKALOWANIE MPI
# =============================================================================

def plot_4_mpi_scaling(df):
    """Wykres skalowania MPI"""
    print("\n[INFO] Generating plot 4: MPI scaling...")

    mpi_data = df[df['mode'] == 'mpi'].copy()

    if mpi_data.empty:
        print("[WARN] No MPI data found, skipping plot 4")
        return

    mpi_data = mpi_data.sort_values('mpi_procs')

    # Czas bazowy (1 proces)
    baseline = mpi_data[mpi_data['mpi_procs'] == 1]['duration_ms'].values
    if len(baseline) == 0:
        print("[WARN] No baseline (1 process) found, skipping plot 4")
        return

    baseline_time = baseline[0]

    # Oblicz speedup
    mpi_data['speedup'] = baseline_time / mpi_data['duration_ms']

    # Linia idealna
    ideal_speedup = mpi_data['mpi_procs'].values

    plt.figure(figsize=(10, 6))
    plt.plot(mpi_data['mpi_procs'], mpi_data['speedup'],
             marker='s', linewidth=2, markersize=8, label='Rzeczywiste przyspieszenie')
    plt.plot(mpi_data['mpi_procs'], ideal_speedup,
             '--', linewidth=2, alpha=0.7, label='Przyspieszenie idealne')

    plt.xlabel('Liczba procesów MPI', fontsize=12)
    plt.ylabel('Przyspieszenie (speedup)', fontsize=12)
    plt.title('Skalowanie MPI', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Dodaj wartości
    for _, row in mpi_data.iterrows():
        plt.annotate(f"{row['speedup']:.2f}x",
                    (row['mpi_procs'], row['speedup']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=9)

    save_plot('plot_4_mpi_scaling.png')

# =============================================================================
# WYKRES 5: EFEKTYWNOŚĆ RÓWNOLEGŁOŚCI
# =============================================================================

def plot_5_efficiency(df):
    """Wykres efektywności równoległości"""
    print("\n[INFO] Generating plot 5: Parallel efficiency...")

    cpu_data = df[(df['mode'] == 'cpu') & (df['mpi_procs'] == 1)].copy()

    if cpu_data.empty:
        print("[WARN] No CPU data found, skipping plot 5")
        return

    cpu_data = cpu_data.sort_values('threads')

    # Czas bazowy (1 wątek)
    baseline = cpu_data[cpu_data['threads'] == 1]['duration_ms'].values
    if len(baseline) == 0:
        print("[WARN] No baseline found, skipping plot 5")
        return

    baseline_time = baseline[0]

    # Oblicz speedup i efektywność
    cpu_data['speedup'] = baseline_time / cpu_data['duration_ms']
    cpu_data['efficiency'] = (cpu_data['speedup'] / cpu_data['threads']) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(cpu_data['threads'], cpu_data['efficiency'],
             marker='o', linewidth=2, markersize=8, color='green')
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Efektywność idealna (100%)')

    plt.xlabel('Liczba wątków OpenMP', fontsize=12)
    plt.ylabel('Efektywność (%)', fontsize=12)
    plt.title('Efektywność równoległości OpenMP', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 110)
    plt.legend()

    # Dodaj wartości
    for _, row in cpu_data.iterrows():
        plt.annotate(f"{row['efficiency']:.1f}%",
                    (row['threads'], row['efficiency']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=9)

    save_plot('plot_5_efficiency.png')

# =============================================================================
# WYKRES 6: HYBRID MPI+OPENMP (MAPA CIEPŁA)
# =============================================================================

def plot_6_hybrid_heatmap(df):
    """Mapa ciepła dla kombinacji MPI × OpenMP"""
    print("\n[INFO] Generating plot 6: Hybrid MPI+OpenMP heatmap...")

    hybrid_data = df[df['mode'] == 'hybrid'].copy()

    if hybrid_data.empty:
        print("[WARN] No hybrid data found, skipping plot 6")
        return

    # Pivot table: MPI processes × OpenMP threads
    pivot = hybrid_data.pivot_table(
        values='duration_ms',
        index='mpi_procs',
        columns='threads'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd_r',
                cbar_kws={'label': 'Czas wykonania (ms)'})

    plt.xlabel('Liczba wątków OpenMP', fontsize=12)
    plt.ylabel('Liczba procesów MPI', fontsize=12)
    plt.title('Czas wykonania: MPI × OpenMP (Hybrid)', fontsize=14, fontweight='bold')

    save_plot('plot_6_hybrid_heatmap.png')

# =============================================================================
# WYKRES 7: PRZEPUSTOWOŚĆ (GB/s)
# =============================================================================

def plot_7_throughput(df):
    """Wykres przepustowości w GB/s"""
    print("\n[INFO] Generating plot 7: Throughput...")

    # Weź wszystkie testy CPU i GPU
    data = df[(df['mode'].isin(['cpu', 'gpu'])) & (df['mpi_procs'] == 1)].copy()

    if data.empty:
        print("[WARN] No data found, skipping plot 7")
        return

    # Oblicz przepustowość: GB/s = FILESIZE_GB / (time_ms / 1000)
    data['throughput_gbps'] = FILESIZE_GB / (data['duration_ms'] / 1000)

    # Przygotuj dane
    cpu_data = data[data['mode'] == 'cpu'].sort_values('threads')
    gpu_data = data[data['mode'] == 'gpu'].sort_values('threads')

    plt.figure(figsize=(12, 6))

    if not cpu_data.empty:
        plt.plot(cpu_data['threads'], cpu_data['throughput_gbps'],
                marker='o', linewidth=2, markersize=8, label='CPU')

    if not gpu_data.empty:
        plt.plot(gpu_data['threads'], gpu_data['throughput_gbps'],
                marker='s', linewidth=2, markersize=8, label='GPU')

    plt.xlabel('Liczba wątków', fontsize=12)
    plt.ylabel('Przepustowość (GB/s)', fontsize=12)
    plt.title('Przepustowość przetwarzania danych', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot('plot_7_throughput.png')

# =============================================================================
# DODATKOWE STATYSTYKI
# =============================================================================

def generate_summary_stats(df):
    """Generuje plik tekstowy z podsumowaniem statystyk"""
    print("\n[INFO] Generating summary statistics...")

    output_file = os.path.join(OUTPUT_DIR, 'summary_stats.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PRIR PROJECT - BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Ogólne statystyki
        f.write("OGÓLNE STATYSTYKI:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Łączna liczba testów: {len(df)}\n")
        f.write(f"Rozmiar pliku testowego: {FILESIZE_GB} GB\n\n")

        # Najlepsze wyniki
        f.write("NAJLEPSZE WYNIKI:\n")
        f.write("-" * 80 + "\n")

        fastest = df.loc[df['duration_ms'].idxmin()]
        f.write(f"Najszybszy test: {fastest['test_name']}\n")
        f.write(f"  - Czas: {fastest['duration_ms']:.2f} ms\n")
        f.write(f"  - Konfiguracja: threads={fastest['threads']}, mode={fastest['mode']}\n")
        f.write(f"  - Przepustowość: {FILESIZE_GB / (fastest['duration_ms'] / 1000):.2f} GB/s\n\n")

        # CPU
        cpu_data = df[(df['mode'] == 'cpu') & (df['mpi_procs'] == 1)]
        if not cpu_data.empty:
            fastest_cpu = cpu_data.loc[cpu_data['duration_ms'].idxmin()]
            f.write(f"Najszybszy CPU: {fastest_cpu['threads']} wątków\n")
            f.write(f"  - Czas: {fastest_cpu['duration_ms']:.2f} ms\n")

            # Speedup
            baseline_cpu = cpu_data[cpu_data['threads'] == 1]['duration_ms'].values
            if len(baseline_cpu) > 0:
                speedup = baseline_cpu[0] / fastest_cpu['duration_ms']
                f.write(f"  - Przyspieszenie vs 1 wątek: {speedup:.2f}x\n\n")

        # GPU
        gpu_data = df[(df['mode'] == 'gpu') & (df['mpi_procs'] == 1)]
        if not gpu_data.empty:
            fastest_gpu = gpu_data.loc[gpu_data['duration_ms'].idxmin()]
            f.write(f"Najszybszy GPU: {fastest_gpu['threads']} wątków\n")
            f.write(f"  - Czas: {fastest_gpu['duration_ms']:.2f} ms\n\n")

        # Porównanie CPU vs GPU
        if not cpu_data.empty and not gpu_data.empty:
            f.write("PORÓWNANIE CPU vs GPU:\n")
            f.write("-" * 80 + "\n")

            for threads in sorted(cpu_data['threads'].unique()):
                cpu_time = cpu_data[cpu_data['threads'] == threads]['duration_ms'].values
                gpu_time = gpu_data[gpu_data['threads'] == threads]['duration_ms'].values

                if len(cpu_time) > 0 and len(gpu_time) > 0:
                    cpu_time = cpu_time[0]
                    gpu_time = gpu_time[0]
                    ratio = cpu_time / gpu_time
                    winner = "GPU" if gpu_time < cpu_time else "CPU"

                    f.write(f"{threads} wątków: CPU={cpu_time:.0f}ms, GPU={gpu_time:.0f}ms ")
                    f.write(f"(Winner: {winner}, ratio: {ratio:.2f}x)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"[OK] Saved: {output_file}")

# =============================================================================
# GŁÓWNA FUNKCJA
# =============================================================================

def main():
    """Główna funkcja - generuje wszystkie wykresy"""
    print("=" * 80)
    print("PRIR PROJECT - PLOT GENERATOR")
    print("=" * 80)
    print()

    # Wczytaj dane
    df = load_data()

    # Generuj wykresy
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_1_cpu_threads(df)
    plot_2_cpu_vs_gpu(df)
    plot_3_speedup_cpu(df)
    plot_4_mpi_scaling(df)
    plot_5_efficiency(df)
    plot_6_hybrid_heatmap(df)
    plot_7_throughput(df)

    # Generuj statystyki
    generate_summary_stats(df)

    # Podsumowanie
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print(f"Output directory: {OUTPUT_DIR}/")
    print("Files created:")
    print("  - plot_1_cpu_threads.png")
    print("  - plot_2_cpu_vs_gpu.png")
    print("  - plot_3_speedup_cpu.png")
    print("  - plot_4_mpi_scaling.png")
    print("  - plot_5_efficiency.png")
    print("  - plot_6_hybrid_heatmap.png")
    print("  - plot_7_throughput.png")
    print("  - summary_stats.txt")
    print()
    print("Next steps:")
    print("  1. Review plots in the plots/ directory")
    print("  2. Include relevant plots in your report")
    print("  3. Analyze summary_stats.txt for key findings")
    print()

if __name__ == "__main__":
    main()
