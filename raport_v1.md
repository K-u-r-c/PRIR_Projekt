# Równoległy analizator logów systemowych

## Raport z projektu z przedmiotu Programowanie Równoległe i Rozproszone

**Temat**: Analiza logów systemowych (zliczanie i filtrowanie)

**Technologie**: OpenMP, MPI, CUDA

**Autorzy**: Kurc Jakub, Kusper Karol - DS2

**Data**: [Data]

---

## Spis treści

1. [Wstęp](#1-wstęp)
2. [Architektura i technologie](#2-architektura-i-technologie)
3. [Implementacja - analiza kodu](#3-implementacja---analiza-kodu)
4. [Instrukcja obsługi aplikacji](#4-instrukcja-obsługi-aplikacji)
5. [Testy i analiza wydajności](#5-testy-i-analiza-wydajności)
6. [Analiza wyników i wnioski](#6-analiza-wyników-i-wnioski)
7. [Podsumowanie](#7-podsumowanie)
8. [Załączniki](#załączniki)

---

# 1. WSTĘP

## 1.1. Cel projektu

Celem projektu jest opracowanie efektywnego narzędzia do równoległej analizy wielkich zbiorów danych tekstowych, w szczególności plików logów systemowych. Współczesne systemy informatyczne generują ogromne ilości danych diagnostycznych, których analiza w czasie rzeczywistym staje się kluczowym wyzwaniem dla administratorów systemów oraz zespołów monitorujących infrastrukturę IT.

Problem analizy logów dotyczy wielu dziedzin zastosowań. Serwery webowe generują setki gigabajtów zapisów każdego dnia, zawierających informacje o żądaniach HTTP, błędach aplikacji oraz próbach nieautoryzowanego dostępu. Systemy IoT (Internet of Things) produkują strumienie danych z czujników wymagających ciągłej analizy w celu wykrywania anomalii. Aplikacje rozproszone zapisują zdarzenia z wielu węzłów, które należy agregować i korelować w celu diagnostyki problemów wydajnościowych.

Projekt realizuje narzędzie umożliwiające zliczanie częstości występowania określonych słów lub fraz w plikach tekstowych, filtrowanie wierszy według zadanych kryteriów oraz tworzenie statystyk czasowych zdarzeń. Program analizuje pliki logów i pozwala na identyfikację wzorców, takich jak liczba błędów krytycznych w określonym przedziale czasowym, częstotliwość określonych zdarzeń na minutę lub godzinę, oraz wyodrębnianie linii spełniających złożone kryteria filtrowania.

Funkcjonalność programu obejmuje następujące możliwości: zliczanie wystąpień dowolnej liczby fraz jednocześnie, filtrowanie rekordów według poziomu logowania (ERROR, WARNING, INFO, DEBUG), ograniczanie analizy do określonego okna czasowego, generowanie statystyk czasowych z zadaną granularnością (minuty lub godziny), oraz opcjonalne wypisywanie dopasowanych linii na standardowe wyjście lub do pliku.

## 1.2. Założenia projektu

Głównym założeniem projektowym jest wykorzystanie trzech komplementarnych poziomów równoległości w celu maksymalizacji wydajności przetwarzania. Każda z zastosowanych technologii adresuje inny aspekt problemu wydajnościowego.

Technologia OpenMP (Open Multi-Processing) stanowi pierwszy poziom równoległości, realizowany na poziomie wątków procesora w pamięci współdzielonej. OpenMP umożliwia równoległą tokenizację tekstu oraz budowanie lokalnych słowników fraz przez poszczególne wątki, które następnie są scalane w fazie redukcji. Wykorzystanie dyrektyw kompilatora upraszcza implementację i pozwala na szybkie prototypowanie równoległych algorytmów bez konieczności ręcznego zarządzania wątkami.

Drugi poziom równoległości realizuje biblioteka MPI (Message Passing Interface), która pozwala na podział pracy między wiele procesów działających w rozproszonym środowisku. W kontekście analizy plików MPI umożliwia podział dużego pliku na fragmenty, które są przetwarzane niezależnie przez różne procesy. Każdy proces MPI może działać na osobnym rdzeniu procesora lub nawet na innym węźle obliczeniowym w klastrze, co pozwala na skalowanie do bardzo dużych zbiorów danych. Po zakończeniu obliczeń, proces główny (rank 0) zbiera i agreguje wyniki od pozostałych procesów.

Trzeci poziom równoległości stanowi przyspieszenie GPU przy użyciu technologii CUDA (Compute Unified Device Architecture). Karty graficzne oferują tysiące rdzeni obliczeniowych działających równolegle, co czyni je idealnymi do operacji agregacji danych. W projekcie CUDA wykorzystywana jest do budowania histogramów zliczających wystąpienia fraz. Kernel GPU przyjmuje listę indeksów znalezionych dopasowań i przy użyciu operacji atomowych inkrementuje liczniki w pamięci globalnej karty graficznej.

Cel wydajnościowy projektu zakłada osiągnięcie przepustowości przetwarzania rzędu gigabajtów na sekundę dla typowych plików logów. Architektura trójpoziomowa pozwala na adaptację do różnych środowisk obliczeniowych: od laptopa z wielordzeniowym procesorem, przez stację roboczą z kartą graficzną NVIDIA, po klaster obliczeniowy z dziesiątkami węzłów. Modularność rozwiązania umożliwia selektywne wyłączanie poszczególnych poziomów równoległości w zależności od dostępnych zasobów sprzętowych.

## 1.3. Wymagania techniczne

Kompilacja i uruchomienie projektu wymaga środowiska spełniającego następujące wymagania techniczne. Kompilator C++ musi wspierać standard C++20, co zapewnia dostęp do nowoczesnych funkcjonalności języka, takich jak concepts, ranges oraz ulepszone wyrażenia lambda. Przetestowano kompatybilność z kompilatorami GCC w wersji 10 lub nowszej, Clang 12 lub nowszej, oraz Microsoft Visual C++ 2019 lub nowszej.

Biblioteka OpenMP jest zazwyczaj dołączona do kompilatora i nie wymaga osobnej instalacji. Dla GCC wystarczy flaga kompilacji `-fopenmp`, a dla Clang odpowiednio `-fopenmp=libomp`. Biblioteka MPI wymaga zainstalowania implementacji takiej jak OpenMPI lub MPICH. W systemach Linux dystrybucje Debian/Ubuntu udostępniają pakiety `libopenmpi-dev` i `openmpi-bin`, podczas gdy w CentOS/RHEL odpowiednie pakiety to `openmpi-devel`.

Wsparcie dla CUDA jest opcjonalne i wymaga karty graficznej NVIDIA z architekturą compute capability 3.5 lub nowszą. Konieczna jest instalacja CUDA Toolkit w wersji 11.0 lub nowszej, zawierającego kompilator nvcc oraz biblioteki runtime. Sterowniki graficzne NVIDIA muszą być zainstalowane w wersji odpowiadającej używanej wersji CUDA. W systemach bez karty NVIDIA program działa bez ograniczeń funkcjonalnych, wykorzystując wyłącznie obliczenia CPU.

Środowisko testowe i dashboardowe wykorzystuje dodatkowo Python 3.8 lub nowszy oraz Node.js 18 lub nowszy. Backend FastAPI wymaga zainstalowania pakietów: `fastapi`, `uvicorn`, `pydantic`. Frontend React wymaga menedżera pakietów npm oraz następujących zależności: `react`, `typescript`, `vite`. Środowisko deweloperskie może działać na systemach Windows, Linux oraz macOS.

## 1.4. Opis danych testowych

Dane wykorzystane do testowania i walidacji projektu pochodzą z publicznie dostępnego zbioru logów serwerowych udostępnionego w repozytorium Harvard Dataverse. Zbiór danych zawiera rzeczywiste logi z irańskiego serwisu e-commerce zanbil.ir, co czyni go reprezentatywnym dla rzeczywistych scenariuszy produkcyjnych.

Plik `access.log` ma rozmiar 3.3 GB i zawiera zapisy aktywności serwera webowego w standardowym formacie logów Apache/Nginx. Każda linia reprezentuje pojedyncze żądanie HTTP i zawiera następujące informacje: adres IP klienta, znacznik czasowy, metodę HTTP (GET, POST, PUT, DELETE), żądany zasób (URL), kod odpowiedzi HTTP (200, 404, 500 itd.) oraz rozmiar odpowiedzi w bajtach. Format czasowy jest zgodny ze specyfikacją Common Log Format (CLF), co ułatwia parsowanie i ekstrakcję informacji temporalnych.

Struktura pojedynczego rekordu wygląda następująco:
```
192.168.1.100 - - [01/Jan/2019:12:34:56 +0000] "GET /api/products HTTP/1.1" 200 1234
```

Zbiór danych został wybrany ze względu na kilka kluczowych cech. Po pierwsze, rozmiar 3.3 GB jest wystarczający do demonstracji korzyści płynących z równoległego przetwarzania, przy czym czas analizy na pojedynczym wątku jest mierzalny (rzędu dziesiątek sekund), co pozwala na wygodne porównywanie wydajności różnych konfiguracji. Po drugie, dane pochodzą z rzeczywistego środowiska produkcyjnego, co oznacza obecność naturalnych wzorców ruchu, burst-ów aktywności, oraz różnorodności żądań. Po trzecie, dostępność w domenie publicznej (licencja Creative Commons) oraz możliwość cytowania źródła (DOI: 10.7910/DVN/3QBYB5) spełnia wymogi reprodukowalności badań.

Analiza tego zbioru danych pozwala na realizację typowych przypadków użycia, takich jak: zliczanie żądań według metody HTTP (GET vs POST), identyfikacja najpopularniejszych endpointów API, analiza rozkładu kodów błędów (4xx, 5xx), wykrywanie anomalii w ruchu sieciowym, oraz tworzenie profili aktywności użytkowników w czasie.

---

# 2. ARCHITEKTURA I TECHNOLOGIE

## 2.1. Architektura ogólna systemu

System składa się z trzech głównych komponentów tworzących pełne środowisko do analizy logów oraz wizualizacji wyników. Architektura została zaprojektowana w sposób modularny, umożliwiający niezależne działanie poszczególnych elementów.

Centralnym komponentem jest program analityczny napisany w C++, który implementuje algorytmy przetwarzania równoległego z wykorzystaniem OpenMP, MPI oraz CUDA. Program operuje w trybie wsadowym, przyjmując na wejściu ścieżkę do pliku logów oraz parametry określające sposób analizy. Komunikacja odbywa się poprzez standardowe wejście i wyjście systemu operacyjnego, co pozwala na łatwą integrację z innymi narzędziami Unix-owymi poprzez potoki i przekierowania.

Backend aplikacji webowej zaimplementowano przy użyciu frameworka FastAPI w języku Python. Serwer udostępnia interfejs RESTful API umożliwiający zdalne wykonywanie testów wydajnościowych oraz pobieranie wyników. Backend pełni rolę adaptera między interfejsem webowym a programem C++, uruchamiając go jako podproces systemowy i przechwytując jego wyjście. Serwer działa domyślnie na porcie 8000 i obsługuje żądania HTTP z Cross-Origin Resource Sharing (CORS), co pozwala na komunikację z frontendem działającym w oddzielnej domenie.

Frontend stanowi aplikacja jednostronicowa (SPA) zbudowana w React z TypeScript i Vite jako narzędziem budowania. Interfejs użytkownika prezentuje listę predefiniowanych scenariuszy testowych, umożliwia modyfikację parametrów uruchomienia (liczba wątków, włączenie CUDA, dodatkowe argumenty CLI), oraz wyświetla wyniki w formie tabel i wykresów. Aplikacja komunikuje się z backendem poprzez asynchroniczne żądania HTTP wykorzystując Fetch API przeglądarki.

```
┌─────────────────────────────────────────┐
│         Frontend (React)                │
│      http://localhost:5173              │
│                                         │
│  - Lista testów                        │
│  - Konfiguracja parametrów             │
│  - Wizualizacja wyników                │
└────────────┬────────────────────────────┘
             │
             │ HTTP POST/GET
             │ (JSON)
             ▼
┌─────────────────────────────────────────┐
│         Backend (FastAPI)               │
│      http://localhost:8000              │
│                                         │
│  - RESTful API endpoints               │
│  - Uruchamianie testów                 │
│  - Parsowanie wyników                  │
└────────────┬────────────────────────────┘
             │
             │ subprocess.run()
             │ (stdin/stdout)
             ▼
┌─────────────────────────────────────────┐
│      Program C++ (prir)                 │
│      ./build/bin/prir                   │
│                                         │
│  - OpenMP (wątki CPU)                  │
│  - MPI (procesy rozproszone)           │
│  - CUDA (kernel GPU)                   │
│  - Analiza pliku access.log            │
└─────────────────────────────────────────┘
```

Przepływ danych rozpoczyna się od interakcji użytkownika z frontendem, który wysyła żądanie HTTP POST do backendu zawierające identyfikator testu oraz parametry nadpisania. Backend konstruuje odpowiednią komendę CLI, uruchamia program C++ jako podproces, przechwytuje jego standardowe wyjście oraz mierzy czas wykonania. Wyniki są następnie parsowane i formatowane do struktury JSON, która jest zwracana do frontendu. Frontend odbiera odpowiedź i aktualizuje interfejs użytkownika, prezentując wyniki w formie tekstowej oraz opcjonalnie generując wykresy porównawcze.

## 2.2. Wybór technologii równoległości

### 2.2.1. OpenMP

OpenMP (Open Multi-Processing) jest standardem definiującym zestaw dyrektyw kompilatora, funkcji bibliotecznych oraz zmiennych środowiskowych przeznaczonych do programowania równoległego w architekturach pamięci współdzielonej (shared memory). Główną zaletą OpenMP jest prostota implementacji równoległości poprzez adnotacje do istniejącego kodu sekwencyjnego, co znacząco obniża próg wejścia w porównaniu z niskopoziomowym programowaniem wątkowym.

W kontekście projektu OpenMP wykorzystywany jest do równoległego przetwarzania wierszy pliku logów. Dyrektywa `#pragma omp parallel` tworzy zespół wątków, które wykonują kod znajdujący się w następującym bloku. Dyrektywa `#pragma omp for` dzieli iteracje pętli między dostępne wątki, realizując dekompozycję danych (data parallelism). Model fork-join zapewnia, że wszystkie wątki synchronizują się na końcu regionu równoległego przed kontynuacją wykonania sekwencyjnego.

Kluczową funkcjonalnością OpenMP w projekcie jest dynamiczny load balancing realizowany przez klauzulę `schedule(dynamic, chunk_size)`. W przeciwieństwie do statycznego podziału, gdzie każdy wątek otrzymuje z góry określony zakres iteracji, podział dynamiczny pozwala wątkom pobierać kolejne porcje pracy w trakcie wykonania. Jest to istotne, ponieważ linie w pliku logów mają różną długość i wymagają różnego nakładu obliczeniowego (parsowanie daty, dopasowywanie wyrażeń, zliczanie wystąpień). Wątki, które szybciej kończą swoją porcję pracy, mogą natychmiast pobrać kolejną, co minimalizuje czas bezczynności.

Projekt wykorzystuje także sekcje krytyczne (`#pragma omp critical`) do synchronizacji dostępu do współdzielonych struktur danych podczas scalania lokalnych wyników wątków. Sekcja krytyczna gwarantuje, że tylko jeden wątek na raz może wykonywać zawarty w niej kod, co jest niezbędne do zachowania spójności danych przy aktualizacji globalnych liczników oraz map.

Zastosowanie OpenMP w projekcie realizuje pierwszy poziom równoległości, efektywnie wykorzystując wiele rdzeni współczesnych procesorów bez konieczności jawnego zarządzania wątkami czy muteksami. Overhead związany z tworzeniem wątków i synchronizacją jest minimalny w porównaniu z korzyściami płynącymi z równoległego przetwarzania danych.

### 2.2.2. MPI

Message Passing Interface (MPI) jest standardem biblioteki do programowania systemów równoległych i rozproszonych, bazującym na paradygmacie przekazywania komunikatów (message passing). W przeciwieństwie do OpenMP, gdzie wątki współdzielą przestrzeń adresową, procesy MPI posiadają oddzielną pamięć i komunikują się poprzez jawne wysyłanie i odbieranie wiadomości.

W projekcie MPI służy do realizacji równoległości na poziomie procesów, gdzie każdy proces przetwarza niezależny fragment pliku logów. Kluczową funkcjonalnością jest podział pliku między procesy na podstawie offsetów bajtowych. Funkcje `MPI_Comm_size()` i `MPI_Comm_rank()` pozwalają każdemu procesowi ustalić swoją pozycję w grupie oraz liczbę procesów, na podstawie czego obliczany jest zakres bajtów do przetworzenia.

Implementacja podziału opiera się na równomiernej dekompozycji przestrzeni bajtowej pliku. Dla pliku o rozmiarze F bajtów i N procesów, proces o randze r przetwarza fragment od bajtu (F × r) / N do bajtu (F × (r+1)) / N. Taki podział może prowadzić do przecięcia linii w środku, dlatego procesy o randze większej od zera pomijają pierwszą niepełną linię, a proces o randze zero czyta ją w całości. Mechanizm ten zapewnia, że każda linia jest przetwarzana dokładnie raz.

Po zakończeniu lokalnego przetwarzania, każdy proces robotniczy (rank > 0) serializuje swoje wyniki do bufora binarnego i wysyła je do procesu głównego (rank 0) przy użyciu funkcji `MPI_Send()`. Proces główny odbiera wiadomości poprzez `MPI_Recv()` w pętli, scalając przychodzące wyniki poprzez sumowanie liczników oraz agregację statystyk czasowych. Ten wzorzec komunikacji nazywany jest redukcją master-worker.

Wykorzystanie MPI pozwala na skalowanie wydajności proporcjonalne do liczby procesów, przy założeniu wystarczającej granularności problemu. Ponieważ plik jest dużych rozmiarów (gigabajty), a komunikacja ogranicza się do przesłania końcowych wyników (zazwyczaj kilkadziesiąt kilobajtów), overhead komunikacyjny stanowi znikomy ułamek czasu obliczeń. MPI umożliwia także łatwe rozszerzenie rozwiązania na klaster obliczeniowy, gdzie procesy mogą działać na fizycznie rozdzielonych maszynach połączonych siecią.

### 2.2.3. CUDA

CUDA (Compute Unified Device Architecture) jest równoległą platformą obliczeniową oraz modelem programowania opracowanym przez firmę NVIDIA dla kart graficznych. GPU (Graphics Processing Unit) zawiera tysiące prostych rdzeni obliczeniowych zorganizowanych w Streaming Multiprocessors (SM), co czyni je idealnymi do obliczeń typu SIMD (Single Instruction Multiple Data) oraz operacji agregacyjnych.

W projekcie CUDA wykorzystywana jest do akceleracji budowania histogramów zliczających wystąpienia fraz. Po zakończeniu fazy przetwarzania tekstu przez wątki OpenMP, program posiada wektor indeksów reprezentujących dopasowane frazy. Wektor ten jest transferowany do pamięci globalnej GPU, gdzie kernel obliczeniowy inkrementuje odpowiednie liczniki w tablicy wyjściowej.

Kernel CUDA definiowany jest poprzez kwalifikator `__global__`, który oznacza funkcję wykonywaną na GPU i wywoływaną z CPU. Każdy wątek GPU (CUDA thread) identyfikowany jest poprzez unikalne indeksy: `blockIdx.x` (indeks bloku) oraz `threadIdx.x` (indeks wątku w bloku). Globalne ID wątku obliczane jest jako `blockIdx.x * blockDim.x + threadIdx.x`, gdzie `blockDim.x` to liczba wątków w bloku (zazwyczaj 256 lub 512).

Kluczowym elementem implementacji jest użycie operacji atomowych, konkretnie `atomicAdd()`, która gwarantuje poprawną aktualizację współdzielonego licznika w sytuacji, gdy wiele wątków GPU próbuje jednocześnie inkrementować tę samą wartość. Operacja atomowa zapewnia, że odczyt-modyfikacja-zapis odbywa się jako niepodzielna transakcja, eliminując race conditions.

Wykorzystanie GPU wiąże się z nakładem związanym z transferem danych między pamięcią systemową (host) a pamięcią GPU (device). Funkcja `cudaMemcpy()` realizuje kopiowanie danych z flagami określającymi kierunek transferu: `cudaMemcpyHostToDevice` oraz `cudaMemcpyDeviceToHost`. Dla małych zbiorów danych overhead transferu może przewyższać korzyści z równoległego przetwarzania, dlatego CUDA jest opcjonalna i włączana poprzez parametr `--use-cuda`.

W przypadku braku karty NVIDIA lub wystąpienia błędu podczas inicjalizacji CUDA, program automatycznie przełącza się na obliczenia CPU bez przerywania działania. Ten mechanizm fallback zapewnia przenośność aplikacji na różne konfiguracje sprzętowe.

### 2.2.4. FastAPI i React

Backend aplikacji webowej zaimplementowano przy użyciu FastAPI, nowoczesnego frameworka webowego dla języka Python charakteryzującego się wysoką wydajnością oraz automatyczną generacją dokumentacji API. FastAPI bazuje na standardach OpenAPI i JSON Schema, co pozwala na automatyczne tworzenie interaktywnej dokumentacji dostępnej pod adresem `/docs`. Framework wykorzystuje anotacje typów Pythona (type hints) do walidacji danych wejściowych oraz serializacji odpowiedzi, co redukuje liczbę błędów runtime.

Głównym zastosowaniem backendu jest orkiestracja testów wydajnościowych. Endpoint `POST /api/tests/{test_id}/scenarios/{scenario_id}/run` przyjmuje identyfikatory testu i scenariusza oraz opcjonalne nadpisania parametrów (liczba wątków, włączenie CUDA, dodatkowe argumenty CLI). Backend konstruuje komendę systemową, uruchamia program C++ przy użyciu modułu `subprocess`, mierzy czas wykonania oraz parsuje wyjście programu. Dla testów typu perf-test backend wyciąga szczegółowe informacje o czasach CPU i CUDA przy użyciu wyrażeń regularnych.

Frontend aplikacji zbudowano jako Single Page Application (SPA) w bibliotece React z TypeScript jako językiem programowania. React umożliwia tworzenie komponentów UI o określonym stanie (state) oraz reaktywne odświeżanie widoku w odpowiedzi na zmiany danych. TypeScript dodaje statyczne typowanie do JavaScript, co pozwala na wczesne wykrywanie błędów podczas kompilacji oraz lepsze wsparcie IDE poprzez autouzupełnianie i refaktoryzację.

Aplikacja prezentuje listę testów zdefiniowanych w pliku `test_cases.json`, umożliwia wybór scenariusza testowego, modyfikację parametrów poprzez formularz, oraz uruchomienie testu przyciskiem. Po zakończeniu testu wyniki wyświetlane są w czytelnej formie: kod wyjścia, czas wykonania, standardowe wyjście oraz błędy. Dla testów perf-test frontend generuje porównawczą tabelę czasów CPU vs CUDA.

Wybór architektury klient-serwer z RESTful API pozwala na rozdzielenie warstwy prezentacji od logiki biznesowej. Frontend może być hostowany na osobnym serwerze lub udostępniany jako pliki statyczne, podczas gdy backend może działać na maszynie obliczeniowej posiadającej zasoby wymagane do uruchomienia testów (wiele rdzeni CPU, karta GPU). Komunikacja poprzez HTTP/JSON jest standardem przemysłowym, co ułatwia integrację z innymi systemami oraz rozbudowę o dodatkowe funkcjonalności.

## 2.3. Struktura projektu

Organizacja kodu źródłowego projektu jest hierarchiczna i zgodna z najlepszymi praktykami projektów C++ oraz aplikacji webowych. Katalog główny zawiera główne pliki konfiguracyjne oraz podkatalogi z kodem poszczególnych komponentów.

```
PRIR_Projekt/
├── src/
│   ├── main.cpp              # Główna logika programu (OpenMP + MPI)
│   ├── gpu_histogram.cu      # Kernel CUDA dla histogramu GPU
│   └── gpu_interface.hpp     # Interfejs C++ do funkcji CUDA
├── backend/
│   ├── main.py               # Serwer FastAPI
│   ├── requirements.txt      # Zależności Python
│   └── test_data/
│       └── test_cases.json   # Definicje scenariuszy testowych
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Główny komponent React
│   │   ├── types.ts          # Definicje typów TypeScript
│   │   └── main.tsx          # Punkt wejścia aplikacji
│   ├── package.json          # Zależności npm
│   └── vite.config.ts        # Konfiguracja narzędzia budowania
├── build/
│   └── bin/
│       └── prir              # Skompilowany program (po make)
├── plots/                    # Wygenerowane wykresy (po testach)
├── Makefile                  # System budowania
├── run_benchmarks.sh         # Skrypt automatyzacji testów
├── generate_plots.py         # Skrypt generowania wykresów
├── benchmark_results.csv     # Wyniki testów (po uruchomieniu)
├── access.log                # Plik danych testowych (3.3 GB)
├── README.md                 # Dokumentacja użytkownika

```

Katalog `src/` zawiera kod źródłowy programu analitycznego w C++. Plik `main.cpp` implementuje całą logikę aplikacji, łącznie z parsowaniem argumentów CLI, inicjalizacją MPI, czytaniem pliku, równoległym przetwarzaniem oraz agregacją wyników. Plik `gpu_histogram.cu` zawiera implementację kernela CUDA oraz funkcję hosta uruchamiającą obliczenia na GPU. Plik nagłówkowy `gpu_interface.hpp` definiuje interfejs między kodem C++ a CUDA, eksportując funkcję `gpu::histogram()`, która jest dostępna zarówno w trybie z CUDA jak i bez.

Katalog `backend/` zawiera serwer FastAPI wraz z plikiem `requirements.txt` listującym zależności Python. Podkatalog `test_data/` zawiera plik JSON definiujący testy, ich scenariusze, oczekiwane wyniki oraz dane referencyjne wydajności.

Katalog `frontend/` zawiera kod aplikacji React. Podkatalog `src/` zawiera pliki TypeScript definiujące komponenty UI, typy danych oraz logikę stanu aplikacji. Pliki `package.json` oraz `vite.config.ts` konfigurują środowisko budowania i zależności JavaScript.

Katalog `build/` jest tworzony automatycznie podczas kompilacji i zawiera pliki obiektowe oraz finalny plik wykonywalny `build/bin/prir`. Katalog ten jest ignorowany przez system kontroli wersji.

Makefile w katalogu głównym definiuje targets do kompilacji projektu: `make` (release build z optymalizacjami), `make debug` (build debugowy z symbolami), `make clean` (usunięcie plików obiektowych), oraz `make info` (wyświetlenie konfiguracji kompilacji). Makefile obsługuje zmienne sterujące: `USE_MPI`, `USE_OPENMP`, `USE_CUDA`, które pozwalają na selektywne włączanie poszczególnych technologii.

Pliki `run_benchmarks.sh` oraz `generate_plots.py` są skryptami automatyzującymi proces testowania wydajnościowego oraz wizualizacji wyników. Pierwsze uruchamia serię testów dla różnych konfiguracji i zapisuje wyniki do CSV. Drugi generuje wykresy porównawcze oraz plik tekstowy z podsumowaniem statystyk.

---

# 3. IMPLEMENTACJA - ANALIZA KODU

## 3.1. Główny plik: src/main.cpp

Plik `main.cpp` stanowi rdzeń programu i zawiera implementację wszystkich kluczowych algorytmów przetwarzania równoległego. Kod ma 858 linii i jest podzielony na funkcje realizujące poszczególne etapy przetwarzania: parsowanie argumentów, czytanie pliku, analizę danych, agregację wyników oraz wypisywanie statystyk.

### 3.1.1. Parsowanie argumentów CLI

Funkcja `parse_cli()` (linie 282-372) odpowiada za interpretację argumentów wiersza poleceń oraz walidację poprawności konfiguracji. Wykorzystuje strukturę `ProgramConfig` do przechowywania wszystkich opcji:

```cpp
struct ProgramConfig {
  std::string filePath;              // --file
  std::vector<std::string> phrases;  // --phrase (może być wiele)
  bool caseSensitive = false;        // --case-sensitive
  bool useCuda = false;              // --use-cuda
  bool cpuOnly = false;              // --cpu-only
  int threads = 0;                   // --threads (0 = auto)
  std::vector<std::string> severityFilters; // --level
  std::string fromTime;              // --from
  std::string toTime;                // --to
  bool statsEnabled = true;          // --stats / --no-stats
  std::string statsInterval = "hour"; // hour|minute
  bool emitMatches = false;          // --emit
  std::string emitFile;              // --emit-file
  bool countOnlyFiltered = false;    // --count-filtered
  bool perfTest = false;             // --perf-test
};
```

Parser iteruje przez tablicę `argv` i rozpoznaje flagi poprzez porównanie stringów. Dla opcji powtarzalnych (np. `--phrase`, `--level`) wartości są dodawane do wektorów. Parser wykonuje także walidację:

- Sprawdza obecność wymaganych parametrów (`--file`, przynajmniej jedna `--phrase`)
- Waliduje wzajemne wykluczanie się opcji (`--use-cuda` i `--cpu-only` nie mogą wystąpić jednocześnie)
- Weryfikuje dostępność CUDA w przypadku `--use-cuda` poprzez wywołanie `gpu::is_available()`
- Normalizuje frazy do małych liter, jeśli nie ustawiono `--case-sensitive`

Funkcja zwraca strukturę `ProgramConfig` lub kończy program z komunikatem błędu w przypadku niepoprawnych argumentów. Użycie struktury zamiast globalnych zmiennych poprawia testowalność i czytelność kodu.

### 3.1.2. Podział pliku między procesy MPI

Funkcja `read_chunk()` (linie 374-400) realizuje podział dużego pliku między procesy MPI poprzez obliczenie offsetów bajtowych i czytanie przydzielonego fragmentu. Implementacja rozpoczyna się od uzyskania informacji o środowisku MPI:

```cpp
int world = 1, rank = 0;
#ifdef USE_MPI
MPI_Comm_size(MPI_COMM_WORLD, &world);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
```

Dyrektywy preprocesora `#ifdef USE_MPI` pozwalają na kompilację warunkową, umożliwiając budowanie projektu bez MPI. W takim przypadku zmienne `world` i `rank` pozostają z wartościami domyślnymi (1 i 0), co odpowiada wykonaniu sekwencyjnemu.

Następnie program otwiera plik i oblicza jego rozmiar:

```cpp
std::ifstream in(cfg.filePath, std::ios::binary);
in.seekg(0, std::ios::end);
uint64_t fileSize = in.tellg();
```

Wykorzystanie trybu binarnego (`std::ios::binary`) zapewnia, że pozycje w pliku odpowiadają faktycznym offsetom bajtowym, niezależnie od systemu operacyjnego (Windows vs Unix line endings).

Obliczenie zakresu dla danego procesu odbywa się poprzez równomierne podzielenie przestrzeni bajtowej:

```cpp
uint64_t chunkStart = (fileSize * rank) / world;
uint64_t chunkEnd = (fileSize * (rank + 1)) / world;
in.seekg(chunkStart);
```

Kluczowym problemem przy tym podziale jest możliwość przecięcia linii w środku. Rozwiązanie polega na pomijaniu pierwszej niepełnej linii przez procesy o randze większej od zera:

```cpp
if (rank != 0) {
  std::string dummy;
  std::getline(in, dummy);  // Pomiń pierwszą (potencjalnie niepełną) linię
}
```

W ten sposób proces 0 przeczyta tę linię w całości, a proces 1 rozpocznie czytanie od pierwszej pełnej linii w swoim zakresie. Mechanizm ten zapewnia, że żadna linia nie jest pominięta ani zduplikowana.

Funkcja następnie wczytuje linie do wektora, aż do osiągnięcia końca swojego zakresu:

```cpp
FileChunk chunk;
std::string line;
while (in.tellg() < (std::streampos)chunkEnd && std::getline(in, line)) {
  chunk.lines.push_back(std::move(line));
}
return chunk;
```

Użycie semantyki move (`std::move`) unika niepotrzebnego kopiowania stringów, poprawiając wydajność. Wynikowa struktura `FileChunk` zawiera wektor linii gotowych do przetworzenia przez algorytm równoległy.

### 3.1.3. Równoległa analiza linii - OpenMP

Funkcja `analyze_chunk()` (linie 414-537) stanowi serce algorytmu przetwarzania i demonstruje zaawansowane użycie OpenMP. Funkcja przyjmuje wektor linii oraz konfigurację, i zwraca strukturę `LocalResults` zawierającą zliczone frazy, statystyki czasowe oraz dopasowane linie.

Przed rozpoczęciem regionu równoległego funkcja wykonuje przygotowania:

```cpp
LocalResults res;
res.phraseCounts.assign(cfg.phrases.size(), 0);

bool needTime = cfg.statsEnabled || !cfg.fromTime.empty() || !cfg.toTime.empty();
bool needSeverity = !cfg.severityFilters.empty();
```

Inicjalizacja wektora liczników zerami oraz ustalenie, które informacje należy ekstrahować z linii (timestamp, poziom logowania) pozwala uniknąć niepotrzebnych operacji parsowania.

Region równoległy definiowany jest następująco:

```cpp
#pragma omp parallel if (enableParallel)
{
  // Lokalne zmienne wątków
  std::vector<uint64_t> threadCounts(cfg.phrases.size(), 0);
  std::unordered_map<std::string, uint64_t> localBuckets;
  std::vector<std::string> localMatches;
  std::vector<uint32_t> localHits;  // Dla GPU

  #pragma omp for schedule(dynamic, 256) nowait
  for (long long idx = 0; idx < total; ++idx) {
    const std::string &line = chunk.lines[idx];

    // Przetwarzanie linii...
  }

  // Scalanie wyników
  #pragma omp critical
  {
    for (size_t i = 0; i < threadCounts.size(); ++i)
      res.phraseCounts[i] += threadCounts[i];

    for (auto &kv : localBuckets)
      res.timeBuckets[kv.first] += kv.second;

    res.matchingLines.insert(res.matchingLines.end(),
                             localMatches.begin(),
                             localMatches.end());
  }
}
```

Klauzula `if (enableParallel)` pozwala na selektywne wyłączenie równoległości dla małych plików, gdzie overhead tworzenia wątków nie jest uzasadniony. Zmienna `enableParallel` jest ustawiana na `true` dla plików większych niż 1000 linii.

Każdy wątek posiada prywatne kopie struktur danych (`threadCounts`, `localBuckets`, `localMatches`, `localHits`), co eliminuje potrzebę synchronizacji podczas przetwarzania linii. Jest to kluczowe dla wydajności, ponieważ dostęp do współdzielonych danych wymagałby użycia muteksów, drastycznie spowalniając wykonanie.

Dyrektywa `schedule(dynamic, 256)` realizuje dynamiczny load balancing. Pętla for jest dzielona na chunki po 256 iteracji, które są przydzielane wątkom na żądanie. Gdy wątek kończy przetwarzanie swojego chunka, otrzymuje kolejny z puli. Rozmiar chunka (256) jest kompromisem między overhead przydzielania a granularnością balansu obciążenia.

Klauzula `nowait` informuje kompilator, że nie jest wymagana bariera synchronizacyjna na końcu pętli for. Wątki mogą natychmiast przejść do kolejnych instrukcji (w tym przypadku do sekcji critical), co redukuje czas oczekiwania.

Wewnątrz pętli każda linia jest przetwarzana w kilku etapach:

1. **Ekstrakcja timestampu** (jeśli potrzebny):
```cpp
std::time_t ts = 0;
if (needTime) {
  ts = extract_timestamp(line);
}
```

Funkcja `extract_timestamp()` używa wyrażenia regularnego do znalezienia wzorca `YYYY-MM-DD HH:MM:SS` w linii i konwertuje go na `std::time_t`.

2. **Detekcja poziomu logowania** (jeśli potrzebna):
```cpp
std::string severity;
bool severityOk = true;
if (needSeverity) {
  std::string upper = to_upper(line);
  severity = detect_severity(upper, cfg.severityUniverse);
  severityOk = cfg.severityFilters.count(severity) > 0;
}
```

Funkcja `detect_severity()` szuka słów kluczowych (ERROR, WARNING, INFO, DEBUG, CRITICAL) w linii i zwraca pierwszy znaleziony poziom.

3. **Weryfikacja filtrów**:
```cpp
bool windowOk = time_in_window(cfg, ts);
bool selected = severityOk && windowOk;
```

Linia jest uznawana za dopasowaną, jeśli spełnia wszystkie kryteria filtrowania.

4. **Aktualizacja statystyk** (dla dopasowanych linii):
```cpp
if (selected) {
  ++localMatched;

  if (cfg.statsEnabled) {
    std::string bucketKey = make_bucket(ts, cfg.statsInterval);
    ++localBuckets[bucketKey];
  }

  if (cfg.emitMatches) {
    localMatches.push_back(line);
  }
}
```

Funkcja `make_bucket()` zaokrągla timestamp do początku godziny lub minuty, tworząc klucz typu `"2019-01-15 14:00"`.

5. **Zliczanie fraz**:
```cpp
bool allowCounts = !cfg.countOnlyFiltered || selected;
if (allowCounts) {
  std::string lowered = cfg.caseSensitive ? line : to_lower(line);

  for (size_t p = 0; p < cfg.phrasesNormalized.size(); ++p) {
    size_t matches = count_occurrences(lowered, cfg.phrasesNormalized[p]);

    if (cfg.useCuda) {
      for (size_t k = 0; k < matches; ++k) {
        localHits.push_back(p);  // Zapisz indeks frazy dla GPU
      }
    } else {
      threadCounts[p] += matches;  // Zlicz od razu na CPU
    }
  }
}
```

Funkcja `count_occurrences()` używa metody `std::string_view::find()` w pętli do zliczania wszystkich wystąpień podciągu w tekście. W trybie GPU wystąpienia są zapisywane jako indeksy do późniejszego przetworzenia przez kernel, w trybie CPU są natychmiast dodawane do liczników.

Po zakończeniu pętli każdy wątek ma swoje lokalne wyniki, które należy scalić. Sekcja krytyczna zapewnia wzajemne wykluczanie:

```cpp
#pragma omp critical
{
  for (size_t i = 0; i < threadCounts.size(); ++i)
    res.phraseCounts[i] += threadCounts[i];

  for (auto &kv : localBuckets)
    res.timeBuckets[kv.first] += kv.second;

  res.matchingLines.insert(res.matchingLines.end(),
                           localMatches.begin(),
                           localMatches.end());

  if (cfg.useCuda) {
    res.gpuHits.insert(res.gpuHits.end(),
                      localHits.begin(),
                      localHits.end());
  }
}
```

Sekcja krytyczna jest potencjalnym wąskim gardłem, ponieważ tylko jeden wątek może wykonywać ten kod w danym momencie. Jednak dzięki temu, że każdy wątek wykonuje tę sekcję tylko raz (na końcu przetwarzania), overhead synchronizacji jest minimalny w porównaniu do całkowitego czasu obliczeń.

### 3.1.4. Histogram na GPU - CUDA

Interfejs między C++ a CUDA definiowany jest w pliku `gpu_interface.hpp`, który eksportuje namespace `gpu` z dwiema funkcjami:

```cpp
namespace gpu {

bool is_available() {
  // Sprawdza dostępność CUDA
#ifdef USE_CUDA
  return check_cuda_device();
#else
  return false;
#endif
}

bool histogram(const std::vector<uint32_t> &values,
               size_t bucketCount,
               std::vector<uint64_t> &out) {
  // Wywołuje kernel CUDA lub fallback do CPU
#ifdef USE_CUDA
  return gpu_histogram_count(values.data(), values.size(),
                             bucketCount, out.data());
#else
  return false;  // Fallback w main.cpp
#endif
}

}
```

Implementacja kernela znajduje się w pliku `gpu_histogram.cu`:

```cpp
__global__ void histogram_kernel(const uint32_t *values,
                                 size_t count,
                                 unsigned long long *out)
{
  size_t stride = blockDim.x * gridDim.x;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = gid; i < count; i += stride) {
    uint32_t bucket = values[i];
    atomicAdd(&out[bucket], 1ULL);
  }
}
```

Kernel wykorzystuje grid-stride loop pattern, gdzie każdy wątek przetwarza wiele elementów tablicy wejściowej. Zmienna `stride` reprezentuje całkowitą liczbę wątków w gridzie (liczba bloków × liczba wątków na blok), a `gid` to globalny identyfikator wątku.

W pętli for każdy wątek zaczyna od swojego indeksu `gid` i przeskakuje o `stride` elementów w każdej iteracji. Taki wzorzec zapewnia dobrą koalescencję dostępu do pamięci oraz umożliwia przetwarzanie dowolnej liczby elementów niezależnie od liczby wątków.

Operacja `atomicAdd()` jest kluczowa dla poprawności algorytmu. Funkcja ta gwarantuje, że operacja odczyt-inkrementacja-zapis jest wykonywana atomowo, co oznacza, że w sytuacji, gdy wiele wątków jednocześnie próbuje zinkrementować ten sam licznik, wszystkie operacje są wykonywane sekwencyjnie i żadna aktualizacja nie jest tracona.

Funkcja hosta `gpu_histogram_count()` zarządza cyklem życia pamięci GPU:

```cpp
extern "C" bool gpu_histogram_count(const uint32_t *values,
                                    size_t count,
                                    size_t bucketCount,
                                    uint64_t *out)
{
  // 1. Alokacja pamięci GPU
  uint32_t *d_values = nullptr;
  unsigned long long *d_out = nullptr;

  cudaMalloc(&d_values, count * sizeof(uint32_t));
  cudaMalloc(&d_out, bucketCount * sizeof(unsigned long long));

  // 2. Inicjalizacja liczników zerami
  cudaMemset(d_out, 0, bucketCount * sizeof(unsigned long long));

  // 3. Transfer CPU -> GPU
  cudaMemcpy(d_values, values, count * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  // 4. Uruchomienie kernela
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  histogram_kernel<<<numBlocks, blockSize>>>(d_values, count, d_out);

  // 5. Synchronizacja
  cudaDeviceSynchronize();

  // 6. Transfer GPU -> CPU
  cudaMemcpy(out, d_out, bucketCount * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  // 7. Zwolnienie pamięci
  cudaFree(d_values);
  cudaFree(d_out);

  return true;
}
```

Sekwencja operacji jest typowa dla aplikacji CUDA. Pamięć jest alokowana osobno na hoście i device, dane są kopiowane do GPU, kernel jest uruchamiany asynchronicznie, program czeka na zakończenie obliczeń, wyniki są kopiowane z powrotem do pamięci systemowej, a na końcu pamięć GPU jest zwalniana.

Dobór parametrów uruchomienia kernela (`blockSize` = 256 wątków na blok, `numBlocks` zapewniające pokrycie całej tablicy) jest oparty na heurystykach. Rozmiar bloku 256 jest kompromisem między zajętością (occupancy) a dostępnymi zasobami SM (shared memory, rejestry).

### 3.1.5. Agregacja wyników MPI

Po zakończeniu lokalnego przetwarzania każdy proces MPI posiada strukturę `LocalResults` zawierającą wyniki dla swojego fragmentu pliku. Proces główny (rank 0) musi zebrać i zsumować wyniki od wszystkich procesów. Implementacja wzorca master-worker znajduje się w funkcji `main()`:

```cpp
LocalResults aggregated = local;  // Zaczynaj od lokalnych wyników rank 0

#ifdef USE_MPI
if (rank == 0) {
  // Proces główny: odbieraj i agreguj
  for (int src = 1; src < world; ++src) {
    LocalResults incoming = mpi_recv_results(src, cfg);
    merge_results(aggregated, incoming, cfg);
  }
} else {
  // Procesy robocze: wyślij wyniki do rank 0
  mpi_send_results(local, 0, cfg);
}
#endif
```

Funkcja `mpi_send_results()` (linie 644-668) serializuje strukturę `LocalResults` do bufora binarnego i wysyła go poprzez MPI:

```cpp
void mpi_send_results(const LocalResults &res, int dest,
                     const ProgramConfig &cfg) {
#ifdef USE_MPI
  // 1. Wyślij liczniki fraz (stały rozmiar)
  std::vector<uint64_t> counts = res.phraseCounts;
  MPI_Send(counts.data(), counts.size(), MPI_UINT64_T,
           dest, TAG_PHRASE_COUNTS, MPI_COMM_WORLD);

  // 2. Serializuj mapę timeBuckets do wektorów
  std::vector<std::string> keys;
  std::vector<uint64_t> values;
  for (const auto &kv : res.timeBuckets) {
    keys.push_back(kv.first);
    values.push_back(kv.second);
  }

  // Wyślij rozmiar, potem klucze, potem wartości
  int mapSize = keys.size();
  MPI_Send(&mapSize, 1, MPI_INT, dest, TAG_MAP_SIZE, MPI_COMM_WORLD);

  for (const auto &key : keys) {
    int keySize = key.size();
    MPI_Send(&keySize, 1, MPI_INT, dest, TAG_KEY_SIZE, MPI_COMM_WORLD);
    MPI_Send(key.data(), keySize, MPI_CHAR, dest, TAG_KEY, MPI_COMM_WORLD);
  }

  MPI_Send(values.data(), values.size(), MPI_UINT64_T,
           dest, TAG_VALUES, MPI_COMM_WORLD);

  // 3. Serializuj wektor dopasowanych linii
  int numLines = res.matchingLines.size();
  MPI_Send(&numLines, 1, MPI_INT, dest, TAG_NUM_LINES, MPI_COMM_WORLD);

  for (const auto &line : res.matchingLines) {
    int lineSize = line.size();
    MPI_Send(&lineSize, 1, MPI_INT, dest, TAG_LINE_SIZE, MPI_COMM_WORLD);
    MPI_Send(line.data(), lineSize, MPI_CHAR, dest, TAG_LINE, MPI_COMM_WORLD);
  }
#endif
}
```

Każda wiadomość MPI jest oznaczona unikalnym tagiem (TAG_PHRASE_COUNTS, TAG_MAP_SIZE, itd.), co pozwala odbiorcy na prawidłową interpretację przychodzących danych. Dla typów prostych (liczby) używany jest bezpośredni transfer bufora. Dla struktur złożonych (mapy, wektory stringów) stosowana jest ręczna serializacja polegająca na najpierw wysłaniu rozmiaru, a następnie zawartości.

Funkcja `mpi_recv_results()` (linie 670-703) realizuje operację odwrotną, odbierając dane i rekonstruując strukturę `LocalResults`:

```cpp
LocalResults mpi_recv_results(int src, const ProgramConfig &cfg) {
  LocalResults res;

#ifdef USE_MPI
  // 1. Odbierz liczniki fraz
  res.phraseCounts.resize(cfg.phrases.size());
  MPI_Recv(res.phraseCounts.data(), res.phraseCounts.size(), MPI_UINT64_T,
           src, TAG_PHRASE_COUNTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // 2. Odbierz mapę timeBuckets
  int mapSize;
  MPI_Recv(&mapSize, 1, MPI_INT, src, TAG_MAP_SIZE,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int i = 0; i < mapSize; ++i) {
    int keySize;
    MPI_Recv(&keySize, 1, MPI_INT, src, TAG_KEY_SIZE,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::string key(keySize, '\0');
    MPI_Recv(key.data(), keySize, MPI_CHAR, src, TAG_KEY,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    uint64_t value;
    MPI_Recv(&value, 1, MPI_UINT64_T, src, TAG_VALUES,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    res.timeBuckets[key] = value;
  }

  // 3. Odbierz dopasowane linie
  int numLines;
  MPI_Recv(&numLines, 1, MPI_INT, src, TAG_NUM_LINES,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int i = 0; i < numLines; ++i) {
    int lineSize;
    MPI_Recv(&lineSize, 1, MPI_INT, src, TAG_LINE_SIZE,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::string line(lineSize, '\0');
    MPI_Recv(line.data(), lineSize, MPI_CHAR, src, TAG_LINE,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    res.matchingLines.push_back(std::move(line));
  }
#endif

  return res;
}
```

Funkcja `merge_results()` (linie 539-555) scala przychodzące wyniki z akumulatorem:

```cpp
void merge_results(LocalResults &base, const LocalResults &other,
                  const ProgramConfig &cfg) {
  // Sumuj liczniki fraz
  for (size_t i = 0; i < other.phraseCounts.size(); ++i) {
    base.phraseCounts[i] += other.phraseCounts[i];
  }

  // Sumuj buckety czasowe
  for (const auto &kv : other.timeBuckets) {
    base.timeBuckets[kv.first] += kv.second;
  }

  // Dopisz dopasowane linie
  base.matchingLines.insert(base.matchingLines.end(),
                           other.matchingLines.begin(),
                           other.matchingLines.end());

  // Zsumuj statystyki
  base.processedLines += other.processedLines;
  base.matchedLines += other.matchedLines;
}
```

Scalanie jest prostą operacją addytywną dla wszystkich liczników oraz konkatenacją dla wektorów linii. Kluczowym aspektem jest to, że komunikacja MPI zachodzi tylko raz na proces, na samym końcu obliczeń. Oznacza to, że stosunek czasu komunikacji do czasu obliczeń jest bardzo korzystny, co jest kluczowe dla efektywnego wykorzystania MPI.

## 3.2. System budowania - Makefile

Makefile zarządza procesem kompilacji i pozwala na łatwą konfigurację poprzez zmienne sterujące. Główna część Makefile definiuje zmienne konfiguracyjne:

```makefile
# Konfiguracja (można nadpisać: make USE_CUDA=1)
USE_MPI ?= 1
USE_OPENMP ?= 1
USE_CUDA ?= 0

# Kompilatory
CXX ?= mpicxx
NVCC ?= nvcc

# Katalogi
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
SRC_DIR := src
```

Symbol `?=` oznacza przypisanie warunkowe: zmienna jest ustawiana tylko jeśli nie została wcześniej zdefiniowana. Pozwala to na nadpisywanie wartości z linii poleceń, np. `make USE_CUDA=1`.

### 3.2.1. Konfiguracja kompilacji

Flagi kompilacji są budowane warunkowo w zależności od wartości zmiennych sterujących:

```makefile
# Flagi bazowe
CXXFLAGS := -std=c++20 -Wall -Wextra

# Optymalizacje (release)
CXXFLAGS += -O3 -march=native -DNDEBUG

# OpenMP
ifeq ($(USE_OPENMP),1)
  CXXFLAGS += -fopenmp -DUSE_OPENMP
  LDFLAGS += -fopenmp
endif

# MPI
ifeq ($(USE_MPI),1)
  CXXFLAGS += -DUSE_MPI
  # mpicxx już zawiera linki do MPI
endif

# CUDA
ifeq ($(USE_CUDA),1)
  CXXFLAGS += -DUSE_CUDA
  NVCCFLAGS := -O3 -std=c++17 --compiler-options "-fPIC"
  LDFLAGS += -lcudart -L/usr/local/cuda/lib64
endif
```

Flaga `-O3` włącza agresywne optymalizacje kompilatora, łącznie z inlining funkcji, loop unrolling, oraz wektoryzacją. Flaga `-march=native` instruuje kompilator do wykorzystania wszystkich instrukcji dostępnych w procesorze maszyny budującej, co może zwiększyć wydajność o 10-20%, kosztem przenośności binarki.

Definicje preprocesora (`-DUSE_OPENMP`, `-DUSE_MPI`, `-DUSE_CUDA`) są używane w kodzie źródłowym do kompilacji warunkowej bloków kodu zależnych od danej technologii.

### 3.2.2. Kompilacja warunkowa

Target `all` buduje finalny program:

```makefile
# Lista plików źródłowych
SOURCES_CPP := $(wildcard $(SRC_DIR)/*.cpp)
SOURCES_CU :=

ifeq ($(USE_CUDA),1)
  SOURCES_CU := $(wildcard $(SRC_DIR)/*.cu)
endif

# Pliki obiektowe
OBJECTS_CPP := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES_CPP))
OBJECTS_CU := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SOURCES_CU))
OBJECTS := $(OBJECTS_CPP) $(OBJECTS_CU)

# Target główny
all: $(BIN_DIR)/prir

# Linkowanie
$(BIN_DIR)/prir: $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Kompilacja C++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Kompilacja CUDA
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
```

Makefile automatycznie wykrywa pliki `.cpp` i `.cu` w katalogu `src/` i generuje odpowiednie reguły kompilacji. Pliki CUDA są kompilowane tylko jeśli `USE_CUDA=1`.

Target `clean` usuwa wszystkie wygenerowane pliki:

```makefile
clean:
	rm -rf $(BUILD_DIR)
```

Target `debug` buduje wersję debug z symbolami i bez optymalizacji:

```makefile
debug: CXXFLAGS := -std=c++20 -Wall -Wextra -g3 -O0 -DUSE_OPENMP -DUSE_MPI
debug: NVCCFLAGS := -g -G -O0 --compiler-options "-fPIC"
debug: all
```

Target `info` wypisuje aktualną konfigurację:

```makefile
info:
	@echo "Configuration:"
	@echo "  CXX          = $(CXX)"
	@echo "  NVCC         = $(NVCC)"
	@echo "  USE_MPI      = $(USE_MPI)"
	@echo "  USE_OPENMP   = $(USE_OPENMP)"
	@echo "  USE_CUDA     = $(USE_CUDA)"
	@echo "  SOURCES CPP  = $(SOURCES_CPP)"
	@echo "  SOURCES CU   = $(SOURCES_CU)"
```

Makefile zapewnia wygodny sposób budowania projektu w różnych konfiguracjach bez modyfikowania kodu źródłowego.

## 3.3. Backend - FastAPI

Backend FastAPI implementuje RESTful API umożliwiające zdalne uruchamianie testów wydajnościowych oraz pobieranie wyników. Kod znajduje się w pliku `backend/main.py`.

### 3.3.1. Endpointy API

Główna aplikacja FastAPI definiowana jest następująco:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PRIR Test API",
    description="API do uruchamiania testów wydajnościowych analizatora logów",
    version="1.0.0"
)

# CORS middleware - pozwala na żądania z frontendu
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Middleware CORS jest niezbędny, ponieważ frontend działa na innym porcie (5173) niż backend (8000), co jest traktowane przez przeglądarki jako różne originy. Konfiguracja `allow_origins=["*"]` pozwala na żądania z dowolnej domeny, co jest akceptowalne w środowisku deweloperskim.

Endpoint pobierający listę testów:

```python
@app.get("/api/tests")
def get_tests() -> dict:
    """Zwraca pełną konfigurację testów z test_cases.json"""
    test_file = Path(__file__).parent / "test_data" / "test_cases.json"

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "suite": data["suite"],
        "tests": data["tests"]
    }
```

Plik `test_cases.json` zawiera definicje testów w formacie:

```json
{
  "suite": {
    "title": "PRIR Parallel Analyzer Test Bench",
    "binaryPath": "./build/bin/prir",
    "logFile": "access.log"
  },
  "tests": [
    {
      "id": "cuda-benchmark",
      "name": "CPU vs CUDA benchmark",
      "scenarios": [
        {
          "id": "cpu-gpu-perf-test",
          "label": "Run perf-test (CPU vs CUDA)",
          "command": "./build/bin/prir --file access.log --phrase GET --perf-test",
          "perfTest": true
        }
      ]
    }
  ]
}
```

Endpoint uruchamiający test:

```python
@app.post("/api/tests/{test_id}/scenarios/{scenario_id}/run")
def execute_scenario(test_id: str, scenario_id: str,
                    overrides: RunOverrides) -> RunResult:
    """
    Uruchamia scenariusz testowy z opcjonalnymi nadpisaniami parametrów.

    Args:
        test_id: Identyfikator testu
        scenario_id: Identyfikator scenariusza
        overrides: Parametry nadpisania (threads, useCuda, extraArgs)

    Returns:
        RunResult z czasem wykonania, kodem wyjścia i outputem
    """
    # 1. Znajdź test i scenariusz
    test = find_test_by_id(test_id)
    scenario = find_scenario_by_id(test, scenario_id)

    # 2. Aplikuj nadpisania
    final_command = _apply_overrides(scenario["command"], overrides)

    # 3. Uruchom komendę i zmierz czas
    start_time = time.time()
    result = run_command(final_command)
    duration_ms = (time.time() - start_time) * 1000

    # 4. Parsuj output (jeśli perf-test)
    perf_summary = None
    if scenario.get("perfTest"):
        perf_summary = parse_perf_test_output(result.stdout)

    # 5. Zwróć wynik
    return RunResult(
        testId=test_id,
        scenarioId=scenario_id,
        command=final_command,
        exitCode=result.returncode,
        durationMs=duration_ms,
        stdout=result.stdout,
        stderr=result.stderr,
        success=(result.returncode == 0),
        perfTestSummary=perf_summary
    )
```

Struktura `RunOverrides` definiuje możliwe nadpisania:

```python
class RunOverrides(BaseModel):
    threads: Optional[int] = None
    useCuda: Optional[bool] = None
    extraArgs: Optional[str] = None
```

### 3.3.2. Parsowanie wyników perf-test

Funkcja `parse_perf_test_output()` analizuje wyjście programu w trybie `--perf-test` i ekstrahuje informacje o czasach CPU i CUDA:

```python
def parse_perf_test_output(stdout: str) -> Optional[PerfTestSummary]:
    """
    Parsuje output --perf-test w formacie:

    [perf-test] CPU baseline took 3456 ms (8 CPU threads)
    phrase,count
    ERROR,125643

    [perf-test] CUDA pass took 892 ms (GPU histogram)
    phrase,count
    ERROR,125643
    """
    PERF_HEADER_RE = re.compile(
        r"^\[perf-test\]\s+(.*?)\s+took\s+(\d+(?:\.\d+)?)\s+ms"
        r"(?:\s*\((.*?)\))?$"
    )

    entries = []
    lines = stdout.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Szukaj nagłówka perf-test
        match = PERF_HEADER_RE.match(line)
        if match:
            label = match.group(1)  # "CPU baseline" lub "CUDA pass"
            duration_ms = float(match.group(2))
            details = match.group(3) if match.group(3) else ""

            # Następne linie to CSV z frazami
            phrases = []
            i += 1

            # Pomiń linię "phrase,count"
            if i < len(lines) and lines[i].strip() == "phrase,count":
                i += 1

            # Czytaj linie CSV dopóki nie napotkamy pustej
            while i < len(lines) and lines[i].strip():
                parts = lines[i].strip().split(",")
                if len(parts) == 2:
                    phrases.append({
                        "phrase": parts[0],
                        "count": int(parts[1])
                    })
                i += 1

            entries.append({
                "label": label,
                "durationMs": duration_ms,
                "details": details,
                "phrases": phrases
            })

        i += 1

    if entries:
        return {"entries": entries}
    else:
        return None
```

Wyrażenie regularne ekstrahuje trzy grupy: etykietę (CPU baseline / CUDA pass), czas w milisekundach, oraz opcjonalny opis (8 CPU threads / GPU histogram). Parser następnie wczytuje linie CSV zawierające zliczone frazy, aż do napotkania pustej linii.

### 3.3.3. Obsługa nadpisań parametrów

Funkcja `_apply_overrides()` modyfikuje komendę bazową zgodnie z parametrami przekazanymi z frontendu:

```python
def _apply_overrides(base_command: str, overrides: RunOverrides) -> str:
    """
    Aplikuje nadpisania do komendy bazowej.

    Przykład:
        base: "./prir --file log.txt --phrase GET --threads 4"
        overrides: {"threads": 16, "useCuda": True}
        wynik: "./prir --file log.txt --phrase GET --threads 16 --use-cuda"
    """
    import shlex

    # Rozdziel komendę na tokeny
    tokens = shlex.split(base_command)

    # 1. Nadpisz liczbę wątków
    if overrides.threads is not None:
        # Usuń istniejące --threads
        tokens = _strip_flag_with_value(tokens, "--threads")
        # Dodaj nowe
        tokens.extend(["--threads", str(overrides.threads)])

    # 2. Nadpisz CUDA
    if overrides.useCuda is not None:
        # Usuń istniejące --use-cuda i --cpu-only
        tokens = _strip_flag(tokens, "--use-cuda")
        tokens = _strip_flag(tokens, "--cpu-only")

        # Dodaj odpowiednią flagę
        if overrides.useCuda:
            tokens.append("--use-cuda")
        else:
            tokens.append("--cpu-only")

    # 3. Dodaj dodatkowe argumenty
    if overrides.extraArgs:
        extra = shlex.split(overrides.extraArgs)
        tokens.extend(extra)

    # Połącz z powrotem w string
    return shlex.join(tokens)
```

Funkcja wykorzystuje moduł `shlex` do poprawnego parsowania i składania komend z uwzględnieniem cudzysłowów i escape sequences. Pozwala to na bezpieczne przekazywanie parametrów zawierających spacje lub znaki specjalne.

## 3.4. Frontend - React

Frontend aplikacji zbudowano jako Single Page Application w React z TypeScript. Główny komponent znajduje się w pliku `frontend/src/App.tsx`.

### 3.4.1. Komponenty UI

Aplikacja składa się z kilku głównych sekcji renderowanych w komponencie `App`:

```tsx
function App() {
  // Stan aplikacji
  const [suite, setSuite] = useState<SuiteInfo | null>(null);
  const [tests, setTests] = useState<TestCase[]>([]);
  const [selectedTestId, setSelectedTestId] = useState("");
  const [scenarioStates, setScenarioStates] = useState<Record<string, ScenarioState>>({});
  const [scenarioOverrides, setScenarioOverrides] = useState<Record<string, ScenarioOverride>>({});

  // Pobierz testy przy starcie
  useEffect(() => {
    fetchTests();
  }, []);

  return (
    <div className="app">
      <Header suite={suite} />
      <div className="app__content">
        <TestList tests={tests} selectedId={selectedTestId}
                 onSelect={setSelectedTestId} />
        <TestDetail test={selectedTest}
                   scenarioStates={scenarioStates}
                   onRunScenario={handleRunScenario} />
      </div>
    </div>
  );
}
```

Komponent `TestList` wyświetla listę testów po lewej stronie:

```tsx
function TestList({ tests, selectedId, onSelect }) {
  return (
    <aside className="test-list">
      {tests.map((test) => (
        <button
          key={test.id}
          className={`test-card ${selectedId === test.id ? 'is-active' : ''}`}
          onClick={() => onSelect(test.id)}
        >
          <h3>{test.name}</h3>
          <p>{test.shortDescription}</p>
          <StatusBadge status={getTestStatus(test)} />
        </button>
      ))}
    </aside>
  );
}
```

Komponent `ScenarioCard` renderuje pojedynczy scenariusz z formularzem nadpisań:

```tsx
function ScenarioCard({ scenario, test, onRun }) {
  const [threads, setThreads] = useState("");
  const [useCuda, setUseCuda] = useState(false);
  const [extraArgs, setExtraArgs] = useState("");

  return (
    <article className="scenario-card">
      <header>
        <h4>{scenario.label}</h4>
        <button onClick={() => onRun(scenario)}>Run scenario</button>
      </header>

      <pre>{scenario.command}</pre>

      <div className="overrides">
        <label>
          Threads:
          <input type="number" value={threads}
                onChange={(e) => setThreads(e.target.value)} />
        </label>

        <label>
          <input type="checkbox" checked={useCuda}
                onChange={(e) => setUseCuda(e.target.checked)} />
          Use CUDA
        </label>

        <label>
          Extra args:
          <input type="text" value={extraArgs}
                onChange={(e) => setExtraArgs(e.target.value)} />
        </label>
      </div>

      {lastResult && <ResultDisplay result={lastResult} />}
    </article>
  );
}
```

### 3.4.2. Stan aplikacji

Aplikacja zarządza stanem przy użyciu React hooks. Hook `useState` deklaruje zmienne stanu:

```tsx
// Dane z backendu
const [suite, setSuite] = useState<SuiteInfo | null>(null);
const [tests, setTests] = useState<TestCase[]>([]);

// Stan scenariuszy (wyniki i status running)
const [scenarioStates, setScenarioStates] = useState<Record<string, ScenarioState>>({});
```

Struktura `ScenarioState` przechowuje stan dla każdego scenariusza:

```typescript
interface ScenarioState {
  running: boolean;
  result?: RunResult;
}
```

Kluczem w mapie jest konkatenacja `testId:scenarioId`, co pozwala na unikalną identyfikację każdego scenariusza.

Funkcja `handleRunScenario` obsługuje uruchomienie testu:

```tsx
const handleRunScenario = async (test: TestCase, scenario: TestScenario) => {
  const key = `${test.id}:${scenario.id}`;
  const override = scenarioOverrides[key] || getDefaultOverride(scenario);

  // Walidacja
  const error = validateOverride(override);
  if (error) {
    alert(error);
    return;
  }

  // Ustaw stan "running"
  setScenarioStates(prev => ({
    ...prev,
    [key]: { ...prev[key], running: true }
  }));

  // Przygotuj payload
  const payload: any = {};
  if (override.threads.trim()) {
    payload.threads = Number(override.threads);
  }
  if (!scenario.perfTest) {
    payload.useCuda = override.useCuda;
  }
  if (override.extraArgs.trim()) {
    payload.extraArgs = override.extraArgs.trim();
  }

  // Wyślij request
  try {
    const response = await fetch(
      `${API_BASE_URL}/tests/${test.id}/scenarios/${scenario.id}/run`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result: RunResult = await response.json();

    // Zapisz wynik
    setScenarioStates(prev => ({
      ...prev,
      [key]: { running: false, result }
    }));
  } catch (error) {
    // Obsłuż błąd
    setScenarioStates(prev => ({
      ...prev,
      [key]: {
        running: false,
        result: {
          testId: test.id,
          scenarioId: scenario.id,
          command: scenario.command,
          exitCode: -1,
          durationMs: 0,
          stdout: "",
          stderr: "",
          success: false,
          errorMessage: error.message
        }
      }
    }));
  }
};
```

Funkcja używa async/await do asynchronicznej komunikacji z backendem. Aktualizacje stanu są wykonywane poprzez funkcję callback przyjmującą poprzedni stan i zwracającą nowy, co zapewnia poprawność w przypadku wielu jednoczesnych aktualizacji.

---

# 4. INSTRUKCJA OBSŁUGI APLIKACJI

## 4.1. Kompilacja programu C++

Program analityczny wymaga skompilowania przed pierwszym użyciem. Proces budowania wykorzystuje narzędzie Make i jest zautomatyzowany poprzez Makefile.

Podstawowa kompilacja w trybie release z domyślnymi ustawieniami:

```bash
make clean
make
```

Polecenie `make clean` usuwa poprzednio skompilowane pliki obiektowe, co zapewnia, że wszystkie zmiany w kodzie źródłowym lub konfiguracji zostaną uwzględnione. Polecenie `make` buduje program z następującymi domyślnymi ustawieniami: OpenMP włączone, MPI włączone, CUDA wyłączone.

Kompilacja z wsparciem CUDA wymaga karty graficznej NVIDIA oraz zainstalowanego CUDA Toolkit:

```bash
make clean
make USE_CUDA=1
```

Flaga `USE_CUDA=1` instruuje system budowania do włączenia kodu kerneli GPU oraz linkowania biblioteki CUDA runtime. Po udanej kompilacji program będzie wspierał parametr `--use-cuda`.

Kompilacja bez MPI jest przydatna na systemach gdzie OpenMPI nie jest zainstalowane:

```bash
make clean
make USE_MPI=0 CXX=g++
```

Parametr `CXX=g++` nadpisuje domyślny kompilator `mpicxx` standardowym g++, ponieważ wrapper MPI nie jest potrzebny. Program w tym trybie będzie działał jako pojedynczy proces.

Kompilacja w trybie debug z symbolami debugowania i bez optymalizacji:

```bash
make debug
```

Tryb debug wyłącza optymalizacje kompilatora (`-O0`) i dodaje symbole debugowania (`-g3`), co pozwala na używanie debuggera GDB do śledzenia wykonania programu oraz inspekcji zmiennych.

Weryfikacja konfiguracji kompilacji przed budowaniem:

```bash
make info
```

Polecenie wyświetla aktualną konfigurację Makefile, łącznie z używanymi kompilatorami, flagami oraz listą plików źródłowych, co pozwala zweryfikować poprawność ustawień przed czasochłonną kompilacją.

Po udanej kompilacji plik wykonywalny znajduje się w lokalizacji `./build/bin/prir`. Weryfikacja poprawności budowania:

```bash
./build/bin/prir --help
```

Polecenie wyświetla pomoc CLI, potwierdzając że binarka została poprawnie zbudowana i jest funkcjonalna.

## 4.2. Uruchomienie backendu

Backend FastAPI wymaga środowiska Python z zainstalowanymi zależnościami. Proces uruchomienia składa się z utworzenia wirtualnego środowiska, instalacji pakietów oraz startu serwera.

Przejście do katalogu backendu:

```bash
cd backend
```

Utworzenie wirtualnego środowiska Python (wykonywane raz):

```bash
python3 -m venv .venv
```

Komenda tworzy odizolowane środowisko Python w katalogu `.venv`, co pozwala na instalację pakietów bez modyfikowania systemowej instalacji Python.

Aktywacja wirtualnego środowiska zależy od systemu operacyjnego:

```bash
# Linux/macOS:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Po aktywacji prompt terminala jest poprzedzony `(.venv)`, co potwierdza, że środowisko jest aktywne.

Instalacja zależności z pliku requirements.txt:

```bash
pip install -r requirements.txt
```

Plik requirements.txt zawiera wszystkie niezbędne pakiety: fastapi, uvicorn, pydantic. Instalacja pobiera pakiety z Python Package Index (PyPI) i umieszcza je w wirtualnym środowisku.

Uruchomienie serwera deweloperskiego z auto-reload:

```bash
uvicorn main:app --reload
```

Serwer startuje na domyślnym adresie `http://127.0.0.1:8000`. Flaga `--reload` powoduje automatyczne restartowanie serwera przy wykryciu zmian w plikach źródłowych, co jest przydatne podczas developmentu.

Weryfikacja działania backendu poprzez otwarcie w przeglądarce:

```
http://localhost:8000/docs
```

FastAPI automatycznie generuje interaktywną dokumentację API w formacie Swagger UI, która pozwala na testowanie endpointów bez użycia frontendu.

Test podstawowego endpointu z linii poleceń:

```bash
curl http://localhost:8000/api/tests
```

Polecenie powinno zwrócić JSON zawierający listę dostępnych testów, co potwierdza poprawne działanie backendu oraz odczyt pliku `test_cases.json`.

## 4.3. Uruchomienie frontendu

Frontend React wymaga środowiska Node.js oraz menedżera pakietów npm. Proces uruchomienia obejmuje instalację zależności JavaScript oraz start serwera deweloperskiego.

Przejście do katalogu frontendu (w nowym terminalu):

```bash
cd frontend
```

Instalacja zależności npm (wykonywane raz):

```bash
npm install
```

Polecenie odczytuje plik `package.json`, pobiera wszystkie zależności oraz ich zależności rekurencyjne, i umieszcza je w katalogu `node_modules`. Proces może potrwać kilka minut przy pierwszym uruchomieniu.

Uruchomienie serwera deweloperskiego Vite:

```bash
npm run dev
```

Vite startuje serwer na domyślnym porcie 5173 i wyświetla komunikat:

```
  VITE v5.0.0  ready in 523 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

Otwarcie aplikacji w przeglądarce:

```
http://localhost:5173
```

Przeglądarka powinna wyświetlić dashboard z listą testów po lewej stronie. Aplikacja automatycznie łączy się z backendem na `http://localhost:8000` i pobiera konfigurację testów.

W przypadku gdy backend działa na innym adresie, należy utworzyć plik `.env.local` w katalogu `frontend/`:

```
VITE_API_BASE_URL=http://192.168.1.100:8000/api
```

Zmienna środowiskowa `VITE_API_BASE_URL` nadpisuje domyślny URL backendu. Po zmianie pliku `.env.local` należy zrestartować serwer deweloperski.

## 4.4. Użycie web dashboardu

Interfejs webowy pozwala na wygodne uruchamianie testów wydajnościowych oraz wizualizację wyników bez konieczności używania wiersza poleceń.

### Ekran główny aplikacji

![Screenshot - Strona główna](screeny_z_apki/screen_z_apki_1.png)

Po załadowaniu aplikacji prezentowana jest lista dostępnych testów w panelu bocznym. Każdy test zawiera nazwę, krótki opis oraz wskaźnik statusu (pending, running, passed). Po prawej stronie wyświetlane są szczegóły wybranego testu wraz z listą scenariuszy.

### Uruchamianie scenariusza testowego

Każdy scenariusz zawiera następujące elementy:
- Etykietę opisującą cel testu
- Komendę CLI która będzie wykonana
- Formularz nadpisań parametrów
- Przycisk uruchomienia

Formularz nadpisań umożliwia modyfikację:
- Liczby wątków OpenMP (pole tekstowe)
- Włączenia/wyłączenia CUDA (checkbox)
- Dodatkowych argumentów CLI (pole tekstowe)

Po kliknięciu przycisku "Run scenario" aplikacja wysyła żądanie do backendu i wyświetla wskaźnik ładowania. Po zakończeniu testu wyniki są prezentowane bezpośrednio pod formularzem.

### Wyniki testów

![Screenshot - Wynik testu wydajnościowego](screeny_z_apki/wynik_testu_2.png)

Wyniki zawierają:
- Kod wyjścia programu (0 = sukces)
- Czas wykonania w sekundach
- Timestamp uruchomienia
- Rozwijaną sekcję ze standardowym wyjściem
- Rozwijaną sekcję z błędami (jeśli wystąpiły)

Dla testów typu perf-test (porównanie CPU vs CUDA) wyświetlana jest dodatkowa tabela porównawcza:

![Screenshot - Porównanie CPU vs GPU](screeny_z_apki/wynik_testu_3.png)

Tabela prezentuje:
- Etykietę przebiegu (CPU baseline / CUDA pass)
- Czas wykonania w milisekundach
- Szczegóły konfiguracji (liczba wątków / GPU histogram)
- Zliczone frazy z wartościami

### Tabela wydajności

![Screenshot - Tabela wydajności](screeny_z_apki/wynik_testu_4.png)

Każdy test może zawierać predefiniowaną tabelę wydajności pokazującą referencyjne czasy wykonania dla różnych konfiguracji. Tabela prezentuje:
- Liczbę wątków
- Czas CPU w milisekundach
- Czas GPU w milisekundach
- Obliczony stosunek wydajności

Poniżej tabeli wyświetlane jest automatyczne podsumowanie:
- Najszybszy run CPU (wątki i czas)
- Najlepszy delta GPU vs CPU (procent przyspieszenia)

### Formularz własnych pomiarów

![Screenshot - Formularz pomiarów](screeny_z_apki/wynik_testu_5.png)

Dashboard umożliwia zapisywanie własnych pomiarów wydajnościowych poprzez dedykowany formularz:
- Wybór trybu (CPU / GPU)
- Liczba wątków
- Zmierzony czas w sekundach
- Opcjonalne notatki

Zapisane pomiary są wyświetlane w tabeli poniżej formularza i mogą służyć do porównania wyników z różnych konfiguracji sprzętowych lub wersji programu.

### Logi backendu

![Screenshot - Logi backendu](screeny_z_apki/logi_z_backendu.png)

Terminal backendu wyświetla szczegółowe logi operacji:
- Żądania HTTP z metodą, ścieżką i kodem odpowiedzi
- Czasy przetwarzania żądań
- Informacje o uruchamianych testach
- Błędy kompilacji lub wykonania

Logi są przydatne do diagnozowania problemów oraz monitorowania aktywności systemu.

## 4.5. Użycie CLI (wiersz poleceń)

Program analityczny może być uruchamiany bezpośrednio z wiersza poleceń, co pozwala na integrację z skryptami shell oraz automatyzację procesów.

### 4.5.1. Podstawowe zliczanie

Najprostsze użycie polega na zliczeniu wystąpień jednej lub więcej fraz:

```bash
./build/bin/prir --file access.log --phrase GET --phrase POST
```

Program przetwarza plik `access.log`, zlicza wystąpienia słów "GET" i "POST", oraz wypisuje wyniki w formacie CSV:

```
phrase,count
GET,10283350
POST,2156789

interval,count
2019-01-15 08:00,45234
2019-01-15 09:00,52341
...
```

Domyślnie program generuje także statystyki czasowe z granularnością godzinową.

### 4.5.2. Filtrowanie po czasie

Ograniczenie analizy do określonego okna czasowego:

```bash
./build/bin/prir --file access.log --phrase ERROR \
  --from 2019-01-15T08:00:00 --to 2019-01-15T18:00:00
```

Program przetworzy tylko linie zawierające timestamp w przedziale od 8:00 do 18:00 dnia 15 stycznia 2019. Linie spoza tego zakresu są pomijane zarówno przy zliczaniu fraz jak i w statystykach czasowych.

### 4.5.3. Statystyki co minutę

Zmiana granularności statystyk czasowych:

```bash
./build/bin/prir --file access.log --phrase GET --stats minute
```

Parametr `--stats minute` powoduje zaokrąglanie timestampów do początku minuty zamiast godziny, co skutkuje bardziej szczegółowymi statystykami:

```
interval,count
2019-01-15 08:00,756
2019-01-15 08:01,812
2019-01-15 08:02,789
...
```

### 4.5.4. Uruchomienie z MPI

Wykorzystanie wielu procesów do przyspieszenia analizy dużych plików:

```bash
mpirun -np 8 ./build/bin/prir --file access.log --phrase GET --threads 4
```

Komenda uruchamia 8 procesów MPI, z których każdy używa 4 wątków OpenMP, co daje łącznie 32 wątki pracujące równolegle. Parametr `-np` (number of processes) kontroluje liczbę procesów MPI.

### 4.5.5. Test wydajności CPU vs GPU

Automatyczne porównanie czasów wykonania na CPU i GPU:

```bash
./build/bin/prir --file access.log --phrase GET --perf-test
```

Tryb `--perf-test` uruchamia analizę dwukrotnie: raz z obliczeniami CPU (`--cpu-only`), i drugi raz z obliczeniami GPU (`--use-cuda`). Program wypisuje czasy dla obu przebiegów oraz oblicza przyspieszenie:

```
[perf-test] CPU baseline took 9470 ms (8 CPU threads)
phrase,count
GET,10283350

[perf-test] CUDA pass took 10080 ms (GPU histogram)
phrase,count
GET,10283350

Speedup: 0.94x (GPU slower)
```

---

# 5. TESTY I ANALIZA WYDAJNOŚCI

## 5.1. Środowisko testowe

Testy wydajnościowe przeprowadzono w kontrolowanym środowisku sprzętowo-programowym w celu zapewnienia reprodukowalności wyników oraz wiarygodności porównań.

### Specyfikacja sprzętowa

[DO UZUPEŁNIENIA - Podaj szczegóły konfiguracji sprzętowej]

- **Procesor**: [model, liczba rdzeni fizycznych, liczba wątków logicznych, częstotliwość]
- **RAM**: [ilość, typ, częstotliwość]
- **GPU**: [model karty NVIDIA, architektura compute capability, liczba rdzeni CUDA] lub N/A
- **Dysk**: [SSD/HDD, model, interfejs, szybkość odczytu sekwencyjnego]
- **System operacyjny**: [Windows/Linux, wersja]

### Dane testowe

Zbiór danych wykorzystany do testów pochodzi z publicznie dostępnego repozytorium Harvard Dataverse i zawiera rzeczywiste logi z serwisu e-commerce zanbil.ir.

- **Plik**: `access.log`
- **Rozmiar**: 3.3 GB (3,502,440,823 bajtów)
- **Liczba linii**: [DO UZUPEŁNIENIA - wc -l access.log]
- **Format**: Apache/Nginx Combined Log Format
- **Źródło**: Zaker, Farzin, 2019, "Online Shopping Store - Web Server Logs", DOI: 10.7910/DVN/3QBYB5

Struktura pojedynczego rekordu:
```
192.168.1.100 - - [15/Jan/2019:12:34:56 +0000] "GET /api/products HTTP/1.1" 200 1234
```

Zbiór zawiera zapisy żądań HTTP z następującymi charakterystykami:
- Adres IP klienta
- Timestamp w formacie CLF z dokładnością do sekundy
- Metoda HTTP (GET, POST, PUT, DELETE)
- Żądany zasób (URL)
- Kod odpowiedzi HTTP (2xx, 3xx, 4xx, 5xx)
- Rozmiar odpowiedzi w bajtach

## 5.2. Metodologia testowania

### 5.2.1. Scenariusze testowe

Zaprojektowano cztery podstawowe scenariusze testowe reprezentujące typowe przypadki użycia programu:

**Scenariusz 1: Zliczanie pojedynczej frazy**
- Fraza: "GET"
- Cel: Pomiar bazowej wydajności przetwarzania tekstu
- Charakterystyka: Prosta operacja zliczania bez złożonych filtrów

**Scenariusz 2: Zliczanie wielu fraz**
- Frazy: "GET", "POST", "PUT", "DELETE"
- Cel: Ocena skalowania przy zwiększeniu liczby wyszukiwanych fraz
- Charakterystyka: Więcej operacji dopasowywania na linię

**Scenariusz 3: Filtrowanie po kodzie odpowiedzi**
- Frazy: "200", "404", "500"
- Filtry: `--level` z identyfikacją kodów HTTP
- Cel: Pomiar narzutu filtrowania
- Charakterystyka: Wymaga parsowania poziomu logowania

**Scenariusz 4: Statystyki czasowe**
- Fraza: "GET"
- Parametry: `--stats minute`
- Cel: Ocena wydajności ekstrahowania timestampów oraz agregacji czasowej
- Charakterystyka: Parsowanie daty, budowanie histogramów czasowych

### 5.2.2. Zmienne testowe

Testy przeprowadzono dla następujących konfiguracji parametrów:

**Liczba wątków OpenMP**: 1, 2, 4, 8, 16
- Cel: Ocena skalowania równoległości shared-memory
- Kontrola: Parametr `--threads N`

**Liczba procesów MPI**: 1, 2, 4, 8
- Cel: Ocena skalowania równoległości distributed-memory
- Kontrola: Parametr `mpirun -np N`

**Tryb wykonania**: CPU-only, GPU (CUDA)
- Cel: Porównanie wydajności obliczeń CPU vs GPU
- Kontrola: Parametry `--cpu-only` oraz `--use-cuda`

Każdy test przeprowadzono trzykrotnie i obliczono średnią arytmetyczną czasu wykonania, co redukuje wpływ zmienności wynikającej z procesów systemowych oraz cache effects.

## 5.3. Skrypt automatyzujący testy

W celu zapewnienia reprodukowalności oraz automatyzacji procesu testowania opracowano zestaw skryptów wykonujących wszystkie konfiguracje testowe oraz generujących wykresy porównawcze.

### 5.3.1. Skrypt bash do uruchamiania testów

Skrypt `run_benchmarks.sh` automatyzuje proces uruchamiania testów dla różnych kombinacji parametrów i zapisuje wyniki do pliku CSV.

Struktura skryptu obejmuje:

1. **Weryfikację środowiska**
   - Sprawdzenie istnienia binarki programu
   - Sprawdzenie dostępności pliku danych
   - Detekcja MPI (sprawdzenie `mpirun`)
   - Detekcja CUDA (sprawdzenie flagi w output `--help`)

2. **Serie testowe**

   **TEST 1: Skalowanie OpenMP (CPU)**
   - Parametry: wątki od 1 do 16, `--cpu-only`, `--no-stats`
   - Cel: Pomiar czystej wydajności OpenMP bez narzutu I/O statystyk

   **TEST 2: GPU (CUDA) dla różnych wątków**
   - Parametry: wątki od 1 do 16, `--use-cuda`, `--no-stats`
   - Cel: Pomiar wydajności GPU w funkcji liczby wątków pre-processing

   **TEST 3: Skalowanie MPI**
   - Parametry: procesy od 1 do 8, 4 wątki OpenMP na proces
   - Cel: Ocena skalowania distributed parallelism

   **TEST 4: Kombinacja MPI + OpenMP**
   - Parametry: kombinacje 2×2, 2×4, 4×2, 4×4 (procesy × wątki)
   - Cel: Znajdowanie optymalnej konfiguracji hybrydowej

   **TEST 5: Wiele fraz (CPU vs GPU)**
   - Parametry: 4 frazy, 8 wątków, CPU i GPU
   - Cel: Ocena wpływu liczby fraz na relatywną wydajność GPU

3. **Zapisywanie wyników**

   Każdy test zapisuje wiersz w pliku `benchmark_results.csv`:
   ```
   threads,mode,duration_ms,mpi_procs,test_name
   8,cpu,9470,1,"OpenMP CPU threads=8"
   ```

### 5.3.2. Skrypt Python do generowania wykresów

Skrypt `generate_plots.py` wczytuje plik CSV z wynikami i generuje zestaw wykresów porównawczych przy użyciu bibliotek pandas i matplotlib.

Generowane wykresy:

1. **plot_1_cpu_threads.png**: Wykres liniowy czasu wykonania vs liczba wątków dla CPU
2. **plot_2_cpu_vs_gpu.png**: Wykres słupkowy porównujący czasy CPU i GPU
3. **plot_3_speedup_cpu.png**: Wykres przyspieszenia (speedup) OpenMP z linią idealną
4. **plot_4_mpi_scaling.png**: Wykres skalowania MPI z linią idealną
5. **plot_5_efficiency.png**: Wykres efektywności równoległości w procentach
6. **plot_6_hybrid_heatmap.png**: Mapa ciepła dla kombinacji MPI × OpenMP
7. **plot_7_throughput.png**: Wykres przepustowości w GB/s

Dodatkowo skrypt generuje plik tekstowy `summary_stats.txt` zawierający:
- Najszybszy test (konfiguracja i czas)
- Najszybszy run CPU i GPU osobno
- Porównanie CPU vs GPU dla każdej liczby wątków
- Obliczone przyspieszenia i efektywności

Przykład użycia skryptów:

```bash
# 1. Uruchom testy (może potrwać 5-15 minut)
chmod +x run_benchmarks.sh
./run_benchmarks.sh

# 2. Wygeneruj wykresy
python3 generate_plots.py

# 3. Wyniki znajdują się w:
#    - benchmark_results.csv (surowe dane)
#    - plots/ (wykresy PNG + statystyki TXT)
```

## 5.4. Wyniki testów

> **[DO UZUPEŁNIENIA]** Wyniki zostaną wygenerowane po uruchomieniu skryptów `run_benchmarks.sh` i `generate_plots.py`

### 5.4.1. Porównanie CPU vs GPU

[Miejsce na tabelę wyników i wykres słupkowy]

### 5.4.2. Skalowanie OpenMP (liczba wątków)

[Miejsce na wykres liniowy czasu oraz wykres słupkowy speedup]

### 5.4.3. Skalowanie MPI (liczba procesów)

[Miejsce na tabelę wyników i wykres z linią idealną]

### 5.4.4. Kombinacja MPI + OpenMP

[Miejsce na mapę ciepła i analizę optymalnej konfiguracji]

### 5.4.5. Przepustowość (GB/s)

[Miejsce na obliczenia i wykres przepustowości]

### 5.4.6. Top-N fraz

[Miejsce na wykres słupkowy najczęściej występujących słów]

## 5.5. Screenshoty z testów

Dokumentacja wizualna procesu testowania:

- **Screenshot**: Terminal z uruchomieniem `--perf-test` pokazujący czasy CPU vs CUDA
- **Screenshot**: Wyjście programu z wygenerowanymi statystykami czasowymi
- **Screenshot**: Uruchomienie z MPI `mpirun -np 8` oraz output agregacji
- **Screenshot**: Dashboard pokazujący wyniki testów wydajnościowych dla różnych konfiguracji
- **Screenshot**: Terminal backendu z logami przetwarzania żądań
- **Screenshot**: Wyniki z różnymi liczbami wątków w tabeli wydajności

---

# 6. ANALIZA WYNIKÓW I WNIOSKI

> **[DO UZUPEŁNIENIA]** Analiza zostanie przeprowadzona po uzyskaniu konkretnych wyników z testów

## 6.1. Analiza wydajności OpenMP

### 6.1.1. Przyspieszenie przy zwiększaniu liczby wątków

[Analiza krzywej skalowania, identyfikacja punktu nasycenia]

### 6.1.2. Efektywność równoległości

[Obliczenia efektywności dla różnych liczb wątków, analiza spadku]

## 6.2. Analiza wydajności MPI

### 6.2.1. Skalowanie między procesami

[Porównanie MPI vs OpenMP, wyjaśnienie różnic]

### 6.2.2. Ograniczenia MPI

[Overhead komunikacji, wymagania infrastrukturalne]

## 6.3. Analiza wydajności CUDA

### 6.3.1. Kiedy GPU jest szybsze?

[Identyfikacja scenariuszy GPU-favorable i CPU-favorable]

### 6.3.2. Overhead GPU

[Pomiar czasu transferów, analiza break-even point]

## 6.4. Wąskie gardła

### 6.4.1. I/O dyskowe

[Identyfikacja I/O jako bottleneck, propozycje rozwiązań]

### 6.4.2. Memory bandwidth

[Analiza wpływu przepustowości pamięci]

### 6.4.3. Synchronizacja

[Overhead sekcji krytycznych OpenMP]

## 6.5. Porównanie z innymi narzędziami

### 6.5.1. vs grep/awk

[Porównanie czasów z klasycznymi narzędziami Unix]

### 6.5.2. vs GNU Parallel

[Analiza różnic w podejściu i wydajności]

## 6.6. Wnioski końcowe

### 6.6.1. Osiągnięte cele

[Podsumowanie zrealizowanych funkcjonalności]

### 6.6.2. Mocne strony projektu

[Lista zalet implementacji]

### 6.6.3. Słabe strony i możliwe usprawnienia

[Identyfikacja ograniczeń, propozycje ulepszeń]

### 6.6.4. Możliwe rozszerzenia

[Potencjalne kierunki rozwoju projektu]

---

# 7. PODSUMOWANIE

> **[DO UZUPEŁNIENIA]** Synteza najważniejszych wniosków z projektu

## 7.1. Co udało się osiągnąć

[Pełna lista zrealizowanych celów]

## 7.2. Wyniki ilościowe

[Kluczowe metryki wydajności]

## 7.3. Wnioski ogólne

[Ocena technologii, rekomendacje]

---

# ZAŁĄCZNIKI

## A. Kod źródłowy

Kluczowe fragmenty kodu zostały przedstawione i omówione w Rozdziale 3.

## B. Screenshoty

Wszystkie screenshoty z aplikacji oraz testów zostały dołączone w Rozdziałach 4 i 5.

## C. Wyniki surowe (CSV)

Pełne wyniki testów znajdują się w pliku `benchmark_results.csv` dołączonym do projektu.

## D. Skrypty

- `run_benchmarks.sh` - automatyzacja testów
- `generate_plots.py` - generowanie wykresów

## E. Instrukcja kompilacji

Szczegółowa instrukcja kompilacji dla różnych platform została przedstawiona w Rozdziale 4.1.

---

## Bibliografia

1. OpenMP Architecture Review Board. *OpenMP Application Programming Interface*, Version 5.2, November 2021. https://www.openmp.org/specifications/

2. Message Passing Interface Forum. *MPI: A Message-Passing Interface Standard*, Version 4.0, June 2021. https://www.mpi-forum.org/docs/

3. NVIDIA Corporation. *CUDA C++ Programming Guide*, Version 12.0, 2023. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

4. Tiangolo, Sebastián. *FastAPI Framework*, 2018-2024. https://fastapi.tiangolo.com/

5. Facebook Inc. *React - A JavaScript library for building user interfaces*, 2013-2024. https://react.dev/

6. Zaker, Farzin. "Online Shopping Store - Web Server Logs", Harvard Dataverse, V1, 2019. https://doi.org/10.7910/DVN/3QBYB5

7. Chapman, Barbara, Gabriele Jost, and Ruud van der Pas. *Using OpenMP: Portable Shared Memory Parallel Programming*. MIT Press, 2008.

8. Gropp, William, Ewing Lusk, and Anthony Skjellum. *Using MPI: Portable Parallel Programming with the Message-Passing Interface*. MIT Press, 2014.

9. Sanders, Jason, and Edward Kandrot. *CUDA by Example: An Introduction to General-Purpose GPU Programming*. Addison-Wesley Professional, 2010.

10. Kaggle Dataset Repository. *Web Server Access Logs*, 2019. https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs
