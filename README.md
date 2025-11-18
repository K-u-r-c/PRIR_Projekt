# PRIR Projekt - Rownolegly analizator logow

Narzedzie do badania bardzo duzych plikow tekstowych (dzienniki systemowe,
raporty z serwerow, dane IoT). Program potrafi:

- zliczac wystapienia zdefiniowanych slow lub fraz (np. `ERROR`, `WARNING`),
- odfiltrowac wiersze spelniajace zadane kryteria (poziom, okno czasowe),
- przygotowac statystyki czasowe (liczba zdarzen na godzine/minute),
- wykonac powyzsze zadania rownolegle z wykorzystaniem OpenMP + MPI + CUDA.

## Wymagania

- kompilator C++20
- OpenMP (np. g++/clang++)
- MPI (np. OpenMPI, MPICH)
- opcjonalnie CUDA 11+ (nvcc + libcudart)

## Budowanie

```
make            # Release (OpenMP + MPI + CUDA jesli dostepne)
make debug      # Debug
```

Najwazniejsze przelaczniki `Makefile`:

| Zmienna         | Domyslnie | Opis                                      |
| --------------- | --------- | ----------------------------------------- |
| `USE_MPI=0/1`   | `1`       | wlacza/wylacza kompilacje z MPI           |
| `USE_OPENMP=0/1`| `1`       | steruje obecnoscia flag OpenMP            |
| `USE_CUDA=0/1`  | `1`       | wlacza modul CUDA                         |
| `CXX=...`       | `mpicxx`  | inny kompilator gdy MPI nieuzywane        |
| `NVCC=...`      | `nvcc`    | sciezka do kompilatora CUDA               |

Przyklady:

```
make USE_CUDA=0                # tylko CPU (OpenMP + MPI)
make USE_MPI=0 CXX=g++         # samodzielna binarka
make USE_OPENMP=0              # bez OpenMP
```

Wynik: `build/bin/prir`.

## Uruchamianie

```
./build/bin/prir --file logs.txt --phrase ERROR --phrase WARNING

mpirun -np 8 ./build/bin/prir --file huge.log \
  --phrase ERROR --phrase "disk full" --level ERROR --stats minute --use-cuda
```

## Opcje CLI

```
./prir --file PATH --phrase TEXT [opcje]

Wymagane:
  --file PATH            analizowany plik (tekstowy)
  --phrase TEXT          fraza/slowo do zliczenia (opcja powtarzalna)

Filtrowanie:
  --case-sensitive       rozroznianie wielkosci liter
  --level NAME           dopuszczalny poziom (mozna powtarzac / lista csv)
  --from YYYY-MM-DDTHH:MM:SS   poczatek okna czasowego
  --to   YYYY-MM-DDTHH:MM:SS   koniec okna czasowego
  --count-filtered       licz frazy tylko na liniach spelniajacych filtry

Statystyki i wyjscie:
  --stats hour|minute    wielkosc przedzialow czasowych (domyslnie hour)
  --no-stats             wylacza statystyki czasowe
  --emit                 wypisz dopasowane linie na stdout
  --emit-file PATH       dopisz dopasowane linie do pliku

Wydajnosc:
  --threads N            wymusza liczbe watkow OpenMP
  --use-cuda             wlacza histogram GPU

Inne:
  --help                 krotki opis
```

## Format wyjscia

- Liczniki fraz: CSV `phrase,count` (stdout)
- Statystyki czasowe: pusty wiersz, CSV `interval,count`
- Dopasowania: stdout i/lub wskazany plik

## Architektura

1. **MPI** - dzieli plik na porcje, kazdy proces analizuje swoj fragment i
   odsyla zredukowane wyniki do rangi 0.
2. **OpenMP** - w ramach procesu dzieli linie na watki; kazdy buduje lokalne
   slowniki fraz, bucketow czasowych i liste dopasowan, nastepnie nastepuje
   redukcja.
3. **CUDA** - opcjonalnie przyspiesza histogram fraz poprzez kernel GPU,
   operujacy na indeksach dopasowan.

Kazdy z komponentow mozna wylaczyc (`USE_MPI`, `USE_OPENMP`, `USE_CUDA`).

## Przyklady

```
./prir --file sys.log --phrase ERROR --phrase WARNING \
       --level ERROR --from 2025-03-04T00:00:00 --to 2025-03-05T00:00:00 \
       --emit-file errors.txt

mpirun -np 4 ./prir --file service.log --phrase "user login" \
       --level INFO --count-filtered --stats minute

mpirun -np 16 ./prir --file telemetry.log --phrase ALERT --use-cuda --threads 8
```

## Daty/czasy

Program rozpoznaje fragmenty `YYYY-MM-DD HH:MM:SS` lub `YYYY-MM-DDTHH:MM:SS`.
Linie bez poprawnego czasu sa pomijane podczas filtrow czasowych i w statystykach.

## Licencja

MIT.