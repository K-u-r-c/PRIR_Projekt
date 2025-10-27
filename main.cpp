
#include <bits/stdc++.h>
#include <thread>

using namespace std;

// --- Prosty parser Apache/Nginx (combined/common) ---
// Przyjmuje linię i wyciąga: ip, minuteKey "YYYY-MM-DD HH:MM", status (int).
struct Parsed {
  string ip;
  string minuteKey;
  int status = -1;
  bool ok = false;
};

static inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

static inline Parsed parse_line(const string_view &s) {
  Parsed out;
  // IP = token przed pierwszą spacją
  size_t p = s.find(' ');
  if (p == string_view::npos)
    return out;
  out.ip = string(s.substr(0, p));

  // Szukaj daty: występuje po '[' i kończy się na ']'
  size_t lb = s.find('[', p);
  if (lb == string_view::npos)
    return out;
  size_t rb = s.find(']', lb + 1);
  if (rb == string_view::npos)
    return out;
  // Przykład: 10/Oct/2000:13:55:36 -0700
  auto ts = s.substr(lb + 1, rb - lb - 1);

  // Zrób klucz minutowy: YYYY-MM-DD HH:MM
  // Parsujemy dd/Mon/yyyy:HH:MM
  // Mapowanie miesiąca:
  static unordered_map<string, int> mon{{"Jan", 1},  {"Feb", 2},  {"Mar", 3},
                                        {"Apr", 4},  {"May", 5},  {"Jun", 6},
                                        {"Jul", 7},  {"Aug", 8},  {"Sep", 9},
                                        {"Oct", 10}, {"Nov", 11}, {"Dec", 12}};
  // dd
  if (ts.size() < 17)
    return out;
  int dd = stoi(string(ts.substr(0, 2)));
  string monStr = string(ts.substr(3, 3));
  int mm = mon.count(monStr) ? mon[monStr] : 0;
  int yyyy = stoi(string(ts.substr(7, 4)));
  string HH = string(ts.substr(12, 2));
  string MM = string(ts.substr(15, 2));
  auto two = [](int x) {
    string s = to_string(x);
    return (s.size() == 1 ? "0" + s : s);
  };
  out.minuteKey =
      to_string(yyyy) + "-" + two(mm) + "-" + two(dd) + " " + HH + ":" + MM;

  // Status: po cudzysłowie z metodą/ścieżką
  // ... "GET /path HTTP/1.1" 200 123
  size_t q2 = s.rfind('"');
  if (q2 == string_view::npos)
    return out;
  // znajdź liczbę po spacji po q2
  size_t pos = s.find_first_not_of(' ', q2 + 1);
  if (pos == string_view::npos)
    return out;
  // status to pierwsza liczba
  size_t endpos = pos;
  while (endpos < s.size() && is_digit(s[endpos]))
    endpos++;
  if (endpos == pos)
    return out;
  out.status = stoi(string(s.substr(pos, endpos - pos)));

  out.ok = true;
  return out;
}

// --- Narzędzia: bezpieczne wczytanie pliku i podział na kawałki ---
struct Chunk {
  size_t begin;
  size_t end;
}; // [begin,end)
static vector<Chunk> chunk_file(const string &path, int threads) {
  ifstream f(path, ios::binary);
  f.seekg(0, ios::end);
  size_t size = (size_t)f.tellg();
  vector<Chunk> chunks;
  if (size == 0 || threads <= 1) {
    chunks.push_back({0, size});
    return chunks;
  }
  size_t base = size / threads;
  size_t start = 0;
  f.clear();
  for (int i = 0; i < threads; i++) {
    size_t tentativeEnd = (i == threads - 1) ? size : (start + base);
    // Wyrównaj do końca linii
    if (tentativeEnd < size) {
      f.seekg(tentativeEnd);
      string dummy;
      getline(f, dummy); // doskocz do końca bieżącej linii
      tentativeEnd = (size_t)f.tellg();
      if (tentativeEnd == 0)
        tentativeEnd = size; // fallback
    }
    chunks.push_back({start, tentativeEnd});
    start = tentativeEnd;
  }
  return chunks;
}

// Wczytuje kawałek i przetwarza linia po linii.
template <class Fn>
static void process_chunk(const string &path, Chunk c, Fn fnLine) {
  ifstream f(path, ios::binary);
  f.seekg(c.begin);
  string line;
  // Jeśli nie zaczynamy od 0, odrzuć niedomkniętą pierwszą linię
  if (c.begin != 0) {
    getline(f, line);
  }
  while ((size_t)f.tellg() < c.end && std::getline(f, line)) {
    fnLine(line);
  }
}

// --- Operacje równoległe ---
// Każdy wątek buduje lokalne mapy, potem redukujemy.

struct Results {
  unordered_map<int, uint64_t> statusCount;
  unordered_map<string, uint64_t> ipCount;
  unordered_map<string, uint64_t> perMinute;
};

struct Options {
  bool doStatus = false, doTopIps = false, doPerMinute = false,
       doSpikes = false;
  int topK = 10;
  double spikeThresh = 3.0; // T
  int threads =
      thread::hardware_concurrency() ? thread::hardware_concurrency() : 4;
  string path;
};

static Results analyze_parallel(const Options &opt) {
  vector<Chunk> chunks = chunk_file(opt.path, opt.threads);
  mutex gmut;
  Results globalR;

  auto worker = [&](Chunk c) {
    unordered_map<int, uint64_t> sc;
    unordered_map<string, uint64_t> ic;
    unordered_map<string, uint64_t> pm;
    process_chunk(opt.path, c, [&](const string &line) {
      auto p = parse_line(string_view(line));
      if (!p.ok)
        return;
      if (opt.doStatus)
        sc[p.status]++;
      if (opt.doTopIps)
        ic[p.ip]++;
      if (opt.doPerMinute || opt.doSpikes)
        pm[p.minuteKey]++;
    });
    lock_guard lk(gmut);
    for (auto &[k, v] : sc)
      globalR.statusCount[k] += v;
    for (auto &[k, v] : ic)
      globalR.ipCount[k] += v;
    for (auto &[k, v] : pm)
      globalR.perMinute[k] += v;
  };

  vector<thread> pool;
  for (auto &c : chunks)
    pool.emplace_back(worker, c);
  return globalR;
}

static void print_status(const Results &r) {
  cout << "status,count\n";
  vector<pair<int, uint64_t>> v(r.statusCount.begin(), r.statusCount.end());
  sort(v.begin(), v.end());
  for (auto &[k, c] : v)
    cout << k << "," << c << "\n";
}

static void print_top_ips(const Results &r, int K) {
  cout << "ip,count\n";
  vector<pair<string, uint64_t>> v(r.ipCount.begin(), r.ipCount.end());
  partial_sort(v.begin(), v.begin() + min<int>(K, v.size()), v.end(),
               [](auto &a, auto &b) { return a.second > b.second; });
  for (int i = 0; i < (int)min<int>(K, v.size()); ++i)
    cout << v[i].first << "," << v[i].second << "\n";
}

static void print_per_minute(const Results &r) {
  cout << "minute,count\n";
  vector<pair<string, uint64_t>> v(r.perMinute.begin(), r.perMinute.end());
  sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first < b.first; });
  for (auto &[k, c] : v)
    cout << k << "," << c << "\n";
}

static void print_spikes(const Results &r, double T) {
  // Proste wykrywanie spike: licz minutę jako spike, jeśli count > T *
  // mediana(z 15 min okna) Dla prostoty: globalna mediana zamiast okna
  // (wystarczy do projektu).
  vector<uint64_t> counts;
  counts.reserve(r.perMinute.size());
  for (auto &kv : r.perMinute)
    counts.push_back(kv.second);
  if (counts.empty()) {
    cout << "minute,count,is_spike\n";
    return;
  }
  nth_element(counts.begin(), counts.begin() + counts.size() / 2, counts.end());
  double med = counts[counts.size() / 2];

  cout << "minute,count,is_spike\n";
  vector<pair<string, uint64_t>> v(r.perMinute.begin(), r.perMinute.end());
  sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first < b.first; });
  for (auto &[k, c] : v) {
    bool spike = (med > 0.0) && (c > T * med);
    cout << k << "," << c << "," << (spike ? "1" : "0") << "\n";
  }
}

static void usage(const char *argv0) {
  cerr << "Usage: " << argv0 << " --file path [--threads N] "
       << "[--status] [--top-ips K] [--per-minute] [--spikes T]\n";
}

int main(int argc, char **argv) {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  Options opt;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    if (a == "--file" && i + 1 < argc)
      opt.path = argv[++i];
    else if (a == "--threads" && i + 1 < argc)
      opt.threads = stoi(argv[++i]);
    else if (a == "--status")
      opt.doStatus = true;
    else if (a == "--top-ips" && i + 1 < argc) {
      opt.doTopIps = true;
      opt.topK = stoi(argv[++i]);
    } else if (a == "--per-minute")
      opt.doPerMinute = true;
    else if (a == "--spikes" && i + 1 < argc) {
      opt.doSpikes = true;
      opt.spikeThresh = stod(argv[++i]);
    } else {
      usage(argv[0]);
      return 1;
    }
  }
  if (opt.path.empty()) {
    usage(argv[0]);
    return 1;
  }

  auto res = analyze_parallel(opt);

  if (opt.doStatus)
    print_status(res);
  if (opt.doTopIps)
    print_top_ips(res, opt.topK);
  if (opt.doPerMinute)
    print_per_minute(res);
  if (opt.doSpikes)
    print_spikes(res, opt.spikeThresh);
  return 0;
}
