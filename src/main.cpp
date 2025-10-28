// main.cpp — FastLog/PRIR: parallel web log analyzer
// Build: make (produces build/bin/prir)
// Run examples:
//   ./build/bin/prir --file access.log --threads 4 --status
//   ./build/bin/prir --file access.log --threads 8 --top-ips 20
//   ./build/bin/prir --file access.log --threads 8 --per-minute
//   ./build/bin/prir --file access.log --threads 8 --spikes 3.0
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct Parsed {
  string ip;
  string minuteKey;
  int status = -1;
  bool ok = false;
};

// fast, no-throw int parse
static inline bool sv_to_int(string_view sv, int &out) {
  const char *b = sv.data();
  const char *e = b + sv.size();
  while (b < e && isspace(static_cast<unsigned char>(*b)))
    ++b;
  auto res = from_chars(b, e, out);
  return res.ec == errc{}; // success
}

static inline int month3_to_int(string_view m) {
  switch (m.size() ? ((m[0] << 16) | (m[1] << 8) | m[2]) : 0) {
  case ('J' << 16 | 'a' << 8 | 'n'):
    return 1;
  case ('F' << 16 | 'e' << 8 | 'b'):
    return 2;
  case ('M' << 16 | 'a' << 8 | 'r'):
    return 3;
  case ('A' << 16 | 'p' << 8 | 'r'):
    return 4;
  case ('M' << 16 | 'a' << 8 | 'y'):
    return 5;
  case ('J' << 16 | 'u' << 8 | 'n'):
    return 6;
  case ('J' << 16 | 'u' << 8 | 'l'):
    return 7;
  case ('A' << 16 | 'u' << 8 | 'g'):
    return 8;
  case ('S' << 16 | 'e' << 8 | 'p'):
    return 9;
  case ('O' << 16 | 'c' << 8 | 't'):
    return 10;
  case ('N' << 16 | 'o' << 8 | 'v'):
    return 11;
  case ('D' << 16 | 'e' << 8 | 'c'):
    return 12;
  default:
    return 0;
  }
}

// Extract: IP (first token), minuteKey "YYYY-MM-DD HH:MM", status (int).
// Works for common/combined. Returns ok=false if critical fields missing.
static inline Parsed parse_line(string_view s) {
  Parsed out;

  // 1) IP: token before first space
  size_t sp = s.find(' ');
  if (sp == string_view::npos || sp == 0)
    return out;
  out.ip.assign(s.substr(0, sp));

  // 2) Timestamp between [ ... ]  e.g. 10/Oct/2000:13:55:36 -0700
  size_t lb = s.find('[', sp);
  if (lb == string_view::npos)
    return out;
  size_t rb = s.find(']', lb + 1);
  if (rb == string_view::npos)
    return out;
  string_view ts = s.substr(lb + 1, rb - lb - 1);
  // Expect dd/Mon/yyyy:HH:MM
  if (ts.size() < 17)
    return out;

  // dd
  int dd = 0, yyyy = 0;
  if (!sv_to_int(ts.substr(0, 2), dd))
    return out;
  int mm = month3_to_int(ts.substr(3, 3));
  if (mm == 0)
    return out;
  if (!sv_to_int(ts.substr(7, 4), yyyy))
    return out;
  string_view HH = ts.substr(12, 2);
  string_view MM = ts.substr(15, 2);

  auto two = [](int x) {
    char buf[3];
    buf[2] = 0;
    buf[1] = char('0' + (x % 10));
    buf[0] = char('0' + ((x / 10) % 10));
    return string(buf);
  };
  auto two_sv = [](string_view sv) {
    if (sv.size() == 2)
      return string(sv);
    string r = "00";
    for (size_t i = 0; i < min<size_t>(2, sv.size()); ++i)
      r[i] = sv[i];
    return r;
  };
  auto pad2 = [&](int x) {
    return (x < 10 ? "0" + to_string(x) : to_string(x));
  };

  out.minuteKey = to_string(yyyy) + "-" + pad2(mm) + "-" + pad2(dd) + " " +
                  two_sv(HH) + ":" + two_sv(MM);

  // 3) Status: token after closing quote of request
  // Find " ... " then status
  size_t q1 = s.find('"', rb);
  if (q1 == string_view::npos)
    return out;
  size_t q2 = s.find('"', q1 + 1);
  if (q2 == string_view::npos)
    return out;

  size_t pos = s.find_first_not_of(' ', q2 + 1);
  if (pos == string_view::npos)
    return out;
  size_t posEnd = pos;
  while (posEnd < s.size() && isdigit(static_cast<unsigned char>(s[posEnd])))
    ++posEnd;
  int st = -1;
  if (posEnd == pos || !sv_to_int(s.substr(pos, posEnd - pos), st))
    return out;

  out.status = st;
  out.ok = true;
  return out;
}

struct Chunk {
  streampos begin;
  streampos end;
}; // [begin,end)

static vector<Chunk> chunk_file(const string &path, int threads) {
  ifstream f(path, ios::binary);
  vector<Chunk> one{{0, 0}};
  if (!f)
    return one;

  f.seekg(0, ios::end);
  streampos size = f.tellg();
  if (size <= 0)
    return vector<Chunk>{{0, 0}};
  if (threads <= 1)
    return vector<Chunk>{{0, size}};

  vector<Chunk> chunks;
  chunks.reserve(threads);
  streampos base = size / threads;
  streampos start = 0;

  for (int i = 0; i < threads; ++i) {
    streampos tentativeEnd = (i == threads - 1) ? size : (start + base);

    f.clear();
    f.seekg(tentativeEnd);
    if (!f) {
      f.clear();
      tentativeEnd = size;
    } else if (tentativeEnd < size) {
      string dummy;
      getline(f, dummy); // advance to next '\n'
      if (!f) {
        f.clear();
        tentativeEnd = size;
      } else {
        streampos p = f.tellg();
        if (p < 0)
          tentativeEnd = size;
        else
          tentativeEnd = p;
      }
    }

    chunks.push_back({start, tentativeEnd});
    start = tentativeEnd;
  }

  // Compact & drop empties
  vector<Chunk> compact;
  compact.reserve(chunks.size());
  for (auto &c : chunks) {
    if (c.end > c.begin)
      compact.push_back(c);
  }
  if (compact.empty())
    compact.push_back({0, size});
  return compact;
}

template <class Fn>
static void process_chunk(const string &path, Chunk c, Fn fnLine) {
  if (c.end <= c.begin)
    return;
  ifstream f(path, ios::binary);
  if (!f)
    return;
  f.seekg(c.begin);

  string line;
  if (c.begin > 0) {
    getline(f, line);
  } // drop partial first line

  while (f) {
    streampos posBefore = f.tellg();
    if (posBefore < 0 || posBefore >= c.end)
      break;
    if (!getline(f, line))
      break;
    // If reading went beyond chunk end, stop (we may have read the next chunk's
    // first line).
    streampos posAfter = f.tellg();
    if (posAfter >= c.end && posAfter != -1) {
      // still process this 'line' because it belongs to our chunk (read
      // completed within bounds) but do not continue further Note: getline
      // stops at '\n' which was before posAfter; safe to process
      fnLine(line);
      break;
    }
    fnLine(line);
  }
}

struct Results {
  unordered_map<int, uint64_t> statusCount;
  unordered_map<string, uint64_t> ipCount;
  unordered_map<string, uint64_t> perMinute;
};

struct Options {
  bool doStatus = false, doTopIps = false, doPerMinute = false,
       doSpikes = false;
  int topK = 10;
  double spikeThresh = 3.0;
  int threads = (int)thread::hardware_concurrency()
                    ? (int)thread::hardware_concurrency()
                    : 4;
  string path;
};

static Results analyze_parallel(const Options &opt) {
  Results globalR;
  auto chunks = chunk_file(opt.path, opt.threads);
  mutex gmut;

  auto worker = [&](Chunk c) {
    unordered_map<int, uint64_t> sc;
    unordered_map<string, uint64_t> ic;
    unordered_map<string, uint64_t> pm;
    process_chunk(opt.path, c, [&](const string &line) {
      Parsed p = parse_line(string_view(line));
      if (!p.ok)
        return;
      if (opt.doStatus)
        sc[p.status]++;
      if (opt.doTopIps)
        ic[p.ip]++;
      if (opt.doPerMinute || opt.doSpikes)
        pm[p.minuteKey]++;
    });
    lock_guard<mutex> lk(gmut);
    for (auto &[k, v] : sc)
      globalR.statusCount[k] += v;
    for (auto &[k, v] : ic)
      globalR.ipCount[k] += v;
    for (auto &[k, v] : pm)
      globalR.perMinute[k] += v;
  };

  vector<thread> pool;
  pool.reserve(chunks.size());
  for (auto &c : chunks)
    pool.emplace_back(worker, c);
  for (auto &t : pool)
    t.join();
  return globalR;
}

static void print_status(const Results &r) {
  cout << "status,count\n";
  vector<pair<int, uint64_t>> v(r.statusCount.begin(), r.statusCount.end());
  sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first < b.first; });
  for (auto &[k, c] : v)
    cout << k << "," << c << "\n";
}

static void print_top_ips(const Results &r, int K) {
  cout << "ip,count\n";
  vector<pair<string, uint64_t>> v(r.ipCount.begin(), r.ipCount.end());
  partial_sort(v.begin(), v.begin() + min<int>(K, (int)v.size()), v.end(),
               [](auto &a, auto &b) { return a.second > b.second; });
  int limit = min<int>(K, (int)v.size());
  for (int i = 0; i < limit; i++)
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
  cout << "minute,count,is_spike\n";
  vector<pair<string, uint64_t>> v(r.perMinute.begin(), r.perMinute.end());
  sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first < b.first; });

  if (v.empty())
    return;
  vector<uint64_t> counts;
  counts.reserve(v.size());
  for (auto &kv : v)
    counts.push_back(kv.second);
  nth_element(counts.begin(), counts.begin() + counts.size() / 2, counts.end());
  double med = (double)counts[counts.size() / 2];

  for (auto &[k, c] : v) {
    bool spike = (med > 0.0) && (c > T * med);
    cout << k << "," << c << "," << (spike ? "1" : "0") << "\n";
  }
}

static void usage(const char *argv0) {
  cerr << "Usage: " << argv0
       << " --file path [--threads N] [--status] [--top-ips K] [--per-minute] "
          "[--spikes T]\n";
}

int main(int argc, char **argv) {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  try {
    Options opt;
    for (int i = 1; i < argc; i++) {
      string a = argv[i];
      if (a == "--file" && i + 1 < argc)
        opt.path = argv[++i];
      else if (a == "--threads" && i + 1 < argc)
        opt.threads = max(1, stoi(argv[++i]));
      else if (a == "--status")
        opt.doStatus = true;
      else if (a == "--top-ips" && i + 1 < argc) {
        opt.doTopIps = true;
        opt.topK = max(1, stoi(argv[++i]));
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

    // File checks
    {
      ifstream test(opt.path, ios::binary);
      if (!test.is_open()) {
        cerr << "Error: cannot open file: " << opt.path << "\n";
        return 2;
      }
    }

    if (!opt.doStatus && !opt.doTopIps && !opt.doPerMinute && !opt.doSpikes) {
      cerr << "No operation selected. Try one of: "
              "--status --top-ips K --per-minute --spikes T\n";
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

  } catch (const exception &e) {
    cerr << "Fatal: " << e.what() << "\n";
    return 3;
  } catch (...) {
    cerr << "Fatal: unknown error\n";
    return 4;
  }
}
