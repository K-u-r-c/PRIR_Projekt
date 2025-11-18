#include "gpu_interface.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(USE_MPI)
#include <mpi.h>
#endif

namespace {

using sys_seconds =
    std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>;

enum class StatInterval { Minute, Hour };

struct ProgramConfig {
  std::string filePath;
  std::vector<std::string> phrases;
  std::vector<std::string> phrasesNormalized;
  std::vector<std::string> severityUniverse;
  std::unordered_set<std::string> severityFilters;
  bool filterBySeverity = false;
  bool caseSensitive = false;
  bool emitMatches = false;
  std::string emitFile;
  bool emitStdout = false;
  bool countOnlyFiltered = false;
  bool useCuda = false;
  bool statsEnabled = true;
  StatInterval interval = StatInterval::Hour;
  std::optional<sys_seconds> fromTs;
  std::optional<sys_seconds> toTs;
  int requestedThreads = 0;
};

struct FileChunk {
  std::vector<std::string> lines;
};

struct LocalResults {
  std::vector<std::uint64_t> phraseCounts;
  std::unordered_map<std::string, std::uint64_t> timeBuckets;
  std::vector<std::string> matchingLines;
  std::vector<std::uint32_t> gpuHits;
  std::uint64_t processedLines = 0;
  std::uint64_t matchedLines = 0;
  bool gpuUsed = false;
};

struct DateTimeParts {
  int year = 0;
  int month = 0;
  int day = 0;
  int hour = 0;
  int minute = 0;
  int second = 0;
};

std::string to_lower(std::string_view s) {
  std::string out(s.begin(), s.end());
  for (char &ch : out)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

std::string to_upper(std::string_view s) {
  std::string out(s.begin(), s.end());
  for (char &ch : out)
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return out;
}

std::vector<std::string> split_list(std::string_view src) {
  std::vector<std::string> items;
  std::string current;
  for (char ch : src) {
    if (ch == ',' || ch == ';') {
      if (!current.empty()) {
        items.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(ch);
    }
  }
  if (!current.empty())
    items.push_back(current);
  for (auto &item : items) {
    size_t start = 0;
    while (start < item.size() &&
           std::isspace(static_cast<unsigned char>(item[start])))
      ++start;
    size_t end = item.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(item[end - 1])))
      --end;
    item = item.substr(start, end - start);
  }
  items.erase(std::remove_if(items.begin(), items.end(),
                             [](const std::string &s) { return s.empty(); }),
              items.end());
  return items;
}

bool parse_date_time(std::string_view text, DateTimeParts &out) {
  if (text.size() < 19)
    return false;
  auto is_digit = [](char ch) { return ch >= '0' && ch <= '9'; };
  for (int i = 0; i < 4; ++i)
    if (!is_digit(text[i]))
      return false;
  if (text[4] != '-')
    return false;
  if (!is_digit(text[5]) || !is_digit(text[6]))
    return false;
  if (text[7] != '-')
    return false;
  if (!is_digit(text[8]) || !is_digit(text[9]))
    return false;
  char sep = text[10];
  if (sep != 'T' && sep != ' ')
    return false;
  if (!is_digit(text[11]) || !is_digit(text[12]))
    return false;
  if (text[13] != ':')
    return false;
  if (!is_digit(text[14]) || !is_digit(text[15]))
    return false;
  if (text[16] != ':')
    return false;
  if (!is_digit(text[17]) || !is_digit(text[18]))
    return false;

  auto to_int = [&](int idx, int len) {
    int value = 0;
    for (int j = 0; j < len; ++j)
      value = value * 10 + (text[idx + j] - '0');
    return value;
  };

  out.year = to_int(0, 4);
  out.month = to_int(5, 2);
  out.day = to_int(8, 2);
  out.hour = to_int(11, 2);
  out.minute = to_int(14, 2);
  out.second = to_int(17, 2);
  return true;
}

std::optional<sys_seconds> to_time_point(const DateTimeParts &parts) {
  using namespace std::chrono;
  if (parts.month < 1 || parts.month > 12)
    return std::nullopt;
  if (parts.day < 1 || parts.day > 31)
    return std::nullopt;
  year y{parts.year};
  month m{static_cast<unsigned>(parts.month)};
  day d{static_cast<unsigned>(parts.day)};
  year_month_day ymd{y, m, d};
  if (!ymd.ok())
    return std::nullopt;
  if (parts.hour < 0 || parts.hour > 23)
    return std::nullopt;
  if (parts.minute < 0 || parts.minute > 59)
    return std::nullopt;
  if (parts.second < 0 || parts.second > 60)
    return std::nullopt;
  sys_days base{ymd};
  sys_seconds tp = base + hours(parts.hour) + minutes(parts.minute) +
                   seconds(parts.second);
  return tp;
}

std::optional<sys_seconds> parse_datetime_string(std::string_view text) {
  DateTimeParts parts{};
  if (!parse_date_time(text, parts))
    return std::nullopt;
  return to_time_point(parts);
}

std::optional<sys_seconds> extract_timestamp(std::string_view line) {
  for (size_t i = 0; i + 19 <= line.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(line[i])))
      continue;
    DateTimeParts parts{};
    if (!parse_date_time(line.substr(i), parts))
      continue;
    auto tp = to_time_point(parts);
    if (tp)
      return tp;
  }
  return std::nullopt;
}

std::string make_bucket(const sys_seconds &tp, StatInterval interval) {
  using namespace std::chrono;
  auto dayFloor = floor<days>(tp);
  year_month_day ymd{dayFloor};
  auto dayTime = tp - dayFloor;
  auto h = duration_cast<hours>(dayTime);
  auto m = duration_cast<minutes>(dayTime - h);

  std::ostringstream oss;
  oss << std::setw(4) << std::setfill('0') << int(ymd.year()) << "-"
      << std::setw(2) << unsigned(ymd.month()) << "-" << std::setw(2)
      << unsigned(ymd.day()) << " " << std::setw(2) << h.count();
  if (interval == StatInterval::Minute)
    oss << ":" << std::setw(2) << m.count();
  else
    oss << ":00";
  return oss.str();
}

size_t count_occurrences(std::string_view haystack,
                         std::string_view needle) {
  if (needle.empty())
    return 0;
  size_t count = 0;
  size_t pos = haystack.find(needle);
  size_t step = std::max<size_t>(1, needle.size());
  while (pos != std::string_view::npos) {
    ++count;
    pos = haystack.find(needle, pos + step);
  }
  return count;
}

std::string detect_severity(const std::string &upperLine,
                            const std::vector<std::string> &universe) {
  for (const auto &candidate : universe) {
    if (candidate.empty())
      continue;
    if (upperLine.find(candidate) != std::string::npos)
      return candidate;
  }
  return {};
}

void usage(const char *prog) {
  std::cerr << "Parallel log/text analyzer
"
            << "Usage: " << prog
            << " --file path --phrase WORD [options]

"
            << "Options:
"
            << "  --phrase TEXT           Phrase to count (repeatable).
"
            << "  --case-sensitive        Keep original casing when matching.
"
            << "  --level NAME            Severity to keep (repeatable).
"
            << "  --from YYYY-MM-DDTHH:MM:SS   Start of time window.
"
            << "  --to   YYYY-MM-DDTHH:MM:SS   End of time window.
"
            << "  --stats [hour|minute]   Time bucket granularity (default hour).
"
            << "  --no-stats              Disable time bucket statistics.
"
            << "  --count-filtered        Count phrases only for filtered lines.
"
            << "  --emit                  Print matching lines to stdout.
"
            << "  --emit-file path        Save matching lines to a file.
"
            << "  --threads N             Force OpenMP thread count.
"
            << "  --use-cuda              Use CUDA histogram backend.
"
            << "  --help                  Show this message.
";
}

std::optional<ProgramConfig> parse_cli(int argc, char **argv, bool isRoot) {
  ProgramConfig cfg;
  cfg.severityUniverse = {"CRITICAL", "ERROR", "WARNING", "WARN",
                          "INFO",     "DEBUG"};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      if (isRoot)
        usage(argv[0]);
      return std::nullopt;
    } else if (arg == "--file" && i + 1 < argc) {
      cfg.filePath = argv[++i];
    } else if (arg == "--phrase" && i + 1 < argc) {
      std::string phrase = argv[++i];
      if (phrase.empty())
        throw std::runtime_error("phrase cannot be empty");
      cfg.phrases.push_back(phrase);
    } else if (arg == "--case-sensitive") {
      cfg.caseSensitive = true;
    } else if ((arg == "--level" || arg == "--severity") && i + 1 < argc) {
      auto list = split_list(argv[++i]);
      for (auto &item : list) {
        auto upper = to_upper(item);
        if (upper.empty())
          continue;
        cfg.severityFilters.insert(upper);
        cfg.severityUniverse.push_back(upper);
      }
      cfg.filterBySeverity = true;
    } else if (arg == "--from" && i + 1 < argc) {
      auto ts = parse_datetime_string(argv[++i]);
      if (!ts)
        throw std::runtime_error("Invalid --from timestamp");
      cfg.fromTs = ts;
    } else if (arg == "--to" && i + 1 < argc) {
      auto ts = parse_datetime_string(argv[++i]);
      if (!ts)
        throw std::runtime_error("Invalid --to timestamp");
      cfg.toTs = ts;
    } else if (arg == "--stats" && i + 1 < argc) {
      std::string mode = argv[++i];
      if (mode == "hour")
        cfg.interval = StatInterval::Hour;
      else if (mode == "minute")
        cfg.interval = StatInterval::Minute;
      else
        throw std::runtime_error("Unsupported --stats interval");
      cfg.statsEnabled = true;
    } else if (arg == "--no-stats") {
      cfg.statsEnabled = false;
    } else if (arg == "--count-filtered") {
      cfg.countOnlyFiltered = true;
    } else if (arg == "--emit") {
      cfg.emitMatches = true;
      cfg.emitStdout = true;
    } else if (arg == "--emit-file" && i + 1 < argc) {
      cfg.emitMatches = true;
      cfg.emitFile = argv[++i];
    } else if (arg == "--threads" && i + 1 < argc) {
      cfg.requestedThreads = std::stoi(argv[++i]);
    } else if (arg == "--use-cuda") {
      cfg.useCuda = true;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (cfg.filePath.empty())
    throw std::runtime_error("--file is required");
  if (cfg.phrases.empty())
    throw std::runtime_error("At least one --phrase is required");
  cfg.phrasesNormalized.clear();
  for (const auto &phrase : cfg.phrases) {
    cfg.phrasesNormalized.push_back(cfg.caseSensitive ? phrase : to_lower(phrase));
  }
  return cfg;
}

FileChunk read_chunk(const ProgramConfig &cfg, int rank, int world) {
  FileChunk chunk;
  std::ifstream in(cfg.filePath, std::ios::binary);
  if (!in.is_open())
    throw std::runtime_error("Cannot open input file: " + cfg.filePath);
  in.seekg(0, std::ios::end);
  std::streamoff total = in.tellg();
  if (total < 0)
    total = 0;
  std::streamoff start = total * rank / world;
  std::streamoff stop = total * (rank + 1) / world;
  in.seekg(start, std::ios::beg);
  if (rank != 0) {
    std::string dummy;
    std::getline(in, dummy);
  }
  std::string line;
  while (std::getline(in, line)) {
    chunk.lines.push_back(line);
    auto pos = in.tellg();
    if (pos == -1)
      break;
    if (pos >= stop && (rank + 1) < world)
      break;
  }
  return chunk;
}

bool time_in_window(const ProgramConfig &cfg, const std::optional<sys_seconds> &ts) {
  if (!cfg.fromTs && !cfg.toTs)
    return true;
  if (!ts)
    return false;
  if (cfg.fromTs && *ts < *cfg.fromTs)
    return false;
  if (cfg.toTs && *ts > *cfg.toTs)
    return false;
  return true;
}

LocalResults analyze_chunk(const FileChunk &chunk, const ProgramConfig &cfg) {
  LocalResults res;
  res.processedLines = chunk.lines.size();
  res.phraseCounts.assign(cfg.phrases.size(), 0);

  bool needTime = cfg.statsEnabled || cfg.fromTs || cfg.toTs;
  bool needSeverity = cfg.filterBySeverity;

#pragma omp parallel if (chunk.lines.size() > 1024)
  {
    std::vector<std::uint64_t> threadCounts(cfg.phrases.size(), 0);
    std::unordered_map<std::string, std::uint64_t> localBuckets;
    std::vector<std::string> localMatches;
    std::vector<std::uint32_t> localHits;
    std::uint64_t localMatched = 0;

    if (cfg.emitMatches)
      localMatches.reserve(256);
    if (cfg.statsEnabled)
      localBuckets.reserve(256);
    if (cfg.useCuda)
      localHits.reserve(1024);

    long long total = static_cast<long long>(chunk.lines.size());
#pragma omp for schedule(dynamic, 256) nowait
    for (long long idx = 0; idx < total; ++idx) {
      const std::string &line = chunk.lines[static_cast<size_t>(idx)];
      std::optional<sys_seconds> ts;
      std::string bucketKey;
      if (needTime) {
        ts = extract_timestamp(line);
        if (ts && cfg.statsEnabled)
          bucketKey = make_bucket(*ts, cfg.interval);
      }

      bool severityOk = true;
      std::string severity;
      if (needSeverity) {
        std::string upper = to_upper(line);
        severity = detect_severity(upper, cfg.severityUniverse);
        severityOk = cfg.severityFilters.count(severity) > 0;
      }
      bool windowOk = time_in_window(cfg, ts);
      bool selected = severityOk && windowOk;

      if (selected) {
        ++localMatched;
        if (cfg.statsEnabled && !bucketKey.empty())
          ++localBuckets[bucketKey];
        if (cfg.emitMatches)
          localMatches.push_back(line);
      }

      bool allowCounts = !cfg.countOnlyFiltered || selected;
      if (!allowCounts)
        continue;

      std::string lowered;
      const std::string *searchPtr = &line;
      if (!cfg.caseSensitive) {
        lowered = to_lower(line);
        searchPtr = &lowered;
      }
      std::string_view haystack(*searchPtr);

      for (size_t p = 0; p < cfg.phrasesNormalized.size(); ++p) {
        size_t matches = count_occurrences(haystack, cfg.phrasesNormalized[p]);
        if (!matches)
          continue;
        if (cfg.useCuda) {
          for (size_t k = 0; k < matches; ++k)
            localHits.push_back(static_cast<std::uint32_t>(p));
        } else {
          threadCounts[p] += matches;
        }
      }
    }
#pragma omp critical
    {
      for (size_t i = 0; i < threadCounts.size(); ++i)
        res.phraseCounts[i] += threadCounts[i];
      if (cfg.statsEnabled) {
        for (auto &kv : localBuckets)
          res.timeBuckets[kv.first] += kv.second;
      }
      if (cfg.emitMatches)
        res.matchingLines.insert(res.matchingLines.end(), localMatches.begin(),
                                 localMatches.end());
      res.matchedLines += localMatched;
      if (cfg.useCuda)
        res.gpuHits.insert(res.gpuHits.end(), localHits.begin(),
                           localHits.end());
    }
  }

  if (cfg.useCuda) {
    if (gpu::is_available()) {
      if (gpu::histogram(res.gpuHits, cfg.phrases.size(), res.phraseCounts))
        res.gpuUsed = true;
      else
        std::cerr << "[warn] GPU histogram failed, falling back to CPU
";
    } else {
      std::cerr << "[warn] CUDA backend not available, falling back to CPU
";
    }
    if (!res.gpuUsed) {
      res.phraseCounts.assign(cfg.phrases.size(), 0);
      for (std::uint32_t idx : res.gpuHits) {
        if (idx < res.phrases.size())
          ++res.phraseCounts[idx];
      }
    }
    res.gpuHits.clear();
  }

  return res;
}

void merge_results(LocalResults &base, const LocalResults &other,
                   const ProgramConfig &cfg) {
  if (base.phraseCounts.size() != other.phraseCounts.size())
    base.phraseCounts.resize(other.phraseCounts.size());
  for (size_t i = 0; i < other.phraseCounts.size(); ++i)
    base.phraseCounts[i] += other.phraseCounts[i];
  if (cfg.statsEnabled) {
    for (const auto &kv : other.timeBuckets)
      base.timeBuckets[kv.first] += kv.second;
  }
  base.matchedLines += other.matchedLines;
  base.processedLines += other.processedLines;
  if (cfg.emitMatches) {
    base.matchingLines.insert(base.matchingLines.end(), other.matchingLines.begin(),
                              other.matchingLines.end());
  }
}

std::vector<char>
serialize_pairs(const std::unordered_map<std::string, std::uint64_t> &map) {
  std::vector<char> buf;
  std::uint64_t entries = map.size();
  buf.insert(buf.end(), reinterpret_cast<const char *>(&entries),
             reinterpret_cast<const char *>(&entries) + sizeof(entries));
  for (const auto &kv : map) {
    std::uint64_t len = kv.first.size();
    buf.insert(buf.end(), reinterpret_cast<const char *>(&len),
               reinterpret_cast<const char *>(&len) + sizeof(len));
    buf.insert(buf.end(), kv.first.data(), kv.first.data() + len);
    buf.insert(buf.end(), reinterpret_cast<const char *>(&kv.second),
               reinterpret_cast<const char *>(&kv.second) + sizeof(kv.second));
  }
  return buf;
}

std::unordered_map<std::string, std::uint64_t>
deserialize_pairs(const std::vector<char> &buf) {
  std::unordered_map<std::string, std::uint64_t> map;
  const char *ptr = buf.data();
  const char *end = buf.data() + buf.size();
  if (ptr + sizeof(std::uint64_t) > end)
    return map;
  std::uint64_t entries = *reinterpret_cast<const std::uint64_t *>(ptr);
  ptr += sizeof(std::uint64_t);
  for (std::uint64_t i = 0; i < entries; ++i) {
    if (ptr + sizeof(std::uint64_t) > end)
      break;
    std::uint64_t len = *reinterpret_cast<const std::uint64_t *>(ptr);
    ptr += sizeof(std::uint64_t);
    if (ptr + len > end)
      break;
    std::string key(ptr, ptr + len);
    ptr += len;
    if (ptr + sizeof(std::uint64_t) > end)
      break;
    std::uint64_t value = *reinterpret_cast<const std::uint64_t *>(ptr);
    ptr += sizeof(std::uint64_t);
    map[key] += value;
  }
  return map;
}

std::vector<char> serialize_strings(const std::vector<std::string> &items) {
  std::vector<char> buf;
  std::uint64_t count = items.size();
  buf.insert(buf.end(), reinterpret_cast<const char *>(&count),
             reinterpret_cast<const char *>(&count) + sizeof(count));
  for (const auto &line : items) {
    std::uint64_t len = line.size();
    buf.insert(buf.end(), reinterpret_cast<const char *>(&len),
               reinterpret_cast<const char *>(&len) + sizeof(len));
    buf.insert(buf.end(), line.data(), line.data() + len);
  }
  return buf;
}

std::vector<std::string> deserialize_strings(const std::vector<char> &buf) {
  std::vector<std::string> out;
  const char *ptr = buf.data();
  const char *end = buf.data() + buf.size();
  if (ptr + sizeof(std::uint64_t) > end)
    return out;
  std::uint64_t count = *reinterpret_cast<const std::uint64_t *>(ptr);
  ptr += sizeof(std::uint64_t);
  for (std::uint64_t i = 0; i < count; ++i) {
    if (ptr + sizeof(std::uint64_t) > end)
      break;
    std::uint64_t len = *reinterpret_cast<const std::uint64_t *>(ptr);
    ptr += sizeof(std::uint64_t);
    if (ptr + len > end)
      break;
    out.emplace_back(ptr, ptr + len);
    ptr += len;
  }
  return out;
}

#if defined(USE_MPI)
const int TAG_COUNTS = 100;
const int TAG_BUCKETS_META = 101;
const int TAG_BUCKETS_DATA = 102;
const int TAG_LINES_META = 103;
const int TAG_LINES_DATA = 104;
const int TAG_SUMMARY = 105;

void mpi_send_results(const LocalResults &res, int dest, MPI_Comm comm,
                      const ProgramConfig &cfg) {
  const auto countSize = static_cast<int>(res.phraseCounts.size());
  MPI_Send(res.phraseCounts.data(), countSize, MPI_UINT64_T, dest,
           TAG_COUNTS, comm);

  auto bucketBuf = cfg.statsEnabled ? serialize_pairs(res.timeBuckets)
                                    : std::vector<char>();
  std::uint64_t bucketBytes = bucketBuf.size();
  MPI_Send(&bucketBytes, 1, MPI_UINT64_T, dest, TAG_BUCKETS_META, comm);
  if (bucketBytes)
    MPI_Send(bucketBuf.data(), static_cast<int>(bucketBytes), MPI_CHAR, dest,
             TAG_BUCKETS_DATA, comm);

  auto lineBuf = cfg.emitMatches ? serialize_strings(res.matchingLines)
                                 : std::vector<char>();
  std::uint64_t lineBytes = lineBuf.size();
  MPI_Send(&lineBytes, 1, MPI_UINT64_T, dest, TAG_LINES_META, comm);
  if (lineBytes)
    MPI_Send(lineBuf.data(), static_cast<int>(lineBytes), MPI_CHAR, dest,
             TAG_LINES_DATA, comm);

  std::uint64_t summary[2] = {res.processedLines, res.matchedLines};
  MPI_Send(summary, 2, MPI_UINT64_T, dest, TAG_SUMMARY, comm);
}

LocalResults mpi_recv_results(int src, size_t phrases, MPI_Comm comm,
                              const ProgramConfig &cfg) {
  LocalResults res;
  res.phraseCounts.assign(phrases, 0);
  MPI_Status st{};
  MPI_Recv(res.phraseCounts.data(), static_cast<int>(phrases), MPI_UINT64_T,
           src, TAG_COUNTS, comm, &st);

  std::uint64_t bucketBytes = 0;
  MPI_Recv(&bucketBytes, 1, MPI_UINT64_T, src, TAG_BUCKETS_META, comm, &st);
  if (bucketBytes) {
    std::vector<char> buf(bucketBytes);
    MPI_Recv(buf.data(), static_cast<int>(bucketBytes), MPI_CHAR, src,
             TAG_BUCKETS_DATA, comm, &st);
    if (cfg.statsEnabled)
      res.timeBuckets = deserialize_pairs(buf);
  }

  std::uint64_t lineBytes = 0;
  MPI_Recv(&lineBytes, 1, MPI_UINT64_T, src, TAG_LINES_META, comm, &st);
  if (lineBytes) {
    std::vector<char> buf(lineBytes);
    MPI_Recv(buf.data(), static_cast<int>(lineBytes), MPI_CHAR, src,
             TAG_LINES_DATA, comm, &st);
    if (cfg.emitMatches)
      res.matchingLines = deserialize_strings(buf);
  }

  std::uint64_t summary[2] = {0, 0};
  MPI_Recv(summary, 2, MPI_UINT64_T, src, TAG_SUMMARY, comm, &st);
  res.processedLines = summary[0];
  res.matchedLines = summary[1];
  return res;
}
#endif

void dump_results(const LocalResults &res, const ProgramConfig &cfg) {
  std::cout << "phrase,count
";
  for (size_t i = 0; i < cfg.phrases.size(); ++i)
    std::cout << cfg.phrases[i] << "," << res.phraseCounts[i] << "
";
  if (cfg.statsEnabled) {
    std::cout << "
interval,count
";
    std::vector<std::pair<std::string, std::uint64_t>> rows(res.timeBuckets.begin(),
                                                       res.timeBuckets.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (auto &kv : rows)
      std::cout << kv.first << "," << kv.second << "
";
  }
}

void emit_matches_root(const LocalResults &res, const ProgramConfig &cfg) {
  if (!cfg.emitMatches)
    return;
  if (cfg.emitStdout) {
    for (const auto &line : res.matchingLines)
      std::cout << line << "
";
  }
  if (!cfg.emitFile.empty()) {
    std::ofstream out(cfg.emitFile, std::ios::app);
    for (const auto &line : res.matchingLines)
      out << line << "
";
  }
}

} // namespace

int main(int argc, char **argv) {
  int rank = 0;
  int world = 1;
#if defined(USE_MPI)
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
#endif

  try {
    auto cfgOpt = parse_cli(argc, argv, rank == 0);
    if (!cfgOpt) {
#if defined(USE_MPI)
      MPI_Finalize();
#endif
      return 0;
    }
    ProgramConfig cfg = *cfgOpt;

#ifdef _OPENMP
    if (cfg.requestedThreads > 0)
      omp_set_num_threads(cfg.requestedThreads);
#endif

    FileChunk chunk = read_chunk(cfg, rank, world);
    LocalResults local = analyze_chunk(chunk, cfg);

#if defined(USE_MPI)
    if (world > 1) {
      if (rank == 0) {
        LocalResults aggregated = local;
        for (int src = 1; src < world; ++src) {
          LocalResults incoming =
              mpi_recv_results(src, cfg.phrases.size(), MPI_COMM_WORLD, cfg);
          merge_results(aggregated, incoming, cfg);
        }
        dump_results(aggregated, cfg);
        emit_matches_root(aggregated, cfg);
      } else {
        mpi_send_results(local, 0, MPI_COMM_WORLD, cfg);
      }
      MPI_Finalize();
      return 0;
    }
#endif

    if (rank == 0) {
      dump_results(local, cfg);
      emit_matches_root(local, cfg);
    }

#if defined(USE_MPI)
    MPI_Finalize();
#endif
    return 0;
  } catch (const std::exception &e) {
    if (rank == 0)
      std::cerr << "Error: " << e.what() << "
";
#if defined(USE_MPI)
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    return 1;
#endif
  }
  return 0;
}
