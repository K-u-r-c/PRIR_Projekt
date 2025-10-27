````markdown
# PRIR – Parallel Log Analyzer (C++20)

A simple, fast, **multithreaded analyzer** for Apache/Nginx access logs written in modern **C++20**.  
Processes logs in parallel and supports common analytics like top IPs, HTTP status histograms, per-minute traffic, and spike detection.

---

## 🚀 Build

```bash
make          # Release build (default)
make debug    # Debug build
```
````

The compiled binary will appear at:

```
build/bin/prir
```

---

## ⚙️ Usage

```bash
./build/bin/prir --file <path> [--threads N] [--status] [--top-ips K] [--per-minute] [--spikes T]
```

### Options

| Flag            | Description                                     |
| --------------- | ----------------------------------------------- |
| `--file <path>` | Path to the access log file (required).         |
| `--threads N`   | Number of threads to use (default = CPU cores). |
| `--status`      | Count HTTP response statuses (e.g., 200, 404).  |
| `--top-ips K`   | Show top-K most active IP addresses.            |
| `--per-minute`  | Show requests aggregated per minute.            |
| `--spikes T`    | Detect traffic spikes (T × median rule).        |

---

## 🧩 Examples

```bash
# Count HTTP statuses
./build/bin/prir --file access.log --threads 4 --status

# Top 20 IPs
./build/bin/prir --file access.log --threads 8 --top-ips 20

# Requests per minute
./build/bin/prir --file access.log --threads 4 --per-minute

# Detect spikes where requests > 3× median
./build/bin/prir --file access.log --threads 8 --spikes 3.0
```

---

## 🏁 Output Format

All results are printed as **CSV** to `stdout`.
Example (status counts):

```
status,count
200,12423
404,52
500,3
```

---

## 🧠 Implementation Notes

- Uses **thread-based chunking** of large log files.
- Each thread parses independently, then results are merged.
- Safe I/O, exception-guarded, and skips malformed lines.

---

## 📜 License

MIT – free to use, modify, and share.

```
