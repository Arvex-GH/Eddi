#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Eddi Core: Hyperdimensional Writable Knowledge + 1M-Token Disk-Backed Context Engine
// Zero external dependencies. Compiles to a native binary.

namespace eddi {

static constexpr size_t HV_WORDS = 16; // 16 * 64 = 1024-bit packed hypervectors (L1-cache friendly)
static constexpr size_t L1_WINDOW = 2048;

struct HyperVector {
    uint64_t words[HV_WORDS]{};

    static HyperVector from_seed(uint64_t seed) {
        HyperVector hv;
        uint64_t x = seed + 0x9e3779b97f4a7c15ULL;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            hv.words[i] = x;
        }
        return hv;
    }

    // Bind two hypervectors (XOR in binary space)
    HyperVector bind(const HyperVector& other) const {
        HyperVector out;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            out.words[i] = words[i] ^ other.words[i];
        }
        return out;
    }

    // Hamming similarity in [0.0, 1.0]
    double similarity(const HyperVector& other) const {
        uint32_t diff = 0;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            uint64_t x = words[i] ^ other.words[i];
#ifdef _MSC_VER
            diff += static_cast<uint32_t>(__popcnt64(x));
#else
            diff += __builtin_popcountll(x);
#endif
        }
        return 1.0 - (static_cast<double>(diff) / (HV_WORDS * 64.0));
    }
};

static uint64_t fnv1a_64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

struct DiskTokenRecord {
    uint64_t token_hash;
    uint64_t sig; // 64-bit SimHash signature for O(1) LSH bucketing
};

class MMapContextStore {
public:
    MMapContextStore(const std::string& path, size_t max_tokens = 1200000)
        : path_(path), max_tokens_(max_tokens) {
        size_t bytes = max_tokens_ * sizeof(DiskTokenRecord);
#ifdef _WIN32
        hFile_ = CreateFileA(path_.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile_ != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(bytes);
            hMap_ = CreateFileMappingA(hFile_, nullptr, PAGE_READWRITE, li.HighPart, li.LowPart, nullptr);
            if (hMap_) {
                data_ = static_cast<DiskTokenRecord*>(MapViewOfFile(hMap_, FILE_MAP_ALL_ACCESS, 0, 0, bytes));
            }
        }
#else
        fd_ = open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ >= 0) {
            if (ftruncate(fd_, bytes) == 0) {
                data_ = static_cast<DiskTokenRecord*>(
                    mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
                if (data_ == MAP_FAILED) {
                    data_ = nullptr;
                }
            }
        }
#endif
    }

    ~MMapContextStore() {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (hMap_) CloseHandle(hMap_);
        if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#else
        if (data_) {
            munmap(data_, max_tokens_ * sizeof(DiskTokenRecord));
        }
        if (fd_ >= 0) {
            close(fd_);
        }
#endif
    }

    void append(uint64_t token_hash, uint64_t sig) {
        if (!data_ || count_ >= max_tokens_) return;
        data_[count_] = {token_hash, sig};
        // Index into 16-bit LSH bucket for fast retrieval
        uint16_t bucket = static_cast<uint16_t>(sig & 0xFFFF);
        buckets_[bucket].push_back(count_);
        if (l1_window_.size() >= L1_WINDOW) {
            l1_window_.erase(l1_window_.begin());
        }
        l1_window_.push_back(token_hash);
        ++count_;
    }

    size_t size() const { return count_; }

    std::vector<size_t> query_bucket(uint64_t sig, size_t limit = 8) const {
        uint16_t bucket = static_cast<uint16_t>(sig & 0xFFFF);
        auto it = buckets_.find(bucket);
        if (it == buckets_.end()) return {};
        const auto& vec = it->second;
        std::vector<size_t> out;
        size_t start = vec.size() > limit ? vec.size() - limit : 0;
        for (size_t i = start; i < vec.size(); ++i) {
            out.push_back(vec[i]);
        }
        return out;
    }

private:
    std::string path_;
    size_t max_tokens_;
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
    HANDLE hMap_ = nullptr;
#else
    int fd_ = -1;
#endif
    DiskTokenRecord* data_ = nullptr;
    size_t count_ = 0;
    std::vector<uint64_t> l1_window_;
    std::unordered_map<uint16_t, std::vector<size_t>> buckets_;
};

} // namespace eddi

int main(int argc, char** argv) {
    using namespace eddi;
    if (argc < 2) {
        std::cout << "Eddi Core Engine v0.1\n"
                  << "Usage:\n"
                  << "  eddi_core bench <num_tokens>   Run 1M-token ingestion & retrieval benchmark\n";
        return 0;
    }

    std::string cmd = argv[1];
    if (cmd == "bench") {
        size_t n = (argc >= 3) ? std::stoull(argv[2]) : 1000000ULL;
        MMapContextStore store("/tmp/eddi_context.mmap", n + 1024);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {
            uint64_t h = i * 0x9e3779b97f4a7c15ULL;
            uint64_t sig = h ^ (h >> 17);
            store.append(h, sig);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double tps = static_cast<double>(n) / secs;

        std::cout << "Ingested " << store.size() << " tokens into disk-backed mmap context in "
                  << secs << "s (" << static_cast<uint64_t>(tps) << " tokens/sec)\n";

        // Query test
        auto q0 = std::chrono::high_resolution_clock::now();
        size_t hits = 0;
        for (size_t i = 0; i < 1000; ++i) {
            uint64_t h = (i * 997) * 0x9e3779b97f4a7c15ULL;
            uint64_t sig = h ^ (h >> 17);
            hits += store.query_bucket(sig, 4).size();
        }
        auto q1 = std::chrono::high_resolution_clock::now();
        double qsecs = std::chrono::duration<double>(q1 - q0).count();
        std::cout << "1000 LSH context queries completed in " << (qsecs * 1000.0)
                  << " ms (hits=" << hits << ")\n";
        return 0;
    }

    return 0;
}
