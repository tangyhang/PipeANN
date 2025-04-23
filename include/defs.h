#ifndef PREDEFS_H_
#define PREDEFS_H_

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

constexpr int kCacheLineSize = 64;
#define SECTOR_LEN 4096

#endif  // PREDEFS_H_