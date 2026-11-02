/* kgvecd: serve a KGVM vector matrix over stdin/stdout.
 *
 * Coarse pass: Hamming distance over the bit matrix. Fine pass: int8 dot product
 * over the shortlist. No libc on ARM, so it runs on the router with no packages;
 * reached through an SSH pipe, so it opens no network port.
 *
 * File:     char magic[4] "KGVM"; u32 version, count, dim; f32 scale; char model[32]; pad to 64
 *           u32 ids[count]; u8 bits[count * dim/8]; i8 values[count * dim]
 * Ready:    u32 MAGIC_READY, count, dim
 * Request:  u32 MAGIC_REQ, k, shortlist; u8 bits[dim/8]; i8 values[dim]
 * Response: u32 MAGIC_RESP, n, micros; { u32 id; i32 score; } results[n]
 *
 * Results are ranked by score, then by matrix position. vectors.py implements the
 * same ranking and a test holds the two to it.
 */
typedef unsigned char u8; typedef signed char i8; typedef short i16;
typedef unsigned short u16; typedef unsigned int u32; typedef int i32;

#define MAGIC_READY 0x5244474bu
#define MAGIC_REQ 0x5152474bu
#define MAGIC_RESP 0x5352474bu
#define HEADER 64u
#define MAX_K 256u
#define MAX_DIM 8192u

#ifdef __arm__
#include <arm_neon.h>
static inline long sc3(long n, long a, long b, long c) {
  register long r7 __asm__("r7") = n, r0 __asm__("r0") = a, r1 __asm__("r1") = b, r2 __asm__("r2") = c;
  __asm__ volatile("svc 0" : "+r"(r0) : "r"(r7), "r"(r1), "r"(r2) : "memory");
  return r0;
}
static inline long sc6(long n, long a, long b, long c, long d, long e, long f) {
  register long r7 __asm__("r7") = n, r0 __asm__("r0") = a, r1 __asm__("r1") = b, r2 __asm__("r2") = c,
                r3 __asm__("r3") = d, r4 __asm__("r4") = e, r5 __asm__("r5") = f;
  __asm__ volatile("svc 0" : "+r"(r0) : "r"(r7), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5) : "memory");
  return r0;
}
static long sys_read(int f, void *b, long n) { return sc3(3, f, (long)b, n); }
static long sys_write(int f, const void *b, long n) { return sc3(4, f, (long)b, n); }
static int sys_open(const char *p) { return (int)sc3(5, (long)p, 0, 0); }
static long sys_size(int f) { return sc3(19, f, 0, 2); }
static void sys_exit(int c) { sc3(1, c, 0, 0); for (;;) {} }
static void *map_check(long r) { return (r < 0 && r > -4096) ? (void *)0 : (void *)r; }
static void *sys_alloc(long n) { return map_check(sc6(192, 0, n, 3, 0x22, -1, 0)); }
static void *sys_mapfile(int f, long n) { return map_check(sc6(192, 0, n, 1, 1, f, 0)); }
static u32 micros(void) { struct { long s, n; } t; sc3(263, 1, (long)&t, 0); return (u32)t.s * 1000000u + (u32)t.n / 1000u; }
void *memset(void *d, int c, unsigned n) { volatile u8 *p = d; while (n--) *p++ = (u8)c; return d; }
void *memcpy(void *d, const void *s, unsigned n) { volatile u8 *p = d; const u8 *q = s; while (n--) *p++ = *q++; return d; }
void __aeabi_memclr(void *d, unsigned n) { memset(d, 0, n); }
void __aeabi_memclr4(void *d, unsigned n) { memset(d, 0, n); }
void __aeabi_memclr8(void *d, unsigned n) { memset(d, 0, n); }
void __aeabi_memcpy(void *d, const void *s, unsigned n) { memcpy(d, s, n); }
void __aeabi_memcpy4(void *d, const void *s, unsigned n) { memcpy(d, s, n); }
void __aeabi_memcpy8(void *d, const void *s, unsigned n) { memcpy(d, s, n); }
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
static long sys_read(int f, void *b, long n) { return read(f, b, n); }
static long sys_write(int f, const void *b, long n) { return write(f, b, n); }
static int sys_open(const char *p) { return open(p, O_RDONLY); }
static long sys_size(int f) { return lseek(f, 0, SEEK_END); }
static void sys_exit(int c) { _exit(c); }
static void *sys_alloc(long n) { void *r = mmap(0, n, 3, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); return r == MAP_FAILED ? (void *)0 : r; }
static void *sys_mapfile(int f, long n) { void *r = mmap(0, n, 1, MAP_SHARED, f, 0); return r == MAP_FAILED ? (void *)0 : r; }
static u32 micros(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (u32)t.tv_sec * 1000000u + (u32)(t.tv_nsec / 1000); }
#endif

static void die(const char *s) {
  int n = 0;
  while (s[n]) n++;
  sys_write(2, "kgvecd: ", 8);
  sys_write(2, s, n);
  sys_write(2, "\n", 1);
  sys_exit(1);
}
static void *xalloc(long n) { void *p = sys_alloc(n); if (!p) die("out of memory"); return p; }

/* A clean EOF between requests returns 0; EOF inside one is an error. */
static int read_all(void *buf, long n, int eof_ok) {
  u8 *p = buf;
  long got = 0;
  while (got < n) {
    long r = sys_read(0, p + got, n - got);
    if (r == 0 && got == 0 && eof_ok) return 0;
    if (r <= 0) die("short read on stdin");
    got += r;
  }
  return 1;
}
static void write_all(const void *buf, long n) {
  const u8 *p = buf;
  while (n > 0) {
    long r = sys_write(1, p, n);
    if (r <= 0) die("write failed");
    p += r;
    n -= r;
  }
}

__attribute__((noinline)) static u32 hamming(const u8 *a, const u8 *b, u32 n) {
  u32 s = 0, i = 0;
#ifdef __arm__
  uint32x4_t acc = vdupq_n_u32(0);
  for (; i + 16 <= n; i += 16) {
    uint8x16_t x = veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i));
    acc = vpadalq_u16(acc, vpaddlq_u8(vcntq_u8(x)));
  }
  s = vgetq_lane_u32(acc, 0) + vgetq_lane_u32(acc, 1) + vgetq_lane_u32(acc, 2) + vgetq_lane_u32(acc, 3);
#endif
  for (; i < n; i++) s += (u32)__builtin_popcount((u32)(a[i] ^ b[i]));
  return s;
}
__attribute__((noinline)) static i32 dot8(const i8 *restrict a, const i8 *restrict b, u32 n) {
  i32 s = 0;
  for (u32 i = 0; i < n; i++) s += (i16)a[i] * (i16)b[i];
  return s;
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: kgvecd MATRIX");
  int fd = sys_open(argv[1]);
  if (fd < 0) die("cannot open matrix");
  long size = sys_size(fd);
  if (size < (long)HEADER) die("matrix too small");
  const u8 *file = sys_mapfile(fd, size);
  if (!file) die("cannot map matrix");
  const u32 *h = (const u32 *)file;
  if (file[0] != 'K' || file[1] != 'G' || file[2] != 'V' || file[3] != 'M') die("bad magic");
  if (h[1] != 1u) die("unsupported version");
  u32 N = h[2], D = h[3], Db = D / 8u;
  if (D == 0 || D > MAX_DIM || D % 8u) die("bad dimension");
  if ((unsigned long long)size != (unsigned long long)HEADER + 4ull * N + (unsigned long long)N * Db + (unsigned long long)N * D)
    die("matrix size does not match its header");
  const u32 *ids = (const u32 *)(file + HEADER);
  const u8 *bits = file + HEADER + 4u * N;
  const i8 *vals = (const i8 *)(bits + N * Db);

  /* Pull the coarse matrix into the page cache so the first query doesn't wait on USB. */
  volatile u8 touch = 0;
  for (u32 i = 0; i < N * Db; i += 4096u) touch ^= bits[i];

  u16 *dist = xalloc((long)(N ? N : 1u) * 2);
  u32 *hist = xalloc((long)(D + 2u) * 4);
  u8 *qb = xalloc(Db);
  i8 *qv = xalloc(D);
  struct { u32 id; i32 score; } *top = xalloc(sizeof(*top) * MAX_K);
  u32 *pos = xalloc(sizeof(u32) * MAX_K);

  u32 ready[3] = {MAGIC_READY, N, D};
  write_all(ready, sizeof ready);

  for (;;) {
    u32 req[3];
    if (!read_all(req, sizeof req, 1)) return 0;
    if (req[0] != MAGIC_REQ) die("bad request magic");
    read_all(qb, Db, 0);
    read_all(qv, D, 0);
    u32 t0 = micros();
    u32 k = req[1] > MAX_K ? MAX_K : req[1];
    if (k > N) k = N;
    u32 want = req[2] > N ? N : req[2];
    if (want < k) want = k;

    for (u32 i = 0; i <= D; i++) hist[i] = 0;
    for (u32 n = 0; n < N; n++) {
      u32 d = hamming(qb, bits + n * Db, Db);
      dist[n] = (u16)d;
      hist[d]++;
    }
    u32 limit = 0, below = 0;
    while (limit < D && below + hist[limit] < want) below += hist[limit++];
    u32 ties = want > below ? want - below : 0; /* how many at distance == limit, lowest position first */

    u32 found = 0;
    for (u32 n = 0; n < N; n++) {
      if (dist[n] > limit) continue;
      if (dist[n] == limit) {
        if (!ties) continue;
        ties--;
      }
      i32 s = dot8(qv, vals + n * D, D);
      if (found < k) {
        u32 j = found++;
        while (j > 0 && top[j - 1].score < s) { top[j] = top[j - 1]; pos[j] = pos[j - 1]; j--; }
        top[j].id = ids[n]; top[j].score = s; pos[j] = n;
      } else if (k && s > top[k - 1].score) {
        u32 j = k - 1;
        while (j > 0 && top[j - 1].score < s) { top[j] = top[j - 1]; pos[j] = pos[j - 1]; j--; }
        top[j].id = ids[n]; top[j].score = s; pos[j] = n;
      }
    }
    u32 resp[3] = {MAGIC_RESP, found, micros() - t0};
    write_all(resp, sizeof resp);
    if (found) write_all(top, (long)sizeof(*top) * found);
  }
}

#ifdef __arm__
__asm__(".text\n.global _start\n_start:\n mov r0, sp\n bl cstart\n");
__attribute__((used)) void cstart(long *sp) { sys_exit(main((int)sp[0], (char **)(sp + 1))); }
#endif
