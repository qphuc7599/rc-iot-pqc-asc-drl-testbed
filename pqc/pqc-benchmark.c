/*
 * pqc-benchmark.c — ML-DSA (Dilithium) Benchmark cho IoT Containers
 *
 * Compile: gcc -static -O2 pqc-benchmark.c -I/opt/liboqs/include -L/opt/liboqs/lib -loqs -lm -o pqc-benchmark
 *
 * Usage:
 *   ./pqc-benchmark keygen [--algo ML-DSA-44] [--iterations 100]
 *   ./pqc-benchmark sign   [--algo ML-DSA-44] [--iterations 100]
 *   ./pqc-benchmark verify [--algo ML-DSA-44] [--iterations 100]
 *   ./pqc-benchmark all    [--algo ML-DSA-44] [--iterations 100] [--node-id 1]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <oqs/oqs.h>

/* ---- Energy model: per-device power profiles (mW) ----
 * Sources: Nordic DS v1.7, Espressif DS, ST RM0090/RM0433,
 *          Raspberry Pi RP2040 DS
 * P_active = dynamic power during crypto computation
 */
typedef struct { const char *name; double power_mw; } DevicePower;
static const DevicePower DEVICE_POWER[] = {
    {"nrf52840",    23.0},   /* Nordic nRF52840 @ 64MHz, 3.0V  */
    {"stm32l4",     30.0},   /* STM32L4 Cortex-M4 @ 80MHz      */
    {"rp2040",      45.0},   /* Raspberry Pi RP2040 @ 133MHz   */
    {"stm32f4",     50.0},   /* STM32F4 Cortex-M4 @ 168MHz     */
    {"stm32h7",     90.0},   /* STM32H7 Cortex-M7 @ 480MHz     */
    {"esp32",      160.0},   /* ESP32 @ 240MHz, WiFi active     */
    {"esp32s3",    170.0},   /* ESP32-S3 @ 240MHz              */
};
#define NUM_DEVICES 7

/*
 * pqm4 reference cycle counts for ML-DSA-44 (NIST Level 2)
 * Source: https://github.com/mupq/pqm4
 *
 *   m4f optimized:  KeyGen 1,426k  Sign 4,336k  Verify 1,579k
 *   clean reference: KeyGen 1,870k  Sign 7,280k  Verify 2,060k
 */
typedef struct {
    const char *name;
    double clock_mhz;
    double sign_cycles_k;    /* kilo-cycles for sign */
    double verify_cycles_k;  /* kilo-cycles for verify */
    double keygen_cycles_k;  /* kilo-cycles for keygen */
} ArmDevice;

static const ArmDevice ARM_DEVICES[] = {
    /* name          MHz     sign_k   verify_k keygen_k  (source)               */
    {"nrf52840",      64.0,  4336.0,  1579.0,  1426.0},  /* Cortex-M4, pqm4 m4f */
    {"stm32l4",       80.0,  4336.0,  1579.0,  1426.0},  /* Cortex-M4, pqm4 m4f */
    {"stm32f4",      168.0,  4336.0,  1579.0,  1426.0},  /* Cortex-M4, pqm4 m4f */
    {"stm32h7",      480.0,  4336.0,  1579.0,  1426.0},  /* Cortex-M7, pqm4 m4f */
    {"rp2040",       133.0,  7280.0,  2060.0,  1870.0},  /* Cortex-M0+, pqm4 clean */
    {"esp32",        240.0,  9464.0,  2678.0,  2431.0},  /* Xtensa, est 1.3x clean */
    {"esp32s3",      240.0,  8736.0,  2472.0,  2244.0},  /* Xtensa-S3, est 1.2x clean */
};
#define NUM_ARM_DEVICES 7

/* ---- Helpers ---- */

static double timespec_diff_us(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1e6 +
           (end->tv_nsec - start->tv_nsec) / 1e3;
}

typedef struct {
    double min, max, sum, sum_sq;
    int count;
} Stats;

static void stats_init(Stats *s) {
    s->min = 1e18; s->max = 0; s->sum = 0; s->sum_sq = 0; s->count = 0;
}

static void stats_add(Stats *s, double val) {
    if (val < s->min) s->min = val;
    if (val > s->max) s->max = val;
    s->sum += val;
    s->sum_sq += val * val;
    s->count++;
}

static double stats_avg(Stats *s) {
    return s->count > 0 ? s->sum / s->count : 0;
}

static double stats_stddev(Stats *s) {
    if (s->count < 2) return 0;
    double avg = stats_avg(s);
    return sqrt(s->sum_sq / s->count - avg * avg);
}

static double energy_mj(double time_us, double power_mw) {
    /* E(mJ) = T(s) * P(mW) = T(us) * 1e-6 * P(mW) */
    return time_us * 1e-6 * power_mw;
}

/* ---- Benchmark functions ---- */

static int bench_keygen(const char *algo_name, int iterations, Stats *st) {
    OQS_SIG *sig = OQS_SIG_new(algo_name);
    if (!sig) {
        fprintf(stderr, "ERROR: Algorithm '%s' not supported\n", algo_name);
        return -1;
    }

    uint8_t *pk = malloc(sig->length_public_key);
    uint8_t *sk = malloc(sig->length_secret_key);
    if (!pk || !sk) { fprintf(stderr, "ERROR: OOM\n"); return -1; }

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        OQS_STATUS rc = OQS_SIG_keypair(sig, pk, sk);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (rc != OQS_SUCCESS) {
            fprintf(stderr, "ERROR: keygen failed at iteration %d\n", i);
            free(pk); free(sk); OQS_SIG_free(sig);
            return -1;
        }
        stats_add(st, timespec_diff_us(&t0, &t1));
    }

    free(pk); free(sk); OQS_SIG_free(sig);
    return 0;
}

static int bench_sign(const char *algo_name, int iterations, Stats *st) {
    OQS_SIG *sig = OQS_SIG_new(algo_name);
    if (!sig) return -1;

    uint8_t *pk = malloc(sig->length_public_key);
    uint8_t *sk = malloc(sig->length_secret_key);
    uint8_t *signature = malloc(sig->length_signature);
    size_t sig_len = 0;

    /* Fixed message (32 bytes, like a hash) */
    uint8_t message[32];
    memset(message, 0xAB, sizeof(message));

    if (!pk || !sk || !signature) { fprintf(stderr, "ERROR: OOM\n"); return -1; }

    /* Generate keypair once */
    if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) return -1;

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        OQS_STATUS rc = OQS_SIG_sign(sig, signature, &sig_len, message, sizeof(message), sk);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (rc != OQS_SUCCESS) {
            fprintf(stderr, "ERROR: sign failed\n");
            free(pk); free(sk); free(signature); OQS_SIG_free(sig);
            return -1;
        }
        stats_add(st, timespec_diff_us(&t0, &t1));
    }

    free(pk); free(sk); free(signature); OQS_SIG_free(sig);
    return 0;
}

static int bench_verify(const char *algo_name, int iterations, Stats *st) {
    OQS_SIG *sig = OQS_SIG_new(algo_name);
    if (!sig) return -1;

    uint8_t *pk = malloc(sig->length_public_key);
    uint8_t *sk = malloc(sig->length_secret_key);
    uint8_t *signature = malloc(sig->length_signature);
    size_t sig_len = 0;

    uint8_t message[32];
    memset(message, 0xAB, sizeof(message));

    if (!pk || !sk || !signature) return -1;
    if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) return -1;
    if (OQS_SIG_sign(sig, signature, &sig_len, message, sizeof(message), sk) != OQS_SUCCESS) return -1;

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        OQS_STATUS rc = OQS_SIG_verify(sig, message, sizeof(message), signature, sig_len, pk);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (rc != OQS_SUCCESS) {
            fprintf(stderr, "ERROR: verify failed\n");
            free(pk); free(sk); free(signature); OQS_SIG_free(sig);
            return -1;
        }
        stats_add(st, timespec_diff_us(&t0, &t1));
    }

    free(pk); free(sk); free(signature); OQS_SIG_free(sig);
    return 0;
}

/* ---- Print sizes ---- */
static void print_sizes(const char *algo_name) {
    OQS_SIG *sig = OQS_SIG_new(algo_name);
    if (!sig) return;
    printf("  \"pk_bytes\": %zu,\n", sig->length_public_key);
    printf("  \"sk_bytes\": %zu,\n", sig->length_secret_key);
    printf("  \"sig_bytes\": %zu,\n", sig->length_signature);
    OQS_SIG_free(sig);
}

/* ---- JSON output ---- */
static void print_stats_json(const char *name, Stats *st) {
    printf("  \"%s_us\": {\"min\": %.1f, \"avg\": %.1f, \"max\": %.1f, \"stddev\": %.1f}",
           name, st->min, stats_avg(st), st->max, stats_stddev(st));
}

static void print_energy_json(const char *op, Stats *st) {
    double avg_us = stats_avg(st);
    for (int i = 0; i < NUM_DEVICES; i++) {
        if (i > 0) printf(",\n");
        printf("    \"%s_%s\": %.4f",
               op, DEVICE_POWER[i].name,
               energy_mj(avg_us, DEVICE_POWER[i].power_mw));
    }
}

/* ---- Main ---- */
int main(int argc, char *argv[]) {
    const char *algo = "ML-DSA-44";
    const char *mode = "all";
    int iterations = 100;
    int node_id = 0;

    /* Parse args */
    if (argc >= 2) mode = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) algo = argv[++i];
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) node_id = atoi(argv[++i]);
    }

    if (strcmp(mode, "list") == 0) {
        printf("Supported algorithms:\n");
        printf("  ML-DSA-44  (NIST Level 2)\n");
        printf("  ML-DSA-65  (NIST Level 3)\n");
        printf("  ML-DSA-87  (NIST Level 5)\n");
        return 0;
    }

    Stats keygen_st, sign_st, verify_st;
    int do_keygen = (strcmp(mode, "keygen") == 0 || strcmp(mode, "all") == 0);
    int do_sign   = (strcmp(mode, "sign") == 0   || strcmp(mode, "all") == 0);
    int do_verify = (strcmp(mode, "verify") == 0 || strcmp(mode, "all") == 0);

    if (do_keygen && bench_keygen(algo, iterations, &keygen_st) != 0) return 1;
    if (do_sign   && bench_sign(algo, iterations, &sign_st) != 0) return 1;
    if (do_verify && bench_verify(algo, iterations, &verify_st) != 0) return 1;

    /* Output JSON */
    printf("{\n");
    printf("  \"algorithm\": \"%s\",\n", algo);
    printf("  \"node_id\": %d,\n", node_id);
    printf("  \"iterations\": %d,\n", iterations);
    print_sizes(algo);

    int first = 1;
    if (do_keygen) {
        if (!first) printf(",\n"); first = 0;
        print_stats_json("keygen", &keygen_st);
    }
    if (do_sign) {
        if (!first) printf(",\n"); first = 0;
        print_stats_json("sign", &sign_st);
    }
    if (do_verify) {
        if (!first) printf(",\n"); first = 0;
        print_stats_json("verify", &verify_st);
    }

    printf(",\n  \"energy_mj\": {\n");
    first = 1;
    if (do_keygen) {
        if (!first) printf(",\n"); first = 0;
        print_energy_json("keygen", &keygen_st);
    }
    if (do_sign) {
        if (!first) printf(",\n"); first = 0;
        print_energy_json("sign", &sign_st);
    }
    if (do_verify) {
        if (!first) printf(",\n"); first = 0;
        print_energy_json("verify", &verify_st);
    }
    printf("\n  },\n");

    /* ARM reference latencies from pqm4 */
    printf("  \"arm_pqm4_reference\": [\n");
    for (int i = 0; i < NUM_ARM_DEVICES; i++) {
        const ArmDevice *ad = &ARM_DEVICES[i];
        double sign_us   = ad->sign_cycles_k * 1000.0 / ad->clock_mhz;
        double verify_us = ad->verify_cycles_k * 1000.0 / ad->clock_mhz;
        double keygen_us = ad->keygen_cycles_k * 1000.0 / ad->clock_mhz;
        /* Find matching power device */
        double pw = 0;
        for (int j = 0; j < NUM_DEVICES; j++) {
            if (strcmp(ad->name, DEVICE_POWER[j].name) == 0) { pw = DEVICE_POWER[j].power_mw; break; }
        }
        printf("    {\"device\": \"%s\", \"clock_mhz\": %.0f, "
               "\"sign_us\": %.0f, \"verify_us\": %.0f, \"keygen_us\": %.0f, "
               "\"sign_energy_mj\": %.4f, \"verify_energy_mj\": %.4f}",
               ad->name, ad->clock_mhz, sign_us, verify_us, keygen_us,
               sign_us * 1e-6 * pw, verify_us * 1e-6 * pw);
        if (i < NUM_ARM_DEVICES - 1) printf(",");
        printf("\n");
    }
    printf("  ]\n");
    printf("}\n");

    return 0;
}
