/*
 * bilinear-benchmark.c — BLS12-381 (Bilinear Pairings) Benchmark
 *
 * Baseline: Chữ ký dựa trên Bilinear Pairings (bài báo gốc)
 * So sánh với ML-DSA (Lattice) trong hệ thống đề xuất.
 *
 * Sử dụng thư viện blst (BLS12-381) — https://github.com/supranational/blst
 *
 * Compile:
 *   gcc -static -O2 bilinear-benchmark.c -I/blst/bindings -L/blst -lblst -o bilinear-benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "blst.h"

/* ---- Energy model constants (mW) ---- */
#define POWER_ESP32_MW      160.0
#define POWER_CORTEX_M4_MW   50.0

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
    return time_us * 1e-6 * power_mw;
}

/* ---- BLS12-381 Benchmark ---- */

/* KeyGen: generate secret scalar + public key on G1 */
static int bench_keygen(int iterations, Stats *st) {
    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        /* Generate random 32-byte secret (IKM) */
        byte ikm[32];
        for (int j = 0; j < 32; j++) ikm[j] = (byte)(rand() & 0xFF);

        blst_scalar sk;
        blst_p1 pk;
        blst_p1_affine pk_aff;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Derive secret key from IKM */
        blst_keygen(&sk, ikm, 32, NULL, 0);
        /* Compute public key: pk = sk * G1 */
        blst_sk_to_pk_in_g1(&pk_aff, &sk);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        stats_add(st, timespec_diff_us(&t0, &t1));
    }
    return 0;
}

/* Sign: BLS sign a message using secret key */
static int bench_sign(int iterations, Stats *st) {
    /* Generate a keypair first */
    byte ikm[32];
    for (int j = 0; j < 32; j++) ikm[j] = (byte)(42 + j);

    blst_scalar sk;
    blst_keygen(&sk, ikm, 32, NULL, 0);

    /* Message to sign */
    byte message[32];
    memset(message, 0xAB, sizeof(message));
    const char *dst = "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        blst_p2 sig_point;
        blst_p2_affine sig_aff;
        byte sig_bytes[96];

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Hash message to G2 curve point */
        blst_hash_to_g2(&sig_point, message, sizeof(message),
                         (const byte*)dst, strlen(dst), NULL, 0);
        /* Multiply by secret key: sig = sk * H(m) */
        blst_sign_pk_in_g1(&sig_point, &sig_point, &sk);
        /* Compress to bytes */
        blst_p2_compress(sig_bytes, &sig_point);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        stats_add(st, timespec_diff_us(&t0, &t1));
    }
    return 0;
}

/* Verify: BLS verify requires the pairing check — the expensive part */
static int bench_verify(int iterations, Stats *st) {
    /* Generate keypair */
    byte ikm[32];
    for (int j = 0; j < 32; j++) ikm[j] = (byte)(42 + j);

    blst_scalar sk;
    blst_keygen(&sk, ikm, 32, NULL, 0);

    blst_p1_affine pk_aff;
    blst_sk_to_pk_in_g1(&pk_aff, &sk);

    /* Sign a message */
    byte message[32];
    memset(message, 0xAB, sizeof(message));
    const char *dst = "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

    blst_p2 sig_point;
    blst_hash_to_g2(&sig_point, message, sizeof(message),
                     (const byte*)dst, strlen(dst), NULL, 0);
    blst_sign_pk_in_g1(&sig_point, &sig_point, &sk);

    blst_p2_affine sig_aff;
    blst_p2_to_affine(&sig_aff, &sig_point);

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Core verify: pairing check e(pk, H(m)) == e(G1, sig) */
        BLST_ERROR result = blst_core_verify_pk_in_g1(
            &pk_aff, &sig_aff,
            1,  /* hash_or_encode = 1 (hash) */
            message, sizeof(message),
            (const byte*)dst, strlen(dst),
            NULL, 0
        );

        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (result != BLST_SUCCESS) {
            fprintf(stderr, "ERROR: BLS verify failed (code %d)\n", result);
            return -1;
        }
        stats_add(st, timespec_diff_us(&t0, &t1));
    }
    return 0;
}

/* ---- Main ---- */
int main(int argc, char *argv[]) {
    int iterations = 100;
    int node_id = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) node_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "list") == 0) {
            printf("BLS12-381 (Bilinear Pairings baseline)\n");
            return 0;
        }
    }

    srand(node_id + 12345);

    Stats keygen_st, sign_st, verify_st;

    if (bench_keygen(iterations, &keygen_st) != 0) return 1;
    if (bench_sign(iterations, &sign_st) != 0) return 1;
    if (bench_verify(iterations, &verify_st) != 0) return 1;

    /* JSON output (same format as pqc-benchmark) */
    printf("{\n");
    printf("  \"algorithm\": \"BLS12-381\",\n");
    printf("  \"node_id\": %d,\n", node_id);
    printf("  \"iterations\": %d,\n", iterations);
    printf("  \"pk_bytes\": 48,\n");
    printf("  \"sk_bytes\": 32,\n");
    printf("  \"sig_bytes\": 96,\n");

    printf("  \"keygen_us\": {\"min\": %.1f, \"avg\": %.1f, \"max\": %.1f, \"stddev\": %.1f},\n",
           keygen_st.min, stats_avg(&keygen_st), keygen_st.max, stats_stddev(&keygen_st));
    printf("  \"sign_us\": {\"min\": %.1f, \"avg\": %.1f, \"max\": %.1f, \"stddev\": %.1f},\n",
           sign_st.min, stats_avg(&sign_st), sign_st.max, stats_stddev(&sign_st));
    printf("  \"verify_us\": {\"min\": %.1f, \"avg\": %.1f, \"max\": %.1f, \"stddev\": %.1f},\n",
           verify_st.min, stats_avg(&verify_st), verify_st.max, stats_stddev(&verify_st));

    printf("  \"energy_mj\": {\n");
    printf("    \"keygen_esp32\": %.4f, \"keygen_cortexm4\": %.4f,\n",
           energy_mj(stats_avg(&keygen_st), POWER_ESP32_MW),
           energy_mj(stats_avg(&keygen_st), POWER_CORTEX_M4_MW));
    printf("    \"sign_esp32\": %.4f, \"sign_cortexm4\": %.4f,\n",
           energy_mj(stats_avg(&sign_st), POWER_ESP32_MW),
           energy_mj(stats_avg(&sign_st), POWER_CORTEX_M4_MW));
    printf("    \"verify_esp32\": %.4f, \"verify_cortexm4\": %.4f\n",
           energy_mj(stats_avg(&verify_st), POWER_ESP32_MW),
           energy_mj(stats_avg(&verify_st), POWER_CORTEX_M4_MW));
    printf("  }\n");
    printf("}\n");

    return 0;
}
