/*
 * ecdsa-benchmark.c — ECDSA-P256 & Ed25519 Benchmark cho IoT Containers
 * Chay CUNG testbed voi ML-DSA de fair comparison
 *
 * Compile: gcc -static -O2 ecdsa-benchmark.c -lssl -lcrypto -lm -o ecdsa-benchmark
 *
 * Usage:
 *   ./ecdsa-benchmark all --algo ecdsa-p256 --iterations 100 --node-id 1
 *   ./ecdsa-benchmark all --algo ed25519    --iterations 100 --node-id 1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <openssl/evp.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/rand.h>

/* ---- Helpers (same as pqc-benchmark) ---- */

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

/* ---- Benchmark using OpenSSL EVP ---- */

static int get_evp_type(const char *algo) {
    if (strcmp(algo, "ecdsa-p256") == 0) return EVP_PKEY_EC;
    if (strcmp(algo, "ed25519") == 0)    return EVP_PKEY_ED25519;
    return -1;
}

static int get_pk_bytes(const char *algo) {
    if (strcmp(algo, "ecdsa-p256") == 0) return 64;
    if (strcmp(algo, "ed25519") == 0)    return 32;
    return 0;
}

static int get_sig_bytes(const char *algo) {
    if (strcmp(algo, "ecdsa-p256") == 0) return 72;  /* DER encoded max */
    if (strcmp(algo, "ed25519") == 0)    return 64;
    return 0;
}

static EVP_PKEY *generate_key(const char *algo) {
    EVP_PKEY *pkey = NULL;
    EVP_PKEY_CTX *ctx = NULL;

    if (strcmp(algo, "ecdsa-p256") == 0) {
        ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, NULL);
        if (!ctx) return NULL;
        EVP_PKEY_keygen_init(ctx);
        EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1);
        EVP_PKEY_keygen(ctx, &pkey);
    } else if (strcmp(algo, "ed25519") == 0) {
        ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, NULL);
        if (!ctx) return NULL;
        EVP_PKEY_keygen_init(ctx);
        EVP_PKEY_keygen(ctx, &pkey);
    }

    if (ctx) EVP_PKEY_CTX_free(ctx);
    return pkey;
}

static int bench_keygen(const char *algo, int iterations, Stats *st) {
    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        EVP_PKEY *key = generate_key(algo);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (!key) {
            fprintf(stderr, "ERROR: keygen failed at %d\n", i);
            return -1;
        }
        stats_add(st, timespec_diff_us(&t0, &t1));
        EVP_PKEY_free(key);
    }
    return 0;
}

static int bench_sign(const char *algo, int iterations, Stats *st) {
    EVP_PKEY *key = generate_key(algo);
    if (!key) return -1;

    unsigned char message[32];
    RAND_bytes(message, sizeof(message));

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
        unsigned char *sig = NULL;
        size_t sig_len = 0;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (strcmp(algo, "ecdsa-p256") == 0) {
            EVP_DigestSignInit(md_ctx, NULL, EVP_sha256(), NULL, key);
        } else {
            EVP_DigestSignInit(md_ctx, NULL, NULL, NULL, key);
        }
        EVP_DigestSign(md_ctx, NULL, &sig_len, message, sizeof(message));
        sig = malloc(sig_len);
        EVP_DigestSign(md_ctx, sig, &sig_len, message, sizeof(message));

        clock_gettime(CLOCK_MONOTONIC, &t1);

        stats_add(st, timespec_diff_us(&t0, &t1));
        free(sig);
        EVP_MD_CTX_free(md_ctx);
    }

    EVP_PKEY_free(key);
    return 0;
}

static int bench_verify(const char *algo, int iterations, Stats *st) {
    EVP_PKEY *key = generate_key(algo);
    if (!key) return -1;

    unsigned char message[32];
    RAND_bytes(message, sizeof(message));

    /* Sign once for verification benchmark */
    EVP_MD_CTX *sign_ctx = EVP_MD_CTX_new();
    unsigned char *sig = NULL;
    size_t sig_len = 0;

    if (strcmp(algo, "ecdsa-p256") == 0) {
        EVP_DigestSignInit(sign_ctx, NULL, EVP_sha256(), NULL, key);
    } else {
        EVP_DigestSignInit(sign_ctx, NULL, NULL, NULL, key);
    }
    EVP_DigestSign(sign_ctx, NULL, &sig_len, message, sizeof(message));
    sig = malloc(sig_len);
    EVP_DigestSign(sign_ctx, sig, &sig_len, message, sizeof(message));
    EVP_MD_CTX_free(sign_ctx);

    stats_init(st);

    for (int i = 0; i < iterations; i++) {
        EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (strcmp(algo, "ecdsa-p256") == 0) {
            EVP_DigestVerifyInit(md_ctx, NULL, EVP_sha256(), NULL, key);
        } else {
            EVP_DigestVerifyInit(md_ctx, NULL, NULL, NULL, key);
        }
        int rc = EVP_DigestVerify(md_ctx, sig, sig_len, message, sizeof(message));

        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (rc != 1) {
            fprintf(stderr, "ERROR: verify failed\n");
            free(sig); EVP_PKEY_free(key); EVP_MD_CTX_free(md_ctx);
            return -1;
        }
        stats_add(st, timespec_diff_us(&t0, &t1));
        EVP_MD_CTX_free(md_ctx);
    }

    free(sig);
    EVP_PKEY_free(key);
    return 0;
}

/* ---- JSON output ---- */
static void print_stats_json(const char *name, Stats *st) {
    printf("  \"%s_us\": {\"min\": %.1f, \"avg\": %.1f, \"max\": %.1f, \"stddev\": %.1f}",
           name, st->min, stats_avg(st), st->max, stats_stddev(st));
}

/* ---- Main ---- */
int main(int argc, char *argv[]) {
    const char *algo = "ecdsa-p256";
    const char *mode = "all";
    int iterations = 100;
    int node_id = 0;

    if (argc >= 2) mode = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) algo = argv[++i];
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) node_id = atoi(argv[++i]);
    }

    if (strcmp(mode, "list") == 0) {
        printf("Supported algorithms:\n");
        printf("  ecdsa-p256  (NIST P-256, secp256r1)\n");
        printf("  ed25519    (EdDSA Curve25519)\n");
        return 0;
    }

    if (get_evp_type(algo) < 0) {
        fprintf(stderr, "ERROR: Unknown algorithm '%s'\n", algo);
        return 1;
    }

    Stats keygen_st, sign_st, verify_st;

    if (bench_keygen(algo, iterations, &keygen_st) != 0) return 1;
    if (bench_sign(algo, iterations, &sign_st) != 0) return 1;
    if (bench_verify(algo, iterations, &verify_st) != 0) return 1;

    /* JSON output */
    printf("{\n");
    printf("  \"algorithm\": \"%s\",\n", algo);
    printf("  \"node_id\": %d,\n", node_id);
    printf("  \"iterations\": %d,\n", iterations);
    printf("  \"pk_bytes\": %d,\n", get_pk_bytes(algo));
    printf("  \"sk_bytes\": %d,\n", get_pk_bytes(algo));  /* approx */
    printf("  \"sig_bytes\": %d,\n", get_sig_bytes(algo));

    print_stats_json("keygen", &keygen_st);
    printf(",\n");
    print_stats_json("sign", &sign_st);
    printf(",\n");
    print_stats_json("verify", &verify_st);

    printf("\n}\n");

    return 0;
}
