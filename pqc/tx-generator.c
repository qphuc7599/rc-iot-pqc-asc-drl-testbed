/*
 * tx-generator.c — Sinh giao dịch PQC + gửi qua mạng NS-3
 *
 * Mỗi IoT node chạy chương trình này để:
 *   1. Tạo cặp khóa ML-DSA (1 lần)
 *   2. Lặp: Tạo transaction data → Ký bằng ML-DSA → Gửi UDP tới Gateway
 *
 * v2: Thêm device-specific CPU scaling factor
 *     Docker CPU throttling không chính xác cho cryptographic ops trên x86
 *     → dùng artificial delay để mô phỏng tốc độ ARM thực tế
 *
 * Usage:
 *   ./tx-generator --node-id 1 --gateway 10.1.1.100 --port 9000
 *                  --rate 10 --duration 60 --algo ML-DSA-44
 *
 * Packet format (binary, v2):
 *   [4B node_id] [8B first_tx_timestamp_us] [8B last_tx_timestamp_us]
 *   [8B timestamp_us] [4B tx_count] [32B state_hash] [4B sig_len] [sig]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <oqs/oqs.h>

/* OpenSSL SHA-256 for state root hashing */
#include <openssl/sha.h>

#define MAX_PKT_SIZE 8192
#define DATA_SIZE 32
#define NUM_DEVICE_TYPES 7
#define BATCH_PER_CHANNEL 50      /* State Channel: gop 50 TX thanh 1 packet */
#define CHANNEL_TIMEOUT_US 2000000 /* 2s timeout — cho phep batch day du 50 TX */

/*
 * State Channel Update packet format:
 *   [4B node_id] [8B first_tx_timestamp] [8B last_tx_timestamp]
 *   [8B timestamp] [4B tx_count] [32B state_hash] [4B sig_len] [sig]
 * Gateway verifies 1 ML-DSA signature on state_hash → accepts tx_count TXs
 */
typedef struct {
    uint32_t node_id;
    uint64_t first_tx_timestamp_us;
    uint64_t last_tx_timestamp_us;
    uint64_t timestamp_us;
    uint32_t tx_count;              /* so TX off-chain trong batch nay */
    uint8_t  state_hash[DATA_SIZE]; /* SHA-256 root hash cua tat ca TX */
    uint32_t sig_len;
    /* followed by signature bytes */
} __attribute__((packed)) TxChannelPacket;

/*
 * Device-specific ML-DSA-44 cryptographic latency (microseconds).
 *
 * Source: pqm4 benchmark framework (https://github.com/mupq/pqm4)
 *
 * Cycle counts for ML-DSA-44 (NIST Level 2):
 *   m4f optimized (Cortex-M4 w/ DSP):  KeyGen 1,426k  Sign 4,336k  Verify 1,579k
 *   clean reference (no DSP):           KeyGen 1,870k  Sign 7,280k  Verify 2,060k
 *
 * Latency = cycles / clock_freq_MHz
 *
 * Cortex-M4 devices:  pqm4 m4f optimized cycles
 * Cortex-M7 devices:  pqm4 m4f cycles (conservative; M7 pipeline may be faster)
 * Cortex-M0+ devices: pqm4 clean reference cycles (no DSP instructions)
 * Xtensa devices:     estimated 1.3x clean cycles (different ISA, no NTT-friendly DSP)
 *
 * v3: Replaced cpu_scale_factor with hard-coded pqm4 latencies.
 *     Previous approach (x86_time * factor) was incorrect because cgroup CPU
 *     throttling does not model ARM micro-architecture (in-order pipeline,
 *     no SIMD/AVX, limited memory bandwidth). Hard-coded delays from pqm4
 *     provide accurate device-specific latencies independent of host CPU.
 */
typedef struct {
    const char *name;
    uint32_t sign_delay_us;     /* ML-DSA-44 sign latency (pqm4) */
    uint32_t verify_delay_us;   /* ML-DSA-44 verify latency (pqm4) */
    double   power_mw;          /* Active power draw (datasheet) */
} DeviceProfile;

static const DeviceProfile DEVICE_PROFILES[NUM_DEVICE_TYPES] = {
    /* name           sign_us  verify_us  power_mw   paper calibration table              */
    {"ESP32",         24645,    8885,     160.0},
    {"ESP32-S3",      21359,    7700,     170.0},
    {"STM32L4-M4",    49289,   17770,      30.0},
    {"STM32F4-M4",    23471,    8462,      50.0},
    {"STM32H7-M7",     4518,    1629,      90.0},
    {"nRF52840",      61611,   22213,      23.0},
    {"RP2040",        54848,   19774,      45.0},
};

static const DeviceProfile* get_device_profile(int node_id) {
    /*
     * Must match generate.py / docker-compose.yml assignment order:
     *   1-25 ESP32, 26-35 ESP32-S3, 36-50 STM32L4,
     *   51-65 STM32F4, 66-75 STM32H7, 76-85 nRF52840,
     *   86-100 RP2040.
     */
    if (node_id <= 25) return &DEVICE_PROFILES[0];
    if (node_id <= 35) return &DEVICE_PROFILES[1];
    if (node_id <= 50) return &DEVICE_PROFILES[2];
    if (node_id <= 65) return &DEVICE_PROFILES[3];
    if (node_id <= 75) return &DEVICE_PROFILES[4];
    if (node_id <= 85) return &DEVICE_PROFILES[5];
    return &DEVICE_PROFILES[6];
}

static uint64_t htonll(uint64_t value) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((uint64_t)htonl((uint32_t)(value & 0xffffffffULL)) << 32) |
           (uint64_t)htonl((uint32_t)(value >> 32));
#else
    return value;
#endif
}

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

static void random_data(uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i++)
        buf[i] = rand() & 0xFF;
}

int main(int argc, char *argv[]) {
    /* Defaults */
    int node_id = 1;
    const char *gateway_ip = "10.1.1.100";
    int port = 9000;
    int rate = 10;       /* tx/sec */
    int duration = 60;   /* seconds */
    const char *algo = "ML-DSA-44";
    int batch_size = BATCH_PER_CHANNEL;  /* 50 = state channel, 1 = on-chain */
    int emulated_sig_bytes = 0;           /* >0: synthetic signature length */
    int emulated_sign_delay_us = -1;      /* >=0: override signing delay */

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--node-id") == 0 && i+1 < argc) node_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gateway") == 0 && i+1 < argc) gateway_ip = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i+1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rate") == 0 && i+1 < argc) rate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--duration") == 0 && i+1 < argc) duration = atoi(argv[++i]);
        else if (strcmp(argv[i], "--algo") == 0 && i+1 < argc) algo = argv[++i];
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--emulated-sig-bytes") == 0 && i+1 < argc) emulated_sig_bytes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--emulated-sign-us") == 0 && i+1 < argc) emulated_sign_delay_us = atoi(argv[++i]);
    }

    srand(time(NULL) ^ node_id);

    /* Get device profile */
    const DeviceProfile *dev = get_device_profile(node_id);

    const int use_emulated_signature = (emulated_sig_bytes > 0);
    if (rate <= 0 || duration <= 0 || batch_size <= 0) {
        fprintf(stderr, "ERROR: rate, duration, and batch must be positive\n");
        return 1;
    }
    if (use_emulated_signature && emulated_sign_delay_us < 0) {
        fprintf(stderr, "ERROR: --emulated-sign-us is required with --emulated-sig-bytes\n");
        return 1;
    }
    OQS_SIG *sig = NULL;
    uint8_t *pk = NULL;
    uint8_t *sk = NULL;

    fprintf(stderr, "[TX-GEN %d] Device=%s, sign_delay=%uus (pqm4), power=%.0fmW\n",
            node_id, dev->name, dev->sign_delay_us, dev->power_mw);

    if (use_emulated_signature) {
        fprintf(stderr, "[TX-GEN %d] Emulated signature mode: sig=%dB, sign_delay=%dus\n",
                node_id, emulated_sig_bytes, emulated_sign_delay_us);
    } else {
        /* Init ML-DSA */
        sig = OQS_SIG_new(algo);
        if (!sig) { fprintf(stderr, "ERROR: algo %s not found\n", algo); return 1; }

        pk = malloc(sig->length_public_key);
        sk = malloc(sig->length_secret_key);
        if (!pk || !sk) { fprintf(stderr, "ERROR: OOM\n"); return 1; }

        /* Keygen once */
        fprintf(stderr, "[TX-GEN %d] Keygen %s...\n", node_id, algo);
        if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) {
            fprintf(stderr, "ERROR: keygen failed\n"); return 1;
        }
    }
    fprintf(stderr, "[TX-GEN %d] Ready. Rate=%d tx/s, Duration=%ds, Gateway=%s:%d\n",
            node_id, rate, duration, gateway_ip, port);

    /* Setup UDP socket */
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); return 1; }

    struct sockaddr_in gw_addr;
    memset(&gw_addr, 0, sizeof(gw_addr));
    gw_addr.sin_family = AF_INET;
    gw_addr.sin_port = htons(port);
    inet_pton(AF_INET, gateway_ip, &gw_addr.sin_addr);

    /* Packet buffer */
    uint8_t pkt_buf[MAX_PKT_SIZE];
    size_t signature_capacity = use_emulated_signature
        ? (size_t)emulated_sig_bytes
        : sig->length_signature;
    if (sizeof(TxChannelPacket) + signature_capacity > MAX_PKT_SIZE) {
        fprintf(stderr, "ERROR: packet too large (%zu > %d)\n",
                sizeof(TxChannelPacket) + signature_capacity, MAX_PKT_SIZE);
        return 1;
    }
    uint8_t *signature = malloc(signature_capacity);
    if (!signature) { fprintf(stderr, "ERROR: OOM\n"); return 1; }

    /*
     * === STATE CHANNEL TX LOOP ===
     * State Root Batching (Layer-2 Rollup pattern):
     *   1. Hash N sensor readings into rolling state_hash (very lightweight)
     *   2. Sign state_hash ONCE with ML-DSA (1 crypto op per batch)
     *   3. Send 1 packet = [tx_count, state_hash, 1_signature]
     *   4. Gateway verifies 1 signature → accepts N TXs → O(1)
     *
     * Dual trigger: batch full (50 TX) OR timeout (200ms)
     */
    uint64_t start = now_us();
    uint64_t end_time = start + (uint64_t)duration * 1000000ULL;
    uint64_t interval_us = 1000000ULL / rate;  /* delay giua cac TX off-chain */
    uint32_t total_tx_count = 0;               /* tong TX da gui */
    uint32_t total_channels = 0;               /* tong state channel updates */
    uint64_t total_sign_us = 0;
    uint64_t total_hash_us = 0;

    /* State Channel state */
    uint32_t channel_tx_count = 0;             /* TX trong channel hien tai */
    uint64_t channel_first_tx_time = 0;
    uint64_t channel_last_tx_time = 0;
    uint8_t  state_hash[DATA_SIZE];            /* rolling hash */
    memset(state_hash, 0, DATA_SIZE);
    uint64_t channel_open_time = now_us();     /* thoi diem mo channel */

    /* Output header to stdout (CSV) */
    printf("channel_id,node_id,device_type,tx_count,first_tx_timestamp_us,last_tx_timestamp_us,timestamp_us,"
           "hash_time_us,sign_time_us,emulated_sign_us,pkt_size\n");

    fprintf(stderr, "[TX-GEN %d] State Channel mode: batch=%d, timeout=%dms\n",
            node_id, batch_size, CHANNEL_TIMEOUT_US / 1000);

    while (now_us() < end_time) {
        uint64_t tx_start = now_us();
        if (channel_tx_count == 0) {
            channel_first_tx_time = tx_start;
        }
        channel_last_tx_time = tx_start;

        /* --- OFF-CHAIN: Tao data va hash vao state (CUC NHE, khong ky) --- */
        uint8_t data[DATA_SIZE];
        random_data(data, DATA_SIZE);

        /* Rolling hash: state_hash = SHA-256(state_hash || data)
         * Chi dung SHA-256 (rat nhe), KHONG ky ML-DSA o day */
        uint64_t hash_start = now_us();
        SHA256_CTX sha_ctx;
        SHA256_Init(&sha_ctx);
        SHA256_Update(&sha_ctx, state_hash, DATA_SIZE);  /* hash cu */
        SHA256_Update(&sha_ctx, data, DATA_SIZE);        /* data moi */
        SHA256_Final(state_hash, &sha_ctx);
        uint64_t hash_time = now_us() - hash_start;
        total_hash_us += hash_time;

        channel_tx_count++;
        total_tx_count++;

        /* --- DUAL TRIGGER: Du batch HOAC qua timeout → dong channel --- */
        uint64_t now = now_us();
        int batch_full = (channel_tx_count >= (uint32_t)batch_size);
        int timed_out = ((now - channel_open_time) >= CHANNEL_TIMEOUT_US);

        if (batch_full || timed_out) {
            size_t sig_len = 0;
            uint64_t sign_start = now_us();
            uint64_t raw_sign_time = 0;

            if (use_emulated_signature) {
                sig_len = (size_t)emulated_sig_bytes;
                for (size_t j = 0; j < sig_len; j++) {
                    signature[j] = (uint8_t)(state_hash[j % DATA_SIZE] ^ (uint8_t)node_id ^ (uint8_t)j);
                }
                raw_sign_time = now_us() - sign_start;
            } else {
                /* === Sign ML-DSA exactly once over state_hash === */
                OQS_STATUS rc = OQS_SIG_sign(sig, signature, &sig_len,
                                              state_hash, DATA_SIZE, sk);
                raw_sign_time = now_us() - sign_start;

                if (rc != OQS_SUCCESS) {
                    fprintf(stderr, "WARN: sign failed at channel %u\n", total_channels);
                    /* Reset channel */
                    channel_tx_count = 0;
                    memset(state_hash, 0, DATA_SIZE);
                    channel_open_time = now_us();
                    continue;
                }
            }

            uint32_t target_sign_us = (emulated_sign_delay_us >= 0)
                ? (uint32_t)emulated_sign_delay_us
                : dev->sign_delay_us;
            if (target_sign_us > (uint32_t)raw_sign_time) {
                usleep(target_sign_us - (unsigned int)raw_sign_time);
            }
            uint64_t emulated_sign_time = (uint64_t)target_sign_us;
            total_sign_us += emulated_sign_time;

            /* Build State Channel Update packet */
            TxChannelPacket *hdr = (TxChannelPacket *)pkt_buf;
            hdr->node_id = htonl(node_id);
            uint64_t timestamp = now_us();
            hdr->first_tx_timestamp_us = htonll(channel_first_tx_time);
            hdr->last_tx_timestamp_us = htonll(channel_last_tx_time);
            hdr->timestamp_us = htonll(timestamp);
            hdr->tx_count = htonl(channel_tx_count);  /* 50 TX trong batch */
            memcpy(hdr->state_hash, state_hash, DATA_SIZE);
            hdr->sig_len = htonl((uint32_t)sig_len);
            memcpy(pkt_buf + sizeof(TxChannelPacket), signature, sig_len);
            size_t pkt_size = sizeof(TxChannelPacket) + sig_len;

            /* Send 1 UDP packet = 50 TX */
            sendto(sockfd, pkt_buf, pkt_size, 0,
                   (struct sockaddr *)&gw_addr, sizeof(gw_addr));

            total_channels++;

            /* CSV log */
            printf("%u,%d,%s,%u,%lu,%lu,%lu,%lu,%lu,%lu,%zu\n",
                   total_channels, node_id, dev->name,
                   channel_tx_count,
                   (unsigned long)channel_first_tx_time,
                   (unsigned long)channel_last_tx_time,
                   (unsigned long)timestamp,
                   (unsigned long)hash_time,
                   (unsigned long)raw_sign_time,
                   (unsigned long)emulated_sign_time,
                   pkt_size);

            /* Reset channel for next batch */
            channel_tx_count = 0;
            channel_first_tx_time = 0;
            channel_last_tx_time = 0;
            memset(state_hash, 0, DATA_SIZE);
            channel_open_time = now_us();
        }

        /* Rate limiting between off-chain TX */
        uint64_t elapsed = now_us() - tx_start;
        if (elapsed < interval_us) {
            usleep(interval_us - elapsed);
        }
    }

    /* Flush remaining TX in channel (partial batch) */
    if (channel_tx_count > 0) {
        size_t sig_len = 0;
        int sign_ok = 1;
        if (use_emulated_signature) {
            sig_len = (size_t)emulated_sig_bytes;
            for (size_t j = 0; j < sig_len; j++) {
                signature[j] = (uint8_t)(state_hash[j % DATA_SIZE] ^ (uint8_t)node_id ^ (uint8_t)j);
            }
            if (emulated_sign_delay_us > 0) {
                usleep((useconds_t)emulated_sign_delay_us);
            }
        } else {
            sign_ok = (OQS_SIG_sign(sig, signature, &sig_len,
                                    state_hash, DATA_SIZE, sk) == OQS_SUCCESS);
        }
        if (sign_ok) {
            TxChannelPacket *hdr = (TxChannelPacket *)pkt_buf;
            hdr->node_id = htonl(node_id);
            hdr->first_tx_timestamp_us = htonll(channel_first_tx_time);
            hdr->last_tx_timestamp_us = htonll(channel_last_tx_time);
            hdr->timestamp_us = htonll(now_us());
            hdr->tx_count = htonl(channel_tx_count);
            memcpy(hdr->state_hash, state_hash, DATA_SIZE);
            hdr->sig_len = htonl((uint32_t)sig_len);
            memcpy(pkt_buf + sizeof(TxChannelPacket), signature, sig_len);
            size_t pkt_size = sizeof(TxChannelPacket) + sig_len;
            sendto(sockfd, pkt_buf, pkt_size, 0,
                   (struct sockaddr *)&gw_addr, sizeof(gw_addr));
            total_channels++;
            total_tx_count += 0; /* already counted */
        }
    }

    /* Summary to stderr */
    double total_sec = (double)(now_us() - start) / 1e6;
    fprintf(stderr, "\n[TX-GEN %d] STATE CHANNEL DONE:\n", node_id);
    fprintf(stderr, "  Total TX (off-chain): %u in %.1fs (%.1f TPS)\n",
            total_tx_count, total_sec, total_tx_count / total_sec);
    fprintf(stderr, "  Channel updates sent: %u (avg %.1f TX/channel)\n",
            total_channels,
            total_channels > 0 ? (double)total_tx_count / total_channels : 0);
    fprintf(stderr, "  Device=%s, pqm4 sign: %u us, Avg actual: %.1f us (1 per channel)\n",
            dev->name, dev->sign_delay_us,
            total_channels > 0 ? (double)total_sign_us / total_channels : 0);
    fprintf(stderr, "  Avg hash/TX: %.1f us (off-chain, no crypto)\n",
            total_tx_count > 0 ? (double)total_hash_us / total_tx_count : 0);
    fprintf(stderr, "  Energy/sign: %.4f mJ (pqm4 latency x device power)\n",
            dev->sign_delay_us * 1e-6 * dev->power_mw);

    free(pk); free(sk); free(signature);
    if (sig) OQS_SIG_free(sig);
    close(sockfd);
    return 0;
}
