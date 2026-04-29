/*
 * renderer_mpi.c
 * MPI-parallel renderer: root reads records and scatters them to ranks;
 * each rank renders its chunk into an RGB buffer (alpha removed, colors opaque);
 * root receives per-rank RGB buffers in rank order and composites by overwriting
 * to reproduce serial ordering. Per-record format is: float32 x,y,radius + uint8 r,g,b.
 * Usage: mpirun -n <procs> ./renderer_mpi <input.bin> <output.png>
 */

#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RECSZ (sizeof(float)*3 + 3)

static inline void fill_span_u32(uint32_t *dst, int n, uint32_t value) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        dst[i + 0] = value;
        dst[i + 1] = value;
        dst[i + 2] = value;
        dst[i + 3] = value;
        dst[i + 4] = value;
        dst[i + 5] = value;
        dst[i + 6] = value;
        dst[i + 7] = value;
    }
    for (; i < n; ++i) dst[i] = value;
}

static inline void fill_span_u64(uint64_t *dst, int n, uint64_t value) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        dst[i + 0] = value;
        dst[i + 1] = value;
        dst[i + 2] = value;
        dst[i + 3] = value;
        dst[i + 4] = value;
        dst[i + 5] = value;
        dst[i + 6] = value;
        dst[i + 7] = value;
    }
    for (; i < n; ++i) dst[i] = value;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input.bin> <output.png>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];

    uint32_t version = 0;
    uint64_t count = 0;
    float bbox[6] = {0};
    int W = 640, H = 480;
    /* overall_start preserved for optional wall-time measurement (root only) */
    double overall_start = 0.0;

    unsigned char *all_records = NULL;

    if (rank == 0) {
        /* record overall start just before opening/reading the input */
        overall_start = MPI_Wtime();
        FILE *f = fopen(inpath, "rb");
        if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
        char magic[5] = {0};
        if (fread(magic, 1, 4, f) != 4) { fprintf(stderr, "failed read magic\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (strncmp(magic, "CRDR", 4) != 0) { fprintf(stderr, "bad magic: %.4s\n", magic); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&version, sizeof(version), 1, f) != 1) { fprintf(stderr, "failed read version\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(&count, sizeof(count), 1, f) != 1) { fprintf(stderr, "failed read count\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(bbox, sizeof(float), 6, f) != 6) { fprintf(stderr, "failed read bbox\n"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }

        W = (int)roundf(bbox[3] - bbox[0]);
        H = (int)roundf(bbox[4] - bbox[1]);
        if (W <= 0) W = 640;
        if (H <= 0) H = 480;

        fprintf(stderr, "rank0: magic=CRDR version=%u count=%llu image %dx%d\n", version, (unsigned long long)count, W, H);

        size_t totsz = (size_t)count * RECSZ;
        all_records = malloc(totsz);
        if (!all_records) { perror("malloc all_records"); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fread(all_records, 1, totsz, f) != totsz) { fprintf(stderr, "failed read records\n"); free(all_records); fclose(f); MPI_Abort(MPI_COMM_WORLD, 1); }
        fclose(f);
    }

    // Broadcast header info to all ranks
    MPI_Bcast(&version, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(bbox, 6, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *sendcounts = malloc(sizeof(int) * nprocs);
    int *displs = malloc(sizeof(int) * nprocs);
    uint64_t *rec_counts = malloc(sizeof(uint64_t) * nprocs);
    if (!sendcounts || !displs || !rec_counts) {
        perror("malloc partition arrays");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Weight-based load balancing: partition by r^2 to equalize rasterization cost
    if (count == 0) {
        for (int i = 0; i < nprocs; ++i) rec_counts[i] = 0;
    } else if ((uint64_t)nprocs <= 1 || count <= (uint64_t)nprocs) {
        uint64_t base = count / (uint64_t)nprocs;
        uint64_t rem = count % (uint64_t)nprocs;
        for (int i = 0; i < nprocs; ++i)
            rec_counts[i] = base + (i < (int)rem ? 1 : 0);
    } else {
        if (rank == 0) {
            double totalw = 0.0;
            for (uint64_t i = 0; i < count; ++i) {
                float r = 0.0f;
                memcpy(&r, all_records + i * RECSZ + sizeof(float) * 2, sizeof(float));
                totalw += (double)r * (double)r;
            }

            double target = totalw / (double)nprocs;
            uint64_t idx = 0;
            for (int p = 0; p < nprocs; ++p) {
                double acc = 0.0;
                uint64_t start = idx;
                while (idx < count && (acc < target || (count - idx) < (uint64_t)(nprocs - p))) {
                    float r = 0.0f;
                    memcpy(&r, all_records + idx * RECSZ + sizeof(float) * 2, sizeof(float));
                    acc += (double)r * (double)r;
                    idx++;
                }
                uint64_t cnt = idx - start;
                if (cnt == 0 && idx < count) { cnt = 1; idx++; }
                rec_counts[p] = cnt;
            }
            uint64_t assigned = 0;
            for (int p = 0; p < nprocs; ++p) assigned += rec_counts[p];
            if (assigned < count) rec_counts[nprocs - 1] += (count - assigned);
        }
    }

    MPI_Bcast(rec_counts, nprocs, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    size_t offset = 0;
    for (int i = 0; i < nprocs; ++i) {
        sendcounts[i] = (int)(rec_counts[i] * RECSZ);
        displs[i] = (int)offset;
        offset += (size_t)sendcounts[i];
    }
    free(rec_counts);

    int mybytes = sendcounts[rank];
    unsigned char *mybuf = NULL;
    if (mybytes > 0) {
        mybuf = malloc((size_t)mybytes);
        if (!mybuf) { perror("malloc mybuf"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Scatterv(all_records, sendcounts, displs, MPI_BYTE, mybuf, mybytes, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) { free(all_records); all_records = NULL; }

    // allocate local compositing buffer (32-bit path halves bandwidth when rank tag fits in 8 bits)
    size_t npix = (size_t)W * (size_t)H;
    int use_u32 = (nprocs <= 255);
    uint32_t *img32 = NULL;
    uint64_t *img64 = NULL;
    if (use_u32) {
        img32 = calloc(npix, sizeof(uint32_t));
        if (!img32) { perror("alloc img32"); MPI_Abort(MPI_COMM_WORLD, 1); }
    } else {
        img64 = calloc(npix, sizeof(uint64_t));
        if (!img64) { perror("alloc img64"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // Rasterize local records (higher tag means later partition in original stream order)
    double local_start = MPI_Wtime();
    uint64_t mycount = (uint64_t)mybytes / RECSZ;
    if (use_u32) {
        for (uint64_t i = 0; i < mycount; ++i) {
            unsigned char *rec = mybuf + i * RECSZ;
            float cx = 0.0f, cy = 0.0f, radius = 0.0f;
            memcpy(&cx, rec, sizeof(float));
            memcpy(&cy, rec + sizeof(float), sizeof(float));
            memcpy(&radius, rec + sizeof(float) * 2, sizeof(float));
            const unsigned char *rgb = rec + sizeof(float) * 3;
            if (radius <= 0.0f) continue;

            int ymin = (int)ceilf(cy - radius - 0.5f);
            int ymax = (int)floorf(cy + radius - 0.5f);
            if (ymin < 0) ymin = 0;
            if (ymax >= H) ymax = H - 1;
            if (ymin > ymax) continue;

            float r2 = radius * radius;
            uint32_t packed = 0;
            if ((rgb[0] | rgb[1] | rgb[2]) != 0) {
                packed = ((uint32_t)(rank + 1) << 24)
                       | ((uint32_t)rgb[0] << 16)
                       | ((uint32_t)rgb[1] << 8)
                       | (uint32_t)rgb[2];
            }

            float dy = ((float)ymin + 0.5f) - cy;
            size_t row = (size_t)ymin * (size_t)W;
            for (int y = ymin; y <= ymax; ++y, dy += 1.0f, row += (size_t)W) {
                float dy2 = dy * dy;
                if (dy2 > r2) continue;

                float dx_lim = sqrtf(r2 - dy2);
                int xmin = (int)ceilf(cx - dx_lim - 0.5f) - 1;
                int xmax = (int)floorf(cx + dx_lim - 0.5f) + 1;
                if (xmin < 0) xmin = 0;
                if (xmax >= W) xmax = W - 1;
                if (xmin > xmax) continue;

                float dx = ((float)xmin + 0.5f) - cx;
                while (xmin <= xmax && (dx * dx + dy2) > r2) {
                    xmin++;
                    dx += 1.0f;
                }
                dx = ((float)xmax + 0.5f) - cx;
                while (xmax >= xmin && (dx * dx + dy2) > r2) {
                    xmax--;
                    dx -= 1.0f;
                }
                if (xmin > xmax) continue;

                int span = xmax - xmin + 1;
                fill_span_u32(img32 + row + (size_t)xmin, span, packed);
            }
        }
    } else {
        for (uint64_t i = 0; i < mycount; ++i) {
            unsigned char *rec = mybuf + i * RECSZ;
            float cx = 0.0f, cy = 0.0f, radius = 0.0f;
            memcpy(&cx, rec, sizeof(float));
            memcpy(&cy, rec + sizeof(float), sizeof(float));
            memcpy(&radius, rec + sizeof(float) * 2, sizeof(float));
            const unsigned char *rgb = rec + sizeof(float) * 3;
            if (radius <= 0.0f) continue;

            int ymin = (int)ceilf(cy - radius - 0.5f);
            int ymax = (int)floorf(cy + radius - 0.5f);
            if (ymin < 0) ymin = 0;
            if (ymax >= H) ymax = H - 1;
            if (ymin > ymax) continue;

            float r2 = radius * radius;
            uint64_t packed = 0;
            if ((rgb[0] | rgb[1] | rgb[2]) != 0) {
                packed = ((uint64_t)(rank + 1) << 32)
                       | ((uint64_t)rgb[0] << 16)
                       | ((uint64_t)rgb[1] << 8)
                       | (uint64_t)rgb[2];
            }

            float dy = ((float)ymin + 0.5f) - cy;
            size_t row = (size_t)ymin * (size_t)W;
            for (int y = ymin; y <= ymax; ++y, dy += 1.0f, row += (size_t)W) {
                float dy2 = dy * dy;
                if (dy2 > r2) continue;

                float dx_lim = sqrtf(r2 - dy2);
                int xmin = (int)ceilf(cx - dx_lim - 0.5f) - 1;
                int xmax = (int)floorf(cx + dx_lim - 0.5f) + 1;
                if (xmin < 0) xmin = 0;
                if (xmax >= W) xmax = W - 1;
                if (xmin > xmax) continue;

                float dx = ((float)xmin + 0.5f) - cx;
                while (xmin <= xmax && (dx * dx + dy2) > r2) {
                    xmin++;
                    dx += 1.0f;
                }
                dx = ((float)xmax + 0.5f) - cx;
                while (xmax >= xmin && (dx * dx + dy2) > r2) {
                    xmax--;
                    dx -= 1.0f;
                }
                if (xmin > xmax) continue;

                int span = xmax - xmin + 1;
                fill_span_u64(img64 + row + (size_t)xmin, span, packed);
            }
        }
    }
    free(mybuf);
    double local_elapsed = MPI_Wtime() - local_start;
    fprintf(stderr, "rank %d: local render time: %.6f s\n", rank, local_elapsed);

    double min_t, max_t, sum_t;
    MPI_Reduce(&local_elapsed, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_elapsed, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        fprintf(stderr, "per-rank render (s): min=%.6f max=%.6f avg=%.6f imbalance=%.2f%%\n",
                min_t, max_t, sum_t/nprocs, (max_t - sum_t/nprocs) / (sum_t/nprocs) * 100.0);

    // One-shot global compositing: higher rank (later record partition) wins per pixel.
    if (npix > (size_t)INT_MAX) {
        if (rank == 0) fprintf(stderr, "image too large for MPI_Reduce count\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (use_u32) {
        uint32_t *acc_img32 = NULL;
        if (rank == 0) acc_img32 = img32;
        MPI_Reduce((rank == 0 ? MPI_IN_PLACE : img32), (rank == 0 ? acc_img32 : img32),
                   (int)npix, MPI_UINT32_T, MPI_MAX, 0, MPI_COMM_WORLD);
    } else {
        uint64_t *acc_img64 = NULL;
        if (rank == 0) acc_img64 = img64;
        MPI_Reduce((rank == 0 ? MPI_IN_PLACE : img64), (rank == 0 ? acc_img64 : img64),
                   (int)npix, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    // Provided converter (kept for convenience)
    if (rank == 0) {
        size_t stride = (size_t)W * 3;
        unsigned char *pixels = malloc((size_t)H * stride);
        if (!pixels) { perror("malloc pixels"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int y = 0; y < H; ++y) {
            unsigned char *dst = pixels + (size_t)y * stride;
            if (use_u32) {
                const uint32_t *src32 = img32 + (size_t)y * (size_t)W;
                for (int x = 0; x < W; ++x) {
                    uint32_t rgb = src32[x] & 0x00FFFFFFu;
                    dst[0] = (unsigned char)((rgb >> 16) & 0xFF);
                    dst[1] = (unsigned char)((rgb >> 8) & 0xFF);
                    dst[2] = (unsigned char)(rgb & 0xFF);
                    dst += 3;
                }
            } else {
                const uint64_t *src64 = img64 + (size_t)y * (size_t)W;
                for (int x = 0; x < W; ++x) {
                    uint32_t rgb = (uint32_t)(src64[x] & 0x00FFFFFFu);
                    dst[0] = (unsigned char)((rgb >> 16) & 0xFF);
                    dst[1] = (unsigned char)((rgb >> 8) & 0xFF);
                    dst[2] = (unsigned char)(rgb & 0xFF);
                    dst += 3;
                }
            }
        }
        if (!stbi_write_png(outpath, W, H, 3, pixels, (int)stride)) {
            fprintf(stderr, "stbi_write_png failed\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* report overall wall time (root only) */
        {
            double overall_end = MPI_Wtime();
            double overall_elapsed = overall_end - overall_start;
            fprintf(stderr, "rank0: Total wall time (before read -> after write): %.6f s\n", overall_elapsed);
        }
        free(pixels);
        fprintf(stderr, "rank0: Wrote PNG %s\n", outpath);
    }

    free(img32);
    free(img64);
    free(sendcounts); free(displs);
    MPI_Finalize();
    return 0;
}
