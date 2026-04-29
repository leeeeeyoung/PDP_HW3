/*
 * renderer.c
 * Read CRDR binary format and render circles into a PNG image (single-node serial)
 * Note: per-record format is now: float32 x,y,radius followed by uint8 r,g,b
 * (depth z and alpha A were removed). Colors are treated as fully opaque.
 * Usage: ./renderer <input.bin> <output.png>
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*
 * STUDENT TODO: implement this function.
 * It should test pixel centers for coverage by the circle centered at (cx,cy)
 * with given radius and composite the color (rgb) into the float `img`
 * buffer of size W*H*3 (row-major). The scaffold below computes the
 * integer bounding box and exposes color components; students must fill
 * in the inner loops to update `img`.
 */
static void rasterize_circle(float *img, int W, int H, float cx, float cy, float radius, unsigned char rgb[3]) {
    int xmin = (int)floorf(cx - radius);
    int xmax = (int)floorf(cx + radius);
    int ymin = (int)floorf(cy - radius);
    int ymax = (int)floorf(cy + radius);
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax >= W) xmax = W - 1;
    if (ymax >= H) ymax = H - 1;

    float Cr = rgb[0] / 255.0f;
    float Cg = rgb[1] / 255.0f;
    float Cb = rgb[2] / 255.0f;

    float r2 = radius * radius;
    for (int y = ymin; y <= ymax; ++y) {
        for (int x = xmin; x <= xmax; ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;
            float dx = px - cx;
            float dy = py - cy;
            if ((dx*dx + dy*dy) <= r2) {
                size_t idx = ((size_t)y * W + x) * 3;
                img[idx + 0] = Cr;
                img[idx + 1] = Cg;
                img[idx + 2] = Cb;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.png>\n", argv[0]);
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];

    FILE *f = fopen(inpath, "rb");
    if (!f) { perror("fopen"); return 1; }

    char magic[5] = {0};
    uint32_t version;
    uint64_t count;
    float bbox[6];

    if (fread(magic, 1, 4, f) != 4) { fprintf(stderr, "failed read magic\n"); fclose(f); return 1; }
    if (strncmp(magic, "CRDR", 4) != 0) { fprintf(stderr, "bad magic: %.4s\n", magic); fclose(f); return 1; }
    if (fread(&version, sizeof(version), 1, f) != 1) { fprintf(stderr, "failed read version\n"); fclose(f); return 1; }
    if (fread(&count, sizeof(count), 1, f) != 1) { fprintf(stderr, "failed read count\n"); fclose(f); return 1; }
    if (fread(bbox, sizeof(float), 6, f) != 6) { fprintf(stderr, "failed read bbox\n"); fclose(f); return 1; }

    // Determine image size from bbox if it looks like pixel-space, otherwise default 640x480
    int W = (int)roundf(bbox[3] - bbox[0]);
    int H = (int)roundf(bbox[4] - bbox[1]);
    if (W <= 0) W = 640;
    if (H <= 0) H = 480;

    fprintf(stderr, "magic=%.4s version=%u count=%llu\n", magic, version, (unsigned long long)count);
    fprintf(stderr, "bbox: [%g %g %g] -> [%g %g %g] -> image %dx%d\n", bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], W, H);

    // float accumulator per pixel RGB
    float *img = calloc((size_t)W * H * 3, sizeof(float));
    if (!img) { perror("calloc"); fclose(f); return 1; }

    // start timing the main render loop
    struct timespec t_start, t_end;
    if (clock_gettime(CLOCK_MONOTONIC, &t_start) != 0) perror("clock_gettime start");

    for (uint64_t i = 0; i < count; ++i) {
        float floats[3];
        unsigned char rgb[3];
        if (fread(floats, sizeof(float), 3, f) != 3) { fprintf(stderr, "failed read record floats at %llu\n", (unsigned long long)i); break; }
        if (fread(rgb, 1, 3, f) != 3) { fprintf(stderr, "failed read record rgb at %llu\n", (unsigned long long)i); break; }
        float cx = floats[0];
        float cy = floats[1];
        float radius = floats[2];

        /* Delegate rasterization to student-implementable function. */
        rasterize_circle(img, W, H, cx, cy, radius, rgb);
    }

    // Convert to 8-bit continuous buffer
    size_t stride = (size_t)W * 3;
    unsigned char *pixels = malloc((size_t)H * stride);
    if (!pixels) { perror("malloc pixels"); free(img); fclose(f); return 1; }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t idx = ((size_t)y * W + x) * 3;
            float fr = img[idx + 0];
            float fg = img[idx + 1];
            float fb = img[idx + 2];
            int ir = (int)roundf(fmaxf(0.0f, fminf(1.0f, fr)) * 255.0f);
            int ig = (int)roundf(fmaxf(0.0f, fminf(1.0f, fg)) * 255.0f);
            int ib = (int)roundf(fmaxf(0.0f, fminf(1.0f, fb)) * 255.0f);
            pixels[(size_t)y * stride + x*3 + 0] = (unsigned char)ir;
            pixels[(size_t)y * stride + x*3 + 1] = (unsigned char)ig;
            pixels[(size_t)y * stride + x*3 + 2] = (unsigned char)ib;
        }
    }

    // Write PNG using stb_image_write
    if (!stbi_write_png(outpath, W, H, 3, pixels, (int)stride)) {
        fprintf(stderr, "stbi_write_png failed\n"); free(pixels); free(img); fclose(f); return 1;
    }

    // stop timing and report
    if (clock_gettime(CLOCK_MONOTONIC, &t_end) == 0) {
        double elapsed = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        fprintf(stderr, "Total render time: %.6f s\n", elapsed);
    } else {
        perror("clock_gettime end");
    }

    free(pixels);
    free(img);
    fclose(f);

    fprintf(stderr, "Wrote PNG %s\n", outpath);
    return 0;
}
