// strata-inkling CUDA: the K/V auto-offload's reserve for the runner's workspaces (strata/core/ink_kv_reserve.hpp).
// The window of the global K/V that stays on the card is 3/4 of what fits after the reserve, so a larger reserve costs window positions directly:
// the model below is the engine's own arithmetic (ink_runner.cuh), fed with the RTX 3060 12 GB numbers measured on 2026-11-24.
#include "strata/core/ink_kv_reserve.hpp"

#include <algorithm>
#include <cstdio>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr size_t MiB = 1ull << 20;

// window = 3/4 of the positions that fit after the SWA ring, ShortConv state and the reserve, rounded down to 256 (ink_runner.cuh)
int window(size_t vfree, size_t swa_and_sconv, size_t reserve, size_t per_pos, int max_ctx) {
    const size_t avail = vfree > swa_and_sconv + reserve ? vfree - swa_and_sconv - reserve : 0;
    const int fit = (int)std::min<size_t>((size_t)max_ctx, (avail / per_pos) & ~(size_t)63);
    return fit >= max_ctx ? max_ctx : (fit >= 1024 ? ((fit * 3 / 4) & ~255) : 0);
}
}  // namespace

int main() {
    // exactly the values v0.2.20 ran with
    CHECK(ink_runner_ws_reserve_bytes(1, false) == 640 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(1, true) == 800 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(7, true) == 800 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(8, false) == 1280 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(64, true) == 1440 * MiB);
    // a running DSpark drafter with the K/V kept on the card: the 586 - 76 MiB it took at start on top (measured, RTX 3060 12 GB)
    CHECK(kInkWsReserveDsparkRunBytes == 512 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(1, true, true) == 1312 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(8, true, true) == 1952 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(1, false, true) == 1152 * MiB);
    CHECK(ink_runner_ws_reserve_bytes(1, true, false) == 800 * MiB);          // the host-offload window: as before

    // The same card with 4.0 / 4.4 GiB of experts on it, 128K, k8v4 (1 600 B a position per visit incl. scales), the K/V kept on the card
    // (--kv-offload off) and the drafter on: 3768 / 3360 MiB free after the experts, SWA ring + ShortConv 495 MiB. v0.1.67 claimed 131 072 /
    // 126 656 and the global K/V ran out of the card at 111 104 / 84 480 tokens; with the drafter's start-up cost reserved, the context is
    // sized below where the card really ran out.
    const size_t v4[2] = {3768 * MiB, 3360 * MiB};
    const int exhausted[2] = {111104, 84480};
    const size_t per_k8v4 = 8 * 1600 + 4096;
    for (int i = 0; i < 2; ++i) {
        const int old_fit = window(v4[i], 495 * MiB, ink_runner_ws_reserve_bytes(1, true), per_k8v4, 131072);
        const int new_fit = window(v4[i], 495 * MiB, ink_runner_ws_reserve_bytes(1, true, true), per_k8v4, 131072);
        CHECK(old_fit > exhausted[i]);                   // claimed more than the card held
        CHECK(new_fit <= exhausted[i]);                  // honest: no request can reach the point where it ran out
        std::printf("RTX 3060, %.1f GiB of experts, 128K k8v4 on the card, DSpark on: context %d -> %d tokens (ran out at %d)\n",
                    v4[i] / (1024.0 * 1024.0) / 1024.0, old_fit, new_fit, exhausted[i]);
    }

    // RTX 3060 12 GB, SQ-10, 32K, 1 slot, 1.60 GiB free at the decision, SWA ring + ShortConv 495 MiB (fitted: it reproduces both measured windows),
    // per-position K/V of the 8 global visits 32 KiB (+4 KiB for the drafter's own)
    const size_t vfree = 1638 * MiB, swa = 495 * MiB, per_pos = 8 * 4096;
    const int max_ctx = 32768;
    const int w_020 = window(vfree, swa, 640 * MiB + 160 * MiB, per_pos + 4096, max_ctx);         // v0.2.20
    const int w_759 = window(vfree, swa, 768 * MiB + 256 * MiB, per_pos + 4096, max_ctx);         // 759ccf6c6
    const int w_now = window(vfree, swa, ink_runner_ws_reserve_bytes(1, true), per_pos + 4096, max_ctx);
    CHECK(w_020 == 7168);                 // measured: global KV = 7168 VRAM + 25600 host (v0.2.20)
    CHECK(w_759 == 2304);                 // measured: 2304 VRAM + 30464 host (rc1)
    CHECK(w_now == w_020);                // this change: v0.2.20's window again
    std::printf("windows (positions): v0.2.20 %d, 759ccf6c6 %d, this change %d\n", w_020, w_759, w_now);

    if (failed) {
        std::fprintf(stderr, "ink_kv_reserve_test: %d failed\n", failed);
        return 1;
    }
    std::printf("ink_kv_reserve_test: ok\n");
    return 0;
}
