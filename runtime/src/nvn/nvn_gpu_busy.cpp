// The GPU's busy time per frame at the real load (NX_GPU_PROFILE=3; docs/nvn/android-perf/PHASE4.md G0). The per-pass
// profile (NX_GPU_PROFILE=1, nvn_debug.cpp) waits for every submit, so the GPU idles between frames and its clock
// drops: it measured a slower GPU than the game runs on. Here each queued part of a submit or present gets a timestamp
// at its start and its end in its slot's own query pool, and the results are read when the slot is retired (its
// fences passed: no wait); synchronous submits (NX_PIPELINED=0) are one part each, read after their fence. Logged
// every 60 presents: busy = the parts' summed lengths (presents: those of submits that replay nothing), span = first
// start to last end of each slot, and the GPU clock from sysfs when readable.
//
// NX_GPU_TIME_FEED=1 (NieR's default, games/nier/game.cpp) takes the same timestamps without the log: each replayed
// frame's GPU time over its replay's length gives gpu_time_scale(), the rate the counter reports' timestamps advance
// at (nvn_draw_state.cpp). The replay writes those reports when it records them, long before the GPU runs the frame,
// so NieR's dynamic resolution measured the replay (~11 ms on the Adreno 610) instead of the GPU (~41 ms in the load
// menu) and never lowered the resolution (PROGRESO 27).
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "nvn_vk.h"

namespace nx::nvn {

using namespace nx::vk;

namespace {

constexpr uint32_t kSlots = 8, kParts = 64;
VkQueryPool g_pools[kSlots] = {};
double g_period_ns = 1.0;
double g_busy_ns = 0, g_span_ns = 0, g_present_ns = 0;  // present: the parts of submits that replay nothing
uint64_t g_parts = 0, g_not_ready = 0;
double g_feed_gpu_ns = 0, g_feed_replay_ns = 0;  // EMAs over replayed frames (NX_GPU_TIME_FEED)
std::atomic<double> g_scale{1.0};
uint64_t g_feed_logs = 0;

bool profile_on() {
    static const bool on = [] {
        const char* e = std::getenv("NX_GPU_PROFILE");
        return e && std::atoi(e) == 3;
    }();
    return on;
}

bool feed_on() {
    static const bool on = [] {
        const char* e = std::getenv("NX_GPU_TIME_FEED");
        return e && e[0] == '1';
    }();
    return on;
}

uint64_t read_sysfs(const char* path) {
    FILE* f = std::fopen(path, "r");
    if (!f) return 0;
    unsigned long long v = 0;
    if (std::fscanf(f, "%llu", &v) != 1) v = 0;
    std::fclose(f);
    return v;
}

}  // namespace

bool gpu_busy_on() { return profile_on() || feed_on(); }

double gpu_time_scale() { return feed_on() ? g_scale.load(std::memory_order_relaxed) : 1.0; }

void gpu_busy_part_begin(uint32_t slot, uint32_t part, VkCommandBuffer cb) {
    if (!gpu_busy_on() || slot >= kSlots || part >= kParts) return;
    if (!g_pools[slot]) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2 * kParts;
        p_vkCreateQueryPool(vk_device(), &qi, nullptr, &g_pools[slot]);
        static bool once = false;
        if (!once) {
            once = true;
            VkPhysicalDeviceProperties props{};
            p_vkGetPhysicalDeviceProperties(vk_physical(), &props);
            g_period_ns = props.limits.timestampPeriod;
            log("gpu busy: timestamps on (%s), period %.2f ns", profile_on() ? "NX_GPU_PROFILE=3" : "NX_GPU_TIME_FEED",
                g_period_ns);
        }
    }
    p_vkCmdResetQueryPool(cb, g_pools[slot], 2 * part, 2);
    p_vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_pools[slot], 2 * part);
}

void gpu_busy_part_end(uint32_t slot, uint32_t part, VkCommandBuffer cb) {
    if (!gpu_busy_on() || slot >= kSlots || part >= kParts || !g_pools[slot]) return;
    p_vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pools[slot], 2 * part + 1);
}

void gpu_busy_retired(uint32_t slot, uint64_t queued, bool replay, uint64_t replay_ns) {
    if (!gpu_busy_on() || slot >= kSlots || !g_pools[slot] || !queued) return;
    uint64_t first = UINT64_MAX, last = 0;
    double slot_ns = 0;
    for (uint32_t p = 0; p < kParts; p++) {
        if (!(queued >> p & 1)) continue;
        uint64_t t[2] = {};
        if (p_vkGetQueryPoolResults(vk_device(), g_pools[slot], 2 * p, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) !=
            VK_SUCCESS) {
            g_not_ready++;
            continue;
        }
        if (t[1] < t[0]) continue;
        g_busy_ns += double(t[1] - t[0]) * g_period_ns, g_parts++;
        slot_ns += double(t[1] - t[0]) * g_period_ns;
        if (!replay) g_present_ns += double(t[1] - t[0]) * g_period_ns;
        first = std::min(first, t[0]), last = std::max(last, t[1]);
    }
    if (last > first) g_span_ns += double(last - first) * g_period_ns;
    if (feed_on() && replay && replay_ns && slot_ns > 0) {
        constexpr double kKeep = 0.9;  // ~10 frames: the engine's own averaging does the rest
        g_feed_gpu_ns = g_feed_gpu_ns ? kKeep * g_feed_gpu_ns + (1 - kKeep) * slot_ns : slot_ns;
        g_feed_replay_ns = g_feed_replay_ns ? kKeep * g_feed_replay_ns + (1 - kKeep) * double(replay_ns) : double(replay_ns);
        g_scale.store(std::clamp(g_feed_gpu_ns / g_feed_replay_ns, 0.05, 8.0), std::memory_order_relaxed);
    }
}

void gpu_busy_log(uint64_t frames) {
    if (!gpu_busy_on() || !frames) return;
    if (feed_on() && g_feed_logs++ % 10 == 0)
        log("gpu time feed: GPU %.2f ms, replay %.2f ms per frame: report timestamps x%.2f", g_feed_gpu_ns / 1e6,
            g_feed_replay_ns / 1e6, g_scale.load());
    if (!profile_on()) {
        g_busy_ns = g_span_ns = g_present_ns = 0, g_parts = g_not_ready = 0;
        return;
    }
    uint64_t clk = read_sysfs("/sys/class/kgsl/kgsl-3d0/gpuclk");
    uint64_t busy_pct = read_sysfs("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage");
    log("gpu busy: %.2f ms/frame (%.1f parts/frame, presents %.2f ms), span %.2f ms/frame, %llu not ready; clock %llu "
        "MHz, busy %llu%%",
        g_busy_ns / 1e6 / double(frames), double(g_parts) / double(frames), g_present_ns / 1e6 / double(frames),
        g_span_ns / 1e6 / double(frames),
        (unsigned long long)g_not_ready, (unsigned long long)(clk / 1000000), (unsigned long long)busy_pct);
    g_busy_ns = g_span_ns = g_present_ns = 0, g_parts = g_not_ready = 0;
}

}  // namespace nx::nvn
