// NieR:Automata The End of YoRHa Edition (010056B015FE8000, USA, update 1.0.3): what the shared runtime needs to know
// about the game, and what only NieR needs. PlatinumGames' engine, Switch version by Virtuos, SDK 12.3.9. Sound is
// Wwise 2021.1 over nn::audio AudioOut (romfs sound/), films are CRI USM through nn::movie (subsdk0, romfs movie/).
// The NVN shaders are DAT archives in romfs data000.cpk (games/nier/tools/nier_shaders.py). Notes: games/nier/PROGRESO.md.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <nx/game.h>
#include <nx/runtime.h>

#include "platform.h"

namespace nx {

const game::Info game::kInfo = {"nier", "NieR Automata", 0x010056B015FE8000ull, "io.github.nierrecomp",
                                "private/nier", 0x1000000 /* main.npdm META +0x14 */, "3C10EC35C5EF4C36" /* 1.0.3 */};

static constexpr const char* kPerfBuildTag = "r10";

namespace {

// Copy every fputs line into the log (assertion messages), then do the real one.
void fputs_logged(CPU& c) {
    static const uint64_t real = find_symbol("fputs");
    const char* s = reinterpret_cast<const char*>(c.x[0]);
    if (s) {
        std::string line(s, strnlen(s, 1024));
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        log("[guest fputs] %s", line.c_str());
    }
    if (real) call(c, real);
    else c.x[0] = 0;
}

// Frame rate. The main loop (main+591100) calls, every frame, the game's time manager main+8fc7a8 (below) and the
// engine's vsync-mode setter main+9bf1d0 with 1. That setter writes the presenter's mode (*(main+0x12c1c48) + 0x504),
// which the per-frame present (main+9baa54) applies when it changes: 0 = present interval 1 (60 fps), 1 = interval 2
// (30 fps, the Switch's), through nvnWindowSetPresentInterval. Both calls go through the function table (main
// recompiled with --hookable 9bf1d0,8fc7a8) and are hooked here. NX_NIER_FPS=30|60 (default: NX_FPS_CAP, the
// launcher's, else 30).
uint64_t g_vsync_setter = 0, g_update = 0, g_main = 0;

int target_fps() {
    static const int fps = [] {
        const char* e = std::getenv("NX_NIER_FPS");
        if (!e) e = std::getenv("NX_FPS_CAP");
        return e && std::atoi(e) >= 60 ? 60 : 30;
    }();
    return fps;
}

void vsync_hook(CPU& c) {
    const uint32_t asked = uint32_t(c.x[0]);
    c.x[0] = target_fps() >= 60 ? 0 : 1;
    call(c, g_vsync_setter);
    // The presenter starts with mode and applied mode 0 (zeroed memory) while the window was made with the device's
    // interval (+0x9c160, 2): asking 0 from the first frame changed nothing it could see. When the device's interval is
    // not the wanted one, the applied mode is made stale so the next present applies it.
    if (const uint64_t presenter = ld<uint64_t>(g_main + 0x12c1c48)) {
        const uint64_t device = ld<uint64_t>(presenter);
        const uint32_t want = target_fps() >= 60 ? 1 : 2;
        if (device && ld<uint32_t>(device + 0x9c160) != want) st<uint32_t>(presenter + 0x508, 0xFFFFFFFFu);
    }
    static int traced = 0;
    if (std::getenv("NX_NIER_TRACE_VSYNC") && traced++ < 12) {
        const uint64_t presenter = ld<uint64_t>(g_main + 0x12c1c48);
        log("nier: vsync mode asked %u, set %u (returned %u): presenter %llx mode %u applied %u", asked,
            target_fps() >= 60 ? 0u : 1u, uint32_t(c.x[0]), (unsigned long long)presenter,
            presenter ? ld<uint32_t>(presenter + 0x504) : 0, presenter ? ld<uint32_t>(presenter + 0x508) : 0);
    }
}

// main+8fc7a8 (time manager in x0): +0xbc the last frame's milliseconds (system ticks), +0xb0 the game's speed in
// 60 fps frames per frame. Adaptive (w1 != 0, the usual): the average frame time / 16.67 ms clamped to 1.0..2.0, so at
// 60 fps it is 1.0 (the engine runs right at 60) and at 30 fps 2.0, but under 30 fps it stays at 2.0: at 15 fps the
// game ran at half speed (the slow motion on the Adreno 610). When it sits at that cap, the last frame's own time is
// used up to NX_NIER_MAX_STEP (4.0 by default: right down to 15 fps; 2 = the game's own). NX_NIER_STEP=<x>: the speed
// fixed (experiments); NX_NIER_TRACE_STEP=1 logs it.
// Dynamic resolution (main+9bf3e0, once a frame, with the GPU's milliseconds: runtime NX_GPU_TIME_FEED): over 45 ms
// twice the 3D scale drops 0.07, over 32 ms 0.01, and it climbs 0.01 under 30 ms; it stays within [min, 1], min being
// 0.8333 (0.4 with a config flag) or the game's own override if lower (the float at *(main+0x12c7430), 1 unless the
// game sets it: main+9bf670). At 720p that floor was 1056x594 and the Adreno 610 stays there. NX_NIER_MIN_SCALE
// (0.3..1, default 0.6) is written into the override every frame: on a GPU that keeps up nothing changes.
void resolution_floor() {
    static const float floor = [] {
        const char* e = std::getenv("NX_NIER_MIN_SCALE");
        if (e) return std::clamp(float(std::atof(e)), 0.3f, 1.0f);
        return 0.45f;
    }();
    if (const uint64_t override_min = g_main ? ld<uint64_t>(g_main + 0x12c7430) : 0) {
        uint32_t u;
        std::memcpy(&u, &floor, 4);
        st<uint32_t>(override_min, u);
    }
}

void update_hook(CPU& c) {
    const uint64_t timer = c.x[0];
    const bool adaptive = uint32_t(c.x[1]) != 0;
    resolution_floor();
    // Synchronize presenter interval and mode directly every frame (bypasses unhookable ADRP at main+9bf1d0)
    if (const uint64_t presenter = g_main ? ld<uint64_t>(g_main + 0x12c1c48) : 0) {
        const uint32_t want_mode = target_fps() >= 60 ? 0 : 1;
        st<uint32_t>(presenter + 0x504, want_mode);
        const uint64_t device = ld<uint64_t>(presenter);
        const uint32_t want_interval = target_fps() >= 60 ? 1 : 2;
        if (device && ld<uint32_t>(device + 0x9c160) != want_interval)
            st<uint32_t>(presenter + 0x508, 0xFFFFFFFFu);
    }
    call(c, g_update);
    static const char* fixed = std::getenv("NX_NIER_STEP");
    static const float max_step = [] {
        const char* e = std::getenv("NX_NIER_MAX_STEP");
        float v = e ? float(std::atof(e)) : 6.0f;
        return std::clamp(v, 2.0f, 8.0f);
    }();
    auto get = [&](uint64_t off) { uint32_t u = ld<uint32_t>(timer + off); float f; std::memcpy(&f, &u, 4); return f; };
    auto put = [&](uint64_t off, float f) { uint32_t u; std::memcpy(&u, &f, 4); st<uint32_t>(timer + off, u); };
    const float before = get(0xb0), frame_ms = get(0xbc);
    if (fixed) put(0xb0, float(std::atof(fixed)));
    else if (adaptive && before >= 2.0f && max_step > 2.0f) put(0xb0, std::clamp(frame_ms / (1000.0f / 60.0f), 2.0f, max_step));
    static int traced = 0;
    if (std::getenv("NX_NIER_TRACE_STEP") && (traced++ < 20 || traced % 300 == 0))
        log("nier: step: adaptive %d, frame %.1f ms, speed %.3f -> %.3f", adaptive, double(frame_ms), double(before),
            double(get(0xb0)));
}

// Ships and structures came out black (PROGRESO 29). main+9d514c draws one mesh range (x4: its draw parameters, x5:
// the index range): instanced when it has 2+ instances, else through the GPU-culling path when the parameters carry
// that buffer (+0x10 set and +0x18[+0x28] not null: main+9c64a4, DrawElementsIndirect from a record at the culling
// struct's +0x48), else a plain DrawElements with the count the CPU knows (main+9c5d1c). The records are a static
// template copied in every frame with CopyBufferToBuffer ({0 1 0 0 0}: count 0) and nothing we replay ever writes
// their counts, so those draws drew nothing: in the colour pass the sea failed the depth test where the depth
// prepass had put the ships and their pixels kept the target's black clear. The hook clears +0x10 for the call: the
// plain draw, every frame (no GPU culling; the occlusion queries' conditional rendering still applies).
// NX_NIER_GPU_CULLING=1: the game's own path.
uint64_t g_mesh_draw = 0;

void mesh_draw_hook(CPU& c) {
    static const bool own = [] { const char* e = std::getenv("NX_NIER_GPU_CULLING"); return e && e[0] == '1'; }();
    const uint64_t params = c.x[4];
    const uint32_t culling = params && !own ? ld<uint32_t>(params + 0x10) : 0;
    if (culling) st<uint32_t>(params + 0x10, 0);
    call(c, g_mesh_draw);
    if (culling) st<uint32_t>(params + 0x10, culling);
}

}  // namespace

NX_HLE("fputs", fputs_logged);

void game::renderer_start() {}
void game::first_frame() {}

void game::after_init() {
    log("nier: low-end build revision %s", kPerfBuildTag);
    // Dynamic resolution: the engine lowers its 3D resolution (1280x720 down to 1056x594 in handheld mode) from the
    // GPU times its timestamp counters measure. Those are the replay's unless the runtime feeds it the GPU's own
    // (runtime nvn_gpu_busy.cpp, PROGRESO 27). config.txt NX_GPU_TIME_FEED=0 turns it off.
#if defined(_WIN32)
    if (!std::getenv("NX_GPU_TIME_FEED")) _putenv_s("NX_GPU_TIME_FEED", "1");
    if (!std::getenv("NX_IMAGE_STORAGE")) _putenv_s("NX_IMAGE_STORAGE", "0");
#else
    setenv("NX_GPU_TIME_FEED", "1", 0);
    setenv("NX_IMAGE_STORAGE", "0", 0);
#if defined(__ANDROID__)
    // Low-end mobile profile defaults if unset
    if (!std::getenv("NX_MEM_LIMIT_GB")) setenv("NX_MEM_LIMIT_GB", "6.0", 0);
    if (!std::getenv("NX_SHADOWS")) setenv("NX_SHADOWS", "0", 0);
    if (!std::getenv("GRAPHICS")) setenv("GRAPHICS", "low", 0);
    if (!std::getenv("NX_NIER_MIN_SCALE")) setenv("NX_NIER_MIN_SCALE", "0.45", 0);
    if (!std::getenv("NX_NIER_MAX_STEP")) setenv("NX_NIER_MAX_STEP", "6.0", 0);
    if (!std::getenv("NX_PIPELINED_NO_SPLIT")) setenv("NX_PIPELINED_NO_SPLIT", "1", 0);
    if (!std::getenv("NX_CHUNK_IN_PASS")) setenv("NX_CHUNK_IN_PASS", "0", 0);
    if (!std::getenv("NX_MAX_ANISO")) setenv("NX_MAX_ANISO", "1", 0);
    if (!std::getenv("NX_PRUNE_POST")) setenv("NX_PRUNE_POST", "1", 0);
    if (!std::getenv("NX_D24S8")) setenv("NX_D24S8", "1", 0);
    if (!std::getenv("NX_ROBUST")) setenv("NX_ROBUST", "0", 0);
#endif
#endif
    const LoadedModule* m = module_at(find_symbol("nnMain"));
    if (!m) return;
    g_main = m->base;
    g_vsync_setter = hook_function(m->base + 0x9bf1d0, vsync_hook);
    g_update = hook_function(m->base + 0x8fc7a8, update_hook);
    if (!g_update) log("nier: update hook not installed (%llx)", (unsigned long long)g_update);
    if (!(g_mesh_draw = hook_function(m->base + 0x9d514c, mesh_draw_hook))) log("nier: mesh draw hook not installed");
}

}  // namespace nx
