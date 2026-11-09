// Compute results within a submit (docs/nvn/GLITCH-PLAN.md G5). A storage buffer a compute shader writes is a copy in
// the upload buffer, written back to guest memory once the submit completed. Each binding used to get a copy of its
// own: later draws and dispatches of the submit read the guest bytes from before the dispatch, two dispatches writing
// one buffer overwrote each other at write-back, and indirect dispatch arguments were read before the GPU wrote them.
// Here the submit keeps one copy per written guest range (the ranges are disjoint, keyed by host address). Writers and
// readers of a range bind that copy; ranges that meet are merged into a new copy on the GPU (vkCmdCopyBuffer, so in
// command order); each range is written back once. NX_COMPUTE_SEPARATE=1: the previous behaviour.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "nvn_draw_state.h"
#include "platform.h"

namespace nx::nvn {
// Write-backs per log window (written_log): count, bytes compared, 8-byte words changed, host ns.
std::atomic<uint64_t> g_wb_count{0}, g_wb_bytes{0}, g_wb_changed{0}, g_wb_ns{0};
uint64_t frame_recording_serial();  // nvn_frames.cpp: the serial the submit being recorded will get
}  // namespace nx::nvn

namespace nx::nvn::replay {

using namespace nx::vk;

namespace {

constexpr uint64_t kAlign = 256;          // the largest buffer offset alignment Vulkan allows (all descriptor kinds)
constexpr uint64_t kTail = 0x10000 + 16;  // zeros after a range: storage bindings are padded to 64 KB (nvn_draw.cpp)

struct Region {
    uint64_t size = 0;  // guest bytes from the key
    uint64_t cap = 0;   // bytes of the copy from up.offset (size, then zeros)
    Upload up;          // the copy of the key's first byte
    std::shared_ptr<std::vector<uint8_t>> before;  // the guest bytes copied in (what the GPU did not change)
    // Carried results (written_carry / written_adopt): the guest bytes this range's chain of adopted regions expects
    // (CPU writes aside; shared along the chain), and this region's write-back link (a successor fills it in).
    std::shared_ptr<std::vector<uint8_t>> expect;
    std::shared_ptr<WriteBackLink> link;
};
using Regions = std::map<const uint8_t*, Region>;
using It = Regions::iterator;
Regions g_regions;
bool g_read = false;  // a copy was bound for reading since the last write_after_read
uint64_t g_binds = 0;  // written_bind calls, ever (nvn_frames.cpp: did a dispatch write storage after the split)
uint64_t g_shared = 0, g_merged = 0, g_composed = 0, g_indirect = 0;  // for written_log

// The regions overlapping [p, p + size), in address order.
std::vector<It> overlapping(const uint8_t* p, uint64_t size) {
    std::vector<It> v;
    It it = g_regions.upper_bound(p);
    if (it != g_regions.begin() && std::prev(it)->first + std::prev(it)->second.size > p) v.push_back(std::prev(it));
    for (; it != g_regions.end() && it->first < p + size; ++it) v.push_back(it);
    return v;
}

// `bytes` of upload memory whose byte `anchor` is kAlign aligned (so it can start a binding).
Upload alloc_anchored(uint64_t bytes, uint64_t anchor) {
    Upload u = upload_alloc(bytes + kAlign, kAlign);
    uint64_t shift = (kAlign - anchor % kAlign) % kAlign;
    u.offset += shift, u.ptr += shift;
    return u;
}

// The parts of `regions` inside [p, p + size) copied on the GPU into `to` (whose first byte is p's), then made
// visible to everything after. What the dispatches recorded so far wrote is there when the copy runs: dispatch()
// ends each one with a barrier to all later commands. Outside render passes.
void copy_regions(VkCommandBuffer cb, const std::vector<It>& regions, const uint8_t* p, uint64_t size, const Upload& to) {
    for (It it : regions) {
        const uint8_t* lo = std::max(it->first, p);
        const uint8_t* hi = std::min(it->first + it->second.size, p + size);
        if (lo >= hi) continue;
        VkBufferCopy c{it->second.up.offset + uint64_t(lo - it->first), to.offset + uint64_t(lo - p), uint64_t(hi - lo)};
        p_vkCmdCopyBuffer(cb, it->second.up.buffer, to.buffer, 1, &c);
    }
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr,
                           0, nullptr);
}

// A dispatch about to write a copy that earlier commands of the submit read (draws binding it as vertices, uniforms,
// storage): they finish first. Earlier writes are already ordered by dispatch()'s barrier.
void write_after_read(VkCommandBuffer cb) {
    if (!g_read) return;
    g_read = false;
    p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                           nullptr, 0, nullptr);
}

// The device's buffer descriptor offset alignment (kAlign at most, by the Vulkan spec).
uint64_t bind_align() {
    static const uint64_t a = [] {
        VkPhysicalDeviceProperties props{};
        p_vkGetPhysicalDeviceProperties(vk_physical(), &props);
        return std::max<uint64_t>({props.limits.minStorageBufferOffsetAlignment,
                                   props.limits.minUniformBufferOffsetAlignment, 16});
    }();
    return a;
}

Upload at(const It& it, uint64_t off) { return {it->second.up.buffer, it->second.up.offset + off, it->second.up.ptr + off}; }

// [p, p + size) as a sub-range of one region, bindable at `align` (at most the device's descriptor alignment; the
// callers ask 256 for descriptors) with `padded` bytes behind it: else end(). A copy is anchored at its first writer,
// so two pointers aligned in guest memory are aligned in the copy.
It containing(const uint8_t* p, uint64_t size, uint64_t padded, uint64_t align, const std::vector<It>& regions) {
    if (regions.size() != 1) return g_regions.end();
    It it = regions[0];
    uint64_t off = uint64_t(p - it->first);
    if (p < it->first || off + size > it->second.size || off + padded > it->second.cap) return g_regions.end();
    if ((it->second.up.offset + off) % std::clamp<uint64_t>(align, 1, bind_align())) return g_regions.end();
    return it;
}

// ---- results carried to the next submit (odd/tasks/wb-carry.md) ------------------------------------------------
// A deferred submit's regions die with its stretch of the upload ring, and their results reach guest memory only when
// its write-backs are applied, after its fence. NieR reads, every frame, what the previous frame's compute wrote (a
// 1920-byte storage binding, 12-byte indirect dispatch arguments): each such read waited for that whole frame (~18 ms
// a frame on the PC, 33-57 on the Adreno 610). So at the end of a deferred submit each region is copied on the GPU into
// an arena of its own (written_carry), and the next submit makes a region of its own from that copy (written_adopt),
// on the GPU, instead of waiting: when the write-back is still pending and the guest bytes are still what the chain
// expects (the CPU did not write them since). NX_WB_CARRY=0: off.

struct Carry {
    uint64_t size = 0;
    uint64_t offset = 0;  // in the arena
    std::shared_ptr<std::vector<uint8_t>> expect;
    std::shared_ptr<WriteBackLink> link;  // the region's write-back's (also in its WriteBack)
};
std::map<const uint8_t*, Carry> g_carries;  // the last deferred submit's, by range start
VkBuffer g_arena = VK_NULL_HANDLE;
VkDeviceMemory g_arena_memory = VK_NULL_HANDLE;
uint64_t g_arena_size = 0, g_arena_head = 0;
// Arena allocations in order, each freed once the submit after its producer (which copied from it) retired.
struct ArenaUse {
    uint64_t start = 0, end = 0;
    uint64_t producer = 0, free_after = UINT64_MAX;  // free_after: that next submit's serial, once known
};
std::vector<ArenaUse> g_arena_uses;  // oldest first
uint64_t g_adopted = 0, g_adopt_cpu = 0, g_carried = 0, g_carry_bytes = 0, g_carry_full = 0;  // for written_log

bool carry_on() {
    static const bool on = !(std::getenv("NX_WB_CARRY") && std::getenv("NX_WB_CARRY")[0] == '0');
    return on && frame_write_backs_deferred() && !written_separate();
}

bool arena_make() {
    if (g_arena) return true;
    static bool failed = false;
    if (failed) return false;
    const char* e = std::getenv("NX_WB_CARRY_MB");
    g_arena_size = uint64_t(std::clamp(e ? std::atoi(e) : 32, 1, 512)) << 20;
    VkDevice d = vk_device();
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = g_arena_size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryRequirements req{};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    if (p_vkCreateBuffer(d, &bi, nullptr, &g_arena) == VK_SUCCESS) {
        p_vkGetBufferMemoryRequirements(d, g_arena, &req);
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (p_vkAllocateMemory(d, &ai, nullptr, &g_arena_memory) == VK_SUCCESS &&
            p_vkBindBufferMemory(d, g_arena, g_arena_memory, 0) == VK_SUCCESS) {
            log("nvn: compute results carried across submits (%llu MB arena)", (unsigned long long)(g_arena_size >> 20));
            return true;
        }
    }
    log("nvn: no arena for carried compute results: write-backs waited for as before");
    if (g_arena) p_vkDestroyBuffer(d, g_arena, nullptr), g_arena = VK_NULL_HANDLE;
    failed = true;
    return false;
}

// `bytes` of the arena (256-aligned) for the submit being recorded, or false when full.
bool arena_alloc(uint64_t bytes, uint64_t& at) {
    const uint64_t done = frame_completed();
    size_t gone = 0;
    while (gone < g_arena_uses.size() && g_arena_uses[gone].free_after <= done) gone++;
    g_arena_uses.erase(g_arena_uses.begin(), g_arena_uses.begin() + gone);
    bytes = (bytes + 255) & ~255ull;
    uint64_t start = 0;
    if (!g_arena_uses.empty()) {
        const uint64_t tail = g_arena_uses.front().start, head = g_arena_head;
        if (head >= tail) {  // live: [tail, head)
            if (head + bytes <= g_arena_size) start = head;
            else if (bytes <= tail) start = 0;
            else return false;
        } else {  // live: [tail, end) and [0, head)
            if (head + bytes <= tail) start = head;
            else return false;
        }
    }
    if (start + bytes > g_arena_size) return false;
    g_arena_uses.push_back({start, start + bytes, frame_recording_serial(), UINT64_MAX});
    g_arena_head = start + bytes;
    at = start;
    return true;
}

// A region of this submit for a carried range covering [p, p + size), made from the carry on the GPU; false when it
// cannot be used (the old path then applies the write-back and waits).
bool adopt(VkCommandBuffer cb, const uint8_t* p, uint64_t size, uint64_t padded) {
    if (g_carries.empty() || !p || !size || !g_arena) return false;
    auto next = g_carries.upper_bound(p);
    if (next == g_carries.begin()) return false;
    const auto& [lo, k] = *std::prev(next);
    if (p < lo || p + size > lo + k.size) return false;
    if (!overlapping(lo, k.size).empty()) return false;       // meets a region of this submit
    if (!frame_write_back_pending(lo, k.size)) return false;  // the result is in guest memory already
    if (!k.expect || k.expect->size() < k.size || std::memcmp(lo, k.expect->data(), k.size) != 0) {
        g_adopt_cpu++;  // the CPU wrote these bytes since: its words must not be lost
        return false;
    }
    const uint64_t off = uint64_t(p - lo);
    Region r;
    r.size = k.size;
    r.cap = std::max(k.size + kTail, off + padded);
    r.up = alloc_anchored(r.cap, off);
    std::memset(r.up.ptr + k.size, 0, r.cap - k.size);
    r.before = std::make_shared<std::vector<uint8_t>>(lo, lo + k.size);  // the GPU's bytes once the write-back lands
    r.expect = k.expect;
    k.link->before = r.before, k.link->expect = r.expect;
    draw_end_rendering(cb, 2);
    VkBufferCopy copy{k.offset, r.up.offset, k.size};
    p_vkCmdCopyBuffer(cb, g_arena, r.up.buffer, 1, &copy);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr,
                           0, nullptr);
    dedup_invalidate(lo, k.size);  // copies of these bytes made earlier in the submit are not what it now reads
    g_regions.emplace(lo, std::move(r));
    g_adopted++;
    return true;
}

}  // namespace

bool written_separate() {
    static const bool on = std::getenv("NX_COMPUTE_SEPARATE") != nullptr;
    return on;
}

uint64_t written_bind_count() { return g_binds; }
void written_count_bind() { g_binds++; }

Upload written_bind(VkCommandBuffer cb, const uint8_t* p, uint64_t size, uint64_t padded) {
    g_binds++;
    std::vector<It> old = overlapping(p, size);
    if (old.empty() && adopt(cb, p, size, padded)) old = overlapping(p, size);  // the last submit's result, on the GPU
    if (It it = containing(p, size, padded, bind_align(), old); it != g_regions.end()) {
        write_after_read(cb);
        g_shared++;
        return at(it, uint64_t(p - it->first));
    }
    // A new range, or one meeting others (or misaligned in its copy): one copy covering all of them, anchored at p.
    const uint8_t* lo = p;
    const uint8_t* hi = p + size;
    for (It it : old) lo = std::min(lo, it->first), hi = std::max(hi, it->first + it->second.size);
    uint64_t n = uint64_t(hi - lo), off = uint64_t(p - lo);
    Region r;
    r.size = n;
    r.cap = std::max(n + kTail, off + padded);
    r.up = alloc_anchored(r.cap, off);
    frame_guest_read(lo, n, kReadWrittenBind);  // deferred write-backs over these bytes land first (nvn_frames.cpp)
    std::memcpy(r.up.ptr, lo, n);
    std::memset(r.up.ptr + n, 0, r.cap - n);
    r.before = std::make_shared<std::vector<uint8_t>>(r.up.ptr, r.up.ptr + n);
    for (It it : old)  // their parts: as they were before their own GPU writes
        if (it->second.before)
            std::memcpy(r.before->data() + (it->first - lo), it->second.before->data(), it->second.before->size());
    if (!old.empty()) {
        copy_regions(cb, old, lo, n, r.up);  // dispatch(): no render pass is open
        for (It it : old) g_regions.erase(it);
        g_merged++;
    }
    dedup_invalidate(lo, n);  // copies of these bytes made earlier in the submit are not what later work reads
    It it = g_regions.emplace(lo, r).first;
    return at(it, off);
}

bool written_read(VkCommandBuffer cb, const uint8_t* p, uint64_t size, uint64_t padded, uint64_t align, Upload& out) {
    if (!p || !size || written_separate() || (g_regions.empty() && g_carries.empty())) return false;
    std::vector<It> regions = overlapping(p, size);
    if (regions.empty() && adopt(cb, p, size, padded)) regions = overlapping(p, size);  // the last submit's result
    if (regions.empty()) return false;
    if (It it = containing(p, size, padded, align, regions); it != g_regions.end()) {
        g_shared++, g_read = true;
        out = at(it, uint64_t(p - it->first));
        return true;
    }
    // Partly written, or not bindable in place: a copy of its own, the written parts copied in on the GPU. This is
    // what the range holds at this point of the submit (later dispatches do not change it).
    draw_end_rendering(cb, 2);
    out = upload_alloc(std::max(padded, size), std::max<uint64_t>(align, 16));
    // Only the bytes no region covers come from the guest: deferred write-backs over the rest need not land first
    // (their region copies replace them below). NieR's particle buffers: a wait for the last frame's GPU on every
    // composed read, ~29 ms a frame on the Adreno 610 in flight (PROGRESO 28).
    const uint8_t* gap = p;
    for (It it : regions) {
        if (it->first > gap) frame_guest_read(gap, uint64_t(it->first - gap), kReadWritten);
        gap = std::max(gap, it->first + it->second.size);
    }
    if (gap < p + size) frame_guest_read(gap, uint64_t(p + size - gap), kReadWritten);
    std::memcpy(out.ptr, p, size);
    if (padded > size) std::memset(out.ptr + size, 0, padded - size);
    copy_regions(cb, regions, p, size, out);
    g_composed++;
    return true;
}

Upload read_upload(VkCommandBuffer cb, const uint8_t* p, uint64_t size, uint64_t padded, uint64_t align) {
    Upload u;
    if (written_read(cb, p, size, padded, align, u)) return u;
    return dedup_upload(p, size, padded, align);
}

bool written_overlaps(const uint8_t* p, uint64_t size) {
    return p && size && !g_regions.empty() && !overlapping(p, size).empty();
}

// The parts of [p, p + size) inside regions of this submit, in address order, and whether they cover all of it.
static bool region_parts(const uint8_t* p, uint64_t size, std::vector<It>& regs) {
    regs = overlapping(p, size);
    uint64_t covered = 0;
    for (It it : regs)
        covered += uint64_t(std::min(it->first + it->second.size, p + size) - std::max(it->first, p));
    return !regs.empty() && covered == size;
}

// The clear in guest memory, after the earlier submits' results over it landed.
static void written_clear_buffer_guest(const uint8_t* p, uint64_t size, uint32_t value) {
    frame_guest_read(p, size, kReadOther);
    uint8_t* q = const_cast<uint8_t*>(p);
    for (uint64_t k = 0; k < size; k += 4) std::memcpy(q + k, &value, std::min<uint64_t>(4, size - k));
    dedup_invalidate(p, size);
}

void written_clear_buffer(VkCommandBuffer cb, const uint8_t* p, uint64_t size, uint32_t value) {
    if (!p || !size) return;
    static uint64_t gpu = 0, cpu = 0;
    if ((gpu + cpu) % 20000 == 0)
        log("nvn: ClearBuffer: %llu on GPU copies, %llu in guest memory", (unsigned long long)gpu, (unsigned long long)cpu);
    std::vector<It> regs;
    if (overlapping(p, size).empty()) written_adopt(cb, p, size);  // the last submit's result, on the GPU
    if (region_parts(p, size, regs) && size % 4 == 0) {  // a dispatch wrote it: clear the copy, in command order
        draw_end_rendering(cb, 2);
        for (It it : regs) {
            const uint8_t* lo = std::max(it->first, p);
            const uint8_t* hi = std::min(it->first + it->second.size, p + size);
            const uint64_t off = it->second.up.offset + uint64_t(lo - it->first);
            if (off % 4 || uint64_t(hi - lo) % 4) return written_clear_buffer_guest(p, size, value), void();
            p_vkCmdFillBuffer(cb, it->second.up.buffer, off, uint64_t(hi - lo), value);
        }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr,
                               0, nullptr);
        gpu++;
        return;
    }
    cpu++;
    written_clear_buffer_guest(p, size, value);
}

void written_copy_buffer(VkCommandBuffer cb, const uint8_t* src, const uint8_t* dst, uint64_t size) {
    if (!src || !dst || !size) return;
    if (overlapping(src, size).empty()) written_adopt(cb, src, size);
    if (overlapping(dst, size).empty()) written_adopt(cb, dst, size);
    const bool gpu_src = !overlapping(src, size).empty(), gpu_dst = !overlapping(dst, size).empty();
    static uint64_t gpu = 0, cpu = 0;
    if ((gpu + cpu) % 20000 == 0) log("nvn: CopyBufferToBuffer: %llu on the GPU, %llu in guest memory", (unsigned long long)gpu, (unsigned long long)cpu);
    if (!gpu_src && !gpu_dst) {  // plain guest memory on both sides: earlier submits' results land first
        cpu++;
        frame_guest_read(src, size, kReadOther);
        frame_guest_read(dst, size, kReadOther);
        std::memmove(const_cast<uint8_t*>(dst), src, size);
        dedup_invalidate(dst, size);
        return;
    }
    // GPU data on one side: the copy goes into a region for dst (written back like a dispatch's output), in order.
    gpu++;
    Upload from;
    if (!written_read(cb, src, size, size, 4, from)) from = dedup_upload(src, size, size, 4);
    Upload to = written_bind(cb, dst, size, size);
    draw_end_rendering(cb, 2);
    VkBufferCopy copy{from.offset, to.offset, size};
    p_vkCmdCopyBuffer(cb, from.buffer, to.buffer, 1, &copy);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr,
                           0, nullptr);
}

bool written_adopt(VkCommandBuffer cb, const uint8_t* p, uint64_t size) {
    return p && size && !written_separate() && overlapping(p, size).empty() && adopt(cb, p, size, size);
}

bool written_indirect(VkCommandBuffer cb, const uint8_t* p) {
    Upload u;
    if (!written_read(cb, p, 12, 16, 4, u)) return false;
    p_vkCmdDispatchIndirect(cb, u.buffer, u.offset);  // {x, y, z} as the earlier dispatches left them
    g_indirect++;
    return true;
}

// The vertices [lo, hi] and the last instance a draw fetches; false when unknown (no count, no indices).
static bool fetched_range(bool indexed, const Cmd& c, uint64_t& lo, uint64_t& hi, uint64_t& inst_lo, uint64_t& inst_hi) {
    uint32_t count = uint32_t(indexed ? c.i[0] : c.i[1]);
    if (!count) return false;
    if (indexed) {
        const uint8_t* p = guest_bytes(c.a[0]);
        if (!p) return false;
        uint32_t isz = index_size(c.mask), restart = isz == 1 ? 0xFF : isz == 2 ? 0xFFFF : 0xFFFFFFFFu;
        uint32_t mn = UINT32_MAX, mx = 0;
        for (uint32_t k = 0; k < count; k++) {
            uint32_t v = isz == 1 ? p[k] : isz == 2 ? reinterpret_cast<const uint16_t*>(p)[k] : reinterpret_cast<const uint32_t*>(p)[k];
            if (v == restart) continue;
            mn = std::min(mn, v), mx = std::max(mx, v);
        }
        if (mn == UINT32_MAX || int64_t(mn) + c.i[1] < 0) return false;
        lo = uint64_t(int64_t(mn) + c.i[1]), hi = uint64_t(int64_t(mx) + c.i[1]);
    } else {
        lo = uint32_t(c.i[0]), hi = uint64_t(uint32_t(c.i[0])) + count - 1;
    }
    inst_lo = uint32_t(c.i[2]);
    inst_hi = inst_lo + uint32_t(std::max(1, c.i[3])) - 1;
    return true;
}

bool draw_inputs(VkCommandBuffer cb, const ShaderInfo* vs, bool indexed, const Cmd& c, DrawInputs& in) {
    // Vertex buffers: the bound ranges of the streams the vertex shader reads (the pipeline declares no others).
    for (const ShaderIo& io : vs->refl.inputs)
        if (io.location < g_state.attribs.size() && g_state.attribs[io.location].stream < 15)
            in.used |= 1u << g_state.attribs[io.location].stream;
    // Only the bytes of the vertices the draw fetches (NieR binds whole vertex pools of several MB per draw: ~80 MB
    // a frame were copied, 1% of it the same as the last copy). The copy keeps its place in the stream, so the
    // binding (offset - lo) and the vertex numbers do not change. Its effect buffer is bound from the same base with
    // a size that grows draw by draw: each draw copied all of it again (~20 times a frame). In the prologue flight at
    // 720p, controlled A/B: 123 -> 63 MB uploaded a frame, 16.5 -> 16.9 fps on the PC (PROGRESO). NX_VERTEX_TRIM=0: off.
    static const bool trim = !(std::getenv("NX_VERTEX_TRIM") && std::getenv("NX_VERTEX_TRIM")[0] == '0');
    uint64_t lo = 0, hi = 0, inst_lo = 0, inst_hi = 0;
    // From the command, not `indexed`: draw_inputs_shared asks for the vertex streams of an indexed draw with false.
    const bool ranged = trim && !g_draw_untrimmed && fetched_range(c.op == Cmd::DrawElements, c, lo, hi, inst_lo, inst_hi);
    for (uint32_t s = 0; s < 15; s++) {
        if (!(in.used >> s & 1)) continue;
        const Range& r = g_state.vertex_buffers[s];
        const uint8_t* p = r.address ? guest_bytes(r.address) : nullptr;
        uint64_t size = p ? std::min<uint64_t>(r.size, 64ull << 20) : 0;
        g_upload_kind = kUploadVertex;
        uint64_t from = 0, end = 0;
        if (ranged && size > 4096 && s < g_state.streams.size()) {
            const VertexStream& st = g_state.streams[s];
            from = UINT64_MAX;
            for (const ShaderIo& io : vs->refl.inputs) {
                if (io.location >= g_state.attribs.size() || g_state.attribs[io.location].stream != s) continue;
                const VertexAttrib& va = g_state.attribs[io.location];
                uint64_t first = st.divisor ? inst_lo / st.divisor : lo, last = st.divisor ? inst_hi / st.divisor : hi;
                from = std::min(from, first * st.stride + va.offset);
                end = std::max(end, last * st.stride + va.offset + vk_block_bytes(vk_vertex_format(va.format)));
            }
            from &= ~uint64_t(255);  // the attributes keep their alignment
        }
        static const uint64_t big = std::getenv("NX_LOG_BIG_VERTEX") ? std::strtoull(std::getenv("NX_LOG_BIG_VERTEX"), nullptr, 10) : 0;
        if (big && size >= big) {  // diagnosis: the large vertex ranges, what the draw fetches of them, GPU-written or not
            static int logged = 0;
            uint64_t l = 0, h = 0, il = 0, ih = 0;
            bool known = fetched_range(c.op == Cmd::DrawElements, c, l, h, il, ih);
            uint32_t stride = s < g_state.streams.size() ? g_state.streams[s].stride : 0;
            if (logged++ < 60)
                log("nvn: big vertex stream %u at %llx: %llu bytes, stride %u, vertices %llu..%llu%s, %s, vs %s", s,
                    (unsigned long long)r.address, (unsigned long long)size, stride, (unsigned long long)l,
                    (unsigned long long)h, known ? "" : " (unknown)", written_overlaps(p, size) ? "GPU-written" : "CPU",
                    vs->hash.c_str());
        }
        if (ranged && from < end && end <= size && (from > 0 || end + 4096 < size) && !written_overlaps(p, size)) {
            Upload u = dedup_upload(p + from, end - from, (end - from + 15) & ~15ull, 16);
            if (u.offset >= from) {
                u.offset -= from, u.ptr -= from;
            } else {  // too near the ring's start to bind at offset - from: a copy of [0, end) with only the fetched part
                u = upload_alloc((end + 15) & ~15ull, 16);
                frame_guest_read(p + from, end - from, kReadVertex);
                std::memcpy(u.ptr + from, p + from, end - from);
            }
            in.vertex[s] = u;
        } else {
            in.vertex[s] = read_upload(cb, p, size, std::max<uint64_t>((size + 15) & ~15ull, 16), 16);
        }
        g_upload_kind = kUploadOther;
    }
    if (!indexed) return true;
    uint32_t count = uint32_t(c.i[0]);
    const uint8_t* p = guest_bytes(c.a[0]);
    if (!p) return false;
    in.index_type = vk_index_type(c.mask);
    if (in.index_type == VK_INDEX_TYPE_UINT8_EXT) {
        // 8-bit indices: Metal has none, and MoltenVK converts them on the GPU at every draw (that was most of the
        // frame's GPU time), so they are widened here, from guest memory (a dispatch writing them is only logged).
        static int logs = 0;
        if (written_overlaps(p, count) && logs++ < 16) log("nvn: 8-bit indices written by a dispatch: stale");
        in.index = upload_alloc(uint64_t(count) * 2, 16);
        auto* d = reinterpret_cast<uint16_t*>(in.index.ptr);
        for (uint32_t k = 0; k < count; k++) d[k] = p[k];
        in.index_type = VK_INDEX_TYPE_UINT16;
        return true;
    }
    uint64_t bytes = uint64_t(count) * index_size(c.mask);
    if (!written_read(cb, p, bytes, bytes, index_size(c.mask), in.index)) in.index = upload(p, bytes, 16);
    return true;
}

void check_vertex_ranges(const ShaderInfo* vs, const ShaderInfo* fs, bool indexed, const Cmd& c, const DrawInputs& in) {
    static const bool on = std::getenv("NX_CHECK_VERTS") != nullptr;
    static uint64_t found = 0;
    if (!on || !vs) return;
    // The vertices and instances the draw fetches.
    uint64_t lo = 0, hi = 0;  // vertex indices [lo, hi]
    uint32_t count = uint32_t(indexed ? c.i[0] : c.i[1]);
    if (!count) return;
    if (indexed) {
        const uint8_t* p = guest_bytes(c.a[0]);
        if (!p) return;
        uint32_t isz = index_size(c.mask), restart = isz == 1 ? 0xFF : isz == 2 ? 0xFFFF : 0xFFFFFFFFu;
        uint32_t mn = UINT32_MAX, mx = 0;
        for (uint32_t k = 0; k < count; k++) {
            uint32_t v = isz == 1 ? p[k] : isz == 2 ? reinterpret_cast<const uint16_t*>(p)[k] : reinterpret_cast<const uint32_t*>(p)[k];
            if (v == restart) continue;
            mn = std::min(mn, v), mx = std::max(mx, v);
        }
        if (mn == UINT32_MAX) return;
        lo = uint64_t(int64_t(mn) + c.i[1]), hi = uint64_t(int64_t(mx) + c.i[1]);
    } else {
        lo = uint32_t(c.i[0]), hi = uint64_t(uint32_t(c.i[0])) + count - 1;
    }
    uint64_t inst_hi = uint64_t(uint32_t(c.i[2])) + uint32_t(std::max(1, c.i[3])) - 1;
    for (const ShaderIo& io : vs->refl.inputs) {
        if (io.location >= g_state.attribs.size()) continue;
        const VertexAttrib& va = g_state.attribs[io.location];
        if (va.stream >= 15 || va.stream >= g_state.streams.size() || !(in.used >> va.stream & 1)) continue;
        const VertexStream& st = g_state.streams[va.stream];
        const Range& r = g_state.vertex_buffers[va.stream];
        uint64_t last = st.divisor ? inst_hi / st.divisor : hi;
        uint64_t end = last * st.stride + va.offset + vk_block_bytes(vk_vertex_format(va.format));
        uint64_t copied = r.address && guest_bytes(r.address) ? std::min<uint64_t>(r.size, 64ull << 20) : 0;
        if (end <= copied) continue;
        found++;
        static std::map<std::string, uint64_t> seen;
        std::string key = vs->hash + (fs ? fs->hash : std::string("-")) + char('a' + va.stream);
        uint64_t n = ++seen[key];
        if (n <= 3 || (n & (n - 1)) == 0)
            log("nvn: vertex fetch past the bound range (%llu so far; #%llu of vs %s fs %s): stream %u loc %u fmt %#x off %u "
                "stride %u div %u, vertices %llu..%llu instances ..%llu, needs %llu bytes, bound %llu at %llx (copied %llu)",
                (unsigned long long)found, (unsigned long long)n, vs->hash.c_str(), fs ? fs->hash.c_str() : "-", va.stream,
                io.location, va.format, va.offset, st.stride, st.divisor, (unsigned long long)lo, (unsigned long long)hi,
                (unsigned long long)inst_hi, (unsigned long long)end, (unsigned long long)r.size,
                (unsigned long long)r.address, (unsigned long long)copied);
        break;
    }
}

void written_cpu_write(const uint8_t* p, uint64_t size) {
    // UpdateUniformBuffer / ReportCounter into a range a dispatch of this submit writes: the copy misses the bytes
    // (they reach guest memory at replay time) and the write-back replaces them. Not handled: logged.
    static int logs = 0;
    if (written_overlaps(p, size) && logs++ < 16)
        log("nvn: guest write into a compute-written range (%llu bytes): lost at write-back", (unsigned long long)size);
}

std::vector<WriteBack> written_take() {
    std::vector<WriteBack> v;
    v.reserve(g_regions.size());
    for (auto& [p, r] : g_regions) v.push_back({r.up.ptr, const_cast<uint8_t*>(p), r.size, r.before, r.link});
    g_regions.clear();
    return v;
}

// End of a deferred submit (nvn_vk.cpp, before frame_submit takes its write-backs): every region copied into the
// arena for the next submit to adopt. The carries of the submit before are dropped: the next one can only adopt
// these (its arena space is kept until the submit after this one, which copied from it, retired).
void written_carry(VkCommandBuffer cb) {
    const uint64_t serial = frame_recording_serial();
    for (ArenaUse& u : g_arena_uses)
        if (u.producer < serial && u.free_after == UINT64_MAX) u.free_after = serial;
    g_carries.clear();
    if (g_regions.empty() || !carry_on() || !arena_make()) return;
    bool any = false;
    for (auto& [p, r] : g_regions) {
        uint64_t at = 0;
        if (!r.before || r.before->size() < r.size || !arena_alloc(r.size, at)) {
            g_carry_full++;
            continue;
        }
        if (!any) draw_end_rendering(cb, 2), any = true;
        VkBufferCopy copy{r.up.offset, at, r.size};
        p_vkCmdCopyBuffer(cb, r.up.buffer, g_arena, 1, &copy);
        if (!r.expect) r.expect = std::make_shared<std::vector<uint8_t>>(r.before->begin(), r.before->begin() + r.size);
        if (!r.link) r.link = std::make_shared<WriteBackLink>();
        g_carries[p] = Carry{r.size, at, r.expect, r.link};
        g_carried++, g_carry_bytes += r.size;
    }
    if (!any) return;
    // What the next submit reads from the arena (later in submission order) waits for these copies.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    p_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr,
                           0, nullptr);
}

void written_reset() { g_regions.clear(), g_read = false; }

void written_log() {
    if (g_shared + g_merged + g_composed + g_indirect)
        log("nvn: compute ranges: %llu bindings of written copies, %llu merged, %llu reads composed on the GPU, %llu indirect",
            (unsigned long long)g_shared, (unsigned long long)g_merged, (unsigned long long)g_composed,
            (unsigned long long)g_indirect);
    if (g_carried + g_adopted + g_adopt_cpu)
        log("nvn: carried compute results: %llu ranges (%.1f MB), %llu adopted by the next submit, %llu refused (CPU "
            "writes), %llu not carried (arena full)", (unsigned long long)g_carried, double(g_carry_bytes) / 1e6,
            (unsigned long long)g_adopted, (unsigned long long)g_adopt_cpu, (unsigned long long)g_carry_full);
    g_shared = g_merged = g_composed = g_indirect = 0;
    g_carried = g_carry_bytes = g_adopted = g_adopt_cpu = g_carry_full = 0;
}

}  // namespace nx::nvn::replay

namespace nx::nvn {

// A bound storage range is often larger than what the shader writes, and the game's recording thread writes the
// rest of those buffers while the submit is replayed. Copying the whole range back put stale bytes over them: meshes
// exploded into beams and a band of stretched triangles at the horizon, for the rest of the run (DEBUG.md #56). Only
// the words the GPU changed go back. NX_WRITE_BACK_WHOLE=1: the old whole-range copy.
// Per frame over the timing window (log_timing, nvn_vk.cpp).
void write_back_log(uint64_t frames) {
    const uint64_t n = g_wb_count.exchange(0), bytes = g_wb_bytes.exchange(0), changed = g_wb_changed.exchange(0),
                   ns = g_wb_ns.exchange(0);
    if (n && frames)
        log("nvn: write-backs per frame: %.0f, %.2f MB compared, %.0f%% of the words changed, %.2f ms", double(n) / double(frames),
            double(bytes) / 1e6 / double(frames), bytes ? 800.0 * double(changed) / double(bytes) : 0.0,
            double(ns) / 1e6 / double(frames));
}

void apply_write_back(const WriteBack& w) {
    if (w.size == 0) return;
    static const bool whole = std::getenv("NX_WRITE_BACK_WHOLE") != nullptr;
    const uint64_t t0 = plat::monotonic_ns();
    g_wb_count++, g_wb_bytes += w.size;
    // The next submit's region adopted from this result (written_adopt): its initial content is the GPU's bytes, and
    // the guest bytes it expects change where this write-back changes them.
    std::vector<uint8_t>* next_before = w.next && w.next->before && w.next->before->size() >= w.size ? w.next->before.get() : nullptr;
    std::vector<uint8_t>* next_expect = w.next && w.next->expect && w.next->expect->size() >= w.size ? w.next->expect.get() : nullptr;
    if (whole || !w.before || w.before->size() < w.size) {
        std::memcpy(w.to, w.from, w.size);
        if (next_before) std::memcpy(next_before->data(), w.to, w.size);
        if (next_expect) std::memcpy(next_expect->data(), w.to, w.size);
        g_wb_ns += plat::monotonic_ns() - t0;
        return;
    }
    const uint8_t* was = w.before->data();
    // The GPU's bytes copied out of the upload ring in one memcpy first: compared word by word in place, they were
    // read at ~34 MB/s on the Adreno 610 (55 ms a frame on the 3D title, 7.7 fps; with the copy 6 ms, 13.9 fps;
    // a host-cached ring alone changed nothing). NX_WB_BULK=0: in place, as before.
    static const bool bulk = !(std::getenv("NX_WB_BULK") && std::getenv("NX_WB_BULK")[0] == '0');
    static thread_local std::vector<uint8_t> copy;
    const uint8_t* from = w.from;
    if (bulk) {
        if (copy.size() < w.size) copy.resize(w.size);
        std::memcpy(copy.data(), w.from, w.size);
        from = copy.data();
    }
    // Early vectorized check: eliminate ~5 ms CPU stall per frame when data is unchanged
    if (std::memcmp(from, was, w.size) == 0) {
        if (next_before) std::memcpy(next_before->data(), from, w.size);
        g_wb_ns += plat::monotonic_ns() - t0;
        return;
    }
    uint8_t* expect = next_expect ? next_expect->data() : nullptr;
    uint64_t k = 0, changed = 0;
    for (; k + 8 <= w.size; k += 8) {
        if (std::memcmp(from + k, was + k, 8) == 0) continue;
        changed++;
        for (uint64_t b = k; b < k + 8; b += 4)
            if (std::memcmp(from + b, was + b, 4) != 0) {
                std::memcpy(w.to + b, from + b, 4);
                if (expect) std::memcpy(expect + b, from + b, 4);
            }
    }
    for (; k < w.size; k++)
        if (from[k] != was[k]) {
            w.to[k] = from[k];
            if (expect) expect[k] = from[k];
        }
    if (next_before) std::memcpy(next_before->data(), from, w.size);
    // NX_LOG_WB_SIZE=<bytes>: the first floats of write-backs of that size, now and then (what a compute produced).
    static const uint64_t log_size = std::getenv("NX_LOG_WB_SIZE") ? std::strtoull(std::getenv("NX_LOG_WB_SIZE"), nullptr, 10) : 0;
    static uint64_t log_n = 0;
    if (log_size && w.size == log_size && (log_n++ % 120) == 0) {
        std::string line;
        for (uint64_t k = 0; k < 24 && 4 * k + 4 <= w.size; k++) {
            float f;
            std::memcpy(&f, from + 4 * k, 4);
            char b[24];
            std::snprintf(b, sizeof b, " %.4g", double(f));
            line += b;
        }
        log("nvn: write-back %p (%llu bytes), %llu words changed:%s", (void*)w.to, (unsigned long long)w.size,
            (unsigned long long)changed, line.c_str());
    }
    g_wb_changed += changed;
    g_wb_ns += plat::monotonic_ns() - t0;
}

}  // namespace nx::nvn
