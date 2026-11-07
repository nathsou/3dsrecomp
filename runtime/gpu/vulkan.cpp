// Vulkan backend: metal.mm's design on Vulkan (rasterization and the whole fragment pipeline on the GPU at an internal
// resolution scale, optionally multisampled; guest memory authoritative, render targets synchronised with it before
// drawing and written back lazily; texture copies and tiled-to-tiled transfers of GPU images kept on the GPU), for
// Linux (and MoltenVK on a Mac). Fragment shaders come from fsgen.cpp in GLSL, compiled with shaderc; the fixed
// shaders are in vk_shaders.h. Vertex programs are shaded by the CPU (GPU kernels: to come).
//
// What Metal did for us, done here by hand:
// - Ordering: one command buffer is recorded at a time (rec(), the GX thread's), submitted with the next value of
//   a timeline semaphore; waiting for a value waits for everything submitted before it. Every image stays in the
//   GENERAL layout, and a full memory barrier separates the operations that Metal kept in separate encoders.
// - Lifetimes: buffers and images are shared_ptrs whose last release schedules their destruction after the GPU work
//   recorded so far (retire()).
// - Unified memory: what the CPU writes for the GPU (vertices, uniforms, LUTs, uploads) goes through host-visible
//   chunks (arena()), what the CPU reads back through host-cached buffers.
#include <vulkan/vulkan.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include "vma/vk_mem_alloc.h"
#pragma clang diagnostic pop

#include <shaderc/shaderc.h>

#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "gpu/fsgen.h"
#include "gpu/pica.h"
#include "gpu/vk_shaders.h"
#include "gpu/vk_texfilter.h"
#include "gpu/vsblob.h"
#include "loader.h"

extern std::string g_save_base;

namespace gpu {

namespace {

// ---------------------------------------------------------------- device
VkInstance inst;
VkPhysicalDevice phys;
VkDevice dev;
VkQueue queue;
u32 qfam = 0;
// Presenting on this device (gpu_iface.h's present_device): SDL's Vulkan renderer submits to queue 0 of qfam from
// the UI thread, our work then goes to queue 1. queue_mtx: around our submissions (present_hold keeps them back
// while SDL waits for the whole device).
bool present_ok = false;
VkQueue sdl_queue = VK_NULL_HANDLE;
std::mutex &queue_mtx = *new std::mutex;
VmaAllocator vma;
VkPhysicalDeviceProperties props;
PFN_vkCmdPushDescriptorSetKHR vkPushDescriptors;
VkFormat depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
bool has_stencil_export = false, has_local_read = false, has_logic_op = false;
u32 S = 1;                     // internal resolution scale
u32 MS = 1;                    // MSAA samples (GPU_MSAA)
u32 aniso = 1;                 // max anisotropy (GPU_ANISO)
bool scale_fixed = false;      // GPU_SCALE given
constexpr u32 kDefaultMSAA = 4, kDefaultAniso = 16;
std::atomic<u32> want_scale{0};
bool active = false, tried = false;
bool log_timing = false, eager_writeback = false, dump_shaders = false, sync_shaders = false, verify_keys = false;
bool warned_shadow = false, warned_proctex = false, warned_fetch = false;

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        const VkResult r_ = (x);                                                                                       \
        if (r_ != VK_SUCCESS) FATAL("gpu vulkan: %s failed (%d)\n", #x, (int)r_);                                      \
    } while (0)

// ---------------------------------------------------------------- submissions and lifetimes
VkSemaphore timeline;
std::atomic<u64> tl_submitted{0};  // value of the last submission
std::mutex &garbage_mtx = *new std::mutex;
std::deque<std::pair<u64, std::function<void()>>> &garbage = *new std::deque<std::pair<u64, std::function<void()>>>;
std::atomic<bool> recording{false};  // (GX thread) a command buffer is being recorded

// f runs once the GPU is done with everything recorded so far
void retire(std::function<void()> f) {
    const u64 v = tl_submitted.load() + (recording ? 1 : 0);
    std::lock_guard<std::mutex> lk(garbage_mtx);
    garbage.emplace_back(v, std::move(f));
}

u64 completed_value() {
    u64 v = 0;
    vkGetSemaphoreCounterValue(dev, timeline, &v);
    return v;
}

struct Buf {
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocation a = nullptr;
    u8 *p = nullptr;  // mapped (host-visible ones)
    size_t size = 0;
    ~Buf() {
        if (!b) return;
        VkBuffer bb = b;
        VmaAllocation aa = a;
        retire([bb, aa] { vmaDestroyBuffer(vma, bb, aa); });
    }
    // after the GPU wrote it (host-cached memory need not be coherent)
    void invalidate() const { vmaInvalidateAllocation(vma, a, 0, VK_WHOLE_SIZE); }
    void flush() const { vmaFlushAllocation(vma, a, 0, VK_WHOLE_SIZE); }
};
using BufP = std::shared_ptr<Buf>;
enum class Mem { Device, Upload, Readback };
constexpr VkBufferUsageFlags kBufUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
BufP make_buf(size_t n, Mem m) {
    auto b = std::make_shared<Buf>();
    b->size = std::max<size_t>(n, 16);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = b->size;
    bi.usage = kBufUsage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    if (m == Mem::Upload) ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (m == Mem::Readback) ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(vma, &bi, &ai, &b->b, &b->a, &info));
    b->p = (u8 *)info.pMappedData;
    return b;
}

struct Img {
    VkImage img = VK_NULL_HANDLE;
    VmaAllocation a = nullptr;
    VkImageView view = VK_NULL_HANDLE;                     // all aspects (attachments, storage, sampling colour)
    VkImageView vdepth = VK_NULL_HANDLE, vstencil = VK_NULL_HANDLE;  // depth images: one aspect each (sampling)
    u32 w = 0, h = 0, levels = 1;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    std::vector<VkImageView> lviews;  // one mip level each (storage images: texture filters), made when needed
    ~Img() {
        if (!img) return;
        VkImage i = img;
        VmaAllocation aa = a;
        std::vector<VkImageView> v = {view, vdepth, vstencil};
        v.insert(v.end(), lviews.begin(), lviews.end());
        retire([i, aa, v] {
            for (VkImageView x : v)
                if (x) vkDestroyImageView(dev, x, nullptr);
            vmaDestroyImage(vma, i, aa);
        });
    }
};
using ImgP = std::shared_ptr<Img>;

// ---------------------------------------------------------------- command buffers
// One command buffer recorded at a time; submitted at the end of a command list (vk_end_list) or before work that
// must come after it, then reused once its timeline value is reached. The chunks of host memory it used for its
// uploads go back to chunk_free then.
struct Cmd {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    u64 value = 0;
    u32 qslot = 0;  // timestamp queries 2*qslot, 2*qslot+1
    std::vector<BufP> chunks;
};
VkCommandPool cmd_pool;
std::deque<Cmd *> inflight;
std::vector<Cmd *> free_cmds;
Cmd *cur = nullptr;
std::vector<BufP> chunk_free;
size_t chunk_off = 0;
size_t chunk_last = SIZE_MAX;  // offset of the last slice taken from the current chunk (the only one arena_trim may shrink)
constexpr size_t kChunkBytes = 4u << 20, kMaxFreeChunks = 16;
VkQueryPool qpool = VK_NULL_HANDLE;
constexpr u32 kQuerySlots = 256;
u32 qnext = 0;
bool ts_ok = false;
double ts_ns = 1;  // nanoseconds per timestamp tick

// Time the GPU spends on our command buffers (union of their execution intervals): the frontend's automatic resolution
// backs off when the GPU is saturated (busy), and RECOMP_STATS logs it every 2 s
std::atomic<float> busy_frac{0};
double gpu_busy = 0, busy_end = 0;  // seconds (GX thread)
std::atomic<u64> cbs_made{0}, flushes{0};

void reap(u64 done) {
    while (!inflight.empty() && inflight.front()->value <= done) {
        Cmd *c = inflight.front();
        inflight.pop_front();
        if (ts_ok) {
            u64 t[2] = {0, 0};
            if (vkGetQueryPoolResults(dev, qpool, c->qslot * 2, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
                t[1] > t[0]) {
                const double t0 = t[0] * ts_ns * 1e-9, t1 = t[1] * ts_ns * 1e-9;
                const double b = std::max(t0, busy_end);
                if (t1 > b) gpu_busy += t1 - b;
                busy_end = std::max(busy_end, t1);
            }
        }
        for (BufP &b : c->chunks)
            if (chunk_free.size() < kMaxFreeChunks) chunk_free.push_back(std::move(b));
        c->chunks.clear();
        vkResetCommandBuffer(c->cb, 0);
        free_cmds.push_back(c);
    }
    std::vector<std::function<void()>> run;
    {
        std::lock_guard<std::mutex> lk(garbage_mtx);
        while (!garbage.empty() && garbage.front().first <= done) {
            run.push_back(std::move(garbage.front().second));
            garbage.pop_front();
        }
    }
    for (auto &f : run) f();
}
void reap() { reap(completed_value()); }

// waits for timeline value v (GX thread: reaps what completed)
void wait_value(u64 v, const char *why = nullptr, u32 addr = 0) {
    if (!v) return;
    const double t0 = g_stats ? now_ms() : 0;
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    VK_CHECK(vkWaitSemaphores(dev, &wi, UINT64_MAX));
    if (g_stats && why) {
        const double t = now_ms() - t0;
        if (t > 0.05) LOG("gpu vulkan: waited %.2f ms for the GPU (%s %08x)\n", t, why, addr);
    }
}

// a full barrier: everything recorded before (and submitted earlier) happens before everything after
void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
}

// the command buffer being recorded (begun if none)
VkCommandBuffer rec() {
    if (cur) return cur->cb;
    if (free_cmds.empty() && inflight.size() >= kQuerySlots - 1) wait_value(inflight.front()->value), reap();
    if (free_cmds.empty()) {
        Cmd *c = new Cmd;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cmd_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &c->cb));
        free_cmds.push_back(c);
    }
    cur = free_cmds.back();
    free_cmds.pop_back();
    cur->qslot = qnext++ % kQuerySlots;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cur->cb, &bi));
    recording = true;
    if (ts_ok) {
        vkCmdResetQueryPool(cur->cb, qpool, cur->qslot * 2, 2);
        vkCmdWriteTimestamp2(cur->cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, qpool, cur->qslot * 2);
    }
    barrier(cur->cb);  // (after the work submitted before)
    cbs_made.fetch_add(1, std::memory_order_relaxed);
    return cur->cb;
}

// submits the command buffer being recorded; its timeline value (the last one if nothing was recorded)
u64 submit() {
    if (!cur) return tl_submitted;
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};  // results the CPU reads once it completes
    mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cur->cb, &di);
    if (ts_ok) vkCmdWriteTimestamp2(cur->cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, qpool, cur->qslot * 2 + 1);
    VK_CHECK(vkEndCommandBuffer(cur->cb));
    const u64 v = tl_submitted.load() + 1;
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = timeline;
    sig.value = v;
    sig.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = cur->cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &sig;
    {
        std::lock_guard<std::mutex> lk(queue_mtx);
        VK_CHECK(vkQueueSubmit2(queue, 1, &si, VK_NULL_HANDLE));
    }
    tl_submitted = v;
    cur->value = v;
    inflight.push_back(cur);
    cur = nullptr;
    recording = false;
    chunk_off = 0;
    return v;
}

// metal.mm's new_cb(): what is recorded from here on runs after everything recorded so far (which goes now)
VkCommandBuffer new_cb() {
    submit();
    return rec();
}

struct Slice {
    Buf *buf;
    size_t off;
    u8 *p;
};
// n bytes of host-visible memory for the command buffer being recorded (256-byte aligned)
Slice arena(size_t n) {
    n = (std::max<size_t>(n, 16) + 255) & ~(size_t)255;
    rec();
    auto &ch = cur->chunks;
    if (ch.empty() || chunk_off + n > ch.back()->size) {
        BufP b;
        for (size_t i = 0; i < chunk_free.size(); i++)
            if (chunk_free[i]->size >= n) {
                b = std::move(chunk_free[i]);
                chunk_free.erase(chunk_free.begin() + i);
                break;
            }
        if (!b) b = make_buf(std::max(kChunkBytes, n), Mem::Upload);
        ch.push_back(std::move(b));
        chunk_off = 0;
    }
    Slice s{ch.back().get(), chunk_off, ch.back()->p + chunk_off};
    chunk_last = chunk_off;
    chunk_off += n;
    return s;
}
// gives back the end of the last slice taken (only `used` bytes of it were needed); a slice with others taken after
// it keeps its size (they lie beyond its end)
void arena_trim(const Slice &s, size_t used) {
    if (cur && !cur->chunks.empty() && s.buf == cur->chunks.back().get() && s.off == chunk_last)
        chunk_off = std::min(chunk_off, s.off + ((used + 255) & ~(size_t)255));
}
void ensure_buffer(BufP &b, size_t n, Mem m = Mem::Readback) {
    if (!b || b->size < n) b = make_buf(n + n / 2 + 4096, m);
}

// ---------------------------------------------------------------- images
VkImageView make_view(VkImage img, VkFormat fmt, VkImageAspectFlags aspect, u32 levels) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {aspect, 0, levels, 0, 1};
    VkImageView v;
    VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &v));
    return v;
}
// a view of one mip level of a colour image
VkImageView level_view(Img &im, u32 l) {
    if (im.lviews.size() < im.levels) im.lviews.resize(im.levels, VK_NULL_HANDLE);
    if (!im.lviews[l]) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = im.fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &im.lviews[l]));
    }
    return im.lviews[l];
}
bool is_depth(VkFormat f) { return f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT; }

// a new image, in the GENERAL layout (from the command buffer being recorded on)
ImgP make_img(u32 w, u32 h, VkFormat fmt, u32 levels, VkImageUsageFlags usage, u32 samples = 1) {
    auto im = std::make_shared<Img>();
    im->w = w;
    im->h = h;
    im->levels = levels;
    im->fmt = fmt;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {std::max(w, 1u), std::max(h, 1u), 1};
    ci.mipLevels = levels;
    ci.arrayLayers = 1;
    ci.samples = (VkSampleCountFlagBits)samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateImage(vma, &ci, &ai, &im->img, &im->a, nullptr));
    const bool depth = is_depth(fmt);
    const VkImageAspectFlags all = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    im->view = make_view(im->img, fmt, all, levels);
    if (depth && samples == 1 && (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        im->vdepth = make_view(im->img, fmt, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
        im->vstencil = make_view(im->img, fmt, VK_IMAGE_ASPECT_STENCIL_BIT, 1);
    }
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im->img;
    b.subresourceRange = {all, 0, levels, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(rec(), &di);
    return im;
}

constexpr VkImageUsageFlags kColorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
constexpr VkImageUsageFlags kDepthUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageUsageFlags kUpscaledUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
constexpr VkImageUsageFlags kTexUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

// ---------------------------------------------------------------- shaders
std::vector<u32> compile_glsl(const std::string &src, shaderc_shader_kind kind, const char *name) {
    thread_local shaderc_compiler_t sc = shaderc_compiler_initialize();
    thread_local shaderc_compile_options_t opt = [] {
        shaderc_compile_options_t o = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(o, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        return o;
    }();
    shaderc_compilation_result_t r = shaderc_compile_into_spv(sc, src.data(), src.size(), kind, name, "main", opt);
    std::vector<u32> spv;
    if (shaderc_result_get_compilation_status(r) == shaderc_compilation_status_success) {
        const size_t n = shaderc_result_get_length(r);
        spv.resize(n / 4);
        memcpy(spv.data(), shaderc_result_get_bytes(r), n);
    } else {
        LOG("gpu vulkan: %s does not compile:\n%s\n%s\n", name, shaderc_result_get_error_message(r), src.c_str());
    }
    shaderc_result_release(r);
    return spv;
}
VkShaderModule make_module(const std::vector<u32> &spv) {
    if (spv.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spv.size() * 4;
    ci.pCode = spv.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

// compute kernels: storage buffers 0-3, storage images 4-5, push constants p, q (uvec4 each)
enum Kern { K_ROT, K_ENC, K_DEC, K_DEC_DEPTH, K_ENC_DEPTH, K_TCONV, K_XFER, K_SNAP, K_COUNT };
VkDescriptorSetLayout k_dsl;
VkPipelineLayout k_layout;
VkPipeline kpipe[K_COUNT];

struct KBind {
    u32 binding;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize off = 0, range = VK_WHOLE_SIZE;
    VkImageView view = VK_NULL_HANDLE;
};
KBind kb(u32 binding, const Buf &b, VkDeviceSize off = 0, VkDeviceSize range = VK_WHOLE_SIZE) {
    return {binding, b.b, off, range, VK_NULL_HANDLE};
}
KBind kb(u32 binding, const Slice &s, VkDeviceSize range) { return {binding, s.buf->b, s.off, range, VK_NULL_HANDLE}; }
KBind ki(u32 binding, const Img &im) { return {binding, VK_NULL_HANDLE, 0, 0, im.view}; }

void dispatch(VkCommandBuffer cb, Kern k, std::initializer_list<KBind> binds, const u32 p[4], const u32 *q, u32 w, u32 h) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kpipe[k]);
    VkWriteDescriptorSet wr[6];
    VkDescriptorBufferInfo bi[6];
    VkDescriptorImageInfo ii[6];
    u32 n = 0;
    for (const KBind &b : binds) {
        VkWriteDescriptorSet &w0 = wr[n];
        w0 = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w0.dstBinding = b.binding;
        w0.descriptorCount = 1;
        if (b.view) {
            ii[n] = {VK_NULL_HANDLE, b.view, VK_IMAGE_LAYOUT_GENERAL};
            w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w0.pImageInfo = &ii[n];
        } else {
            bi[n] = {b.buf, b.off, b.range};
            w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w0.pBufferInfo = &bi[n];
        }
        n++;
    }
    vkPushDescriptors(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k_layout, 0, n, wr);
    u32 pc[8] = {};
    memcpy(pc, p, 16);
    if (q) memcpy(pc + 4, q, 16);
    vkCmdPushConstants(cb, k_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 15) / 16, (h + 15) / 16, 1);
}

// ---------------------------------------------------------------- texture filters (vk_texfilter.cpp)
// texfilter.mm's encoding on Vulkan: input texture at binding 0, output storage image at 1 (one mip level), the
// rectangles at 2 (grid z), TFP as push constants
VkDescriptorSetLayout tf_dsl;
VkPipelineLayout tf_layout;
VkPipeline tf_pipe[kVkTfKernCount];
bool tf_ok = false;  // the kernels were built
struct TfRect {
    u32 x, y, w, h;
};
struct TFP {
    u32 sw, sh, dw, dh, wrap_s, wrap_t, factor, pad;
};

// one dispatch over the rectangles rc (source rectangles scaled by num / den, in output texels)
void tf_dispatch(VkCommandBuffer cb, VkTfKernel k, VkImageView in, VkImageView out, const TFP &p, const TfRect *rc,
                 u32 n, u32 num, u32 den) {
    const Slice rs = arena((size_t)n * 16);
    u32 *v = (u32 *)rs.p;
    u32 mw = 1, mh = 1;
    for (u32 i = 0; i < n; i++) {
        v[i * 4] = rc[i].x * num / den;
        v[i * 4 + 1] = rc[i].y * num / den;
        v[i * 4 + 2] = std::max(1u, rc[i].w * num / den);
        v[i * 4 + 3] = std::max(1u, rc[i].h * num / den);
        mw = std::max(mw, v[i * 4 + 2]);
        mh = std::max(mh, v[i * 4 + 3]);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tf_pipe[k]);
    VkDescriptorImageInfo ii[2] = {{VK_NULL_HANDLE, in, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, out, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi = {rs.buf->b, rs.off, (VkDeviceSize)n * 16};
    VkWriteDescriptorSet w[3];
    for (u32 i = 0; i < 3; i++) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &ii[0];
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo = &ii[1];
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &bi;
    vkPushDescriptors(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tf_layout, 0, 3, w);
    vkCmdPushConstants(cb, tf_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
    vkCmdDispatch(cb, (mw + 15) / 16, (mh + 15) / 16, n);
}

// the upscaling of src (level 0, w x h) by factor (2 or 4) into level 0 of dst, then dst's levels 1..log2(factor)-1
// by alpha-aware 2x2 reduction; as texfilter_encode (rects: source texels, all if none; mid: the sharp filter's 2x
// image at 4x, a partial update needing the previous one's). False without the kernels.
bool tf_encode(int mode, Img &src, Img &dst, Img *mid, u32 factor, u32 wrap_s, u32 wrap_t, const TfRect *rects = nullptr,
               u32 nrects = 0) {
    if (!tf_ok || (factor != 2 && factor != 4)) return false;
    const u32 w = src.w, h = src.h;
    const TfRect all = {0, 0, w, h};
    if (!rects || !nrects) {
        rects = &all;
        nrects = 1;
    }
    VkCommandBuffer cb = rec();
    barrier(cb);
    VkImageView top = level_view(dst, 0);
    ImgP tmp;
    if (mode == kVkTfSmooth) {
        const TFP p = {w, h, w * factor, h * factor, wrap_s, wrap_t, factor, 0};
        tf_dispatch(cb, kVkTfKernSmooth, src.view, top, p, rects, nrects, factor, 1);
    } else {
        // 2x, or 2x twice through the intermediate image mid
        if (factor == 4 && !mid) {
            tmp = make_img(w * 2, h * 2, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
            mid = tmp.get();
            barrier(cb);
        }
        VkImageView in = src.view;
        u32 iw = w, ih = h, scale = 1;
        for (u32 f = factor; f > 1; f /= 2) {
            VkImageView out = f > 2 ? mid->view : top;
            const TFP q = {iw, ih, iw * 2, ih * 2, wrap_s, wrap_t, 2, 0};
            tf_dispatch(cb, kVkTfKernSharp, in, out, q, rects, nrects, scale, 1);  // an invocation per input texel
            scale *= 2;
            if (f > 2) barrier(cb);
            in = out;
            iw *= 2;
            ih *= 2;
        }
    }
    u32 ul = 0;
    while ((1u << ul) < factor) ul++;
    for (u32 l = 1; l < ul; l++) {
        barrier(cb);
        const TFP q = {0, 0, 0, 0, 0, 0, 0, 0};
        tf_dispatch(cb, kVkTfKernDown, level_view(dst, l - 1), level_view(dst, l), q, rects, nrects, factor, 1u << l);
    }
    return true;
}

// vertex program kernels: storage buffers 0-4 (vertex arrays, ids, parameter blocks, output, each thread's draw), the
// number of threads as a push constant
VkDescriptorSetLayout v_dsl;
VkPipelineLayout v_layout;
BufP gbuf_dev;  // vertices the kernels shaded (device-local; reused flush after flush, the barriers order the uses)

// graphics pipelines (fragment shaders from fsgen.h): bindings as gen_fs() documents, 15-17 the vertex shader's
VkDescriptorSetLayout g_dsl;
VkPipelineLayout g_layout;
VkShaderModule vs_module, full_module;
VkPipeline p_rcolor = VK_NULL_HANDLE, p_rdepth = VK_NULL_HANDLE;  // MSAA restores
VkPipeline p_rdepth_only = VK_NULL_HANDLE, p_rstencil_bit = VK_NULL_HANDLE;  // (without stencil export)
VkPipelineCache pcache = VK_NULL_HANDLE;
std::mutex pcache_mtx;  // (saving it)
std::atomic<bool> pcache_dirty{false};
std::string pcache_path;
VkSampler dummy_sampler;
BufP dummy_buf;
ImgP black;  // 1x1 transparent black texture, also the dummy one

// C++ mirror of the GLSL uniform block (fsgen.cpp's Uni, std140)
struct alignas(16) LightU {
    float spec0[4], spec1[4], diffuse[4], ambient[4], pos[4], spot[4], att[4];
};
struct alignas(16) Uni {
    float rt[4];
    float depth[4];
    s32 scissor[4];
    u32 konst[8];
    u32 tevc[4];
    u32 misc[4];
    u32 border[4];
    float bias[4];
    float lscale[8];
    float ambient[4];
    LightU lights[8];
};
static_assert(sizeof(Uni) == 16 * 12 + 8 * 112, "uniform layout");
struct VsDraw {
    float rt[4], vp[4], clip[4];
    u32 flags[4];
};

// the pipeline of the full-screen restores and of the draws (fs: a fragment shader module; blend state from ds)
VkPipeline make_pipeline(VkShaderModule vs, VkShaderModule fs, const DrawState *ds, bool depth_only) {
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = fs;
    st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = VK_TRUE;  // (Metal's depth clip mode "clamp")
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = (VkSampleCountFlagBits)MS;
    VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState ca{};
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &ca;
    if (depth_only) {
        ca.colorWriteMask = 0;
    } else if (!ds) {
        ca.colorWriteMask = 0xf;
    } else {
        const FsOptions fo{ShaderLang::GLSL, has_logic_op, MS};
        if (ds->color_write && ds->need_shade) {
            if (!fs_hw_blend(*ds, fo)) {
                ca.colorWriteMask = 0xf;  // masking done in the shader
            } else {
                if (ds->color_mask & 0xff) ca.colorWriteMask |= VK_COLOR_COMPONENT_R_BIT;
                if (ds->color_mask & 0xff00) ca.colorWriteMask |= VK_COLOR_COMPONENT_G_BIT;
                if (ds->color_mask & 0xff0000) ca.colorWriteMask |= VK_COLOR_COMPONENT_B_BIT;
                if (ds->color_mask & 0xff000000u) ca.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
            }
        }
        if (ds->need_shade && ds->blend && hw_blend(*ds)) {
            static const VkBlendFactor fac[15] = {
                VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA,
                VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_FACTOR_CONSTANT_ALPHA,
                VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE};
            static const VkBlendOp ops[5] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT,
                                             VK_BLEND_OP_MIN, VK_BLEND_OP_MAX};
            auto F = [](u32 f) { return fac[f < 15 ? f : 2]; };
            ca.blendEnable = VK_TRUE;
            ca.colorBlendOp = ops[std::min<u32>(ds->eq_rgb, 4)];
            ca.alphaBlendOp = ops[std::min<u32>(ds->eq_a, 4)];
            ca.srcColorBlendFactor = F(ds->sf_rgb);
            ca.dstColorBlendFactor = F(ds->df_rgb);
            ca.srcAlphaBlendFactor = F(ds->sf_a);
            ca.dstAlphaBlendFactor = F(ds->df_a);
        } else if (ds->need_shade && !ds->blend && ds->logic_op != 3 && has_logic_op) {
            // PICA logic ops (in its numbering) by the blender
            static const VkLogicOp lop[16] = {
                VK_LOGIC_OP_CLEAR,         VK_LOGIC_OP_AND,     VK_LOGIC_OP_AND_REVERSE, VK_LOGIC_OP_COPY,
                VK_LOGIC_OP_SET,           VK_LOGIC_OP_COPY_INVERTED, VK_LOGIC_OP_NO_OP, VK_LOGIC_OP_INVERT,
                VK_LOGIC_OP_NAND,          VK_LOGIC_OP_OR,      VK_LOGIC_OP_NOR,         VK_LOGIC_OP_XOR,
                VK_LOGIC_OP_EQUIVALENT,    VK_LOGIC_OP_AND_INVERTED, VK_LOGIC_OP_OR_REVERSE, VK_LOGIC_OP_OR_INVERTED};
            cb.logicOpEnable = VK_TRUE;
            cb.logicOp = lop[ds->logic_op & 15];
        }
    }
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT,          VK_DYNAMIC_STATE_SCISSOR,
                                  VK_DYNAMIC_STATE_CULL_MODE,         VK_DYNAMIC_STATE_FRONT_FACE,
                                  VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                                  VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,  VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                                  VK_DYNAMIC_STATE_STENCIL_OP,        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                                  VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                  VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = sizeof dyn / sizeof dyn[0];
    dy.pDynamicStates = dyn;
    const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = 1;
    ri.pColorAttachmentFormats = &cfmt;
    ri.depthAttachmentFormat = depth_format;
    ri.stencilAttachmentFormat = depth_format;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pi.pNext = &ri;
    pi.stageCount = 2;
    pi.pStages = st;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &dss;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dy;
    pi.layout = g_layout;
    VkPipeline p = VK_NULL_HANDLE;
    const VkResult r = vkCreateGraphicsPipelines(dev, pcache, 1, &pi, nullptr, &p);
    if (r != VK_SUCCESS) {
        LOG("gpu vulkan: pipeline creation failed (%d)\n", (int)r);
        return VK_NULL_HANDLE;
    }
    pcache_dirty = true;
    return p;
}

void save_pipeline_cache() {
    std::lock_guard<std::mutex> lk(pcache_mtx);
    if (!pcache || pcache_path.empty() || !pcache_dirty.load()) return;
    pcache_dirty = false;
    size_t n = 0;
    if (vkGetPipelineCacheData(dev, pcache, &n, nullptr) != VK_SUCCESS || !n) return;
    std::vector<u8> data(n);
    if (vkGetPipelineCacheData(dev, pcache, &n, data.data()) != VK_SUCCESS) return;
    const std::string tmp = pcache_path + ".tmp";
    if (FILE *f = fopen(tmp.c_str(), "wb")) {
        fwrite(data.data(), 1, n, f);
        fclose(f);
        std::error_code ec;
        std::filesystem::rename(tmp, pcache_path, ec);
    }
}

// ---------------------------------------------------------------- render targets
struct GRT {
    u32 addr = 0, w = 0, h = 0, fmt = 0;
    bool depth = false;
    ImgP tex;                // single-sample copy (what everything else reads)
    ImgP ms;                 // with MSAA: the multisampled target, resolved into tex at the end of each pass
    bool ms_valid = false;   // ms matches tex (else it is restored from tex before drawing)
    bool synced = false;     // GPU copy corresponds to guest memory as hashed below
    u64 hash = 0;
    u64 checked = 0;         // g_frame when guest memory last matched the hash
    u64 id = 0;              // unique among all targets ever made
    u64 version = 0;         // changes whenever the GPU copy does
    bool gpu_newer = false;  // changes not yet written to guest memory
    u32 skipped = 0;         // draws left out since its last display transfer (pipelines still compiling)
    bool clear_next = false; // set_clear_after_top_transfer: cleared to transparent before it is drawn into again,
    bool own = false;        // ... and its GPU copy kept (guest memory ignored) until its own next display transfer
};
std::atomic<bool> clear_after_top{false};
std::atomic<bool> key_black_bottom{false};
std::vector<std::unique_ptr<GRT>> rts;

u32 rt_bytes(const GRT &r) {
    if (r.depth) return r.w * r.h * (r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4);
    return r.w * r.h * pixel_bytes(cbuf_to_pixfmt(r.fmt));
}
bool overlaps(u32 a, u32 an, u32 b, u32 bn) { return a < b + bn && b < a + an; }

void clear_overlapping(GRT &r) {
    for (auto &o : rts)
        if (!o->depth && overlaps(o->addr, rt_bytes(*o), r.addr, rt_bytes(r))) o->clear_next = o->own = true;
}

GRT &get_rt(u32 addr, u32 w, u32 h, u32 fmt, bool depth) {
    for (auto &r : rts)
        if (r->addr == addr && r->w == w && r->h == h && r->fmt == fmt && r->depth == depth) return *r;
    static u64 next_id;
    auto r = std::make_unique<GRT>();
    r->id = ++next_id;
    r->addr = addr;
    r->w = w;
    r->h = h;
    r->fmt = fmt;
    r->depth = depth;
    const VkFormat f = depth ? depth_format : VK_FORMAT_R8G8B8A8_UNORM;
    r->tex = make_img(w * S, h * S, f, 1, depth ? kDepthUsage : kColorUsage);
    if (MS > 1)
        r->ms = make_img(w * S, h * S, f, 1,
                         depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                               : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                         MS);
    rts.push_back(std::move(r));
    return *rts.back();
}

// scratch depth target for render targets without a guest depth buffer (cleared at the start of each pass)
std::unordered_map<u64, ImgP> scratch_depth;
const Img &get_scratch_depth(u32 w, u32 h) {
    const u64 key = (u64)w << 32 | h;
    auto it = scratch_depth.find(key);
    if (it != scratch_depth.end()) return *it->second;
    ImgP t = make_img(w * S, h * S, depth_format, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, MS);
    scratch_depth[key] = t;
    return *t;
}

BufP stage_color, stage_depth, stage_stencil, stage_dbytes;

void encode_guest_color(VkCommandBuffer cb, const GRT &r, const Buf &dst, VkDeviceSize off = 0) {
    const u32 prm[4] = {r.w, r.h, S, cbuf_to_pixfmt(r.fmt)};
    dispatch(cb, K_ENC, {ki(4, *r.tex), kb(0, dst, off)}, prm, nullptr, r.w, r.h);
}

struct LoadAction {
    bool clear = false;
    u32 color = 0;
    float depth = 0;
    u32 stencil = 0;
};
inline float depth_quantum(u32 fmt) { return fmt == 0 ? 1.0f / 65536.0f : 1.0f / 16777216.0f; }

VkBufferImageCopy buf_copy(VkDeviceSize off, u32 w, u32 h, VkImageAspectFlags aspect, u32 level = 0, s32 x = 0,
                           s32 y = 0) {
    VkBufferImageCopy c{};
    c.bufferOffset = off;
    c.bufferRowLength = w;
    c.bufferImageHeight = h;
    c.imageSubresource = {aspect, level, 0, 1};
    c.imageOffset = {x, y, 0};
    c.imageExtent = {w, h, 1};
    return c;
}

// make the GPU copy of a render target match guest memory (if the guest changed since the last sync); uniform
// contents become a clear of the render pass
LoadAction sync_rt(GRT &r) {
    LoadAction la;
    const u8 *p = pptr(r.addr);
    if (!p) return la;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (r.synced && r.checked == frame) return la;
    const u64 hv = hash_mem(p, rt_bytes(r));
    r.checked = frame;
    if (r.synced && hv == r.hash) return la;
    if (r.own) {
        r.hash = hv;
        r.synced = true;
        return la;
    }
    r.synced = true;
    r.hash = hv;
    r.gpu_newer = false;
    const u32 rbpp = r.depth ? (r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4) : pixel_bytes(cbuf_to_pixfmt(r.fmt));
    if (rt_bytes(r) > rbpp && memcmp(p, p + rbpp, rt_bytes(r) - rbpp) == 0) {
        la.clear = true;
        if (!r.depth) {
            la.color = decode_pixel(cbuf_to_pixfmt(r.fmt), p);
        } else {
            const u32 z = r.fmt == 0 ? (u32)(p[0] | p[1] << 8) : (u32)(p[0] | p[1] << 8 | p[2] << 16);
            la.depth = (float)z * depth_quantum(r.fmt);
            la.stencil = r.fmt == 3 ? p[3] : 0;
        }
        return la;
    }
    const Slice raw = arena(rt_bytes(r));
    memcpy(raw.p, p, rt_bytes(r));
    r.ms_valid = false;
    r.version++;
    VkCommandBuffer cb = rec();
    barrier(cb);
    if (!r.depth) {
        const u32 prm[4] = {r.w, r.h, S, cbuf_to_pixfmt(r.fmt)};
        dispatch(cb, K_DEC, {kb(0, raw, rt_bytes(r)), ki(5, *r.tex)}, prm, nullptr, r.w * S, r.h * S);
        return la;
    }
    // depth / stencil: as scaled rows of values, copied into the target
    const u32 W = r.w * S, H = r.h * S;
    const Slice upd = arena((size_t)W * H * 4), ups = arena((size_t)W * H);
    const u32 prm[4] = {r.w, r.h, S, r.fmt};
    dispatch(cb, K_DEC_DEPTH, {kb(0, raw, rt_bytes(r)), kb(1, upd, (size_t)W * H * 4), kb(2, ups, (size_t)W * H)}, prm,
             nullptr, W, H);
    barrier(cb);
    const VkBufferImageCopy cd = buf_copy(upd.off, W, H, VK_IMAGE_ASPECT_DEPTH_BIT);
    vkCmdCopyBufferToImage(cb, upd.buf->b, r.tex->img, VK_IMAGE_LAYOUT_GENERAL, 1, &cd);
    const VkBufferImageCopy cs = buf_copy(ups.off, W, H, VK_IMAGE_ASPECT_STENCIL_BIT);
    vkCmdCopyBufferToImage(cb, ups.buf->b, r.tex->img, VK_IMAGE_LAYOUT_GENERAL, 1, &cs);
    return la;
}

bool guest_kept(GRT &r) {
    if (hash_mem(pptr(r.addr), rt_bytes(r)) == r.hash) return true;
    static int logged;
    if (logged < 8 || g_stats) {
        logged++;
        LOG("gpu vulkan: %s target %08x (%ux%u) was rewritten by the game since it was drawn; not written back\n",
            r.depth ? "depth" : "colour", r.addr, r.w, r.h);
    }
    r.gpu_newer = false;
    r.synced = false;
    return false;
}

// write a colour target's GPU contents back to guest memory (box-filtered to 1x)
void write_back_color(GRT &r) {
    if (!guest_kept(r)) return;
    ensure_buffer(stage_color, rt_bytes(r));
    VkCommandBuffer cb = new_cb();
    barrier(cb);
    encode_guest_color(cb, r, *stage_color);
    wait_value(submit(), "colour write-back", r.addr);
    stage_color->invalidate();
    r.hash = copy_hash(pptr(r.addr), stage_color->p, rt_bytes(r));
    r.checked = g_frame.load(std::memory_order_relaxed);
    r.synced = true;
    r.gpu_newer = false;
}

// write a depth target's GPU contents back to guest memory (top-left sample of each pixel)
void write_back_depth(GRT &r) {
    if (!guest_kept(r)) return;
    const u32 W = r.w * S, H = r.h * S;
    ensure_buffer(stage_depth, (size_t)W * H * 4, Mem::Device);
    ensure_buffer(stage_stencil, (size_t)W * H, Mem::Device);
    ensure_buffer(stage_dbytes, rt_bytes(r));
    VkCommandBuffer cb = new_cb();
    barrier(cb);
    const VkBufferImageCopy cd = buf_copy(0, W, H, VK_IMAGE_ASPECT_DEPTH_BIT);
    vkCmdCopyImageToBuffer(cb, r.tex->img, VK_IMAGE_LAYOUT_GENERAL, stage_depth->b, 1, &cd);
    const VkBufferImageCopy cs = buf_copy(0, W, H, VK_IMAGE_ASPECT_STENCIL_BIT);
    vkCmdCopyImageToBuffer(cb, r.tex->img, VK_IMAGE_LAYOUT_GENERAL, stage_stencil->b, 1, &cs);
    // into guest memory's format (unchanged where a format has no stencil or fewer depth bytes)
    u8 *p = pptr(r.addr);
    memcpy(stage_dbytes->p, p, rt_bytes(r));
    stage_dbytes->flush();
    barrier(cb);
    const u32 prm[4] = {r.w, r.h, S, r.fmt};
    dispatch(cb, K_ENC_DEPTH, {kb(0, *stage_depth), kb(1, *stage_stencil), kb(2, *stage_dbytes)}, prm, nullptr, r.w, r.h);
    wait_value(submit(), "depth write-back", r.addr);
    stage_dbytes->invalidate();
    r.gpu_newer = false;
    r.hash = copy_hash(p, stage_dbytes->p, rt_bytes(r));
    r.checked = g_frame.load(std::memory_order_relaxed);
    r.synced = true;
}

// ---------------------------------------------------------------- textures and samplers
// Decoded textures as images, by guest address and layout, uploaded through the arena before the pass. With a texture
// filter (set_texture_filter, GPU_TEX_FILTER) the 2D textures the game samples are upscaled once per decoded content
// (vk_texfilter.cpp), as in metal.mm: level 0 is the upscaled image, the next levels down to the original size are
// reduced from it, and the game's own levels follow (its level l is level l + ul), so get_sampler shifts the LOD clamps.
struct VTex {
    u64 hash = 0;
    ImgP tex;
    int req = 0;               // texture filter requested when it was made (0: none)
    u32 ul = 0;                // log2 of the upscale factor (0: not upscaled)
    u32 wraps = 0;             // wrap modes of the neighbourhood it was upscaled with
    size_t bytes = 0;          // GPU memory of an upscaled texture (with src and mid)
    u64 last_use = 0;          // flush_no
    u32 changes = 0;           // content changes seen
    u64 win_start = 0;         // complete re-upscales since win_start (flush_no): textures the game keeps
    u32 win_changes = 0;       // rewriting completely (video) are no longer upscaled
    // a texture the game has rewritten (animated tiles in an atlas) keeps what its upscaled image was made from,
    // so that the next updates only redo the 8x8 tiles that changed
    ImgP src, mid;             // level 0 as decoded; the sharp filter's 2x image (at 4x)
    std::vector<u32> prev;     // level 0 as decoded (CPU copy, compared with the new contents)
};
std::unordered_map<std::string, VTex> tex_cache;
std::atomic<int> want_filter{0};  // set_texture_filter (any thread), applied by the GX thread at the next flush
int tf_mode = 0;
u64 flush_no = 0;
size_t tf_bytes = 0;                      // upscaled textures' memory
constexpr size_t kTfBudget = 512u << 20;  // beyond: least recently used ones evicted, else no more upscaling
u32 st_uploads = 0, st_upscaled = 0, st_partial = 0, st_tiles = 0, st_fetch = 0;  // RECOMP_STATS
double st_upload_ms = 0, st_upscale_ms = 0;

// Guest memory holding GPU output (render targets, or transfer / copy outputs made from them, as long as the bytes
// are unchanged): render-to-texture, not upscaled. Only tracked while a texture filter is on.
struct GpuOut {
    u32 addr, size;
    u64 hash;
};
std::deque<GpuOut> gpu_outs;
bool gpu_made(u32 addr, u32 size) {
    for (auto &r : rts)
        if (overlaps(r->addr, rt_bytes(*r), addr, size) && (r->gpu_newer || hash_mem(pptr(r->addr), rt_bytes(*r)) == r->hash))
            return true;
    for (const GpuOut &g : gpu_outs)
        if (overlaps(g.addr, g.size, addr, size) && hash_mem(pptr(g.addr), g.size) == g.hash) return true;
    return false;
}
void note_gpu_out(u32 addr, u32 size) {
    if (!pptr(addr) || !pptr(addr + size - 1)) return;
    for (auto it = gpu_outs.begin(); it != gpu_outs.end(); ++it)
        if (it->addr == addr) {
            gpu_outs.erase(it);
            break;
        }
    gpu_outs.push_back({addr, size, hash_mem(pptr(addr), size)});
    if (gpu_outs.size() > 64) gpu_outs.pop_front();
}

// upscale factor of a texture (0: not upscaled): 4x at internal scales above 2, else 2x, up to 1024x1024; never
// lookup / gradient tables (thin or tiny) or HILO8 (vectors, not colours)
u32 tf_factor(const TexEntry *e) {
    const u32 w = e->w, h = e->h, n = w * h;
    if (e->fmt == 6 || n < 256 || w >= 8 * h || h >= 8 * w || n > 1024 * 1024) return 0;
    return S > 2 ? 4 : 2;
}
u32 tf_wrap(u32 wrap) { return wrap == 2 || wrap == 6 || wrap == 7 ? 1 : wrap == 3 ? 2 : 0; }
void tf_drop_all() {
    tex_cache.clear();
    tf_bytes = 0;
}

// The 8x8 tiles of an image (w x h texels) that differ between prev and cur, grown by one tile (the filters read up
// to 3 texels around: tiles next to a changed one change too; across the edges when the texture repeats), one
// rectangle per tile in rc; copies them from cur to prev. Returns the number of tiles.
u32 changed_tiles(std::vector<u32> &prev, const std::vector<u32> &cur, u32 w, u32 h, u32 wraps, std::vector<TfRect> &rc) {
    const u32 tw = w / 8, th = h / 8;
    std::vector<u8> d((size_t)tw * th), g((size_t)tw * th);
    bool any = false;
    for (u32 ty = 0; ty < th; ty++)
        for (u32 tx = 0; tx < tw; tx++)
            for (u32 y = 0; y < 8; y++) {
                const size_t o = (size_t)(ty * 8 + y) * w + tx * 8;
                if (memcmp(&prev[o], &cur[o], 32)) {
                    d[ty * tw + tx] = 1;
                    any = true;
                    break;
                }
            }
    if (!any) return 0;
    const bool rs = (wraps & 3) == 1, rt = (wraps >> 2) == 1;
    for (u32 ty = 0; ty < th; ty++)
        for (u32 tx = 0; tx < tw; tx++) {
            if (!d[ty * tw + tx]) continue;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int x = (int)tx + dx, y = (int)ty + dy;
                    if (x < 0 || x >= (int)tw) {
                        if (!rs) continue;
                        x = (x + tw) % tw;
                    }
                    if (y < 0 || y >= (int)th) {
                        if (!rt) continue;
                        y = (y + th) % th;
                    }
                    g[y * tw + x] = 1;
                }
        }
    u32 n = 0;
    for (u32 ty = 0; ty < th; ty++)
        for (u32 tx = 0; tx < tw;) {
            if (!g[ty * tw + tx]) {
                tx++;
                continue;
            }
            u32 e = tx;
            while (e < tw && g[ty * tw + e]) e++;
            for (u32 x = tx; x < e; x++) rc.push_back({x * 8, ty * 8, 8, 8});
            for (u32 y = 0; y < 8; y++) {
                const size_t o = (size_t)(ty * 8 + y) * w + tx * 8;
                memcpy(&prev[o], &cur[o], (e - tx) * 32);
            }
            n += e - tx;
            tx = e;
        }
    return n;
}

// the texture's decoded levels into dst's levels (from level `at`), through the arena
void upload_levels(const TexEntry *e, u32 nl, Img &dst, u32 at = 0) {
    size_t total = 0;
    for (u32 l = 0; l < nl; l++) total += (size_t)e->lv[l].w * e->lv[l].h * 4;
    const Slice s = arena(total);
    std::vector<VkBufferImageCopy> cp;
    size_t off = 0;
    for (u32 l = 0; l < nl; l++) {
        const TexImage &im = e->lv[l];
        if (!im.w || !im.h) continue;
        memcpy(s.p + off, im.data.data(), (size_t)im.w * im.h * 4);
        cp.push_back(buf_copy(s.off + off, im.w, im.h, VK_IMAGE_ASPECT_COLOR_BIT, l + at));
        off += (size_t)im.w * im.h * 4;
    }
    VkCommandBuffer cb = rec();
    barrier(cb);
    vkCmdCopyBufferToImage(cb, s.buf->b, dst.img, VK_IMAGE_LAYOUT_GENERAL, (u32)cp.size(), cp.data());
}

// GPU_TF_VERIFY=1: each partial update of an upscaled texture is checked against upscaling all of it again (both
// read back and compared, every level), for tests (tools/gpu_test's gpu_replay --scribble makes textures change)
bool tf_verify = false;
void verify_partial(const TexEntry *e, const VTex &m, u32 f, u32 u, u32 wraps) {
    const u32 w = e->w, h = e->h;
    ImgP t = make_img(w * f, h * f, VK_FORMAT_R8G8B8A8_UNORM, 1 + u, kUpscaledUsage);
    ImgP src = make_img(w, h, VK_FORMAT_R8G8B8A8_UNORM, 1, kTexUsage);
    upload_levels(e, 1, *src);
    ImgP mid = m.mid ? make_img(w * 2, h * 2, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)
                     : nullptr;
    tf_encode(tf_mode, *src, *t, mid.get(), f, wraps & 3, wraps >> 2);
    VkCommandBuffer cb = rec();
    barrier(cb);
    VkImageCopy c{};
    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, u, 0, 1};
    c.extent = {w, h, 1};
    vkCmdCopyImage(cb, src->img, VK_IMAGE_LAYOUT_GENERAL, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    barrier(cb);
    size_t total = 0;
    for (u32 l = 0; l <= u; l++) total += (size_t)((w * f) >> l) * ((h * f) >> l) * 4;
    BufP a = make_buf(total, Mem::Readback), b = make_buf(total, Mem::Readback);
    size_t off = 0;
    for (u32 l = 0; l <= u; l++) {
        const u32 lw = (w * f) >> l, lh = (h * f) >> l;
        const VkBufferImageCopy bc = buf_copy(off, lw, lh, VK_IMAGE_ASPECT_COLOR_BIT, l);
        vkCmdCopyImageToBuffer(cb, m.tex->img, VK_IMAGE_LAYOUT_GENERAL, a->b, 1, &bc);
        vkCmdCopyImageToBuffer(cb, t->img, VK_IMAGE_LAYOUT_GENERAL, b->b, 1, &bc);
        off += (size_t)lw * lh * 4;
    }
    wait_value(submit());
    a->invalidate();
    b->invalidate();
    off = 0;
    for (u32 l = 0; l <= u; l++) {
        const u32 lw = (w * f) >> l, lh = (h * f) >> l;
        u32 bad = 0;
        for (size_t i = 0; i < (size_t)lw * lh; i++) bad += ((const u32 *)(a->p + off))[i] != ((const u32 *)(b->p + off))[i];
        LOG("gpu vulkan: GPU_TF_VERIFY texture %08x %ux%u level %u (%ux%u): %u texels differ from a full upscale\n", e->addr,
            w, h, l, lw, lh, bad);
        off += (size_t)lw * lh * 4;
    }
}

// up: the unit sampling it, if it may be upscaled (2D / projected textures); *ul: log2 of the upscale factor
const Img *get_texture(const TexEntry *e, const TexUnitState *up = nullptr, u32 *ul = nullptr) {
    if (ul) *ul = 0;
    if (!e || e->lv.empty() || e->lv[0].data.empty()) return nullptr;
    char key[64];
    snprintf(key, sizeof key, "%08x:%u:%u:%u:%u", e->addr, e->w, e->h, e->fmt, e->levels);
    VTex &m = tex_cache[key];
    const int want = up ? tf_mode : 0;
    if (m.tex && m.hash == e->hash && m.req == want) {
        m.last_use = flush_no;
        if (ul) *ul = m.ul;
        return m.tex.get();
    }
    const double t0 = log_timing ? now_ms() : 0;
    const bool changed = m.tex && m.hash != e->hash;
    if (changed) m.changes++;
    m.last_use = flush_no;
    u32 nl = 0;
    while (nl < e->lv.size() && !e->lv[nl].data.empty() && e->lv[nl].w >= 1) nl++;
    u32 f = want ? tf_factor(e) : 0;
    const u32 u = f == 4 ? 2 : f == 2 ? 1 : 0;
    const u32 wraps = up ? tf_wrap(up->wrap_s) | tf_wrap(up->wrap_t) << 2 : 0;
    const u32 w = e->w, h = e->h;

    // an upscaled texture of which only some tiles changed: redo those
    if (changed && f && m.req == want && m.ul == u && m.wraps == wraps && m.src && nl == 1 && m.prev.size() == (size_t)w * h) {
        std::vector<TfRect> rc;
        const u32 nt = changed_tiles(m.prev, e->lv[0].data, w, h, wraps, rc);
        if (nt * 2 <= (w / 8) * (h / 8)) {
            if (nt) {
                size_t total = 0;
                for (const TfRect &r : rc) total += (size_t)r.w * r.h * 4;
                const Slice sb = arena(total);
                std::vector<VkBufferImageCopy> cp;
                size_t off = 0;
                for (const TfRect &r : rc) {
                    for (u32 y = 0; y < r.h; y++)
                        memcpy(sb.p + off + (size_t)y * r.w * 4, &e->lv[0].data[(size_t)(r.y + y) * w + r.x], r.w * 4);
                    cp.push_back(buf_copy(sb.off + off, r.w, r.h, VK_IMAGE_ASPECT_COLOR_BIT, 0, (s32)r.x, (s32)r.y));
                    off += (size_t)r.w * r.h * 4;
                }
                VkCommandBuffer cb = rec();
                barrier(cb);
                vkCmdCopyBufferToImage(cb, sb.buf->b, m.src->img, VK_IMAGE_LAYOUT_GENERAL, (u32)cp.size(), cp.data());
                tf_encode(tf_mode, *m.src, *m.tex, m.mid.get(), f, wraps & 3, wraps >> 2, rc.data(), (u32)rc.size());
                std::vector<VkImageCopy> ic;  // the game's level 0
                for (const TfRect &r : rc) {
                    VkImageCopy c{};
                    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    c.srcOffset = {(s32)r.x, (s32)r.y, 0};
                    c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, u, 0, 1};
                    c.dstOffset = {(s32)r.x, (s32)r.y, 0};
                    c.extent = {r.w, r.h, 1};
                    ic.push_back(c);
                }
                barrier(cb);
                vkCmdCopyImage(cb, m.src->img, VK_IMAGE_LAYOUT_GENERAL, m.tex->img, VK_IMAGE_LAYOUT_GENERAL, (u32)ic.size(),
                               ic.data());
                if (tf_verify) verify_partial(e, m, f, u, wraps);
            }
            m.hash = e->hash;
            if (ul) *ul = u;
            if (g_stats) LOG("gpu vulkan: texture %08x %ux%u: %u tiles upscaled again\n", e->addr, w, h, nt);
            if (log_timing) {
                st_partial++;
                st_tiles += nt;
                st_upscale_ms += now_ms() - t0;
            }
            return m.tex.get();
        }
    }

    if (changed) {
        if (flush_no - m.win_start > 600) {
            m.win_start = flush_no;
            m.win_changes = 0;
        }
        m.win_changes++;
    }
    tf_bytes -= m.bytes;
    m.bytes = 0;
    m.ul = 0;
    m.req = want;
    m.wraps = wraps;
    m.hash = e->hash;
    m.src = m.mid = nullptr;
    m.prev.clear();
    m.prev.shrink_to_fit();
    if (f && (m.win_changes > 6 || gpu_made(e->addr, e->size))) f = 0;
    const bool keep = f && m.changes > 0 && nl == 1;  // rewritten before: keep what partial updates need
    size_t bytes = 0;
    for (u32 l = 0; l < nl + u; l++) bytes += (size_t)std::max(1u, (w * f) >> l) * std::max(1u, (h * f) >> l) * 4;
    if (keep) bytes += (size_t)w * h * 4 * (tf_mode == kVkTfSharp && f == 4 ? 5 : 1);
    if (f && tf_bytes + bytes > kTfBudget) {
        for (auto it = tex_cache.begin(); it != tex_cache.end();)
            if (it->second.bytes && it->second.last_use + 120 < flush_no) {
                tf_bytes -= it->second.bytes;
                it = tex_cache.erase(it);
            } else {
                ++it;
            }
        if (tf_bytes + bytes > kTfBudget) f = 0;
    }
    if (f && tf_ok) {
        ImgP t = make_img(w * f, h * f, VK_FORMAT_R8G8B8A8_UNORM, nl + u, kUpscaledUsage);
        // the decoded texture, upscaled from and copied below the upscaled levels
        ImgP src = make_img(w, h, VK_FORMAT_R8G8B8A8_UNORM, nl, kTexUsage);
        upload_levels(e, nl, *src);
        ImgP mid = keep && tf_mode == kVkTfSharp && f == 4
                       ? make_img(w * 2, h * 2, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)
                       : nullptr;
        tf_encode(tf_mode, *src, *t, mid.get(), f, wraps & 3, wraps >> 2);
        std::vector<VkImageCopy> ic;
        for (u32 l = 0; l < nl; l++) {
            const TexImage &im = e->lv[l];
            if (!im.w || !im.h) continue;
            VkImageCopy c{};
            c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l + u, 0, 1};
            c.extent = {im.w, im.h, 1};
            ic.push_back(c);
        }
        VkCommandBuffer cb = rec();
        barrier(cb);
        vkCmdCopyImage(cb, src->img, VK_IMAGE_LAYOUT_GENERAL, t->img, VK_IMAGE_LAYOUT_GENERAL, (u32)ic.size(), ic.data());
        m.tex = t;
        m.ul = u;
        m.bytes = bytes;
        tf_bytes += bytes;
        if (keep) {
            m.src = src;
            m.mid = mid;
            m.prev = e->lv[0].data;
        }
        if (ul) *ul = u;
        if (log_timing) {
            st_upscaled++;
            st_upscale_ms += now_ms() - t0;
        }
        if (g_stats) LOG("gpu vulkan: texture %08x %ux%u (%u levels) upscaled %ux%s\n", e->addr, w, h, nl, f, keep ? ", kept for partial updates" : "");
        return m.tex.get();
    }
    if (f) {  // no kernels: filter off
        want_filter = tf_mode = 0;
        m.req = 0;
    }
    m.tex = make_img(w, h, VK_FORMAT_R8G8B8A8_UNORM, nl, kTexUsage);
    upload_levels(e, nl, *m.tex);
    if (log_timing) {
        st_uploads++;
        st_upload_ms += now_ms() - t0;
    }
    return m.tex.get();
}

std::unordered_map<u64, VkSampler> sampler_cache;
VkSamplerAddressMode addr_mode(u32 wrap) {
    switch (wrap) {
    case 2:
    case 6:
    case 7: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 3: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;  // border modes are emulated in the shader
    }
}
// ul: log2 of the texture's upscale factor (its first ul levels come before the game's level 0)
VkSampler get_sampler(const TexUnitState &tu, u32 levels, u32 ul = 0) {
    const u64 key = (u64)ul << 28 | (u64)aniso << 22 | (u64)tu.mag_linear | (u64)tu.min_linear << 1 |
                    (u64)tu.mip_linear << 2 | (u64)tu.wrap_s << 3 | (u64)tu.wrap_t << 6 | (u64)levels << 9 |
                    (u64)(tu.min_level & 15) << 14 | (u64)(tu.max_level & 15) << 18;
    auto it = sampler_cache.find(key);
    if (it != sampler_cache.end()) return it->second;
    const u32 glevels = levels - ul;  // the game's
    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter = tu.mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = tu.min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    if (glevels > 1) {
        ci.mipmapMode = tu.mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.minLod = (float)tu.min_level;
        ci.maxLod = (float)std::max(tu.min_level, tu.max_level);
    } else if (!ul) {
        // not mipmapped: level 0 only, while the LOD still picks the minification or magnification filter
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.minLod = 0;
        ci.maxLod = 0.25f;
    }
    if (ul) {
        // upscaled: what the game magnifies may now be minified (LOD 0..ul), so the minification filter follows the
        // magnification one; the game's levels moved down by ul
        if (tu.mag_linear) ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = glevels > 1 && !tu.mip_linear ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        if (!tu.mag_linear && !tu.min_linear) ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.minLod = tu.min_level > 0 ? (float)(tu.min_level + ul) : 0.0f;
        ci.maxLod = (glevels > 1 ? (float)std::max(tu.min_level, tu.max_level) : 0.0f) + (float)ul;
    }
    ci.addressModeU = addr_mode(tu.wrap_s);
    ci.addressModeV = addr_mode(tu.wrap_t);
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    // anisotropic filtering where the game asked for linear minification of a mipmapped texture
    if (aniso > 1 && tu.min_linear && glevels > 1) {
        ci.anisotropyEnable = VK_TRUE;
        ci.maxAnisotropy = (float)aniso;
    }
    VkSampler s;
    VK_CHECK(vkCreateSampler(dev, &ci, nullptr, &s));
    sampler_cache[key] = s;
    return s;
}

// ---------------------------------------------------------------- pipeline cache
// As metal.mm's: a pipeline per fs_key(), known ones (shipped and the user's lists) built at startup on background
// threads, new ones in the background (their draws skipped meanwhile) unless GPU_SYNC_SHADERS=1.
struct Pipe {
    VkPipeline ps = VK_NULL_HANDLE;
    int state = 0;       // 0 queued, 1 compiling, 2 done (ps is null if it failed)
    bool known = false;  // from a pipeline list: wait for it rather than skipping its draws
};
// (never destroyed: the workers wait on them until the process ends, and glibc's condition variables cannot be
// destroyed while waited on)
std::mutex &pipe_mtx = *new std::mutex;
std::condition_variable &pipe_cv = *new std::condition_variable;
std::unordered_map<std::string, std::unique_ptr<Pipe>> &pipe_cache = *new std::unordered_map<std::string, std::unique_ptr<Pipe>>;
std::deque<std::string> &pipe_queue = *new std::deque<std::string>;
int pipe_workers = 0;
FILE *key_log;
struct {
    u32 built = 0, fresh = 0, waited = 0, skipped = 0;
    double ms = 0, max_ms = 0, wait_ms = 0;
} pstats;

VkPipeline build_pipeline(const DrawState &ds) {
    const double t0 = now_ms();
    const FsOptions fo{ShaderLang::GLSL, has_logic_op, MS};
    const std::string src = gen_fs(ds, fo);
    if (dump_shaders) LOG("gpu vulkan: new fragment shader:\n%s\n", src.c_str());
    VkShaderModule fs = make_module(compile_glsl(src, shaderc_fragment_shader, "fragment shader"));
    if (!fs) return VK_NULL_HANDLE;
    VkPipeline p = make_pipeline(vs_module, fs, &ds, false);
    vkDestroyShaderModule(dev, fs, nullptr);
    const double dt = now_ms() - t0;
    std::lock_guard<std::mutex> lk(pipe_mtx);
    pstats.built++;
    pstats.ms += dt;
    pstats.max_ms = std::max(pstats.max_ms, dt);
    return p;
}

// (as platform::set_thread_name, which the gpu_test tools do not link)
void name_thread(const char *n) {
#ifdef __APPLE__
    pthread_setname_np(n);
#else
    pthread_setname_np(pthread_self(), n);
#endif
}
void pipe_worker() {
    name_thread("vk-pipelines");
    std::unique_lock<std::mutex> lk(pipe_mtx);
    for (;;) {
        pipe_cv.wait(lk, [] { return !pipe_queue.empty(); });
        const std::string key = std::move(pipe_queue.front());
        pipe_queue.pop_front();
        Pipe &p = *pipe_cache[key];
        if (p.state != 0) continue;
        p.state = 1;
        lk.unlock();
        VkPipeline ps = build_pipeline(decode_key(key));
        lk.lock();
        p.ps = ps;
        p.state = 2;
        pipe_cv.notify_all();
        static bool reported = false;
        if (!reported && pipe_queue.empty() && pstats.built == pipe_cache.size() && (reported = true)) {
            LOG("gpu vulkan: %zu pipelines ready (%.1f ms avg, slowest %.1f ms; draws waited %u times, %.1f ms in all)\n",
                pipe_cache.size(), pstats.built ? pstats.ms / pstats.built : 0.0, pstats.max_ms, pstats.waited,
                pstats.wait_ms);
            lk.unlock();
            save_pipeline_cache();
            lk.lock();
        }
    }
}
void start_workers() {  // (pipe_mtx held)
    if (pipe_workers) return;
    pipe_workers = (int)std::clamp(std::thread::hardware_concurrency() / 2, 2u, 6u);
    for (int i = 0; i < pipe_workers; i++) std::thread(pipe_worker).detach();
}
void record_key(const std::string &key) {  // (pipe_mtx held)
    if (!key_log) return;
    const u32 n = (u32)key.size();
    fwrite(&n, 4, 1, key_log);
    fwrite(key.data(), 1, n, key_log);
    fflush(key_log);
}
void verify_key(const std::string &key, const DrawState &ds) {
    const FsOptions fo{ShaderLang::GLSL, has_logic_op, MS};
    const DrawState d = decode_key(key);
    if (fs_key(d) != key) LOG("gpu vulkan: KEY MISMATCH: fs_key(decode_key(k)) != k\n");
    else if (gen_fs(d, fo) != gen_fs(ds, fo)) LOG("gpu vulkan: KEY MISMATCH: decoded state generates another shader\n");
}

// ---------------------------------------------------------------- vertex program kernels
// As metal.mm's: programs (and their specializations for the uniforms their flow control reads) recorded in
// <saves>/shadercache/<title id>.vs.bin, the shipped and recorded ones compiled at startup, new ones in the background
// while the CPU shades their draws.
struct VsKern {
    VkPipeline ps = VK_NULL_HANDLE;
    bool done = false;
};
std::recursive_mutex &vk_mtx = *new std::recursive_mutex;
std::unordered_map<u64, std::unique_ptr<VsKern>> &vk_cache = *new std::unordered_map<u64, std::unique_ptr<VsKern>>;
FILE *vk_log;
struct {
    u32 built = 0, fail = 0, cpu_draws = 0;
    double ms = 0;
} vkstats;

VkPipeline build_vs_pipeline(const VsBlob &v);
void build_vs_kernel(VsKern *k, const std::string &blob) {
    const double t0 = now_ms();
    VsBlob v;
    VkPipeline ps = parse_vs_blob(blob, v) ? build_vs_pipeline(v) : VK_NULL_HANDLE;
    std::lock_guard<std::recursive_mutex> lk(vk_mtx);
    k->ps = ps;
    k->done = true;
    if (!ps) vkstats.fail++;
    else vkstats.built++;
    vkstats.ms += now_ms() - t0;
}

// (the pipeline workers build these too)
std::mutex &vs_queue_mtx = *new std::mutex;
std::condition_variable &vs_queue_cv = *new std::condition_variable;
std::deque<std::pair<VsKern *, std::string>> &vs_queue = *new std::deque<std::pair<VsKern *, std::string>>;
void vs_worker() {
    name_thread("vk-vertex-progs");
    std::unique_lock<std::mutex> lk(vs_queue_mtx);
    for (;;) {
        vs_queue_cv.wait(lk, [] { return !vs_queue.empty(); });
        auto job = std::move(vs_queue.front());
        vs_queue.pop_front();
        lk.unlock();
        build_vs_kernel(job.first, job.second);
        lk.lock();
    }
}
void vs_compile_async(VsKern *k, std::string blob) {
    if (sync_shaders) {
        build_vs_kernel(k, blob);
        return;
    }
    std::lock_guard<std::mutex> lk(vs_queue_mtx);
    static bool started = false;
    if (!started) {
        started = true;
        for (int i = 0; i < 2; i++) std::thread(vs_worker).detach();
    }
    vs_queue.emplace_back(k, std::move(blob));
    vs_queue_cv.notify_one();
}
VsKern *vs_add(const std::string &blob) {  // (vk_mtx held)
    VsBlob v;
    if (!parse_vs_blob(blob, v)) return nullptr;
    const u64 key = vk_key(vk_hash(v.code.data(), v.code_size, v.swz.data(), v.swz_size), v.entry);
    auto &k = vk_cache[v.spec ? spec_key(key, v.sp) : key];
    if (k) return nullptr;
    k = std::make_unique<VsKern>();
    vs_compile_async(k.get(), blob);
    return k.get();
}
void vs_kernels_init(const std::string &tid) {
    std::vector<std::string> blobs;
    for (const auto &p : {g_game.dir + "/vsprograms.bin", std::string(RECOMP_SOURCE_DIR) + "/games/" + tid + "/vsprograms.bin"}) {
        if (!std::filesystem::exists(p)) continue;
        blobs = read_blobs(p, kVsMagic, kVsVersion);
        break;
    }
    const size_t shipped = blobs.size();
    const std::string user = g_save_base + "/shadercache/" + tid + ".vs.bin";
    std::vector<std::string> mine = read_blobs(user, kVsMagic, kVsVersion);
    blobs.insert(blobs.end(), mine.begin(), mine.end());
    if (mine.empty()) {
        vk_log = fopen(user.c_str(), "wb");
        const u32 hdr[2] = {kVsMagic, kVsVersion};
        if (vk_log) fwrite(hdr, 4, 2, vk_log);
    } else {
        vk_log = fopen(user.c_str(), "ab");
    }
    std::lock_guard<std::recursive_mutex> lk(vk_mtx);
    for (const auto &b : blobs) vs_add(b);
    LOG("gpu vulkan: %zu known vertex programs (%zu shipped, %zu in %s), compiling in the background\n", vk_cache.size(),
        shipped, mine.size(), user.c_str());
}

void pipelines_init() {
    sync_shaders = getenv("GPU_SYNC_SHADERS") != nullptr;
    verify_keys = getenv("GPU_VERIFY_KEYS") != nullptr;
    char tid[20];
    snprintf(tid, sizeof tid, "%016llx", (unsigned long long)g_game.program_id);
    const double t0 = now_ms();
    const std::string dir = g_save_base + "/shadercache";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    vs_kernels_init(tid);
    // the driver's compiled pipelines from earlier runs
    {
        pcache_path = dir + "/" + tid + ".vkpc";
        std::vector<u8> data;
        if (FILE *f = fopen(pcache_path.c_str(), "rb")) {
            u8 b[65536];
            size_t n;
            while ((n = fread(b, 1, sizeof b, f)) > 0) data.insert(data.end(), b, b + n);
            fclose(f);
        }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = data.size();
        ci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(dev, &ci, nullptr, &pcache) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            vkCreatePipelineCache(dev, &ci, nullptr, &pcache);
        }
    }
    if (getenv("GPU_NO_PRECOMPILE")) {
        LOG("gpu vulkan: GPU_NO_PRECOMPILE: no known pipelines built ahead\n");
        return;
    }
    std::vector<std::string> keys;
    for (const auto &p : {g_game.dir + "/pipelines.bin", std::string(RECOMP_SOURCE_DIR) + "/games/" + tid + "/pipelines.bin"}) {
        if (!std::filesystem::exists(p)) continue;
        keys = read_keys(p);
        LOG("gpu vulkan: %zu pipelines listed in %s\n", keys.size(), p.c_str());
        break;
    }
    const size_t shipped = keys.size();
    const std::string user = dir + "/" + tid + ".bin";
    std::vector<std::string> mine = read_keys(user);
    keys.insert(keys.end(), mine.begin(), mine.end());
    if (mine.empty()) {
        key_log = fopen(user.c_str(), "wb");
        const u32 hdr[2] = {kKeyMagic, kKeyVersion};
        if (key_log) fwrite(hdr, 4, 2, key_log);
    } else {
        key_log = fopen(user.c_str(), "ab");
    }
    std::lock_guard<std::mutex> lk(pipe_mtx);
    for (auto &k : keys) {
        auto &p = pipe_cache[k];
        if (p) continue;
        p = std::make_unique<Pipe>();
        p->known = true;
        pipe_queue.push_back(k);
    }
    if (!pipe_queue.empty()) start_workers();
    pipe_cv.notify_all();
    LOG("gpu vulkan: %zu known pipelines (%zu shipped, %zu in %s), building in the background%s (%.1f ms)\n",
        pipe_cache.size(), shipped, mine.size(), user.c_str(), sync_shaders ? "; new ones synchronously" : "",
        now_ms() - t0);
}

Pipe *get_pipeline(const DrawState &ds) {
    const std::string key = fs_key(ds);
    std::unique_lock<std::mutex> lk(pipe_mtx);
    auto it = pipe_cache.find(key);
    if (it == pipe_cache.end()) {
        if (verify_keys) verify_key(key, ds);
        record_key(key);
        Pipe &p = *(pipe_cache[key] = std::make_unique<Pipe>());
        pstats.fresh++;
        LOG("gpu vulkan: new pipeline %016llx (%u this session)%s\n", (unsigned long long)fnv64(key), pstats.fresh,
            sync_shaders ? "" : ", compiling in the background");
        if (!sync_shaders) {
            pipe_queue.push_front(key);
            start_workers();
            pipe_cv.notify_all();
            pstats.skipped++;
            return nullptr;
        }
        p.state = 1;
        lk.unlock();
        VkPipeline ps = build_pipeline(ds);
        lk.lock();
        p.ps = ps;
        p.state = 2;
        pipe_cv.notify_all();
        return &p;
    }
    Pipe &p = *it->second;
    if (p.state == 2) return &p;
    if (!p.known && !sync_shaders) {
        pstats.skipped++;
        return nullptr;
    }
    const double t0 = now_ms();
    if (p.state == 0) {
        p.state = 1;
        lk.unlock();
        VkPipeline ps = build_pipeline(ds);
        lk.lock();
        p.ps = ps;
        p.state = 2;
        pipe_cv.notify_all();
    } else {
        pipe_cv.wait(lk, [&] { return p.state == 2; });
    }
    pstats.waited++;
    pstats.wait_ms += now_ms() - t0;
    if (g_stats) LOG("gpu vulkan: waited %.1f ms for a known pipeline\n", now_ms() - t0);
    return &p;
}

VkCompareOp cmp_op(u32 f) {
    static const VkCompareOp m[8] = {VK_COMPARE_OP_NEVER,         VK_COMPARE_OP_ALWAYS,    VK_COMPARE_OP_EQUAL,
                                     VK_COMPARE_OP_NOT_EQUAL,     VK_COMPARE_OP_LESS,      VK_COMPARE_OP_LESS_OR_EQUAL,
                                     VK_COMPARE_OP_GREATER,       VK_COMPARE_OP_GREATER_OR_EQUAL};
    return m[f & 7];
}

// ---------------------------------------------------------------- display images
struct DisplayImage {
    u32 out_addr = 0, out_size = 0;
    u32 out_w = 0, out_bpp = 0, hs = 0, vs = 0;
    u32 scale = 1;
    bool incomplete = false;
    u64 out_hash = 0;
    int screen = -1;
    u32 w = 0, h = 0;
    u64 version = 0;
    BufP buf;     // the image (RGBA8 rows), host-cached
    u64 tl = 0;   // while the image is still being made: the timeline value that completes it
    bool direct = false;  // made by vk_display_transfer: guest memory is final already, only the image is pending
};
std::mutex disp_mtx;
std::unordered_map<u32, DisplayImage> displays;
u32 last_screen_addr[2] = {0, 0};
u64 disp_version = 0;

// ---------------------------------------------------------------- deferred display transfers
struct PendingXfer {
    u64 tl = 0;
    BufP gbytes, obytes, disp;
    GRT *rt = nullptr;
    bool rt_writeback = true;
    u32 in_paddr = 0, out_paddr = 0, in_dims = 0, out_dims = 0, flags = 0, out_size = 0;
};
std::deque<std::unique_ptr<PendingXfer>> pending;
std::vector<BufP> xbuf_free;     // GX thread only
std::vector<BufP> disp_free;     // under disp_mtx
std::vector<Buf *> disp_reading;  // under disp_mtx: images the UI reads outside the lock (not reused meanwhile)
bool recyclable(const BufP &b) {
    for (Buf *r : disp_reading)
        if (r == b.get()) return false;
    return disp_free.size() < 6;
}
BufP take_buffer(std::vector<BufP> &fl, size_t n) {
    for (size_t i = 0; i < fl.size(); i++)
        if (fl[i]->size >= n) {
            BufP b = std::move(fl[i]);
            fl.erase(fl.begin() + i);
            return b;
        }
    return make_buf(n, Mem::Readback);
}

void rts_overwritten(u32 paddr, u32 size) {
    for (auto &r : rts)
        if (overlaps(r->addr, rt_bytes(*r), paddr, size)) r->synced = false;
}

void finalize(PendingXfer &op) {
    wait_value(op.tl, "display transfer", op.out_paddr);
    GRT &r = *op.rt;
    const u32 nb = rt_bytes(r);
    if (op.rt_writeback && guest_kept(r)) {
        op.gbytes->invalidate();
        r.hash = copy_hash(pptr(r.addr), op.gbytes->p, nb);
        r.checked = g_frame.load(std::memory_order_relaxed);
    }
    op.obytes->invalidate();
    const u64 out_hash = copy_hash(pptr(op.out_paddr), op.obytes->p, op.out_size);
    texcache_invalidate(op.out_paddr, op.out_size);
    rts_overwritten(op.out_paddr, op.out_size);
    if (tf_mode) note_gpu_out(op.out_paddr, op.out_size);
    xbuf_free.push_back(std::move(op.gbytes));
    xbuf_free.push_back(std::move(op.obytes));
    std::lock_guard<std::mutex> lk(disp_mtx);
    auto it = displays.find(op.out_paddr);
    if (it != displays.end() && it->second.tl == op.tl && it->second.buf == op.disp) {
        DisplayImage &d = it->second;
        d.out_hash = out_hash;  // (d.out_addr, d.out_size are the transfer's output)
        d.tl = 0;
    } else if (recyclable(op.disp)) {
        disp_free.push_back(std::move(op.disp));
    }
}
void drain(size_t i) {
    for (size_t k = 0; k <= i; k++) {
        finalize(*pending.front());
        pending.pop_front();
    }
}
void drain_all() {
    if (!pending.empty()) drain(pending.size() - 1);
}
// keep: a target whose own pending write-back may stay pending (it is about to be drawn into: its GPU copy stays
// newer than what the write-back puts in guest memory, and its next write-back replaces that)
bool op_touches(const PendingXfer &op, u32 paddr, u32 size, const GRT *keep = nullptr) {
    return overlaps(op.out_paddr, op.out_size, paddr, size) ||
           (op.rt_writeback && op.rt != keep && overlaps(op.rt->addr, rt_bytes(*op.rt), paddr, size));
}
void drain_touching(u32 paddr, u32 size, const GRT *keep = nullptr) {
    for (size_t i = pending.size(); i-- > 0;)
        if (op_touches(*pending[i], paddr, size, keep)) {
            drain(i);
            return;
        }
}

// ---------------------------------------------------------------- GPU copies of guest images
inline u32 pix_to_cbuf(u32 pf) {
    static constexpr u8 m[8] = {0, 1, 3, 2, 4, 0, 0, 0};
    return m[pf & 7];
}
inline u32 grt_bpp(const GRT &r) { return pixel_bytes(cbuf_to_pixfmt(r.fmt)); }

bool gpu_valid(const GRT &r) {
    if (r.depth || !r.synced) return false;
    if (r.gpu_newer) return true;
    for (const auto &op : pending)
        if (op->rt == &r && op->rt_writeback) return true;
    return false;
}
GRT *find_view(u32 addr, u32 w, u32 fmt, bool depth, bool any = false) {
    GRT *best = nullptr;
    for (auto &r : rts)
        if (r->depth == depth && (r->synced || any) && r->addr == addr && r->w == w && r->fmt == fmt && (!best || r->h > best->h))
            best = r.get();
    return best;
}
GRT *find_color(u32 addr, u32 w, u32 fmt) { return find_view(addr, w, fmt, false); }

// clears an image now (outside a pass)
void clear_image(const Img &im, bool depth, u32 color, float z, u32 stencil) {
    VkCommandBuffer cb = rec();
    barrier(cb);
    if (depth) {
        const VkClearDepthStencilValue v{z, stencil};
        const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cb, im.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg);
    } else {
        VkClearColorValue v;
        v.float32[0] = (color & 255) / 255.0f;
        v.float32[1] = ((color >> 8) & 255) / 255.0f;
        v.float32[2] = ((color >> 16) & 255) / 255.0f;
        v.float32[3] = (color >> 24) / 255.0f;
        const VkImageSubresourceRange rg{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, im.img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg);
    }
}

void fill_from_guest(GRT &r) {
    const LoadAction la = sync_rt(r);
    if (!la.clear) return;
    clear_image(*r.tex, r.depth, la.color, la.depth, la.stencil);
    r.ms_valid = false;
    r.version++;
}

void copy_image(const Img &src, const Img &dst, s32 sx, s32 sy, u32 w, u32 h, VkImageAspectFlags aspect) {
    VkCommandBuffer cb = rec();
    barrier(cb);
    VkImageCopy c{};
    c.srcSubresource = {aspect, 0, 0, 1};
    c.srcOffset = {sx, sy, 0};
    c.dstSubresource = {aspect, 0, 0, 1};
    c.extent = {w, h, 1};
    vkCmdCopyImage(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, dst.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

GRT *grow(GRT &old, u32 h) {
    for (const auto &op : pending)
        if (op->rt == &old) return nullptr;
    GRT &n = get_rt(old.addr, old.w, h, old.fmt, old.depth);
    n.synced = false;
    fill_from_guest(n);
    copy_image(*old.tex, *n.tex, 0, 0, old.w * S, old.h * S,
               old.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
    n.gpu_newer = old.gpu_newer;
    n.ms_valid = false;
    n.version++;
    old.synced = false;
    old.gpu_newer = false;
    return &n;
}

GRT &target_for(u32 addr, u32 w, u32 h, u32 fmt, bool depth) {
    GRT *t = find_view(addr, w, fmt, depth, true);
    if (t && t->h == h) return *t;
    if (t && t->h < h && t->gpu_newer)
        if (GRT *n = grow(*t, h)) return *n;
    return get_rt(addr, w, h, fmt, depth);
}

void drop_displays(u32 paddr, u32 size, bool definite);

bool claim(GRT *src, GRT *dst, u32 paddr, u32 size) {
    drain_touching(paddr, size);
    bool changed = false;
    for (auto &r : rts) {
        if (r.get() == src || r.get() == dst || !overlaps(r->addr, rt_bytes(*r), paddr, size)) continue;
        if (r->gpu_newer) {
            if (r->addr >= paddr && r->addr + rt_bytes(*r) <= paddr + size) {
                r->gpu_newer = false;
            } else {
                if (r->depth) write_back_depth(*r);
                else write_back_color(*r);
                changed = true;
            }
        }
        r->synced = false;
    }
    return changed;
}

GRT *dest_surface(u32 addr, u32 w, u32 h, u32 fmt, GRT *src, u32 bytes) {
    if (overlaps(addr, bytes, src->addr, rt_bytes(*src))) return nullptr;
    GRT *d = find_color(addr, w, fmt);
    if (d && overlaps(d->addr, rt_bytes(*d), src->addr, rt_bytes(*src))) return nullptr;
    if (claim(src, d, addr, bytes)) return nullptr;
    if (!d) {
        d = &get_rt(addr, w, h, fmt, false);
        fill_from_guest(*d);
    } else if (d->h < h) {
        d = grow(*d, h);
    }
    return d;
}

void gpu_wrote(GRT &d, u32 paddr, u32 size) {
    d.gpu_newer = true;
    d.ms_valid = false;
    d.version++;
    texcache_invalidate(paddr, size);
    drop_displays(paddr, size, true);
}

// A texture as its GPU copy holds it, rows in the decoded textures' order (made once per change of the copy)
struct Snap {
    ImgP tex;
    u64 id = 0, version = 0;
};
std::unordered_map<u64, Snap> snaps;
const Img *surface_snapshot(u32 addr, u32 w, u32 h, u32 fmt) {
    GRT *r = find_color(addr, w, fmt);
    if (!r || r->h < h) return nullptr;
    Snap &sn = snaps[(u64)addr << 32 ^ (u64)w << 20 ^ (u64)h << 4 ^ fmt];
    if (!sn.tex || sn.tex->w != w * S || sn.tex->h != h * S) {
        sn.tex = make_img(w * S, h * S, VK_FORMAT_R8G8B8A8_UNORM, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        sn.id = 0;
    } else if (sn.id == r->id && sn.version == r->version) {
        return sn.tex.get();
    }
    VkCommandBuffer cb = rec();
    barrier(cb);
    const u32 p[4] = {w * S, h * S, fmt == 1 || fmt == 3 ? 1u : 0u, 0};  // RGB8, RGB565: no alpha
    dispatch(cb, K_SNAP, {ki(4, *r->tex), ki(5, *sn.tex)}, p, nullptr, w * S, h * S);
    sn.id = r->id;
    sn.version = r->version;
    return sn.tex.get();
}

// ---------------------------------------------------------------- init
VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
VKAPI_ATTR VkBool32 VKAPI_CALL on_debug(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT,
                                        const VkDebugUtilsMessengerCallbackDataEXT *d, void *) {
    static std::atomic<int> n{0};
    if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT && n++ < 200) LOG("gpu vulkan: [validation] %s\n", d->pMessage);
    return VK_FALSE;
}

bool has_ext(const std::vector<VkExtensionProperties> &exts, const char *name) {
    for (auto &e : exts)
        if (!strcmp(e.extensionName, name)) return true;
    return false;
}

VkPipeline make_kernel(const char *src, const char *name, VkPipelineLayout layout = VK_NULL_HANDLE) {
    VkShaderModule m = make_module(compile_glsl(src, shaderc_compute_shader, name));
    if (!m) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.layout = layout ? layout : k_layout;
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    vkDestroyShaderModule(dev, m, nullptr);
    return p;
}

bool init_device() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "3dsrecomp";
    app.apiVersion = VK_API_VERSION_1_3;
    std::vector<const char *> layers, iexts;
    const bool validate = getenv("GPU_VALIDATE") != nullptr;
    if (validate) {
        u32 n = 0;
        vkEnumerateInstanceLayerProperties(&n, nullptr);
        std::vector<VkLayerProperties> lp(n);
        vkEnumerateInstanceLayerProperties(&n, lp.data());
        for (auto &l : lp)
            if (!strcmp(l.layerName, "VK_LAYER_KHRONOS_validation")) layers.push_back("VK_LAYER_KHRONOS_validation");
        if (layers.empty()) LOG("gpu vulkan: GPU_VALIDATE: no validation layer installed (vulkan-validation-layers)\n");
        iexts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    u32 ni = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ni, nullptr);
    std::vector<VkExtensionProperties> iep(ni);
    vkEnumerateInstanceExtensionProperties(nullptr, &ni, iep.data());
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    // the window system's surface extensions (set_present_extensions), for presenting on this device
    bool present_inst = !present_extensions().empty() && !getenv("GPU_PRESENT_CPU");
    for (const std::string &e : present_extensions()) present_inst = present_inst && has_ext(iep, e.c_str());
    if (present_inst)
        for (const std::string &e : present_extensions())
            if (std::find_if(iexts.begin(), iexts.end(), [&](const char *x) { return e == x; }) == iexts.end()) iexts.push_back(e.c_str());
    if (has_ext(iep, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {  // (MoltenVK)
        iexts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = (u32)layers.size();
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = (u32)iexts.size();
    ici.ppEnabledExtensionNames = iexts.data();
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        LOG("gpu vulkan: no Vulkan instance, using the software renderer\n");
        return false;
    }
    if (validate) {
        auto mk = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mi.pfnUserCallback = on_debug;
        if (mk) mk(inst, &mi, nullptr, &messenger);
    }
    // the device: GPU_DEVICE (an index, or part of its name) or the first discrete one, else the first one
    u32 nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> pds(nd);
    vkEnumeratePhysicalDevices(inst, &nd, pds.data());
    const char *want = getenv("GPU_DEVICE");
    int pick = -1;
    for (u32 i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        if (want && *want) {
            if ((isdigit((unsigned char)want[0]) && (u32)atoi(want) == i) || strstr(p.deviceName, want)) pick = (int)i;
            continue;
        }
        if (pick < 0 || (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
            VkPhysicalDeviceProperties q;
            if (pick >= 0) vkGetPhysicalDeviceProperties(pds[pick], &q);
            if (pick < 0 || q.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pick = (int)i;
        }
    }
    if (pick < 0) {
        LOG("gpu vulkan: no Vulkan 1.3 device%s%s, using the software renderer\n", want ? " named " : "", want ? want : "");
        return false;
    }
    phys = pds[pick];
    vkGetPhysicalDeviceProperties(phys, &props);
    u32 nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    qfam = ~0u;
    for (u32 i = 0; i < nq; i++)
        if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            qfam = i;
            break;
        }
    if (qfam == ~0u) return false;
    ts_ok = qf[qfam].timestampValidBits >= 32 && props.limits.timestampPeriod > 0;
    ts_ns = props.limits.timestampPeriod;
    u32 ne = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> dep(ne);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, dep.data());
    std::vector<const char *> dexts;
    if (!has_ext(dep, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        LOG("gpu vulkan: %s has no push descriptors, using the software renderer\n", props.deviceName);
        return false;
    }
    dexts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    // GPU_VK_DISABLE=stencil_export,logic_op,local_read: the paths for devices without them, tried on one with them
    const char *dis = getenv("GPU_VK_DISABLE");
    auto disabled = [&](const char *f) { return dis && strstr(dis, f); };
    has_stencil_export = has_ext(dep, VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME) && !disabled("stencil_export");
    if (has_stencil_export) dexts.push_back(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    if (has_ext(dep, "VK_KHR_portability_subset")) dexts.push_back("VK_KHR_portability_subset");
    // features
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR lrf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f12.pNext = &f13;
    const bool lr_ext = has_ext(dep, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME);
    if (lr_ext) f13.pNext = &lrf;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore || !f12.storageBuffer8BitAccess ||
        !f2.features.depthClamp || !f2.features.shaderClipDistance) {
        LOG("gpu vulkan: %s lacks features this backend needs, using the software renderer\n", props.deviceName);
        return false;
    }
    has_logic_op = f2.features.logicOp && !disabled("logic_op");
    has_local_read = lr_ext && lrf.dynamicRenderingLocalRead && !disabled("local_read");
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR elr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features e13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 e2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    e2.pNext = &e12;
    e12.pNext = &e13;
    e13.dynamicRendering = VK_TRUE;
    e13.synchronization2 = VK_TRUE;
    e13.maintenance4 = f13.maintenance4;
    e13.shaderDemoteToHelperInvocation = f13.shaderDemoteToHelperInvocation;  // (glslang compiles discard to it for 1.3)
    e12.timelineSemaphore = VK_TRUE;
    e12.storageBuffer8BitAccess = VK_TRUE;
    e12.uniformAndStorageBuffer8BitAccess = f12.uniformAndStorageBuffer8BitAccess;
    e12.hostQueryReset = f12.hostQueryReset;
    e2.features.depthClamp = VK_TRUE;
    e2.features.shaderClipDistance = VK_TRUE;
    e2.features.logicOp = has_logic_op;
    e2.features.samplerAnisotropy = f2.features.samplerAnisotropy;
    e2.features.shaderStorageImageMultisample = f2.features.shaderStorageImageMultisample;
    e2.features.sampleRateShading = f2.features.sampleRateShading;
    if (has_local_read) {
        dexts.push_back(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME);
        elr.dynamicRenderingLocalRead = VK_TRUE;
        e13.pNext = &elr;
    }
    // a second queue for SDL's renderer when presenting on this device
    present_ok = present_inst && qf[qfam].queueCount >= 2 && has_ext(dep, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (present_ok) dexts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    const float prio[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam;
    qci.queueCount = present_ok ? 2 : 1;
    qci.pQueuePriorities = prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &e2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (u32)dexts.size();
    dci.ppEnabledExtensionNames = dexts.data();
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        LOG("gpu vulkan: cannot create a device on %s, using the software renderer\n", props.deviceName);
        return false;
    }
    vkGetDeviceQueue(dev, qfam, present_ok ? 1 : 0, &queue);
    if (present_ok) vkGetDeviceQueue(dev, qfam, 0, &sdl_queue);
    vkPushDescriptors = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(dev, "vkCmdPushDescriptorSetKHR");
    // depth / stencil format
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_D32_SFLOAT_S8_UINT, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        LOG("gpu vulkan: %s has no D32S8 depth format, using the software renderer\n", props.deviceName);
        return false;
    }
    VmaAllocatorCreateInfo vi{};
    vi.physicalDevice = phys;
    vi.device = dev;
    vi.instance = inst;
    vi.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&vi, &vma));
    VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &st;
    VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &timeline));
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qfam;
    VK_CHECK(vkCreateCommandPool(dev, &pci, nullptr, &cmd_pool));
    if (ts_ok) {
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = kQuerySlots * 2;
        if (vkCreateQueryPool(dev, &qp, nullptr, &qpool) != VK_SUCCESS) ts_ok = false;
    }
    // descriptor set layouts (push descriptors)
    {
        VkDescriptorSetLayoutBinding b[6];
        for (u32 i = 0; i < 6; i++)
            b[i] = {i, i < 4 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        ci.bindingCount = 6;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &k_dsl));
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &k_dsl;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pr;
        VK_CHECK(vkCreatePipelineLayout(dev, &li, nullptr, &k_layout));
    }
    {
        const VkDescriptorSetLayoutBinding b[3] = {
            {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        ci.bindingCount = 3;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &tf_dsl));
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TFP)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &tf_dsl;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pr;
        VK_CHECK(vkCreatePipelineLayout(dev, &li, nullptr, &tf_layout));
    }
    {
        VkDescriptorSetLayoutBinding b[5];
        for (u32 i = 0; i < 5; i++) b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        ci.bindingCount = 5;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &v_dsl));
        const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &v_dsl;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pr;
        VK_CHECK(vkCreatePipelineLayout(dev, &li, nullptr, &v_layout));
    }
    {
        VkDescriptorSetLayoutBinding b[18];
        for (u32 i = 0; i < 18; i++) {
            VkDescriptorType t = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            if (i == 0) t = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            else if (i <= 2 || i >= 15) t = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            else if (i >= 6 && i <= 8) t = VK_DESCRIPTOR_TYPE_SAMPLER;
            else if (i == 14) t = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            b[i] = {i, t, 1, (VkShaderStageFlags)(i >= 15 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT), nullptr};
        }
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        ci.bindingCount = 18;
        ci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &g_dsl));
        const VkPushConstantRange pr[2] = {{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VsDraw)},
                                           {VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(VsDraw), 16}};  // (stencil restore)
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &g_dsl;
        li.pushConstantRangeCount = 2;
        li.pPushConstantRanges = pr;
        VK_CHECK(vkCreatePipelineLayout(dev, &li, nullptr, &g_layout));
    }
    return true;
}

}  // namespace

// the backend's hooks (pica.h's HwBackend; the table is at the end of this file)
bool vk_init();
void vk_flush(const RTConfig &rt, const MBatch &b);
const void *vk_vs_kernel(const ShaderProgram *p, u32 entry, const ShaderSetup &s);
void vk_guest_read(u32 paddr, u32 size);
void vk_guest_written(u32 paddr, u32 size, bool definite);
void vk_guest_prewrite(u32 paddr, u32 size);
void vk_writeback_all();
void vk_poll();
bool vk_texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width, u32 in_gap, u32 out_width, u32 out_gap);
bool vk_gpu_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
bool vk_surface_for(u32 addr, u32 w, u32 h, u32 fmt);
void vk_end_list();
bool vk_defer_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags, u32 out_size);
void vk_display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
void vk_set_texture_filter(int mode);
const char *vk_texture_filter_name(int mode);

bool vk_init() {
    if (tried) return active;
    tried = true;
    const char *be = getenv("GPU_BACKEND");
    if (be && (!strcmp(be, "sw") || !strcmp(be, "metal"))) return false;
    if (!init_device()) return false;
    // shaders
    vs_module = make_module(compile_glsl(kVkVertexGLSL, shaderc_vertex_shader, "vertex shader"));
    full_module = make_module(compile_glsl(kVkFullGLSL, shaderc_vertex_shader, "full-screen vertex shader"));
    const char *ksrc[K_COUNT] = {kVkRotGLSL,       kVkEncGLSL,   kVkDecGLSL,  kVkDecDepthGLSL,
                                 kVkEncDepthGLSL,  kVkTconvGLSL, kVkXferGLSL, kVkSnapGLSL};
    const char *kname[K_COUNT] = {"k_rot", "k_enc", "k_dec", "k_dec_depth", "k_enc_depth", "k_tconv", "k_xfer", "k_snap"};
    for (int i = 0; i < K_COUNT; i++)
        if (!(kpipe[i] = make_kernel((std::string(kVkKernelHead) + ksrc[i]).c_str(), kname[i]))) {
            LOG("gpu vulkan: kernel %s failed, using the software renderer\n", kname[i]);
            return false;
        }
    if (!vs_module || !full_module) return false;
    const char *sc = getenv("GPU_SCALE");
    scale_fixed = sc && atoi(sc) > 0;
    S = scale_fixed ? (u32)std::clamp(atoi(sc), 1, 4) : 2;
    const char *ms = getenv("GPU_MSAA");
    MS = ms ? (u32)std::clamp(atoi(ms), 1, 8) : kDefaultMSAA;
    const VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts &
                                      props.limits.framebufferStencilSampleCounts;
    while (MS > 1 && ((MS & (MS - 1)) || !(counts & MS))) MS--;
    if (MS > 1) {
        VkShaderModule rc = make_module(compile_glsl(kVkRestoreColorGLSL, shaderc_fragment_shader, "colour restore"));
        p_rcolor = rc ? make_pipeline(full_module, rc, nullptr, false) : VK_NULL_HANDLE;
        if (rc) vkDestroyShaderModule(dev, rc, nullptr);
        auto restore = [](const char *src, const char *name) {
            VkShaderModule m = make_module(compile_glsl(src, shaderc_fragment_shader, name));
            VkPipeline p = m ? make_pipeline(full_module, m, nullptr, true) : VK_NULL_HANDLE;
            if (m) vkDestroyShaderModule(dev, m, nullptr);
            return p;
        };
        if (has_stencil_export) {
            p_rdepth = restore(kVkRestoreDepthGLSL, "depth restore");
        } else {
            p_rdepth_only = restore(kVkRestoreDepthOnlyGLSL, "depth restore");
            p_rstencil_bit = restore(kVkRestoreStencilBitGLSL, "stencil restore");
        }
        if (!p_rcolor || (!p_rdepth && (!p_rdepth_only || !p_rstencil_bit))) {
            LOG("gpu vulkan: MSAA unavailable (restore pipelines failed)\n");
            MS = 1;
        }
    }
    const char *an = getenv("GPU_ANISO");
    aniso = an ? (u32)std::clamp(atoi(an), 1, 16) : kDefaultAniso;
    aniso = std::min(aniso, (u32)props.limits.maxSamplerAnisotropy);
    dump_shaders = getenv("GPU_DUMP_SHADERS") != nullptr;
    eager_writeback = getenv("GPU_METAL_EAGER") != nullptr || getenv("GPU_EAGER") != nullptr;
    log_timing = getenv("RECOMP_STATS") != nullptr;
    tf_verify = getenv("GPU_TF_VERIFY") != nullptr;
    // texture filters
    {
        const char *tfname[kVkTfKernCount] = {"texture filter (sharp)", "texture filter (smooth)", "texture filter (levels)"};
        tf_ok = true;
        for (int i = 0; i < kVkTfKernCount; i++)
            tf_ok = (tf_pipe[i] = make_kernel(vk_texfilter_glsl((VkTfKernel)i), tfname[i], tf_layout)) && tf_ok;
        if (!tf_ok) LOG("gpu vulkan: texture filter kernels unavailable\n");
        if (const char *tf = getenv("GPU_TEX_FILTER")) vk_set_texture_filter(atoi(tf));
    }
    // dummies for the bindings a pipeline does not use
    {
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(dev, &ci, nullptr, &dummy_sampler));
        dummy_buf = make_buf(64 * 1024, Mem::Device);
        black = make_img(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1, kTexUsage);
        clear_image(*black, false, 0, 0, 0);
        wait_value(submit());
        reap();
    }
    active = true;
    const u32 dv = props.driverVersion;
    const bool nv = props.vendorID == 0x10de;  // (NVIDIA's own version encoding)
    LOG("gpu: Vulkan backend on %s (driver %u.%u.%u), internal scale %s%u, MSAA %ux, anisotropic filtering %ux%s%s\n",
        props.deviceName, nv ? dv >> 22 : VK_API_VERSION_MAJOR(dv), nv ? (dv >> 14) & 0xff : VK_API_VERSION_MINOR(dv),
        nv ? (dv >> 6) & 0xff : VK_API_VERSION_PATCH(dv), scale_fixed ? "" : "auto, now ", S, MS, aniso,
        has_logic_op ? "" : ", no logic ops", has_local_read ? "" : ", no blending in shaders");
    if (!has_stencil_export && MS > 1) LOG("gpu vulkan: no stencil export: MSAA stencil restored a bit at a time\n");
    pipelines_init();
    return true;
}

namespace {
void apply_scale(u32 ns) {
    drain_all();
    for (auto &r : rts)
        if (r->gpu_newer) {
            if (r->depth) write_back_depth(*r);
            else write_back_color(*r);
        }
    rts.clear();
    scratch_depth.clear();
    snaps.clear();
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        disp_free.clear();
    }
    LOG("gpu vulkan: internal scale %u -> %u\n", S, ns);
    if (tf_mode && (S > 2) != (ns > 2)) tf_drop_all();  // upscale factors follow the scale (tf_factor)
    S = ns;
}
}  // namespace

void vk_set_scale(int s) { want_scale = (u32)std::clamp(s, 1, 4); }
int vk_scale() { return active ? (int)(want_scale.load() ? want_scale.load() : S) : 0; }
bool vk_scale_is_auto() { return active && !scale_fixed; }
float vk_busy() { return active ? busy_frac.load() : 0; }
void vk_set_texture_filter(int mode) { want_filter = std::clamp(mode, 0, (int)kVkTfCount - 1); }
int vk_texture_filter_count() { return kVkTfCount; }
const char *vk_texture_filter_name(int mode) {
    static const char *const names[kVkTfCount] = {"Off", "Sharp", "Smooth"};
    return mode >= 0 && mode < kVkTfCount ? names[mode] : "?";
}
bool vk_gpu_made(u32 paddr, u32 size) { return active && tf_mode && size && gpu_made(paddr, size); }
void vk_note_gpu_output(u32 paddr, u32 size) {
    if (active && tf_mode && size) note_gpu_out(paddr, size);
}

namespace {
// the kernel of a vertex program blob: specialized for its uniforms when the program allows, else generic
VkPipeline build_vs_pipeline(const VsBlob &v) {
    ShaderProgram *p = shader_decode(v.code.data(), v.code_size, v.swz.data());
    std::string src = v.spec ? shader_glsl_spec(p, v.entry, v.sp.b, v.sp.iu) : std::string();
    if (src.empty()) src = shader_glsl(p, v.entry);
    shader_free(p);
    if (src.empty()) return VK_NULL_HANDLE;  // (not translatable: the CPU shades it)
    if (dump_shaders) LOG("gpu vulkan: vertex kernel:\n%s\n", src.c_str());
    VkShaderModule m = make_module(compile_glsl(std::string(kVkVsCommonGLSL) + src, shaderc_compute_shader, "vertex kernel"));
    if (!m) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.layout = v_layout;
    VkPipeline ps = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, pcache, 1, &ci, nullptr, &ps) != VK_SUCCESS) ps = VK_NULL_HANDLE;
    vkDestroyShaderModule(dev, m, nullptr);
    return ps;
}
}  // namespace

const void *vk_vs_kernel(const ShaderProgram *p, u32 entry, const ShaderSetup &s) {
    if (!active) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(vk_mtx);
    // the uniforms the program's flow control reads (masks cached per program)
    static std::unordered_map<u64, std::pair<u32, u32>> masks;
    const u64 ph = shader_program_hash(p);
    auto mi = masks.find(ph);
    if (mi == masks.end()) {
        u32 lm;
        const u32 bm = shader_static_uniforms(p, &lm);
        mi = masks.emplace(ph, std::make_pair(bm, lm)).first;
    }
    VsSpec sp;
    for (u32 i = 0; i < 16; i++)
        if (((mi->second.first >> i) & 1) && s.b[i]) sp.b |= 1u << i;
    for (u32 i = 0; i < 4; i++)
        if ((mi->second.second >> i) & 1) memcpy(sp.iu[i], s.i[i], 3);
    const u64 key = vk_key(ph, entry);
    auto it = vk_cache.find(spec_key(key, sp));
    if (it == vk_cache.end()) {
        const std::string blob = vs_blob(s, entry, &sp);
        VsKern *k = vs_add(blob);
        if (k && vk_log) {
            const u32 n = (u32)blob.size();
            fwrite(&n, 4, 1, vk_log);
            fwrite(blob.data(), 1, n, vk_log);
            fflush(vk_log);
        }
        LOG("gpu vulkan: new vertex program %016llx entry %u%s\n", (unsigned long long)ph, entry,
            sync_shaders ? "" : ", compiling in the background");
        if (k && k->done && k->ps) return k;
    } else if (it->second->done && it->second->ps) {
        return it->second.get();
    }
    auto g = vk_cache.find(key);
    if (g != vk_cache.end() && g->second->done && g->second->ps) return g->second.get();
    vkstats.cpu_draws++;
    return nullptr;
}

void vk_flush(const RTConfig &rt, const MBatch &mb) {
    if (!active || mb.empty() || !rt.color_addr || !pptr(rt.color_addr)) return;
    if (rt.color_fmt & 0x80) {
        if (!warned_shadow) LOG("gpu vulkan: shadow-map rendering is not supported, draws skipped\n");
        warned_shadow = true;
        return;
    }
    const double t0 = g_stats ? now_ms() : 0;
    flushes++;
    if (const u32 ws = want_scale.load(); ws && ws != S) apply_scale(ws);
    if (const int wf = want_filter.load(); wf != tf_mode) {
        LOG("gpu vulkan: texture filter %s -> %s\n", vk_texture_filter_name(tf_mode), vk_texture_filter_name(wf));
        tf_mode = wf;
        tf_drop_all();
        gpu_outs.clear();
    }
    flush_no++;
    bool uses_depth = false;
    for (const MSeg &seg : mb.segs) uses_depth |= seg.ds->depth_test || seg.ds->depth_write || seg.ds->stencil;
    const bool has_dep = uses_depth && rt.depth_addr && pptr(rt.depth_addr);
    const GRT *tall_c = find_view(rt.color_addr, rt.w, rt.color_fmt, false, true);
    const GRT *tall_d = has_dep ? find_view(rt.depth_addr, rt.w, rt.depth_fmt, true, true) : nullptr;
    const u32 hc = std::max(rt.h, tall_c ? tall_c->h : 0u), hd = has_dep ? std::max(rt.h, tall_d ? tall_d->h : 0u) : hc;
    const u32 H = hc == hd ? hc : rt.h;
    GRT &col = target_for(rt.color_addr, rt.w, H, rt.color_fmt, false);
    GRT *dep = has_dep ? &target_for(rt.depth_addr, rt.w, H, rt.depth_fmt, true) : nullptr;
    // (a game that draws a frame into the target it just transferred to the screen does not wait for that transfer)
    drain_touching(col.addr, rt_bytes(col), &col);
    if (dep) drain_touching(dep->addr, rt_bytes(*dep), dep);
    for (auto &o : rts) {
        if (o.get() == &col || o.get() == dep) continue;
        const bool ov_c = overlaps(o->addr, rt_bytes(*o), col.addr, rt_bytes(col));
        const bool ov_d = dep && overlaps(o->addr, rt_bytes(*o), dep->addr, rt_bytes(*dep));
        if (!ov_c && !ov_d) continue;
        if (o->gpu_newer) {
            if (o->depth) write_back_depth(*o);
            else write_back_color(*o);
        }
        o->synced = false;
    }
    std::unordered_map<u64, const TexEntry *> surf_decoded;
    auto surf_key = [](const TexUnitState &t) { return (u64)t.surf_addr << 32 ^ (u64)t.w << 20 ^ (u64)t.h << 4 ^ t.surf_fmt; };
    for (const MSeg &seg : mb.segs)
        for (const TexUnitState &t : seg.ds->tex) {
            if (!t.surf || surf_decoded.count(surf_key(t))) continue;
            GRT *r = find_color(t.surf_addr, t.w, t.surf_fmt);
            if (r && r->h >= t.h) continue;
            vk_guest_read(t.surf_addr, t.w * t.h * pixel_bytes(cbuf_to_pixfmt(t.surf_fmt)));
            surf_decoded[surf_key(t)] = texcache_get(t.surf_addr, t.w, t.h, t.surf_fmt, 1);
        }
    LoadAction cla = sync_rt(col);
    if (col.clear_next) {
        col.clear_next = false;
        cla.clear = true;
        cla.color = 0;
    }
    LoadAction dla;
    if (dep) dla = sync_rt(*dep);

    // vertices (shaded by the CPU) and indices
    auto upload = [&](const void *p, size_t n) {
        const Slice sl = arena(n);
        if (n) memcpy(sl.p, p, n);
        return sl;
    };
    const Slice cbuf = upload(mb.cverts.data(), mb.cverts.size() * sizeof(OutVertex));
    const Slice ibuf = upload(mb.idx.data(), mb.idx.size() * 4);
    const size_t cbytes = std::max<size_t>(16, mb.cverts.size() * sizeof(OutVertex)), ibytes = std::max<size_t>(16, mb.idx.size() * 4);
    const size_t gbytes = (size_t)mb.gpu_verts * sizeof(OutVertex);
    if (!mb.jobs.empty() && gbytes) {
        // one dispatch per kernel: its draws' vertices one after the other, with their parameter blocks in a buffer
        // and each thread's draw in tj (draws are often only a few vertices). The draws write disjoint ranges of gbuf.
        ensure_buffer(gbuf_dev, gbytes, Mem::Device);
        const Slice dbuf = upload(mb.vdata.data(), mb.vdata.size());
        // (counted first, then each group's vertex ids, parameter blocks and draw indices written straight into the
        // arena: no vectors to grow and copy)
        struct Group {
            VkPipeline ps;
            u32 njobs = 0, nverts = 0, jw = 0, vw = 0;
            Slice vb{}, pb{}, tb{};
        };
        std::vector<Group> groups;
        thread_local std::vector<u32> job_group;
        job_group.resize(mb.jobs.size());
        for (size_t k = 0; k < mb.jobs.size(); k++) {
            const GpuVsJob &j = mb.jobs[k];
            const VkPipeline ps = ((const VsKern *)j.kernel)->ps;
            u32 gi = 0;
            while (gi < groups.size() && groups[gi].ps != ps) gi++;
            if (gi == groups.size()) groups.push_back({ps});
            groups[gi].njobs++;
            groups[gi].nverts += j.p.nverts;
            job_group[k] = gi;
        }
        for (Group &g : groups) {
            g.vb = arena((size_t)g.nverts * 4);
            g.pb = arena((size_t)g.njobs * sizeof(VsParamsG));
            g.tb = arena((size_t)g.nverts * 4);
        }
        for (size_t k = 0; k < mb.jobs.size(); k++) {
            const GpuVsJob &j = mb.jobs[k];
            Group &g = groups[job_group[k]];
            VsParamsG *P = (VsParamsG *)g.pb.p + g.jw;
            memcpy((void *)P, &j.p, sizeof(VsParamsG));
            P->first = g.vw;
            memcpy(g.vb.p + (size_t)g.vw * 4, &mb.vids[j.vids_off], (size_t)j.p.nverts * 4);
            u32 *t = (u32 *)g.tb.p + g.vw;
            for (u32 i = 0; i < j.p.nverts; i++) t[i] = g.jw;
            g.vw += j.p.nverts;
            g.jw++;
        }
        VkCommandBuffer vcb = rec();
        barrier(vcb);
        for (const Group &g : groups) {
            if (!g.nverts) continue;
            vkCmdBindPipeline(vcb, VK_PIPELINE_BIND_POINT_COMPUTE, g.ps);
            VkDescriptorBufferInfo bi[5] = {{dbuf.buf->b, dbuf.off, std::max<size_t>(16, mb.vdata.size())},
                                            {g.vb.buf->b, g.vb.off, (size_t)g.nverts * 4},
                                            {g.pb.buf->b, g.pb.off, (size_t)g.njobs * sizeof(VsParamsG)},
                                            {gbuf_dev->b, 0, gbytes},
                                            {g.tb.buf->b, g.tb.off, (size_t)g.nverts * 4}};
            VkWriteDescriptorSet w[5];
            for (u32 i = 0; i < 5; i++) {
                w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo = &bi[i];
            }
            vkPushDescriptors(vcb, VK_PIPELINE_BIND_POINT_COMPUTE, v_layout, 0, 5, w);
            const u32 nt[4] = {g.nverts, 0, 0, 0};
            vkCmdPushConstants(vcb, v_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, nt);
            vkCmdDispatch(vcb, (g.nverts + 63) / 64, 1, 1);
        }
    }

    std::unordered_map<u64, const Img *> surf_tex;
    for (const MSeg &seg : mb.segs)
        for (const TexUnitState &t : seg.ds->tex)
            if (t.surf && !surf_decoded.count(surf_key(t)) && !surf_tex.count(surf_key(t)))
                surf_tex[surf_key(t)] = surface_snapshot(t.surf_addr, t.w, t.h, t.surf_fmt);

    // per segment: pipeline, uniforms, LUTs and textures, all made (and uploaded) before the pass begins
    struct Prep {
        const MSeg *seg;
        Pipe *pipe;
        bool fetch;  // the shader reads the destination
        VkRect2D sc;
        size_t uo, lo = 0, fo = 0;
        bool lut = false, fog = false;
        VkImageView tv[3], fv[5];
        VkSampler sm[3];
    };
    std::vector<Prep> preps;
    preps.reserve(mb.segs.size());
    size_t uoff = 0;
    auto ualloc = [&](size_t n) {
        const size_t o = (uoff + 255) & ~(size_t)255;
        uoff = o + n;
        return o;
    };
    const size_t need = mb.segs.size() * (256 + sizeof(Uni) + 256 + 24 * 256 * 8 + 256 + 128 * 8);
    const Slice us = arena(need);
    u8 *ub = us.p;
    std::unordered_map<const void *, size_t> lut_map;
    for (const MSeg &seg : mb.segs) {
        const DrawState &ds = *seg.ds;
        if (ds.skip || ds.shadow_mode || ds.gas) continue;
        Pipe *pipe = get_pipeline(ds);
        if (!pipe || !pipe->ps) {
            col.skipped++;
            continue;
        }
        const bool fetch = ds.need_shade && !fs_hw_blend(ds, {ShaderLang::GLSL, has_logic_op, MS});
        if (fetch && !has_local_read) {
            if (!warned_fetch) LOG("gpu vulkan: blending in the shader is not supported on this device, draws skipped\n");
            warned_fetch = true;
            continue;
        }
        if (ds.need_tex[3] && !warned_proctex) {
            LOG("gpu vulkan: procedural textures are not supported (texture 3 reads as 0)\n");
            warned_proctex = true;
        }
        Prep pr{};
        pr.seg = &seg;
        pr.pipe = pipe;
        pr.fetch = fetch;
        st_fetch += fetch;
        if (fetch && g_stats) LOG("gpu vulkan: draw blending in the shader (logic op %u, blend %u)\n", ds.logic_op, ds.blend);
        pr.sc = {{0, 0}, {rt.w * S, rt.h * S}};
        if (ds.scissor_mode == 3) {
            const s32 x1 = std::clamp<s32>(ds.sc_x1, 0, rt.w), x2 = std::clamp<s32>(ds.sc_x2, 0, rt.w);
            const s32 y1 = std::clamp<s32>(ds.sc_y1, 0, rt.h), y2 = std::clamp<s32>(ds.sc_y2, 0, rt.h);
            if (x2 <= x1 || y2 <= y1) continue;
            pr.sc = {{(s32)(x1 * S), (s32)((rt.h - y2) * S)}, {(u32)(x2 - x1) * S, (u32)(y2 - y1) * S}};
        }
        pr.uo = ualloc(sizeof(Uni));
        Uni &u = *(Uni *)(ub + pr.uo);
        memset(&u, 0, sizeof u);
        u.rt[0] = (float)rt.w;
        u.rt[1] = (float)rt.h;
        u.rt[2] = (float)S;
        u.rt[3] = (float)((1u << ds.depth_bits) - 1);
        u.depth[0] = ds.depth_scale;
        u.depth[1] = ds.depth_offset;
        u.depth[2] = ds.depth_bits == 16 ? 1.0f / 65536.0f : 1.0f / 16777216.0f;
        u.scissor[0] = ds.sc_x1;
        u.scissor[1] = ds.sc_y1;
        u.scissor[2] = ds.sc_x2;
        u.scissor[3] = ds.sc_y2;
        for (int k = 0; k < 6; k++) u.konst[k] = ds.tev[k].konst;
        u.tevc[0] = ds.tev_init_out;
        u.tevc[1] = ds.tev_init_buf;
        u.tevc[2] = ds.tev_init_next;
        u.misc[0] = ds.blend_color;
        u.misc[1] = ds.fog_color;
        u.misc[2] = ds.alpha_ref;
        u.misc[3] = ds.tex[0].shadow_bias;
        for (int k = 0; k < 3; k++) {
            u.border[k] = ds.tex[k].border;
            u.bias[k] = ds.tex[k].bias;
        }
        if (ds.lighting) {
            const LightingState &l = ds.light;
            const LutSel *sels[7] = {&l.d0, &l.d1, &l.sp, &l.fr, &l.rb, &l.rg, &l.rr};
            for (int k = 0; k < 7; k++) u.lscale[k] = sels[k]->scale;
            for (int k = 0; k < 3; k++) u.ambient[k] = l.global_ambient[k];
            for (u32 k = 0; k < l.num_lights; k++) {
                const LightState &s = l.lights[k];
                LightU &d = u.lights[k];
                for (int c = 0; c < 3; c++) {
                    d.spec0[c] = s.spec0[c];
                    d.spec1[c] = s.spec1[c];
                    d.diffuse[c] = s.diffuse[c];
                    d.ambient[c] = s.ambient[c];
                    d.pos[c] = s.pos[c];
                    d.spot[c] = s.spot[c];
                }
                d.att[0] = s.da_scale;
                d.att[1] = s.da_bias;
            }
            const void *lk = l.luts.get();
            auto it = lut_map.find(lk);
            if (it == lut_map.end()) {
                pr.lo = ualloc(24 * 256 * 8);
                float *dst = (float *)(ub + pr.lo);
                for (int t = 0; t < 24; t++)
                    for (int e = 0; e < 256; e++) {
                        dst[(t * 256 + e) * 2] = l.luts->val[t][e];
                        dst[(t * 256 + e) * 2 + 1] = l.luts->diff[t][e];
                    }
                lut_map[lk] = pr.lo;
            } else {
                pr.lo = it->second;
            }
            pr.lut = true;
        }
        if (ds.fog) {
            const void *fk = ds.fog_lut.get();
            auto it = lut_map.find(fk);
            if (it == lut_map.end()) {
                pr.fo = ualloc(128 * 8);
                float *dst = (float *)(ub + pr.fo);
                for (int e = 0; e < 128; e++) {
                    dst[e * 2] = ds.fog_lut->val[e];
                    dst[e * 2 + 1] = ds.fog_lut->diff[e];
                }
                lut_map[fk] = pr.fo;
            } else {
                pr.fo = it->second;
            }
            pr.fog = true;
        }
        for (int k = 0; k < 3; k++) {
            pr.tv[k] = black->view;
            pr.sm[k] = dummy_sampler;
        }
        for (int f = 0; f < 5; f++) pr.fv[f] = black->view;
        if (ds.need_shade) {
            for (int k = 0; k < 3; k++) {
                const TexUnitState &t = ds.tex[k];
                if (!ds.need_tex[k] || t.null_tex) continue;
                // upscaling (texture filter): 2D and projected textures only (not cube or shadow maps)
                const bool up = tf_mode && (k != 0 || t.type == 0 || t.type == 3);
                u32 ul = 0;
                const Img *tex = nullptr;
                if (t.surf) {
                    auto si = surf_tex.find(surf_key(t));
                    if (si != surf_tex.end()) tex = si->second;
                    auto di = surf_decoded.find(surf_key(t));
                    if (di != surf_decoded.end()) tex = get_texture(di->second);
                } else {
                    tex = get_texture(t.face[0], up ? &t : nullptr, &ul);
                }
                if (!tex) tex = black.get();
                pr.tv[k] = tex->view;
                pr.sm[k] = get_sampler(t, tex->levels, ul);
                // the LOD bias of a texture without mipmaps does nothing on the PICA (games leave any value);
                // upscaled, it has levels: no bias then, or a positive one would pick the original level
                if (ul && tex->levels == ul + 1) u.bias[k] = 0;
                if (k == 0 && (t.type == 1 || t.type == 4))
                    for (int f = 1; f < 6; f++) {
                        const Img *ft = get_texture(t.face[f]);
                        pr.fv[f - 1] = ft ? ft->view : tex->view;
                    }
            }
        }
        preps.push_back(pr);
    }
    arena_trim(us, uoff);

    // the pass: with MSAA it draws into the multisampled targets and resolves them into the single-sample ones; a
    // multisampled target older than its resolved copy (guest memory uploaded since) is restored from it first
    const bool msaa = MS > 1;
    const bool restore_c = msaa && !cla.clear && !col.ms_valid;
    const bool restore_d = msaa && dep && !dla.clear && !dep->ms_valid;
    VkCommandBuffer cb = rec();
    barrier(cb);
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = msaa ? col.ms->view : col.tex->view;
    ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.loadOp = cla.clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : restore_c ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color.float32[0] = (cla.color & 255) / 255.0f;
    ca.clearValue.color.float32[1] = ((cla.color >> 8) & 255) / 255.0f;
    ca.clearValue.color.float32[2] = ((cla.color >> 16) & 255) / 255.0f;
    ca.clearValue.color.float32[3] = (cla.color >> 24) / 255.0f;
    if (msaa) {
        ca.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        ca.resolveImageView = col.tex->view;
        ca.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (dep) {
        da.imageView = msaa ? dep->ms->view : dep->tex->view;
        da.loadOp = dla.clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : restore_d ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        da.clearValue.depthStencil = {dla.depth, dla.stencil};
        if (msaa) {
            da.resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
            da.resolveImageView = dep->tex->view;
            da.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
    } else {
        da.imageView = get_scratch_depth(rt.w, H).view;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        da.clearValue.depthStencil = {1.0f, 0};
    }
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {rt.w * S, H * S}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    ri.pStencilAttachment = &da;
    const VkRect2D full = {{0, 0}, {rt.w * S, H * S}};
    auto begin = [&] {
        vkCmdBeginRendering(cb, &ri);
        vkCmdSetScissor(cb, 0, 1, &full);
        const VkViewport v = {0, 0, (float)(rt.w * S), (float)(H * S), 0, 1};
        vkCmdSetViewport(cb, 0, 1, &v);
        vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
        vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    };
    // The restores read the single-sample copies, which the pass resolves into: a pass of their own before (the
    // multisampled targets stored, not resolved), then the draws' pass loads them.
    if (restore_c || restore_d) {
        VkRenderingAttachmentInfo rca = ca, rda = da;
        rca.resolveMode = rda.resolveMode = VK_RESOLVE_MODE_NONE;
        rca.resolveImageView = rda.resolveImageView = VK_NULL_HANDLE;
        if (!restore_c) rca.loadOp = cla.clear ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
        if (dep && !restore_d) rda.loadOp = dla.clear ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD;
        if (!dep) rda.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        VkRenderingInfo rri = ri;
        rri.pColorAttachments = &rca;
        rri.pDepthAttachment = rri.pStencilAttachment = &rda;
        vkCmdBeginRendering(cb, &rri);
        vkCmdSetScissor(cb, 0, 1, &full);
        const VkViewport v = {0, 0, (float)(rt.w * S), (float)(H * S), 0, 1};
        vkCmdSetViewport(cb, 0, 1, &v);
        vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
        vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        auto push_images = [&](const Img &a, VkImageView b_view) {
            VkDescriptorImageInfo ii[2] = {{VK_NULL_HANDLE, a.view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, b_view, VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            for (int i = 0; i < 2; i++) {
                w[i].dstBinding = 3 + i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                w[i].pImageInfo = &ii[i];
            }
            vkPushDescriptors(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, b_view ? 2 : 1, w);
        };
        if (restore_c) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_rcolor);
            vkCmdSetDepthTestEnable(cb, VK_FALSE);
            vkCmdSetDepthWriteEnable(cb, VK_FALSE);
            vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
            vkCmdSetStencilTestEnable(cb, VK_FALSE);
            push_images(*col.tex, VK_NULL_HANDLE);
            vkCmdDraw(cb, 3, 1, 0, 0);
        }
        if (restore_d) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_rdepth ? p_rdepth : p_rdepth_only);
            vkCmdSetDepthTestEnable(cb, VK_TRUE);
            vkCmdSetDepthWriteEnable(cb, VK_TRUE);
            vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
            vkCmdSetStencilTestEnable(cb, VK_TRUE);
            vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                              VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS);
            vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff);
            vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff);
            vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
            VkDescriptorImageInfo ii[2] = {{VK_NULL_HANDLE, dep->tex->vdepth, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, dep->tex->vstencil, VK_IMAGE_LAYOUT_GENERAL}};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            for (int i = 0; i < 2; i++) {
                w[i].dstBinding = 3 + i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                w[i].pImageInfo = &ii[i];
            }
            vkPushDescriptors(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, 2, w);
            vkCmdDraw(cb, 3, 1, 0, 0);  // (without stencil export: depth, stencil 0)
            if (!p_rdepth && dep->fmt == 3) {  // the stencil, bit by bit
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_rstencil_bit);
                vkCmdSetDepthWriteEnable(cb, VK_FALSE);
                vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff);
                for (u32 bit = 0; bit < 8; bit++) {
                    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 1u << bit);
                    vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(VsDraw), 4, &bit);
                    vkCmdDraw(cb, 3, 1, 0, 0);
                }
            }
        }
        vkCmdEndRendering(cb);
        barrier(cb);
        if (restore_c) ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        if (restore_d) da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    begin();
    col.ms_valid = true;
    if (dep) dep->ms_valid = true;
    // the draws: PICA y up, so the viewport is flipped (Metal's NDC y up maps to the image's first rows)
    {
        const VkViewport v = {0, (float)(rt.h * S), (float)(rt.w * S), -(float)(rt.h * S), 0, 1};
        vkCmdSetViewport(cb, 0, 1, &v);
    }
    bool wrote_color = false, wrote_depth = false;
    VkPipeline bound = VK_NULL_HANDLE;
    for (const Prep &pr : preps) {
        const DrawState &ds = *pr.seg->ds;
        if (pr.pipe->ps != bound) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr.pipe->ps);
            bound = pr.pipe->ps;
        }
        vkCmdSetDepthTestEnable(cb, VK_TRUE);
        vkCmdSetDepthCompareOp(cb, ds.depth_test ? cmp_op(ds.depth_func) : VK_COMPARE_OP_ALWAYS);
        vkCmdSetDepthWriteEnable(cb, ds.depth_write ? VK_TRUE : VK_FALSE);
        vkCmdSetStencilTestEnable(cb, ds.stencil ? VK_TRUE : VK_FALSE);
        if (ds.stencil) {
            vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, (VkStencilOp)(ds.st_fail & 7),
                              (VkStencilOp)(ds.st_zpass & 7), (VkStencilOp)(ds.st_zfail & 7), cmp_op(ds.st_func));
            vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, ds.st_mask_in);
            vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, ds.st_mask_w);
            vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, ds.st_ref);
        }
        // PICA culling (by the winding in PICA window coordinates, y up)
        vkCmdSetCullMode(cb, ds.cull == 1 ? VK_CULL_MODE_FRONT_BIT : ds.cull == 2 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE);
        const u32 bc = ds.blend_color;
        const float bcf[4] = {(bc & 255) / 255.0f, ((bc >> 8) & 255) / 255.0f, ((bc >> 16) & 255) / 255.0f, (bc >> 24) / 255.0f};
        vkCmdSetBlendConstants(cb, bcf);
        vkCmdSetScissor(cb, 0, 1, &pr.sc);
        const VsDraw vd = {{(float)rt.w, (float)rt.h, ds.depth_scale, ds.depth_offset},
                           {ds.vp_hw, ds.vp_hh, ds.vp_x, ds.vp_y},
                           {ds.clip[0], ds.clip[1], ds.clip[2], ds.clip[3]},
                           {(u32)ds.clip_enable, 0, 0, 0}};
        vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof vd, &vd);
        // the draw's resources
        VkWriteDescriptorSet w[18];
        VkDescriptorBufferInfo bi[6];
        VkDescriptorImageInfo ii[12];
        u32 nw = 0, nb = 0, nim = 0;
        auto put_buf = [&](u32 binding, VkDescriptorType t, VkBuffer b, VkDeviceSize off, VkDeviceSize range) {
            bi[nb] = {b, off, range};
            w[nw] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[nw].dstBinding = binding;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType = t;
            w[nw].pBufferInfo = &bi[nb++];
            nw++;
        };
        auto put_img = [&](u32 binding, VkDescriptorType t, VkImageView v, VkSampler s) {
            ii[nim] = {s, v, VK_IMAGE_LAYOUT_GENERAL};
            w[nw] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[nw].dstBinding = binding;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType = t;
            w[nw].pImageInfo = &ii[nim++];
            nw++;
        };
        put_buf(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, us.buf->b, us.off + pr.uo, sizeof(Uni));
        if (pr.lut) put_buf(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, us.buf->b, us.off + pr.lo, 24 * 256 * 8);
        else put_buf(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dummy_buf->b, 0, VK_WHOLE_SIZE);
        if (pr.fog) put_buf(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, us.buf->b, us.off + pr.fo, 128 * 8);
        else put_buf(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dummy_buf->b, 0, VK_WHOLE_SIZE);
        for (int k = 0; k < 3; k++) put_img(3 + k, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, pr.tv[k], VK_NULL_HANDLE);
        for (int k = 0; k < 3; k++) put_img(6 + k, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, pr.sm[k]);
        for (int f = 0; f < 5; f++) put_img(9 + f, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, pr.fv[f], VK_NULL_HANDLE);
        if (pr.fetch) put_img(14, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, msaa ? col.ms->view : col.tex->view, VK_NULL_HANDLE);
        if (gbytes) put_buf(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, gbuf_dev->b, 0, gbytes);  // (kernel-shaded vertices)
        else put_buf(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dummy_buf->b, 0, VK_WHOLE_SIZE);
        put_buf(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, cbuf.buf->b, cbuf.off, cbytes);
        put_buf(17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ibuf.buf->b, ibuf.off, ibytes);
        vkPushDescriptors(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, nw, w);
        if (pr.fetch) {  // the destination as earlier draws left it (draws overlapping themselves: not exact)
            VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            mb.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            mb.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cb, &di);
        }
        vkCmdDraw(cb, pr.seg->count, 1, pr.seg->first, 0);
        if (ds.color_write) wrote_color = true;
        if (ds.depth_write || (ds.stencil && ds.st_mask_w)) wrote_depth = true;
    }
    vkCmdEndRendering(cb);
    if (wrote_color) col.gpu_newer = true;
    col.version++;
    if (dep && wrote_depth) dep->gpu_newer = true;
    if (eager_writeback) {
        if (col.gpu_newer) write_back_color(col);
        if (dep && dep->gpu_newer) write_back_depth(*dep);
    }
    if (g_stats)
        LOG("gpu vulkan: flush %zu tris: prepare+encode %.2f ms%s%s%s\n", mb.idx.size() / 3, now_ms() - t0,
            eager_writeback ? " (eager write-back)" : "", restore_c ? ", colour restored" : "", restore_d ? ", depth restored" : "");
}

void vk_poll() {
    if (!active) return;
    reap();
    static double last = now_ms();
    const double t = now_ms();
    if (t - last >= 2000) {
        const double b = gpu_busy;
        gpu_busy = 0;
        busy_frac = (float)(b * 1000 / (t - last));
        if (log_timing) {
            LOG("[gpu] GPU busy %.1f ms/s (scale %u, MSAA %ux), %llu command buffers, %llu passes, %u draws blending in "
                "the shader\n",
                b * 1000 / ((t - last) / 1000), S, MS, (unsigned long long)cbs_made.exchange(0),
                (unsigned long long)flushes.exchange(0), st_fetch);
            st_fetch = 0;
            LOG("[gpu] textures: %u decoded in %.2f ms, %u uploaded in %.2f ms; filter %s: %u upscaled + %u updated "
                "(%u tiles) in %.2f ms, %zu MiB\n",
                g_tex_decodes, g_tex_decode_ms, st_uploads, st_upload_ms, vk_texture_filter_name(tf_mode), st_upscaled,
                st_partial, st_tiles, st_upscale_ms, tf_bytes >> 20);
            g_tex_decodes = st_uploads = st_upscaled = st_partial = st_tiles = 0;
            g_tex_decode_ms = st_upload_ms = st_upscale_ms = 0;
        }
        static double saved = t;
        if (t - saved > 30000) {
            saved = t;
            save_pipeline_cache();
        }
        last = t;
    }
    const u64 done = completed_value();
    while (!pending.empty() && pending.front()->tl <= done) drain(0);
    {  // images vk_display_transfer made that the GPU finished: their buffers can be recycled when replaced
        std::lock_guard<std::mutex> lk(disp_mtx);
        for (auto &[addr, d] : displays)
            if (d.direct && d.tl && d.tl <= done) d.tl = 0;
    }
}

void vk_end_list() {
    if (!active) return;
    submit();
}

void vk_guest_read(u32 paddr, u32 size) {
    if (!active) return;
    drain_touching(paddr, size);
    for (auto &r : rts)
        if (r->gpu_newer && overlaps(r->addr, rt_bytes(*r), paddr, size)) {
            if (r->depth) write_back_depth(*r);
            else write_back_color(*r);
        }
}

void vk_guest_prewrite(u32 paddr, u32 size) {
    if (!active) return;
    for (auto &op : pending)
        if (op->rt_writeback && op->rt->addr >= paddr && op->rt->addr + rt_bytes(*op->rt) <= paddr + size)
            op->rt_writeback = false;
    drain_touching(paddr, size);
    for (auto &r : rts) {
        if (!r->gpu_newer || !overlaps(r->addr, rt_bytes(*r), paddr, size)) continue;
        if (r->addr >= paddr && r->addr + rt_bytes(*r) <= paddr + size) {
            r->gpu_newer = false;
        } else {
            if (r->depth) write_back_depth(*r);
            else write_back_color(*r);
        }
    }
}

void vk_writeback_all() {
    if (!active) return;
    drain_all();
    vk_guest_read(0, 0xffffffffu);
}

void vk_guest_written(u32 paddr, u32 size, bool definite) {
    if (!active) return;
    for (auto &r : rts)
        if (overlaps(r->addr, rt_bytes(*r), paddr, size)) {
            if (definite || !r->synced || hash_mem(pptr(r->addr), rt_bytes(*r)) != r->hash) r->synced = false;
        }
    drop_displays(paddr, size, definite);
}

namespace {
void drop_displays(u32 paddr, u32 size, bool definite) {
    std::lock_guard<std::mutex> lk(disp_mtx);
    for (auto it = displays.begin(); it != displays.end();) {
        DisplayImage &d = it->second;
        if (overlaps(d.out_addr, d.out_size, paddr, size) &&
            (definite || ((!d.tl || d.direct) && hash_mem(pptr(d.out_addr), d.out_size) != d.out_hash))) {
            if (g_stats)
                LOG("gpu vulkan: display image %08x dropped (%s write %08x+%x)\n", d.out_addr, definite ? "GPU" : "CPU",
                    paddr, size);
            if (!d.tl && recyclable(d.buf)) disp_free.push_back(std::move(d.buf));
            it = displays.erase(it);
        } else {
            ++it;
        }
    }
}

GRT *transfer_source(u32 in_paddr, u32 in_dims, u32 out_dims, u32 &row0) {
    const u32 in_w = in_dims & 0xffff, out_h0 = out_dims >> 16;
    GRT *src = nullptr;
    for (auto &r : rts) {
        if (r->depth || !r->synced || r->w != in_w || in_paddr < r->addr) continue;
        const u32 line = r->w * pixel_bytes(cbuf_to_pixfmt(r->fmt));
        const u32 off = in_paddr - r->addr;
        if (off % (line * 8) || off / line + out_h0 > r->h) continue;
        src = r.get();
        row0 = off / line;
    }
    if (src && (out_dims & 0xffff) > src->w) return nullptr;  // (narrower: the transfer keeps the first columns)
    return src;
}

// rows x TW (scaled) of a target as an upright image: TW its first columns (all of them, or fewer for a transfer
// keeping only those), from row row0 (1x)
void encode_rot(VkCommandBuffer cb, GRT &src, const Buf &dst, u32 rows, u32 TW, u32 flags, u32 row0) {
    const u32 key = key_black_bottom && rows == 320 * S ? 2 : 0;
    const u32 prm[4] = {TW, rows, (flags & 1) | key, row0 * S};
    dispatch(cb, K_ROT, {ki(4, *src.tex), kb(0, dst)}, prm, nullptr, rows, TW);
}

DisplayImage &new_display(u32 out_paddr, u32 out_size, u32 out_dims, u32 flags, u32 w, u32 h) {
    const u32 out_h0 = out_dims >> 16;
    for (auto it = displays.begin(); it != displays.end();) {
        if (overlaps(it->second.out_addr, it->second.out_size, out_paddr, out_size)) {
            if (!it->second.tl && recyclable(it->second.buf)) disp_free.push_back(std::move(it->second.buf));
            it = displays.erase(it);
        } else {
            ++it;
        }
    }
    DisplayImage &d = displays[out_paddr];
    d.out_addr = out_paddr;
    d.out_size = out_size;
    d.w = w;
    d.h = h;
    d.hs = ((flags >> 24) & 3) ? 1 : 0;
    d.vs = ((flags >> 24) & 3) == 2 ? 1 : 0;
    d.out_w = (out_dims & 0xffff) >> d.hs;
    d.out_bpp = pixel_bytes((flags >> 12) & 7);
    d.scale = S;
    d.version = ++disp_version;
    d.screen = out_h0 == 400 ? 0 : out_h0 == 320 ? 1 : -1;
    if (d.screen >= 0) last_screen_addr[d.screen] = out_paddr;
    return d;
}

void clear_now(GRT &r) {
    r.clear_next = false;
    clear_image(*r.tex, false, 0, 0, 0);
    r.ms_valid = false;
}
}  // namespace

bool vk_defer_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags, u32 out_size) {
    if (!active || eager_writeback || (flags & 2)) return false;
    u32 row0 = 0;
    GRT *src = transfer_source(in_paddr, in_dims, out_dims, row0);
    if (!src || overlaps(src->addr, rt_bytes(*src), out_paddr, out_size)) return false;
    bool writing_back = false;  // a pending transfer writes the target back already
    for (auto &op : pending)
        if (op->rt_writeback && op->rt == src) writing_back = true;
    // Drawn into since its last write-back: this transfer writes it back (after an earlier pending one: they land
    // in order; vk_flush keeps a target's own write-backs pending while it is drawn into). Otherwise its GPU copy is
    // what guest memory holds once that pending write-back lands, or holds already (its hash unchanged): a second
    // transfer from it is made on the GPU too, without waiting for the first one or converting on the CPU.
    const bool writeback = src->gpu_newer;
    if (!writeback && (src->clear_next || (!writing_back && (!src->synced || hash_mem(pptr(src->addr), rt_bytes(*src)) != src->hash))))
        return false;  // (clear_next: cleared to transparent since; guest memory keeps the picture)
    vk_guest_prewrite(out_paddr, out_size);
    auto op = std::make_unique<PendingXfer>();
    op->rt_writeback = writeback;
    const u32 TW = (out_dims & 0xffff) * S, rows = (out_dims >> 16) * S;  // (the columns transferred)
    VkCommandBuffer cb = new_cb();
    if (src->clear_next) clear_now(*src);
    op->gbytes = take_buffer(xbuf_free, rt_bytes(*src));
    op->obytes = take_buffer(xbuf_free, out_size);
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        op->disp = take_buffer(disp_free, (size_t)TW * rows * 4);
    }
    barrier(cb);
    encode_guest_color(cb, *src, *op->gbytes);
    encode_rot(cb, *src, *op->disp, rows, TW, flags, row0);
    barrier(cb);
    const u32 scaling = (flags >> 24) & 3;
    const u32 out_w = (out_dims & 0xffff) >> (scaling ? 1 : 0), out_h = (out_dims >> 16) >> (scaling == 2 ? 1 : 0);
    const u32 tp[4] = {in_dims & 0xffff, out_w, out_h, flags}, tq[4] = {in_paddr - src->addr, 0, 0, 0};
    dispatch(cb, K_TCONV, {kb(0, *op->gbytes), kb(1, *op->obytes)}, tp, tq, out_w, out_h);
    op->tl = submit();
    src->gpu_newer = false;
    src->own = false;
    if (clear_after_top) clear_overlapping(*src);
    op->rt = src;
    op->in_paddr = in_paddr;
    op->out_paddr = out_paddr;
    op->in_dims = in_dims;
    op->out_dims = out_dims;
    op->flags = flags;
    op->out_size = out_size;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        DisplayImage &d = new_display(out_paddr, out_size, out_dims, flags, rows, TW);
        d.tl = op->tl;
        d.buf = op->disp;
        d.incomplete = src->skipped > 0;
        src->skipped = 0;
    }
    pending.push_back(std::move(op));
    return true;
}

bool vk_texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width, u32 in_gap, u32 out_width, u32 out_gap) {
    if (!active || !size) return false;
    GRT *src = nullptr;
    u32 row = 0, bpp = 0;
    for (auto &r : rts) {
        if (!gpu_valid(*r) || in_paddr < r->addr || in_paddr >= r->addr + rt_bytes(*r)) continue;
        const u32 b = grt_bpp(*r), rw = r->w * 8 * b;
        if ((in_paddr - r->addr) % rw) continue;
        src = r.get();
        row = rw;
        bpp = b;
        break;
    }
    if (!src || src->clear_next) return false;
    if (in_gap && in_width + in_gap != row) return false;
    const u32 line = in_gap ? in_width : row;
    if (line % (64 * bpp) || size % line) return false;
    const u32 cols = line / (8 * bpp), rows = size / line * 8, sy0 = (in_paddr - src->addr) / row * 8;
    if (sy0 + rows > src->h) return false;
    if (out_gap && (out_width != line || (out_width + out_gap) % (64 * bpp))) return false;
    const u32 wd = out_gap ? (out_width + out_gap) / (8 * bpp) : cols;
    const u32 bytes = (rows / 8 - 1) * wd * 8 * bpp + line;
    GRT *d = dest_surface(out_paddr, wd, rows, src->fmt, src, bytes);
    if (!d) return false;
    copy_image(*src->tex, *d->tex, 0, (s32)(sy0 * S), cols * S, rows * S, VK_IMAGE_ASPECT_COLOR_BIT);
    gpu_wrote(*d, out_paddr, bytes);
    return true;
}

bool vk_gpu_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags) {
    if (!active || (flags & 2) || !(flags & 0x20) || (flags & 4)) return false;
    const u32 in_w = in_dims & 0xffff, in_h = in_dims >> 16, out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    const u32 in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7, scaling = (flags >> 24) & 3;
    if (in_w != out_w0 || in_h != out_h0 || in_fmt > 4 || out_fmt > 4 || scaling > 2 || in_w % 8 || in_h % 8) return false;
    const u32 hs = scaling ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    const u32 out_w = out_w0 >> hs, out_h = out_h0 >> vs;
    if (!out_w || !out_h || out_w % 8 || out_h % 8) return false;
    const u32 sfmt = pix_to_cbuf(in_fmt), rw = in_w * 8 * pixel_bytes(in_fmt);
    GRT *src = nullptr;
    u32 sy0 = 0;
    for (auto &r : rts) {
        if (r->w != in_w || r->fmt != sfmt || !gpu_valid(*r) || in_paddr < r->addr || in_paddr >= r->addr + rt_bytes(*r) ||
            (in_paddr - r->addr) % rw)
            continue;
        src = r.get();
        sy0 = (in_paddr - r->addr) / rw * 8;
        break;
    }
    if (!src || src->clear_next) return false;
    if (sy0 + in_h > src->h) {
        if (sy0) return false;
        src = grow(*src, in_h);
        if (!src) return false;
    }
    const u32 bytes = out_w * out_h * pixel_bytes(out_fmt);
    GRT *d = dest_surface(out_paddr, out_w, out_h, pix_to_cbuf(out_fmt), src, bytes);
    if (!d) return false;
    VkCommandBuffer cb = rec();
    barrier(cb);
    const u32 p[4] = {out_w * S, out_h * S, sy0 * S, 0};
    const u32 q[4] = {hs, vs, flags & 1, in_fmt == 1 || in_fmt == 2 ? 1u : 0u};
    dispatch(cb, K_XFER, {ki(4, *src->tex), ki(5, *d->tex)}, p, q, out_w * S, out_h * S);
    gpu_wrote(*d, out_paddr, bytes);
    return true;
}

bool vk_surface_for(u32 addr, u32 w, u32 h, u32 fmt) {
    if (!active || fmt > 4) return false;
    GRT *r = find_color(addr, w, fmt);
    if (!r || !gpu_valid(*r) || r->clear_next) return false;
    if (r->h < h) r = grow(*r, h);
    return r != nullptr;
}

void vk_display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags) {
    if (!active) return;
    if (flags & 2) return;
    u32 row0 = 0;
    GRT *src = transfer_source(in_paddr, in_dims, out_dims, row0);
    if (!src) {
        if (g_stats)
            LOG("gpu vulkan: transfer %08x -> %08x (%08x -> %08x, flags %08x): no matching render target, no display image\n",
                in_paddr, out_paddr, in_dims, out_dims, flags);
        return;
    }
    if (hash_mem(pptr(src->addr), rt_bytes(*src)) != src->hash) {
        if (g_stats) LOG("gpu vulkan: transfer %08x -> %08x: target changed in guest memory, no display image\n", in_paddr, out_paddr);
        return;
    }
    const u32 out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    const u32 TW = out_w0 * S, rows = out_h0 * S;  // (the columns transferred)
    BufP buf;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        buf = take_buffer(disp_free, (size_t)TW * rows * 4);
    }
    VkCommandBuffer cb = new_cb();
    if (src->clear_next) clear_now(*src);
    barrier(cb);
    encode_rot(cb, *src, *buf, rows, TW, flags, row0);
    const u64 tl = submit();  // (not waited for: the UI waits if it reads the image before the GPU made it)
    const u32 hs = ((flags >> 24) & 3) ? 1 : 0, vs = ((flags >> 24) & 3) == 2 ? 1 : 0;
    const u32 out_size = (out_w0 >> hs) * (out_h0 >> vs) * pixel_bytes((flags >> 12) & 7);
    std::lock_guard<std::mutex> lk(disp_mtx);
    DisplayImage &d = new_display(out_paddr, out_size, out_dims, flags, rows, TW);
    d.out_hash = hash_mem(pptr(out_paddr), out_size);
    d.tl = tl;
    d.direct = true;
    d.buf = buf;
    d.incomplete = src->skipped > 0;
    src->skipped = 0;
    src->own = false;
    if (clear_after_top) clear_overlapping(*src);
}

void vk_set_clear_after_top_transfer(bool on) { clear_after_top = on; }
void vk_set_key_black_bottom(bool on) { key_black_bottom = on; }

namespace {
struct DisplayView {
    const DisplayImage *d = nullptr;
    u32 x0 = 0, y0 = 0, w = 0, h = 0;
    u64 key = 0;
};
bool find_disp_view(u32 fb, u32 stride, u32 lines, DisplayView &v) {
    for (const auto &[addr, d] : displays) {
        if (fb < d.out_addr || fb >= d.out_addr + d.out_size) continue;
        const u32 row_bytes = d.out_w * d.out_bpp;
        if (!row_bytes || (stride && stride != row_bytes)) return false;
        const u32 off = fb - d.out_addr, r0 = off / row_bytes, c0 = off % row_bytes / d.out_bpp;
        const u32 fx = d.scale << d.vs, fy = d.scale << d.hs;
        if (c0 + 240 > d.out_w || (r0 + lines) * fx > d.w || (c0 + 240) * fy > d.h) return false;
        v.d = &d;
        v.x0 = r0 * fx;
        v.w = lines * fx;
        v.y0 = d.h - (c0 + 240) * fy;
        v.h = 240 * fy;
        v.key = d.version << 20 ^ off;
        return true;
    }
    return false;
}
// the image for the framebuffer finished on the GPU, waited for without disp_mtx held
bool wait_for_view(u32 fb, u32 stride, u32 lines) {
    u64 tl;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        DisplayView v;
        if (!find_disp_view(fb, stride, lines, v)) return false;
        tl = v.d->tl;
    }
    wait_value(tl);
    return true;
}

// ---------------------------------------------------------------- presenting on this device (gpu_iface.h)
// SDL's renderer draws present images (made here, kept in SHADER_READ_ONLY_OPTIMAL); a new screen image is copied
// into one on the GPU, on SDL's queue from the UI thread (SDL's thread), after the GX work that made it. The copy is
// waited for, so that the display buffer can be reused once use_display_image returns.
struct PresentImage {
    VkImage img = VK_NULL_HANDLE;
    VmaAllocation a = nullptr;
    int w = 0, h = 0;
};
std::mutex present_mtx;
std::unordered_map<u64, PresentImage> present_images;
VkCommandPool ui_pool = VK_NULL_HANDLE;  // (UI thread)
VkCommandBuffer ui_cb = VK_NULL_HANDLE;
VkFence ui_fence = VK_NULL_HANDLE;

VkCommandBuffer ui_begin() {
    if (!ui_pool) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam;
        VK_CHECK(vkCreateCommandPool(dev, &pci, nullptr, &ui_pool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = ui_pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &ui_cb));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev, &fci, nullptr, &ui_fence));
    }
    VK_CHECK(vkResetCommandBuffer(ui_cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(ui_cb, &bi));
    return ui_cb;
}
// submits ui_cb on SDL's queue, after GX timeline value wait_tl (0: none), and waits for it
void ui_submit_wait(u64 wait_tl) {
    VK_CHECK(vkEndCommandBuffer(ui_cb));
    VkSemaphoreSubmitInfo wt{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wt.semaphore = timeline;
    wt.value = wait_tl;
    wt.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = ui_cb;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = wait_tl ? 1 : 0;
    si.pWaitSemaphoreInfos = &wt;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    VK_CHECK(vkQueueSubmit2(sdl_queue, 1, &si, ui_fence));
    VK_CHECK(vkWaitForFences(dev, 1, &ui_fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(dev, 1, &ui_fence));
}
void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss,
                   VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = ss;
    b.srcAccessMask = sa;
    b.dstStageMask = ds;
    b.dstAccessMask = da;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}
// copies the view of a display buffer (rows of dw pixels) into a present image of its size
bool copy_view(const Buf &buf, u32 dw, const DisplayView &v, u64 tl, u64 image) {
    PresentImage pi;
    {
        std::lock_guard<std::mutex> lk(present_mtx);
        auto it = present_images.find(image);
        if (it == present_images.end()) return false;
        pi = it->second;
    }
    if (pi.w != (int)v.w || pi.h != (int)v.h) return false;
    VkCommandBuffer cb = ui_begin();
    image_barrier(cb, pi.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy c{};
    c.bufferOffset = ((VkDeviceSize)v.y0 * dw + v.x0) * 4;
    c.bufferRowLength = dw;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {v.w, v.h, 1};
    vkCmdCopyBufferToImage(cb, buf.b, pi.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    image_barrier(cb, pi.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    ui_submit_wait(tl);
    return true;
}
// fn's ScreenImage for a view of a display buffer (copy_to only when presenting on this device)
void view_pixels(const BufP &buf, u32 dw, u64 tl, const DisplayView &v, const std::function<void(const ScreenImage &)> &fn) {
    ScreenImage si{(const u32 *)buf->p + (size_t)v.y0 * dw + v.x0, (int)v.w, (int)v.h, (int)dw, {}};
    if (present_ok) si.copy_to = [&](u64 image) { return copy_view(*buf, dw, v, tl, image); };
    fn(si);
}
}  // namespace

bool vk_present_device(PresentDevice *pd) {
    if (!active || !present_ok) return false;
    pd->instance = inst;
    pd->physical_device = phys;
    pd->device = dev;
    pd->queue_family = qfam;
    return true;
}
u64 vk_present_image_create(int w, int h) {
    if (!present_ok || w <= 0 || h <= 0) return 0;
    PresentImage pi;
    pi.w = w;
    pi.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {(u32)w, (u32)h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  // (as SDL's own)
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(vma, &ci, &ai, &pi.img, &pi.a, nullptr) != VK_SUCCESS) return 0;
    VkCommandBuffer cb = ui_begin();  // (cleared: shown before the first copy, e.g. while pipelines compile)
    image_barrier(cb, pi.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                  VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const VkClearColorValue black{};
    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, pi.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    image_barrier(cb, pi.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    ui_submit_wait(0);
    std::lock_guard<std::mutex> lk(present_mtx);
    present_images[(u64)pi.img] = pi;
    return (u64)pi.img;
}
void vk_present_image_destroy(u64 image) {
    PresentImage pi;
    {
        std::lock_guard<std::mutex> lk(present_mtx);
        auto it = present_images.find(image);
        if (it == present_images.end()) return;
        pi = it->second;
        present_images.erase(it);
    }
    vkQueueWaitIdle(sdl_queue);  // (the renderer's frames that drew it; called on its thread)
    vmaDestroyImage(vma, pi.img, pi.a);
}
void vk_present_hold(bool on) {
    if (on) queue_mtx.lock();
    else queue_mtx.unlock();
}

bool vk_get_display_image_for(u32 fb_paddr, u32 stride, u32 lines, const u32 **rgba, int *w, int *h) {
    thread_local std::vector<u32> out;
    if (!wait_for_view(fb_paddr, stride, lines)) return false;
    std::lock_guard<std::mutex> lk(disp_mtx);
    DisplayView v;
    if (!find_disp_view(fb_paddr, stride, lines, v)) return false;
    wait_value(v.d->tl);
    v.d->buf->invalidate();
    const u32 *px = (const u32 *)v.d->buf->p + (size_t)v.y0 * v.d->w + v.x0;
    out.resize((size_t)v.w * v.h);
    for (u32 y = 0; y < v.h; y++) memcpy(&out[(size_t)y * v.w], px + (size_t)y * v.d->w, (size_t)v.w * 4);
    *w = (int)v.w;
    *h = (int)v.h;
    *rgba = out.data();
    return true;
}

u64 vk_use_display_image(u32 fb_paddr, u32 stride, u32 lines, u64 have, const std::function<void(const ScreenImage &)> &fn) {
    if (!wait_for_view(fb_paddr, stride, lines)) return 0;
    DisplayView v;
    BufP buf;
    u32 dw;
    u64 tl;
    {
        std::unique_lock<std::mutex> lk(disp_mtx);
        if (!find_disp_view(fb_paddr, stride, lines, v)) return 0;
        if (v.key == have) return have;
        static int held[2];
        int &hc = held[lines == 400 ? 0 : 1];
        if (v.d->incomplete && have && hc < 6) {
            hc++;
            return have;
        }
        hc = 0;
        tl = v.d->tl;
        wait_value(tl);
        buf = v.d->buf;
        dw = v.d->w;
        disp_reading.push_back(buf.get());
    }
    buf->invalidate();
    view_pixels(buf, dw, tl, v, fn);
    std::lock_guard<std::mutex> lk(disp_mtx);
    for (size_t i = 0; i < disp_reading.size(); i++)
        if (disp_reading[i] == buf.get()) {
            disp_reading.erase(disp_reading.begin() + i);
            break;
        }
    return v.key;
}

bool vk_get_display_image(int screen, const u32 **rgba, int *w, int *h) {
    u32 addr;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        if (screen < 0 || screen > 1 || !last_screen_addr[screen]) return false;
        addr = last_screen_addr[screen];
    }
    return vk_get_display_image_for(addr, 0, screen ? 320 : 400, rgba, w, h);
}

const HwBackend *vulkan_backend() {
    static const HwBackend b = {
        .name = "vulkan",
        .init = vk_init,
        .flush = vk_flush,
        .vs_kernel = vk_vs_kernel,
        .guest_read = vk_guest_read,
        .guest_written = vk_guest_written,
        .guest_prewrite = vk_guest_prewrite,
        .writeback_all = vk_writeback_all,
        .poll = vk_poll,
        .texture_copy = vk_texture_copy,
        .gpu_transfer = vk_gpu_transfer,
        .surface_for = vk_surface_for,
        .end_list = vk_end_list,
        .defer_transfer = vk_defer_transfer,
        .display_transfer = vk_display_transfer,
        .gpu_made = vk_gpu_made,
        .note_gpu_output = vk_note_gpu_output,
        .get_display_image_for = vk_get_display_image_for,
        .use_display_image = vk_use_display_image,
        .get_display_image = vk_get_display_image,
        .set_scale = vk_set_scale,
        .scale = vk_scale,
        .scale_is_auto = vk_scale_is_auto,
        .busy = vk_busy,
        .set_texture_filter = vk_set_texture_filter,
        .texture_filter_count = vk_texture_filter_count,
        .texture_filter_name = vk_texture_filter_name,
        .set_clear_after_top_transfer = vk_set_clear_after_top_transfer,
        .set_key_black_bottom = vk_set_key_black_bottom,
        .emit_shaders = nullptr,
        .present_device = vk_present_device,
        .present_image_create = vk_present_image_create,
        .present_image_destroy = vk_present_image_destroy,
        .present_hold = vk_present_hold,
    };
    return &b;
}

}  // namespace gpu
