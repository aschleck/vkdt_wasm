// The Vulkan entry points vkdt's compute pipeline uses, on top of WebGPU.
//
// Vulkan binds images and buffers into large memory allocations. WebGPU has no such thing, so each
// image and buffer becomes its own WebGPU object when it's bound, and memory objects only matter
// for host access: host visible memory keeps a shadow copy in the wasm heap that's uploaded
// the first time a command uses it after an unmap, and refreshed after copies into it complete.
//
// vkdt maps whole allocations and binds several buffers to the same staging memory, so an unmap
// can't tell what the host wrote. Uploading on use skips buffers no command reads, and large
// buffers keep a snapshot of what the GPU holds so that unchanged ones skip the upload.
//
// Command buffers record pointers to our objects. vkQueueSubmit resolves them into the flat op
// list library_wvk.js replays, so descriptor updates after recording still take effect.

#include "vulkan_webgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

// library_wvk.js
extern int wvk_js_init(void);
extern int wvk_js_create_buffer(uint32_t size);
extern int wvk_js_create_texture(uint32_t wd, uint32_t ht, int format, uint32_t mips);
extern int wvk_js_create_array_texture(uint32_t wd, uint32_t ht, int format, uint32_t layers);
extern int wvk_js_layer(int array, uint32_t layer);
extern int wvk_js_create_sampler(int linear);
extern void wvk_js_destroy(int id);
extern int wvk_js_create_shader(const void *text, uint32_t len, const char *label,
                                uint32_t writes[4]);
extern void wvk_js_write_buffer(int id, uint32_t offset, const void *ptr, uint32_t size);
extern void wvk_js_submit(const uint32_t *ops, uint32_t count);
extern int wvk_js_wait(void);

#define WVK_NEW(T) ((T *)calloc(1, sizeof(T)))
#define WVK_HANDLE(T, p) ((T)(uintptr_t)(p))
#define WVK_PTR(T, h) ((T *)(uintptr_t)(h))

typedef struct wvk_buffer_t wvk_buffer_t;

typedef struct wvk_memory_t {
  VkDeviceSize size;
  uint32_t type;
  uint8_t *shadow;  // host visible memory only
  wvk_buffer_t **buffers;  // bound buffers, for uploads on unmap
  int num_buffers;
  int cap_buffers;
} wvk_memory_t;

struct wvk_buffer_t {
  VkDeviceSize size;
  int gpu;
  wvk_memory_t *memory;
  VkDeviceSize offset;
  // the shadow may differ from the GPU buffer, see _use
  int dirty;
  // what the GPU buffer holds, for host visible buffers of at least WVK_SNAPSHOT_MIN bytes
  uint8_t *snapshot;
  // see _memo
  uint64_t serial, version;
};

// Below this, uploading costs less than comparing. Above it are the staging buffers for the raw
// and the outputs, where an upload of the 40 MB raw takes 20 ms and a comparison 2.
#define WVK_SNAPSHOT_MIN (1 << 16)

static int _same(const uint8_t *a, const uint8_t *b, size_t n) {
  size_t i = 0;
#ifdef __wasm_simd128__
  for(; i + 64 <= n; i += 64) {
    const v128_t d = wasm_v128_or(
        wasm_v128_or(wasm_v128_xor(wasm_v128_load(a + i), wasm_v128_load(b + i)),
                     wasm_v128_xor(wasm_v128_load(a + i + 16), wasm_v128_load(b + i + 16))),
        wasm_v128_or(wasm_v128_xor(wasm_v128_load(a + i + 32), wasm_v128_load(b + i + 32)),
                     wasm_v128_xor(wasm_v128_load(a + i + 48), wasm_v128_load(b + i + 48))));
    if(wasm_v128_any_true(d)) return 0;
  }
#endif
  return !memcmp(a + i, b + i, n - i);
}

typedef struct wvk_image_t {
  uint32_t wd, ht, mips;
  VkFormat format;
  wvk_gpu_format_t gpu_format;
  wvk_pack_t pack;
  uint32_t texel_bytes;
  // the texture, or when the image is an element of a descriptor array, a layer of one
  int gpu;
  // the array texture this image is a layer of, see _group_images
  int array;
  uint32_t layer;
  uint32_t layers;
  // see _memo
  uint64_t serial, version;
} wvk_image_t;

typedef struct wvk_image_view_t {
  wvk_image_t *image;
} wvk_image_view_t;

typedef struct wvk_sampler_t {
  int gpu;
} wvk_sampler_t;

typedef struct wvk_dset_layout_t {
  uint32_t num_bindings;
  VkDescriptorSetLayoutBinding *bindings;
} wvk_dset_layout_t;

typedef struct wvk_descriptor_t {
  VkDescriptorType type;
  // elements in a descriptor array, whose images then share an array texture
  uint32_t count;
  wvk_buffer_t *buffer;
  VkDeviceSize offset;
  VkDeviceSize range;
  wvk_image_view_t *view;
  wvk_sampler_t *sampler;
} wvk_descriptor_t;

typedef struct wvk_dset_t {
  wvk_dset_layout_t *layout;
  // first element of each binding, indexed by binding number
  wvk_descriptor_t *descriptors;
  uint32_t max_binding;
} wvk_dset_t;

typedef struct wvk_shader_t {
  int gpu;
  // the bindings the shader may write, one mask per set
  uint32_t writes[4];
} wvk_shader_t;

typedef struct wvk_pipeline_t {
  wvk_shader_t *shader;
} wvk_pipeline_t;

// Push constants are bound as a storage buffer at group 3, binding 0, see glsl_fix.pl. Each
// dispatch gets a slot in one buffer per submit, spaced by WebGPU's storage buffer offset
// alignment.
#define WVK_PUSH_SLOT 256

typedef enum wvk_cmd_type_t {
  WVK_CMD_DISPATCH,
  WVK_CMD_COPY_BUFFER,
  WVK_CMD_BUFFER_TO_IMAGE,
  WVK_CMD_IMAGE_TO_BUFFER,
  WVK_CMD_CLEAR_IMAGE,
  WVK_CMD_FILL_BUFFER,
} wvk_cmd_type_t;

typedef struct wvk_cmd_t {
  wvk_cmd_type_t type;
  wvk_pipeline_t *pipeline;
  wvk_dset_t *dsets[4];
  uint32_t groups[3];
  wvk_buffer_t *buffer;
  wvk_buffer_t *dst_buffer;
  wvk_image_t *image;
  VkDeviceSize offset;
  VkDeviceSize dst_offset;
  VkDeviceSize size;
  uint32_t row_texels;
  uint32_t wd, ht;
  float color[4];
  uint32_t push_size;
  uint8_t push[WVK_PUSH_SLOT];
} wvk_cmd_t;

struct VkCommandBuffer_T {
  wvk_cmd_t *cmds;
  uint32_t num_cmds;
  uint32_t cap_cmds;
  wvk_pipeline_t *pipeline;
  wvk_dset_t *dsets[4];
  uint32_t push_size;
  uint8_t push[WVK_PUSH_SLOT];
};

typedef struct wvk_semaphore_t {
  uint64_t value;
  uint64_t pending;
} wvk_semaphore_t;

static struct VkInstance_T {
  int unused;
} instance;
static struct VkPhysicalDevice_T {
  int unused;
} physical_device;
static struct VkDevice_T {
  int unused;
} device;
static struct VkQueue_T {
  int unused;
} queue;

// The flat op list for one submit, see wvk_js_submit.
static uint32_t *ops;
static uint32_t num_ops, cap_ops;

// Push constants for one submit, WVK_PUSH_SLOT bytes per dispatch.
static uint8_t *push;
static uint32_t num_push, cap_push;

// Skipping dispatches whose results are still there. vkdt reruns the whole graph for any change,
// and most of it, such as everything before the module being edited, sees the same inputs as last
// time. A dispatch's key covers its shader, workgroups, uniforms, push constants and the versions
// of what it reads, and it runs again unless the key matches its last run and what it writes
// still has the versions it left there.
//
// Versions come from one clock, so they're unique across images and buffers.
static uint64_t wvk_clock;
// moves on with every write to a layer of an array texture, which readers of the whole array
// can't otherwise see
static uint64_t wvk_array_epoch;

typedef struct wvk_memo_t {
  uint64_t identity, key;
  uint32_t num_written;
  uint64_t written[16];
} wvk_memo_t;
static wvk_memo_t *memos;
static uint32_t num_memos, cap_memos;
// dispatches since the last wvk_dispatch_stats
static uint32_t wvk_ran, wvk_skipped;

/** How many dispatches ran and how many were skipped since the last call. */
void wvk_dispatch_stats(uint32_t *ran, uint32_t *skipped) {
  *ran = wvk_ran;
  *skipped = wvk_skipped;
  wvk_ran = wvk_skipped = 0;
}

static uint64_t _fnv(uint64_t h, const void *data, size_t n) {
  const uint8_t *p = data;
  for(size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}
#define WVK_MIX(h, v) do { __typeof__(v) _v = (v); h = _fnv(h, &_v, sizeof(_v)); } while(0)

static void _wrote_image(wvk_image_t *img) {
  img->version = ++wvk_clock;
  if(img->array) wvk_array_epoch = wvk_clock;
}

static void _wrote_buffer(wvk_buffer_t *b) {
  b->version = ++wvk_clock;
}

// Textures and buffers vkdt freed, kept until the next submit. vkdt frees and reallocates every
// image of a graph whenever it reruns the graph whole, such as for turning a module off, and a
// fresh texture would have a fresh serial that no memo knows. Handing the same texture back with
// its serial and version lets the memos skip everything before the change. This is safe because a
// version only matches when the texture still holds what that version's write left in it.
typedef struct wvk_pooled_t {
  // 0 for a texture, by its size, format and mips, or else a buffer by its size
  int buffer;
  uint32_t wd, ht, mips;
  wvk_gpu_format_t format;
  VkDeviceSize size;
  int gpu;
  uint8_t *snapshot;
  uint64_t serial, version;
} wvk_pooled_t;
static wvk_pooled_t *pool;
static uint32_t num_pool, cap_pool;

static void _pool_put(wvk_pooled_t p) {
  if(num_pool == cap_pool) {
    cap_pool = cap_pool ? 2 * cap_pool : 64;
    pool = realloc(pool, cap_pool * sizeof(*pool));
  }
  pool[num_pool++] = p;
}

// The first pooled entry like want, in the order they were freed, because vkdt allocates nodes in
// the order it freed them and so each image gets its own texture back.
static int _pool_take(const wvk_pooled_t *want, wvk_pooled_t *out) {
  for(uint32_t i = 0; i < num_pool; i++) {
    const wvk_pooled_t *p = pool + i;
    if(p->buffer != want->buffer) continue;
    if(want->buffer ? p->size != want->size
                    : p->wd != want->wd || p->ht != want->ht || p->format != want->format
                          || p->mips != want->mips)
      continue;
    *out = *p;
    memmove(pool + i, pool + i + 1, (num_pool - i - 1) * sizeof(*pool));
    num_pool--;
    return 1;
  }
  return 0;
}

// Destroys what the allocations since the last submit didn't take back.
static void _pool_trim(void) {
  for(uint32_t i = 0; i < num_pool; i++) {
    wvk_js_destroy(pool[i].gpu);
    free(pool[i].snapshot);
  }
  num_pool = 0;
}

static wvk_memo_t *_memo(uint64_t identity) {
  for(uint32_t i = 0; i < num_memos; i++)
    if(memos[i].identity == identity) return memos + i;
  // Graphs that are gone leave memos nothing matches, so start over now and then.
  if(num_memos == 4096) num_memos = 0;
  if(num_memos == cap_memos) {
    cap_memos = cap_memos ? 2 * cap_memos : 256;
    memos = realloc(memos, cap_memos * sizeof(*memos));
  }
  memos[num_memos] = (wvk_memo_t){.identity = identity};
  return memos + num_memos++;
}

static void _op(uint32_t v) {
  if(num_ops == cap_ops) {
    cap_ops = cap_ops ? 2 * cap_ops : 4096;
    ops = realloc(ops, cap_ops * sizeof(uint32_t));
  }
  ops[num_ops++] = v;
}

static void _format(VkFormat f, wvk_gpu_format_t *gpu, wvk_pack_t *pack, uint32_t *bytes) {
  switch(f) {
    case VK_FORMAT_R16_UNORM: *gpu = WVK_R32F; *pack = WVK_PACK_R16_UNORM; *bytes = 2; return;
    case VK_FORMAT_R16_SFLOAT: *gpu = WVK_R32F; *pack = WVK_PACK_R16_SFLOAT; *bytes = 2; return;
    case VK_FORMAT_R16G16_SFLOAT: *gpu = WVK_RG32F; *pack = WVK_PACK_RG16_SFLOAT; *bytes = 4; return;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      *gpu = WVK_RGBA16F; *pack = WVK_PACK_RGBA16_SFLOAT; *bytes = 8; return;
    case VK_FORMAT_R32_SFLOAT: *gpu = WVK_R32F; *pack = WVK_PACK_R32_SFLOAT; *bytes = 4; return;
    case VK_FORMAT_R32G32_SFLOAT: *gpu = WVK_RG32F; *pack = WVK_PACK_RG32_SFLOAT; *bytes = 8; return;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      *gpu = WVK_RGBA32F; *pack = WVK_PACK_RGBA32_SFLOAT; *bytes = 16; return;
    case VK_FORMAT_R8G8B8A8_UNORM: *gpu = WVK_RGBA8; *pack = WVK_PACK_RGBA8_UNORM; *bytes = 4; return;
    case VK_FORMAT_R8_UNORM: *gpu = WVK_R32F; *pack = WVK_PACK_R8_UNORM; *bytes = 1; return;
    case VK_FORMAT_R16G16B16A16_UNORM:
      *gpu = WVK_RGBA32F; *pack = WVK_PACK_RGBA16_UNORM; *bytes = 8; return;
    case VK_FORMAT_R32_UINT: *gpu = WVK_R32UI; *pack = WVK_PACK_R32_SFLOAT; *bytes = 4; return;
    case VK_FORMAT_R32G32B32A32_UINT:
      *gpu = WVK_RGBA32UI; *pack = WVK_PACK_RGBA32_SFLOAT; *bytes = 16; return;
    default:
      fprintf(stderr, "[wvk] unsupported image format %d, using rgba16f\n", f);
      *gpu = WVK_RGBA16F; *pack = WVK_PACK_RGBA16_SFLOAT; *bytes = 8; return;
  }
}

VkResult wvk_init(void) { return wvk_js_init() ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS; }

// Instance and device. There's exactly one of each.

VkResult vkCreateInstance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *a,
                          VkInstance *out) {
  *out = &instance;
  return VK_SUCCESS;
}
void vkDestroyInstance(VkInstance i, const VkAllocationCallbacks *a) {}

VkResult vkEnumeratePhysicalDevices(VkInstance i, uint32_t *count, VkPhysicalDevice *out) {
  if(out && *count) out[0] = &physical_device;
  *count = 1;
  return VK_SUCCESS;
}

VkResult vkCreateDevice(VkPhysicalDevice pd, const VkDeviceCreateInfo *info,
                        const VkAllocationCallbacks *a, VkDevice *out) {
  *out = &device;
  return VK_SUCCESS;
}
void vkDestroyDevice(VkDevice d, const VkAllocationCallbacks *a) {}
void vkGetDeviceQueue(VkDevice d, uint32_t family, uint32_t index, VkQueue *out) { *out = &queue; }
// Readbacks waiting on the GPU, whose snapshots catch up once they land. See WVK_SNAPSHOT_MIN.
typedef struct wvk_readback_t {
  wvk_buffer_t *buffer;
  VkDeviceSize offset, size;
} wvk_readback_t;
static wvk_readback_t *readbacks;
static uint32_t num_readbacks, cap_readbacks;

/** Waits for all submitted work and readbacks. */
static void _wait(void) {
  wvk_js_wait();
  for(uint32_t i = 0; i < num_readbacks; i++) {
    const wvk_readback_t *r = readbacks + i;
    wvk_buffer_t *b = r->buffer;
    if(!b->snapshot || !b->memory) continue;
    memcpy(b->snapshot + r->offset, b->memory->shadow + b->offset + r->offset, r->size);
  }
  num_readbacks = 0;
}

VkResult vkDeviceWaitIdle(VkDevice d) {
  _wait();
  return VK_SUCCESS;
}
VkResult vkQueueWaitIdle(VkQueue q) {
  _wait();
  return VK_SUCCESS;
}

static void _memory_properties(VkPhysicalDeviceMemoryProperties *p) {
  memset(p, 0, sizeof(*p));
  p->memoryTypeCount = 2;
  p->memoryTypes[WVK_MEMORY_DEVICE].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  p->memoryTypes[WVK_MEMORY_DEVICE].heapIndex = 0;
  p->memoryTypes[WVK_MEMORY_HOST].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                  | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                  | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  p->memoryTypes[WVK_MEMORY_HOST].heapIndex = 1;
  p->memoryHeapCount = 2;
  p->memoryHeaps[0].size = (VkDeviceSize)4 << 30;
  p->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  p->memoryHeaps[1].size = (VkDeviceSize)2 << 30;
}

void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties *p) {
  _memory_properties(p);
}

void vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice pd,
                                          VkPhysicalDeviceMemoryProperties2 *p) {
  _memory_properties(&p->memoryProperties);
  for(VkBaseOutStructure *s = p->pNext; s; s = s->pNext) {
    if(s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) {
      VkPhysicalDeviceMemoryBudgetPropertiesEXT *b = (void *)s;
      for(int i = 0; i < 2; i++) {
        b->heapBudget[i] = p->memoryProperties.memoryHeaps[i].size;
        b->heapUsage[i] = 0;
      }
    }
  }
}

// Memory

VkResult vkAllocateMemory(VkDevice d, const VkMemoryAllocateInfo *info,
                          const VkAllocationCallbacks *a, VkDeviceMemory *out) {
  wvk_memory_t *m = WVK_NEW(wvk_memory_t);
  m->size = info->allocationSize;
  m->type = info->memoryTypeIndex;
  if(m->type == WVK_MEMORY_HOST) {
    m->shadow = calloc(1, m->size);
    if(!m->shadow) {
      free(m);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
  }
  *out = WVK_HANDLE(VkDeviceMemory, m);
  return VK_SUCCESS;
}

void vkFreeMemory(VkDevice d, VkDeviceMemory memory, const VkAllocationCallbacks *a) {
  wvk_memory_t *m = WVK_PTR(wvk_memory_t, memory);
  if(!m) return;
  free(m->shadow);
  free(m->buffers);
  free(m);
}

VkResult vkMapMemory(VkDevice d, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                     VkMemoryMapFlags flags, void **data) {
  wvk_memory_t *m = WVK_PTR(wvk_memory_t, memory);
  if(!m->shadow) return VK_ERROR_MEMORY_MAP_FAILED;
  *data = m->shadow + offset;
  return VK_SUCCESS;
}

// The host may have written anywhere in the mapping.
void vkUnmapMemory(VkDevice d, VkDeviceMemory memory) {
  wvk_memory_t *m = WVK_PTR(wvk_memory_t, memory);
  for(int i = 0; i < m->num_buffers; i++) m->buffers[i]->dirty = 1;
}

/** Uploads a buffer's shadow before a command uses it, if the host may have changed it. */
static void _use(wvk_buffer_t *b) {
  if(!b->dirty || !b->gpu || !b->memory || !b->memory->shadow) return;
  b->dirty = 0;
  const uint8_t *host = b->memory->shadow + b->offset;
  if(b->size >= WVK_SNAPSHOT_MIN) {
    if(b->snapshot && _same(host, b->snapshot, b->size)) return;
    if(!b->snapshot) b->snapshot = malloc(b->size);
    memcpy(b->snapshot, host, b->size);
  }
  wvk_js_write_buffer(b->gpu, 0, host, (uint32_t)b->size);
  _wrote_buffer(b);
}

VkResult vkFlushMappedMemoryRanges(VkDevice d, uint32_t count, const VkMappedMemoryRange *r) {
  for(uint32_t i = 0; i < count; i++) vkUnmapMemory(d, r[i].memory);
  return VK_SUCCESS;
}

// Buffers

VkResult vkCreateBuffer(VkDevice d, const VkBufferCreateInfo *info, const VkAllocationCallbacks *a,
                        VkBuffer *out) {
  wvk_buffer_t *b = WVK_NEW(wvk_buffer_t);
  b->size = info->size;
  *out = WVK_HANDLE(VkBuffer, b);
  return VK_SUCCESS;
}

static void _unbind_buffer(wvk_buffer_t *b) {
  wvk_memory_t *m = b->memory;
  if(m) {
    for(int i = 0; i < m->num_buffers; i++)
      if(m->buffers[i] == b) {
        m->buffers[i] = m->buffers[--m->num_buffers];
        break;
      }
  }
  b->memory = NULL;
  if(b->gpu) {
    _pool_put((wvk_pooled_t){.buffer = 1, .size = b->size, .gpu = b->gpu, .snapshot = b->snapshot,
                             .serial = b->serial, .version = b->version});
  }
  b->gpu = 0;
  b->snapshot = NULL;
}

void vkDestroyBuffer(VkDevice d, VkBuffer buffer, const VkAllocationCallbacks *a) {
  wvk_buffer_t *b = WVK_PTR(wvk_buffer_t, buffer);
  if(!b) return;
  _unbind_buffer(b);
  free(b);
}

void vkGetBufferMemoryRequirements(VkDevice d, VkBuffer buffer, VkMemoryRequirements *r) {
  wvk_buffer_t *b = WVK_PTR(wvk_buffer_t, buffer);
  r->size = (b->size + 255) & ~(VkDeviceSize)255;
  r->alignment = 256;
  r->memoryTypeBits = 3;
}

VkResult vkBindBufferMemory(VkDevice d, VkBuffer buffer, VkDeviceMemory memory,
                            VkDeviceSize offset) {
  wvk_buffer_t *b = WVK_PTR(wvk_buffer_t, buffer);
  wvk_memory_t *m = WVK_PTR(wvk_memory_t, memory);
  _unbind_buffer(b);
  b->memory = m;
  b->offset = offset;
  wvk_pooled_t p;
  if(_pool_take(&(wvk_pooled_t){.buffer = 1, .size = b->size}, &p)) {
    b->gpu = p.gpu;
    b->snapshot = p.snapshot;
    b->serial = p.serial;
    b->version = p.version;
  } else {
    b->gpu = wvk_js_create_buffer((uint32_t)b->size);
    b->serial = b->version = ++wvk_clock;
  }
  if(m->shadow) {
    if(m->num_buffers == m->cap_buffers) {
      m->cap_buffers = m->cap_buffers ? 2 * m->cap_buffers : 8;
      m->buffers = realloc(m->buffers, m->cap_buffers * sizeof(wvk_buffer_t *));
    }
    m->buffers[m->num_buffers++] = b;
  }
  return VK_SUCCESS;
}

// Shaders only use buffer references for storage buffer connectors, which we don't support yet,
// so these just have to be unique.
VkDeviceAddress vkGetBufferDeviceAddress(VkDevice d, const VkBufferDeviceAddressInfo *info) {
  wvk_buffer_t *b = WVK_PTR(wvk_buffer_t, info->buffer);
  return ((VkDeviceAddress)b->gpu) << 32;
}

// Images

VkResult vkCreateImage(VkDevice d, const VkImageCreateInfo *info, const VkAllocationCallbacks *a,
                       VkImage *out) {
  wvk_image_t *img = WVK_NEW(wvk_image_t);
  img->wd = info->extent.width;
  img->ht = info->extent.height;
  img->mips = info->mipLevels;
  img->format = info->format;
  _format(info->format, &img->gpu_format, &img->pack, &img->texel_bytes);
  *out = WVK_HANDLE(VkImage, img);
  return VK_SUCCESS;
}

void vkDestroyImage(VkDevice d, VkImage image, const VkAllocationCallbacks *a) {
  wvk_image_t *img = WVK_PTR(wvk_image_t, image);
  if(!img) return;
  // a layer of an array texture isn't a texture of its own
  if(img->gpu && !img->array) {
    _pool_put((wvk_pooled_t){.wd = img->wd, .ht = img->ht, .mips = img->mips,
                             .format = img->gpu_format, .gpu = img->gpu, .serial = img->serial,
                             .version = img->version});
  } else if(img->gpu) {
    wvk_js_destroy(img->gpu);
  }
  free(img);
}

void vkGetImageMemoryRequirements(VkDevice d, VkImage image, VkMemoryRequirements *r) {
  wvk_image_t *img = WVK_PTR(wvk_image_t, image);
  const VkDeviceSize size = (VkDeviceSize)img->wd * img->ht * img->texel_bytes;
  r->size = (size + 255) & ~(VkDeviceSize)255;
  r->alignment = 256;
  r->memoryTypeBits = 3;
}

void vkGetImageMemoryRequirements2(VkDevice d, const VkImageMemoryRequirementsInfo2 *info,
                                   VkMemoryRequirements2 *r) {
  vkGetImageMemoryRequirements(d, info->image, &r->memoryRequirements);
}

VkResult vkBindImageMemory(VkDevice d, VkImage image, VkDeviceMemory memory, VkDeviceSize offset) {
  wvk_image_t *img = WVK_PTR(wvk_image_t, image);
  if(img->gpu) wvk_js_destroy(img->gpu);
  img->array = 0;
  wvk_pooled_t p;
  const wvk_pooled_t want =
      {.wd = img->wd, .ht = img->ht, .mips = img->mips, .format = img->gpu_format};
  if(_pool_take(&want, &p)) {
    img->gpu = p.gpu;
    img->serial = p.serial;
    img->version = p.version;
  } else {
    img->gpu = wvk_js_create_texture(img->wd, img->ht, img->gpu_format, img->mips);
    img->serial = img->version = ++wvk_clock;
  }
  return VK_SUCCESS;
}

// WebGPU has no arrays of bindings, so the images in a descriptor array become layers of one array
// texture, which the transpiled shaders index by layer (see glsl_fix.pl). This needs every image
// to have the same size and format, which holds for the vkdt modules we build.
static void _group_images(const VkDescriptorImageInfo *infos, uint32_t count) {
  wvk_image_t *first = WVK_PTR(wvk_image_view_t, infos[0].imageView)->image;
  int grouped = first->array != 0;
  for(uint32_t k = 0; k < count && grouped; k++) {
    wvk_image_t *img = WVK_PTR(wvk_image_view_t, infos[k].imageView)->image;
    grouped = img->array == first->array && img->layer == k && img->layers == count;
  }
  if(grouped) return;
  const int array = wvk_js_create_array_texture(first->wd, first->ht, first->gpu_format, count);
  for(uint32_t k = 0; k < count; k++) {
    wvk_image_t *img = WVK_PTR(wvk_image_view_t, infos[k].imageView)->image;
    if(img->wd != first->wd || img->ht != first->ht || img->gpu_format != first->gpu_format)
      fprintf(stderr, "[wvk] descriptor array elements differ in size or format\n");
    if(img->gpu) wvk_js_destroy(img->gpu);
    img->gpu = wvk_js_layer(array, k);
    img->array = array;
    img->layer = k;
    img->layers = count;
  }
}

VkResult vkBindImageMemory2(VkDevice d, uint32_t count, const VkBindImageMemoryInfo *infos) {
  // multi-planar images bind twice, once per plane
  return vkBindImageMemory(d, infos[0].image, infos[0].memory, infos[0].memoryOffset);
}

VkResult vkCreateImageView(VkDevice d, const VkImageViewCreateInfo *info,
                           const VkAllocationCallbacks *a, VkImageView *out) {
  wvk_image_view_t *v = WVK_NEW(wvk_image_view_t);
  v->image = WVK_PTR(wvk_image_t, info->image);
  *out = WVK_HANDLE(VkImageView, v);
  return VK_SUCCESS;
}

void vkDestroyImageView(VkDevice d, VkImageView view, const VkAllocationCallbacks *a) {
  free(WVK_PTR(wvk_image_view_t, view));
}

VkResult vkCreateSampler(VkDevice d, const VkSamplerCreateInfo *info,
                         const VkAllocationCallbacks *a, VkSampler *out) {
  wvk_sampler_t *s = WVK_NEW(wvk_sampler_t);
  s->gpu = wvk_js_create_sampler(info->magFilter == VK_FILTER_LINEAR);
  *out = WVK_HANDLE(VkSampler, s);
  return VK_SUCCESS;
}

void vkDestroySampler(VkDevice d, VkSampler sampler, const VkAllocationCallbacks *a) {
  wvk_sampler_t *s = WVK_PTR(wvk_sampler_t, sampler);
  if(!s) return;
  wvk_js_destroy(s->gpu);
  free(s);
}

// Descriptors

VkResult vkCreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo *info,
                                     const VkAllocationCallbacks *a, VkDescriptorSetLayout *out) {
  wvk_dset_layout_t *l = WVK_NEW(wvk_dset_layout_t);
  l->num_bindings = info->bindingCount;
  l->bindings = calloc(info->bindingCount + 1, sizeof(VkDescriptorSetLayoutBinding));
  memcpy(l->bindings, info->pBindings, info->bindingCount * sizeof(VkDescriptorSetLayoutBinding));
  *out = WVK_HANDLE(VkDescriptorSetLayout, l);
  return VK_SUCCESS;
}

void vkDestroyDescriptorSetLayout(VkDevice d, VkDescriptorSetLayout layout,
                                  const VkAllocationCallbacks *a) {
  wvk_dset_layout_t *l = WVK_PTR(wvk_dset_layout_t, layout);
  if(!l) return;
  free(l->bindings);
  free(l);
}

// Pools only track their sets so they can free them all at once.
typedef struct wvk_pool_t {
  wvk_dset_t **sets;
  uint32_t num_sets;
  uint32_t cap_sets;
} wvk_pool_t;

VkResult vkCreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo *info,
                                const VkAllocationCallbacks *a, VkDescriptorPool *out) {
  *out = WVK_HANDLE(VkDescriptorPool, WVK_NEW(wvk_pool_t));
  return VK_SUCCESS;
}

VkResult vkResetDescriptorPool(VkDevice d, VkDescriptorPool pool, VkDescriptorPoolResetFlags f) {
  wvk_pool_t *p = WVK_PTR(wvk_pool_t, pool);
  for(uint32_t i = 0; i < p->num_sets; i++) {
    free(p->sets[i]->descriptors);
    free(p->sets[i]);
  }
  p->num_sets = 0;
  return VK_SUCCESS;
}

void vkDestroyDescriptorPool(VkDevice d, VkDescriptorPool pool, const VkAllocationCallbacks *a) {
  wvk_pool_t *p = WVK_PTR(wvk_pool_t, pool);
  if(!p) return;
  vkResetDescriptorPool(d, pool, 0);
  free(p->sets);
  free(p);
}

VkResult vkAllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo *info,
                                  VkDescriptorSet *out) {
  wvk_pool_t *p = WVK_PTR(wvk_pool_t, info->descriptorPool);
  for(uint32_t i = 0; i < info->descriptorSetCount; i++) {
    wvk_dset_t *s = WVK_NEW(wvk_dset_t);
    s->layout = WVK_PTR(wvk_dset_layout_t, info->pSetLayouts[i]);
    for(uint32_t b = 0; b < s->layout->num_bindings; b++)
      if(s->layout->bindings[b].binding + 1 > s->max_binding)
        s->max_binding = s->layout->bindings[b].binding + 1;
    s->descriptors = calloc(s->max_binding + 1, sizeof(wvk_descriptor_t));
    if(p->num_sets == p->cap_sets) {
      p->cap_sets = p->cap_sets ? 2 * p->cap_sets : 64;
      p->sets = realloc(p->sets, p->cap_sets * sizeof(wvk_dset_t *));
    }
    p->sets[p->num_sets++] = s;
    out[i] = WVK_HANDLE(VkDescriptorSet, s);
  }
  return VK_SUCCESS;
}

void vkUpdateDescriptorSets(VkDevice d, uint32_t count, const VkWriteDescriptorSet *writes,
                            uint32_t copies, const VkCopyDescriptorSet *c) {
  for(uint32_t i = 0; i < count; i++) {
    const VkWriteDescriptorSet *w = writes + i;
    wvk_dset_t *s = WVK_PTR(wvk_dset_t, w->dstSet);
    if(w->dstArrayElement != 0 || w->dstBinding >= s->max_binding) continue;
    wvk_descriptor_t *dsc = s->descriptors + w->dstBinding;
    dsc->type = w->descriptorType;
    dsc->count = w->descriptorCount;
    if(w->pBufferInfo) {
      dsc->buffer = WVK_PTR(wvk_buffer_t, w->pBufferInfo[0].buffer);
      dsc->offset = w->pBufferInfo[0].offset;
      dsc->range = w->pBufferInfo[0].range;
    }
    if(w->pImageInfo) {
      dsc->view = WVK_PTR(wvk_image_view_t, w->pImageInfo[0].imageView);
      dsc->sampler = WVK_PTR(wvk_sampler_t, w->pImageInfo[0].sampler);
      if(w->descriptorCount > 1 && dsc->view) _group_images(w->pImageInfo, w->descriptorCount);
    }
  }
}

// Pipelines

VkResult vkCreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo *info,
                                const VkAllocationCallbacks *a, VkPipelineLayout *out) {
  *out = WVK_HANDLE(VkPipelineLayout, WVK_NEW(int));
  return VK_SUCCESS;
}

void vkDestroyPipelineLayout(VkDevice d, VkPipelineLayout l, const VkAllocationCallbacks *a) {
  free(WVK_PTR(int, l));
}

// vkdt reads <kernel>.comp.spv, which our build fills with WGSL instead of SPIR-V.
VkResult vkCreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo *info,
                              const VkAllocationCallbacks *a, VkShaderModule *out) {
  wvk_shader_t *s = WVK_NEW(wvk_shader_t);
  s->gpu = wvk_js_create_shader(info->pCode, (uint32_t)info->codeSize, "", s->writes);
  *out = WVK_HANDLE(VkShaderModule, s);
  return VK_SUCCESS;
}

// Pipelines hold on to the shader, so modules stay alive until then.
void vkDestroyShaderModule(VkDevice d, VkShaderModule m, const VkAllocationCallbacks *a) {}

VkResult vkCreateComputePipelines(VkDevice d, VkPipelineCache cache, uint32_t count,
                                  const VkComputePipelineCreateInfo *infos,
                                  const VkAllocationCallbacks *a, VkPipeline *out) {
  for(uint32_t i = 0; i < count; i++) {
    wvk_pipeline_t *p = WVK_NEW(wvk_pipeline_t);
    p->shader = WVK_PTR(wvk_shader_t, infos[i].stage.module);
    out[i] = WVK_HANDLE(VkPipeline, p);
  }
  return VK_SUCCESS;
}

VkResult vkCreateGraphicsPipelines(VkDevice d, VkPipelineCache cache, uint32_t count,
                                   const VkGraphicsPipelineCreateInfo *infos,
                                   const VkAllocationCallbacks *a, VkPipeline *out) {
  fprintf(stderr, "[wvk] graphics pipelines are not supported\n");
  return VK_ERROR_FEATURE_NOT_PRESENT;
}

void vkDestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks *a) {
  free(WVK_PTR(wvk_pipeline_t, p));
}

// Command buffers

VkResult vkCreateCommandPool(VkDevice d, const VkCommandPoolCreateInfo *info,
                             const VkAllocationCallbacks *a, VkCommandPool *out) {
  *out = WVK_HANDLE(VkCommandPool, WVK_NEW(int));
  return VK_SUCCESS;
}
void vkDestroyCommandPool(VkDevice d, VkCommandPool p, const VkAllocationCallbacks *a) {
  free(WVK_PTR(int, p));
}
VkResult vkResetCommandPool(VkDevice d, VkCommandPool p, VkCommandPoolResetFlags f) {
  return VK_SUCCESS;
}

VkResult vkAllocateCommandBuffers(VkDevice d, const VkCommandBufferAllocateInfo *info,
                                  VkCommandBuffer *out) {
  for(uint32_t i = 0; i < info->commandBufferCount; i++) out[i] = WVK_NEW(struct VkCommandBuffer_T);
  return VK_SUCCESS;
}

void vkFreeCommandBuffers(VkDevice d, VkCommandPool p, uint32_t count, const VkCommandBuffer *c) {
  for(uint32_t i = 0; i < count; i++) {
    if(!c[i]) continue;
    free(c[i]->cmds);
    free(c[i]);
  }
}

VkResult vkBeginCommandBuffer(VkCommandBuffer c, const VkCommandBufferBeginInfo *info) {
  c->num_cmds = 0;
  c->pipeline = NULL;
  memset(c->dsets, 0, sizeof(c->dsets));
  return VK_SUCCESS;
}

VkResult vkEndCommandBuffer(VkCommandBuffer c) { return VK_SUCCESS; }

static wvk_cmd_t *_cmd(VkCommandBuffer c, wvk_cmd_type_t type) {
  if(c->num_cmds == c->cap_cmds) {
    c->cap_cmds = c->cap_cmds ? 2 * c->cap_cmds : 64;
    c->cmds = realloc(c->cmds, c->cap_cmds * sizeof(wvk_cmd_t));
  }
  wvk_cmd_t *cmd = c->cmds + c->num_cmds++;
  memset(cmd, 0, sizeof(*cmd));
  cmd->type = type;
  return cmd;
}

void vkCmdBindPipeline(VkCommandBuffer c, VkPipelineBindPoint bp, VkPipeline p) {
  c->pipeline = WVK_PTR(wvk_pipeline_t, p);
}

void vkCmdBindDescriptorSets(VkCommandBuffer c, VkPipelineBindPoint bp, VkPipelineLayout l,
                             uint32_t first, uint32_t count, const VkDescriptorSet *sets,
                             uint32_t dynamic_count, const uint32_t *dynamic_offsets) {
  for(uint32_t i = 0; i < count && first + i < 4; i++)
    c->dsets[first + i] = WVK_PTR(wvk_dset_t, sets[i]);
}

void vkCmdPushConstants(VkCommandBuffer c, VkPipelineLayout l, VkShaderStageFlags stages,
                        uint32_t offset, uint32_t size, const void *values) {
  if(offset + size > WVK_PUSH_SLOT) return;
  memcpy(c->push + offset, values, size);
  if(offset + size > c->push_size) c->push_size = offset + size;
}

void vkCmdDispatch(VkCommandBuffer c, uint32_t x, uint32_t y, uint32_t z) {
  wvk_cmd_t *cmd = _cmd(c, WVK_CMD_DISPATCH);
  cmd->pipeline = c->pipeline;
  memcpy(cmd->dsets, c->dsets, sizeof(cmd->dsets));
  cmd->groups[0] = x;
  cmd->groups[1] = y;
  cmd->groups[2] = z;
  cmd->push_size = c->push_size;
  memcpy(cmd->push, c->push, c->push_size);
}

// WebGPU orders everything within a submission for us.
void vkCmdPipelineBarrier(VkCommandBuffer c, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                          VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb,
                          const VkBufferMemoryBarrier *b, uint32_t ni,
                          const VkImageMemoryBarrier *i) {}

void vkCmdCopyBuffer(VkCommandBuffer c, VkBuffer src, VkBuffer dst, uint32_t count,
                     const VkBufferCopy *regions) {
  for(uint32_t i = 0; i < count; i++) {
    wvk_cmd_t *cmd = _cmd(c, WVK_CMD_COPY_BUFFER);
    cmd->buffer = WVK_PTR(wvk_buffer_t, src);
    cmd->dst_buffer = WVK_PTR(wvk_buffer_t, dst);
    cmd->offset = regions[i].srcOffset;
    cmd->dst_offset = regions[i].dstOffset;
    cmd->size = regions[i].size;
  }
}

static void _image_copy(VkCommandBuffer c, wvk_cmd_type_t type, VkBuffer buffer, VkImage image,
                        uint32_t count, const VkBufferImageCopy *regions) {
  for(uint32_t i = 0; i < count; i++) {
    const VkBufferImageCopy *r = regions + i;
    wvk_cmd_t *cmd = _cmd(c, type);
    cmd->buffer = WVK_PTR(wvk_buffer_t, buffer);
    cmd->image = WVK_PTR(wvk_image_t, image);
    cmd->offset = r->bufferOffset;
    cmd->wd = r->imageExtent.width;
    cmd->ht = r->imageExtent.height;
    cmd->row_texels = r->bufferRowLength ? r->bufferRowLength : r->imageExtent.width;
    if(r->imageOffset.x || r->imageOffset.y || r->imageSubresource.mipLevel)
      fprintf(stderr, "[wvk] partial image copies are not supported\n");
  }
}

void vkCmdCopyBufferToImage(VkCommandBuffer c, VkBuffer src, VkImage dst, VkImageLayout layout,
                            uint32_t count, const VkBufferImageCopy *regions) {
  _image_copy(c, WVK_CMD_BUFFER_TO_IMAGE, src, dst, count, regions);
}

void vkCmdCopyImageToBuffer(VkCommandBuffer c, VkImage src, VkImageLayout layout, VkBuffer dst,
                            uint32_t count, const VkBufferImageCopy *regions) {
  _image_copy(c, WVK_CMD_IMAGE_TO_BUFFER, dst, src, count, regions);
}

void vkCmdCopyImage(VkCommandBuffer c, VkImage src, VkImageLayout sl, VkImage dst,
                    VkImageLayout dl, uint32_t count, const VkImageCopy *regions) {
  fprintf(stderr, "[wvk] image to image copies are not supported\n");
}

void vkCmdClearColorImage(VkCommandBuffer c, VkImage image, VkImageLayout layout,
                          const VkClearColorValue *color, uint32_t count,
                          const VkImageSubresourceRange *ranges) {
  wvk_cmd_t *cmd = _cmd(c, WVK_CMD_CLEAR_IMAGE);
  cmd->image = WVK_PTR(wvk_image_t, image);
  memcpy(cmd->color, color->float32, sizeof(cmd->color));
}

void vkCmdFillBuffer(VkCommandBuffer c, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                     uint32_t data) {
  if(data) fprintf(stderr, "[wvk] only zero fills are supported\n");
  wvk_cmd_t *cmd = _cmd(c, WVK_CMD_FILL_BUFFER);
  cmd->buffer = WVK_PTR(wvk_buffer_t, buffer);
  cmd->offset = offset;
  cmd->size = size == VK_WHOLE_SIZE ? cmd->buffer->size - offset : size;
}

// We have no GPU timers.
void vkCmdWriteTimestamp(VkCommandBuffer c, VkPipelineStageFlagBits stage, VkQueryPool p,
                         uint32_t q) {}
void vkCmdResetQueryPool(VkCommandBuffer c, VkQueryPool p, uint32_t first, uint32_t count) {}
VkResult vkCreateQueryPool(VkDevice d, const VkQueryPoolCreateInfo *info,
                           const VkAllocationCallbacks *a, VkQueryPool *out) {
  *out = WVK_HANDLE(VkQueryPool, WVK_NEW(int));
  return VK_SUCCESS;
}
void vkDestroyQueryPool(VkDevice d, VkQueryPool p, const VkAllocationCallbacks *a) {
  free(WVK_PTR(int, p));
}
VkResult vkGetQueryPoolResults(VkDevice d, VkQueryPool p, uint32_t first, uint32_t count,
                               size_t size, void *data, VkDeviceSize stride,
                               VkQueryResultFlags flags) {
  memset(data, 0, size);
  return VK_NOT_READY;
}

// Synchronization. Every submit is complete once wvk_js_wait returns, so timeline semaphores
// only need to remember what they'll be signaled to.

VkResult vkCreateSemaphore(VkDevice d, const VkSemaphoreCreateInfo *info,
                           const VkAllocationCallbacks *a, VkSemaphore *out) {
  wvk_semaphore_t *s = WVK_NEW(wvk_semaphore_t);
  for(const VkBaseInStructure *n = info->pNext; n; n = n->pNext)
    if(n->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO)
      s->value = s->pending = ((const VkSemaphoreTypeCreateInfo *)n)->initialValue;
  *out = WVK_HANDLE(VkSemaphore, s);
  return VK_SUCCESS;
}

void vkDestroySemaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks *a) {
  free(WVK_PTR(wvk_semaphore_t, s));
}

static wvk_semaphore_t **signaled;
static uint32_t num_signaled, cap_signaled;

static void _complete(void) {
  _wait();
  for(uint32_t i = 0; i < num_signaled; i++) signaled[i]->value = signaled[i]->pending;
  num_signaled = 0;
}

VkResult vkWaitSemaphores(VkDevice d, const VkSemaphoreWaitInfo *info, uint64_t timeout) {
  _complete();
  return VK_SUCCESS;
}

VkResult vkGetSemaphoreCounterValue(VkDevice d, VkSemaphore semaphore, uint64_t *value) {
  wvk_semaphore_t *s = WVK_PTR(wvk_semaphore_t, semaphore);
  if(s->pending != s->value) _complete();
  *value = s->value;
  return VK_SUCCESS;
}

static void _readback(wvk_buffer_t *b, VkDeviceSize offset, VkDeviceSize size) {
  if(!b->memory || !b->memory->shadow) return;
  if(b->snapshot) {
    if(num_readbacks == cap_readbacks) {
      cap_readbacks = cap_readbacks ? 2 * cap_readbacks : 16;
      readbacks = realloc(readbacks, cap_readbacks * sizeof(*readbacks));
    }
    readbacks[num_readbacks++] = (wvk_readback_t){b, offset, size};
  }
  _op(7);
  _op(b->gpu);
  _op((uint32_t)offset);
  _op((uint32_t)size);
  _op((uint32_t)(uintptr_t)(b->memory->shadow + b->offset + offset));
}

// The whole array texture for descriptor arrays, otherwise the image's own texture or layer.
static int _binding_texture(const wvk_descriptor_t *dsc) {
  const wvk_image_t *img = dsc->view->image;
  return dsc->count > 1 && img->array ? img->array : img->gpu;
}

static void _flatten(const wvk_cmd_t *cmd) {
  switch(cmd->type) {
    case WVK_CMD_DISPATCH: {
      if(!cmd->pipeline) return;
      // First, see whether the results are still there
      const wvk_shader_t *shader = cmd->pipeline->shader;
      uint64_t identity = 0xcbf29ce484222325ull, key = identity;
      WVK_MIX(identity, shader->gpu);
      WVK_MIX(key, cmd->groups[0]); WVK_MIX(key, cmd->groups[1]); WVK_MIX(key, cmd->groups[2]);
      wvk_image_t *wimg[16];
      wvk_buffer_t *wbuf[16];
      uint32_t nimg = 0, nbuf = 0;
      int memoizable = 1;
      for(uint32_t set = 0; set < 4; set++) {
        wvk_dset_t *s = cmd->dsets[set];
        if(!s) continue;
        for(uint32_t b = 0; b < s->max_binding; b++) {
          const wvk_descriptor_t *dsc = s->descriptors + b;
          const int writes = b < 32 && (shader->writes[set] & (1u << b));
          WVK_MIX(identity, set); WVK_MIX(identity, b);
          if(dsc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
             dsc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            wvk_buffer_t *buf = dsc->buffer;
            if(!buf || !buf->gpu) continue;
            WVK_MIX(identity, buf->serial); WVK_MIX(identity, dsc->offset); WVK_MIX(identity, dsc->range);
            const VkDeviceSize range =
                dsc->range == VK_WHOLE_SIZE ? buf->size - dsc->offset : dsc->range;
            if(writes) {
              if(nbuf < 16) wbuf[nbuf++] = buf;
              else memoizable = 0;
            } else if(buf->memory && buf->memory->shadow) {
              // uniforms, which the host writes every run, so their bytes are what counts
              key = _fnv(key, buf->memory->shadow + buf->offset + dsc->offset, range);
            } else {
              WVK_MIX(key, buf->version);
            }
          } else if(dsc->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
                    dsc->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            if(!dsc->view) continue;
            wvk_image_t *img = dsc->view->image;
            WVK_MIX(identity, img->serial); WVK_MIX(identity, dsc->count);
            if(writes) {
              // which layers an array binding writes isn't ours to know
              if(dsc->count > 1 || nimg == 16) memoizable = 0;
              else wimg[nimg++] = img;
            } else {
              WVK_MIX(key, dsc->count > 1 ? wvk_array_epoch : img->version);
            }
          }
        }
      }
      if(cmd->push_size) key = _fnv(key, cmd->push, cmd->push_size);
      wvk_memo_t *memo = _memo(identity);
      if(memoizable && memo->key == key && memo->num_written == nimg + nbuf) {
        int same = 1;
        for(uint32_t i = 0; i < nimg; i++) same &= memo->written[i] == wimg[i]->version;
        for(uint32_t i = 0; i < nbuf; i++) same &= memo->written[nimg + i] == wbuf[i]->version;
        if(same) {
          wvk_skipped++;
          return;
        }
      }
      // Run it, and remember what it left
      for(uint32_t i = 0; i < nimg; i++) _wrote_image(wimg[i]);
      for(uint32_t i = 0; i < nbuf; i++) _wrote_buffer(wbuf[i]);
      memo = _memo(identity);
      memo->key = memoizable ? key : 0;
      memo->num_written = nimg + nbuf;
      for(uint32_t i = 0; i < nimg; i++) memo->written[i] = wimg[i]->version;
      for(uint32_t i = 0; i < nbuf; i++) memo->written[nimg + i] = wbuf[i]->version;
      if(!memoizable) {
        // an array binding wrote layers we can't name
        wvk_array_epoch = ++wvk_clock;
        for(uint32_t set = 0; set < 4; set++) {
          wvk_dset_t *s = cmd->dsets[set];
          if(!s) continue;
          for(uint32_t b = 0; b < s->max_binding; b++) {
            const wvk_descriptor_t *dsc = s->descriptors + b;
            if(dsc->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && dsc->view) _wrote_image(dsc->view->image);
            if(dsc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && dsc->buffer) _wrote_buffer(dsc->buffer);
          }
        }
      }
      wvk_ran++;
      _op(1);
      _op(cmd->pipeline->shader->gpu);
      _op(cmd->groups[0]);
      _op(cmd->groups[1]);
      _op(cmd->groups[2]);
      const uint32_t count_at = num_ops;
      _op(0);
      uint32_t n = 0;
      // the binding numbers glsl_fix.pl gives every set
      for(uint32_t set = 0; set < 4; set++) {
        wvk_dset_t *s = cmd->dsets[set];
        if(!s) continue;
        for(uint32_t b = 0; b < s->max_binding; b++) {
          const wvk_descriptor_t *dsc = s->descriptors + b;
          const uint32_t wb = 2 * b;
          switch(dsc->type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
              if(!dsc->buffer || !dsc->buffer->gpu) continue;
              _use(dsc->buffer);
              const VkDeviceSize range =
                  dsc->range == VK_WHOLE_SIZE ? dsc->buffer->size - dsc->offset : dsc->range;
              _op(set), _op(wb), _op(dsc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? 0 : 1);
              _op(dsc->buffer->gpu), _op((uint32_t)dsc->offset), _op((uint32_t)range);
              n++;
              break;
            }
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
              if(!dsc->view || !dsc->view->image->gpu) continue;
              _op(set), _op(wb), _op(3), _op(_binding_texture(dsc)), _op(0), _op(0);
              n++;
              if(dsc->sampler) {
                _op(set), _op(wb + 1), _op(2), _op(dsc->sampler->gpu), _op(0), _op(0);
                n++;
              }
              break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
              if(!dsc->view || !dsc->view->image->gpu) continue;
              _op(set), _op(wb), _op(4), _op(_binding_texture(dsc)), _op(0);
              _op(dsc->view->image->gpu_format);
              n++;
              break;
            default:
              break;
          }
        }
      }
      if(cmd->push_size) {
        if(num_push == cap_push) {
          cap_push = cap_push ? 2 * cap_push : 64;
          push = realloc(push, cap_push * WVK_PUSH_SLOT);
        }
        memcpy(push + num_push * WVK_PUSH_SLOT, cmd->push, cmd->push_size);
        _op(3), _op(0), _op(5), _op(0), _op(num_push * WVK_PUSH_SLOT), _op(WVK_PUSH_SLOT);
        num_push++;
        n++;
      }
      ops[count_at] = n;
      break;
    }
    case WVK_CMD_COPY_BUFFER:
      _use(cmd->buffer);
      // a partial copy leaves the rest as the host wrote it
      _use(cmd->dst_buffer);
      _wrote_buffer(cmd->dst_buffer);
      _op(2);
      _op(cmd->buffer->gpu);
      _op((uint32_t)cmd->offset);
      _op(cmd->dst_buffer->gpu);
      _op((uint32_t)cmd->dst_offset);
      _op((uint32_t)cmd->size);
      _readback(cmd->dst_buffer, cmd->dst_offset, cmd->size);
      break;
    case WVK_CMD_BUFFER_TO_IMAGE:
    case WVK_CMD_IMAGE_TO_BUFFER:
      _use(cmd->buffer);
      if(cmd->type == WVK_CMD_BUFFER_TO_IMAGE) {
        // Sources upload on every run that reallocates, such as the raw, mostly unchanged. The
        // copy is skipped like a dispatch when the image still holds what it copied last time.
        uint64_t identity = 0x84222325cbf29ce4ull, key = 0;
        WVK_MIX(identity, cmd->buffer->serial); WVK_MIX(identity, cmd->image->serial);
        WVK_MIX(identity, cmd->offset); WVK_MIX(identity, cmd->row_texels);
        WVK_MIX(identity, cmd->wd); WVK_MIX(identity, cmd->ht);
        WVK_MIX(key, cmd->buffer->version);
        wvk_memo_t *memo = _memo(identity);
        if(memo->key == key && memo->num_written == 1 && memo->written[0] == cmd->image->version) {
          wvk_skipped++;
          break;
        }
        _wrote_image(cmd->image);
        memo->key = key;
        memo->num_written = 1;
        memo->written[0] = cmd->image->version;
      } else {
        _wrote_buffer(cmd->buffer);
      }
      _op(cmd->type == WVK_CMD_BUFFER_TO_IMAGE ? 3 : 4);
      _op(cmd->buffer->gpu);
      _op((uint32_t)cmd->offset);
      _op(cmd->row_texels);
      _op(cmd->image->gpu);
      _op(cmd->wd);
      _op(cmd->ht);
      _op(cmd->image->pack);
      _op(cmd->image->gpu_format);
      if(cmd->type == WVK_CMD_IMAGE_TO_BUFFER)
        _readback(cmd->buffer, cmd->offset,
                  (VkDeviceSize)cmd->row_texels * cmd->ht * cmd->image->texel_bytes);
      break;
    case WVK_CMD_CLEAR_IMAGE: {
      _wrote_image(cmd->image);
      _op(5);
      _op(cmd->image->gpu);
      _op(cmd->image->gpu_format);
      uint32_t bits[4];
      memcpy(bits, cmd->color, sizeof(bits));
      for(int k = 0; k < 4; k++) _op(bits[k]);
      break;
    }
    case WVK_CMD_FILL_BUFFER:
      _use(cmd->buffer);
      _wrote_buffer(cmd->buffer);
      _op(6);
      _op(cmd->buffer->gpu);
      _op((uint32_t)cmd->offset);
      _op((uint32_t)cmd->size);
      break;
  }
}

VkResult vkQueueSubmit(VkQueue q, uint32_t count, const VkSubmitInfo *submits, VkFence fence) {
  _pool_trim();
  num_ops = 0;
  num_push = 0;
  // filled in below once we know the push constants
  _op(8), _op(0), _op(0);
  for(uint32_t i = 0; i < count; i++) {
    const VkSubmitInfo *s = submits + i;
    for(uint32_t c = 0; c < s->commandBufferCount; c++) {
      VkCommandBuffer cb = s->pCommandBuffers[c];
      for(uint32_t k = 0; k < cb->num_cmds; k++) _flatten(cb->cmds + k);
    }
    const VkTimelineSemaphoreSubmitInfo *timeline = NULL;
    for(const VkBaseInStructure *n = s->pNext; n; n = n->pNext)
      if(n->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) timeline = (const void *)n;
    for(uint32_t k = 0; k < s->signalSemaphoreCount; k++) {
      wvk_semaphore_t *sem = WVK_PTR(wvk_semaphore_t, s->pSignalSemaphores[k]);
      sem->pending = timeline && k < timeline->signalSemaphoreValueCount
                         ? timeline->pSignalSemaphoreValues[k]
                         : sem->pending + 1;
      if(num_signaled == cap_signaled) {
        cap_signaled = cap_signaled ? 2 * cap_signaled : 16;
        signaled = realloc(signaled, cap_signaled * sizeof(wvk_semaphore_t *));
      }
      signaled[num_signaled++] = sem;
    }
  }
  ops[1] = (uint32_t)(uintptr_t)push;
  ops[2] = num_push * WVK_PUSH_SLOT;
  if(num_ops > 3) wvk_js_submit(ops, num_ops);
  return VK_SUCCESS;
}

// Extensions (ray tracing, push descriptors) and graphics nodes aren't available.

PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance i, const char *name) { return NULL; }
PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice d, const char *name) { return NULL; }

void vkCmdBeginRendering(VkCommandBuffer c, const VkRenderingInfo *info) {
  fprintf(stderr, "[wvk] graphics nodes are not supported\n");
}
void vkCmdEndRendering(VkCommandBuffer c) {}
void vkCmdDraw(VkCommandBuffer c, uint32_t vertices, uint32_t instances, uint32_t first_vertex,
               uint32_t first_instance) {}
