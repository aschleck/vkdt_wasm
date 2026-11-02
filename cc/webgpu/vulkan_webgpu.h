#pragma once
// A Vulkan subset implemented on WebGPU, covering what vkdt's compute pipeline uses. See
// library_wvk.js for the WebGPU side.

#include <stdint.h>
#include <vulkan/vulkan.h>

// The texture formats we back Vulkan images with. WebGPU only guarantees storage support for a
// few formats, so narrow ones are widened. Keep in sync with WVK.FORMATS in library_wvk.js.
typedef enum wvk_gpu_format_t {
  WVK_R32F = 0,
  WVK_RG32F = 1,
  WVK_RGBA16F = 2,
  WVK_RGBA32F = 3,
  WVK_RGBA8 = 4,
  WVK_R32UI = 5,
  WVK_RGBA32UI = 6,
} wvk_gpu_format_t;

// How texels are packed in buffers, by Vulkan format. Keep in sync with WVK.PACKS.
typedef enum wvk_pack_t {
  WVK_PACK_R16_UNORM = 0,
  WVK_PACK_R16_SFLOAT = 1,
  WVK_PACK_RG16_SFLOAT = 2,
  WVK_PACK_RGBA16_SFLOAT = 3,
  WVK_PACK_R32_SFLOAT = 4,
  WVK_PACK_RG32_SFLOAT = 5,
  WVK_PACK_RGBA32_SFLOAT = 6,
  WVK_PACK_RGBA8_UNORM = 7,
  WVK_PACK_R8_UNORM = 8,
  WVK_PACK_RGBA16_UNORM = 9,
} wvk_pack_t;

// Memory type indices we report. Keep in sync with wvk's vkGetPhysicalDeviceMemoryProperties.
enum {
  WVK_MEMORY_DEVICE = 0,
  WVK_MEMORY_HOST = 1,
};

/** Acquires the WebGPU device. Blocks until it's ready. */
VkResult wvk_init(void);

/** How many dispatches ran and how many were skipped since the last call. */
void wvk_dispatch_stats(uint32_t *ran, uint32_t *skipped);
