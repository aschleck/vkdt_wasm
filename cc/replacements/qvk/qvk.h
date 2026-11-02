#pragma once

// The device state and helpers vkdt's core and modules use, which //cc/webgpu implements on
// top of WebGPU.

#include <vulkan/vulkan.h>

#include "core/core.h"
#include "core/log.h"
#include "core/threads.h"
#include "pipe/token.h"

#define LENGTH(a) (sizeof(a) / sizeof((a)[0]))

// Each of these logs a failed Vulkan call. The R variants also return its result, and the L
// variants hold mutex, which may be null, around it.
#define QVK_CALL_(mutex, on_error, ...) \
  do { \
    threads_mutex_t *qvk_mutex_ = (mutex); \
    if(qvk_mutex_) threads_mutex_lock(qvk_mutex_); \
    const VkResult qvk_res_ = __VA_ARGS__; \
    if(qvk_mutex_) threads_mutex_unlock(qvk_mutex_); \
    if(qvk_res_ != VK_SUCCESS) { \
      dt_log(s_log_qvk, "error %s executing %s!", qvk_result_to_string(qvk_res_), #__VA_ARGS__); \
      on_error; \
    } \
  } while(0)
#define QVK(...) QVK_CALL_(NULL, (void)0, __VA_ARGS__)
#define QVKR(...) QVK_CALL_(NULL, return qvk_res_, __VA_ARGS__)
#define QVKL(mutex, ...) QVK_CALL_(mutex, (void)0, __VA_ARGS__)
#define QVKLR(mutex, ...) QVK_CALL_(mutex, return qvk_res_, __VA_ARGS__)

#define QVK_LOAD(fn) PFN_##fn q##fn = (PFN_##fn)vkGetInstanceProcAddr(qvk.instance, #fn)

// WebGPU orders the passes in a queue itself, so vulkan_webgpu.c's vkCmdPipelineBarrier does
// nothing and these don't call it.
#define BARRIER_COMPUTE_BUFFER(buf) do {} while(0)
#define BARRIER_IMG_LAYOUT(img, old_layout, new_layout, mip_levels) do {} while(0)

typedef enum qvk_queue_name_t
{
  s_queue_graphics,
  s_queue_compute,
  s_queue_cnt,
}
qvk_queue_name_t;

typedef struct qvk_queue_t
{
  threads_mutex_t mutex;
  VkQueue         queue;
  int32_t         family;
}
qvk_queue_t;

typedef struct qvk_t
{
  VkInstance                       instance;
  VkPhysicalDevice                 physical_device;
  VkPhysicalDeviceMemoryProperties mem_properties;
  VkDevice                         device;
  char                             device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];

  // the queue each qvk_queue_name_t runs on
  int                              qid[s_queue_cnt];
  qvk_queue_t                      queue[s_queue_cnt];
  uint32_t                         queue_family_graphics;
  uint32_t                         queue_family_compute;

  VkSampler                        tex_sampler;
  VkSampler                        tex_sampler_dspy;
  VkSampler                        tex_sampler_nearest;
  VkSampler                        tex_sampler_yuv;
  VkSamplerYcbcrConversion         yuv_conversion;

  float                            ticks_to_nanoseconds;
  uint64_t                         uniform_alignment;
  uint64_t                         max_allocation_size;
  uint64_t                         raytracing_acc_min_align;
  uint32_t                         subgroup_size;
  VkSubgroupFeatureFlags           subgroup_ops;

  int                              raytracing_supported;
  int                              float_atomics_supported;
  int                              subgroup_size_control_supported;
}
qvk_t;

#ifndef VKDT_DSO_BUILD
VKDT_API extern qvk_t qvk;
#endif

// The extension functions vkdt calls through q-prefixed pointers.
#define _VK_EXTENSION_LIST _VK_EXTENSION_DO(vkCmdPushDescriptorSetKHR)
#define _VK_EXTENSION_DO(a) extern PFN_##a q##a;
_VK_EXTENSION_LIST
#undef _VK_EXTENSION_DO

/** Sets up qvk. vkdt picks among Vulkan devices with the arguments, which we ignore. */
VkResult qvk_init(
    const char *preferred_device_name,
    int preferred_device_id,
    int window,
    int enable_hdr_wsi,
    int allow_hdr);
VkResult qvk_cleanup(void);

// indices into qvk.mem_properties.memoryTypes
uint32_t qvk_memory_get_device(void);
uint32_t qvk_memory_get_staging(void);
uint32_t qvk_memory_get_uniform(void);

const char *qvk_result_to_string(VkResult result);
