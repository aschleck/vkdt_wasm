// Implements qvk, vkdt's Vulkan device state, as our WebGPU-backed device. See
// //cc/replacements:qvk/qvk.h.

#include <stdio.h>
#include <string.h>

#include "qvk/qvk.h"
#include "vulkan_webgpu.h"

qvk_t qvk;

#define _VK_EXTENSION_DO(a) PFN_##a q##a;
_VK_EXTENSION_LIST
#undef _VK_EXTENSION_DO

VkResult qvk_init(const char *preferred_device_name, int preferred_device_id, int window,
                  int enable_hdr_wsi, int allow_hdr) {
  memset(&qvk, 0, sizeof(qvk));
  VkResult res = wvk_init();
  if(res != VK_SUCCESS) return res;

  vkCreateInstance(NULL, NULL, &qvk.instance);
  uint32_t count = 1;
  vkEnumeratePhysicalDevices(qvk.instance, &count, &qvk.physical_device);
  vkCreateDevice(qvk.physical_device, NULL, NULL, &qvk.device);
  vkGetPhysicalDeviceMemoryProperties(qvk.physical_device, &qvk.mem_properties);
  for(int i = 0; i < s_queue_cnt; i++) {
    qvk.qid[i] = 0;
    vkGetDeviceQueue(qvk.device, 0, 0, &qvk.queue[i].queue);
  }

  snprintf(qvk.device_name, sizeof(qvk.device_name), "WebGPU");
  // WebGPU's minUniformBufferOffsetAlignment
  qvk.uniform_alignment = 256;
  qvk.max_allocation_size = (uint64_t)2 << 30;
  qvk.ticks_to_nanoseconds = 1.0f;
  // Shaders with subgroup operations aren't transpiled, so don't advertise any.
  qvk.subgroup_size = 32;
  qvk.subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT;

  const VkSamplerCreateInfo linear = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
  };
  const VkSamplerCreateInfo nearest = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
  };
  vkCreateSampler(qvk.device, &linear, NULL, &qvk.tex_sampler);
  vkCreateSampler(qvk.device, &linear, NULL, &qvk.tex_sampler_dspy);
  vkCreateSampler(qvk.device, &nearest, NULL, &qvk.tex_sampler_nearest);
  return VK_SUCCESS;
}

VkResult qvk_cleanup(void) { return VK_SUCCESS; }

uint32_t qvk_memory_get_device(void) { return WVK_MEMORY_DEVICE; }

uint32_t qvk_memory_get_staging(void) { return WVK_MEMORY_HOST; }

uint32_t qvk_memory_get_uniform(void) { return WVK_MEMORY_HOST; }

const char *qvk_result_to_string(VkResult result) {
  // only for logging, and vkdt logs from more than one thread
  static _Thread_local char buf[32];
  snprintf(buf, sizeof(buf), "VkResult %d", result);
  return buf;
}
