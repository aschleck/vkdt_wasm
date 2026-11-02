// A sink that hands finished 8-bit sRGB images to engine.c instead of writing files. The instance
// name tells outputs apart, such as main for the image and hist for its histogram.
#include "modules/api.h"

void wasm_vkdt_output(dt_token_t inst, const void *rgba, int wd, int ht);

void write_sink(dt_module_t *module, void *buf, dt_write_sink_params_t *p) {
  wasm_vkdt_output(module->inst, buf, module->connector[0].roi.wd, module->connector[0].roi.ht);
}
