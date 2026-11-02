// A mask painted with brush strokes, for limiting a module's effect through blend. vkdt's draw
// module does this with a graphics pipeline, which our WebGPU backend lacks, so this one is a
// source whose pixels the worker rasterises on a 2d canvas.
//
// strokes holds the vertex count and then four floats per vertex: x and y as fractions of the
// image in the sensor's orientation, the brush radius as a fraction of the image's longer side,
// and flags, where 1 starts a new stroke and 2 erases.
#include "modules/api.h"

#include <emscripten.h>

// Fills out with the mask at wd by ht. See rasterise in js/brush.ts.
EM_JS(void, wasm_js_rasterise_brush, (const float *strokes, int wd, int ht, float hardness,
                                     float opacity, float *out), {
  const count = HEAPF32[strokes >> 2];
  const vertices = HEAPF32.subarray((strokes >> 2) + 1, (strokes >> 2) + 1 + 4 * count);
  const mask = self.wasmRasteriseBrush(vertices, wd, ht, hardness, opacity);
  HEAPF32.set(mask, out >> 2);
});

void modify_roi_out(dt_graph_t *graph, dt_module_t *module) {
  // we rasterise at whatever size blend asks for
  module->connector[0].roi = (dt_roi_t){.full_wd = 1024, .full_ht = 1024, .marker = s_roi_mark_dontcare};
}

dt_graph_run_t check_params(dt_module_t *module, uint32_t parid, uint32_t num, void *oldval) {
  module->flags |= s_module_request_read_source;
  return s_graph_run_record_cmd_buf;
}

int read_source(dt_module_t *module, void *mapped, dt_read_source_params_t *p) {
  const float *strokes = dt_module_param_float(module, dt_module_get_param(module->so, dt_token("strokes")));
  const float hardness = dt_module_param_float(module, dt_module_get_param(module->so, dt_token("hardness")))[0];
  const float opacity = dt_module_param_float(module, dt_module_get_param(module->so, dt_token("opacity")))[0];
  wasm_js_rasterise_brush(strokes, module->connector[0].roi.wd, module->connector[0].roi.ht, hardness,
                         opacity, mapped);
  module->flags = 0;
  return 0;
}
