// The C API the browser calls. An image is opened the way vkdt's command line exporter runs it
// (see src/cli/main.c), with an output module that hands us pixels instead of writing a file, and
// the graph stays alive afterwards so parameter edits only rerun what vkdt's GUI would.

#include <emscripten.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/log.h"
#include "pipe/global.h"
#include "pipe/graph-io.h"
#include "pipe/graph.h"
#include "pipe/module.h"
#include "pipe/modules/api.h"
#include "pipe/modules/matrices.h"
#include "qvk/qvk.h"
#include "vulkan_webgpu.h"

// One image per o-wasm instance, see modules/o-wasm/main.c.
#define WASM_OUTPUTS 4
static struct {
  dt_token_t inst;
  uint8_t *pixels;
  int wd, ht;
} outputs[WASM_OUTPUTS];

int wasm_noise_profile(const char *path, float *a, float *b);

// Two graphs of the open image, one that fits the screen and one at full resolution for zooming
// past it, so that zooming switches between them rather than rebuilding either.
#define WASM_SLOTS 2
// at least as many as a graph has modules
#define WASM_MODULES 128
static struct {
  dt_graph_t graph;
  int open;
  // demosaics at half size, and the method each module asked for before, see _halve
  int halved;
  int32_t method[WASM_MODULES];
  // what it was built from, see wasm_open
  char *path, *cfg;
  int max_wd, max_ht;
} slots[WASM_SLOTS];
// the slot wasm_open last opened, which the other calls work on
static int cur;
static dt_graph_t *graph = &slots[0].graph;
#define graph_open (slots[cur].open)
// wasm_state_json's result, its length and its allocation
static char *json;
static size_t json_len, json_cap;

static int _output(dt_token_t inst) {
  for(int i = 0; i < WASM_OUTPUTS; i++)
    if(outputs[i].inst == inst) return i;
  for(int i = 0; i < WASM_OUTPUTS; i++)
    if(!outputs[i].inst) {
      outputs[i].inst = inst;
      return i;
    }
  return 0;
}

void wasm_vkdt_output(dt_token_t inst, const void *rgba, int wd, int ht) {
  const int i = _output(inst);
  free(outputs[i].pixels);
  outputs[i].pixels = malloc((size_t)wd * ht * 4);
  memcpy(outputs[i].pixels, rgba, (size_t)wd * ht * 4);
  outputs[i].wd = wd;
  outputs[i].ht = ht;
}

EMSCRIPTEN_KEEPALIVE
int wasm_init(void) {
  dt_log_init(s_log_err);
  if(dt_pipe_global_init()) return 1;
  if(qvk_init(0, -1, 0, 0, 0)) return 2;
  return 0;
}

// demosaic's method that bins each 2x2 block into a pixel
#define HALF_SIZE 2

/**
 * Switches every demosaic in g to half size when the output fits in half the raw's size, keeping
 * the methods they had in method by module. Returns whether it did.
 *
 * This is the only way a graph renders smaller than the raw. i-raw marks its size hard_fwd and
 * demosaic passes that on, which beats the hard_bck size o-wasm asks for (see dt_roi_stronger), so
 * everything after demosaic runs at its size and o-wasm gets that too.
 */
static int _halve(dt_graph_t *g, int32_t *method, int max_wd, int max_ht) {
  const int raw = dt_module_get(g, dt_token("i-raw"), dt_token("main"));
  // only works out sizes, and i-raw keeps the raw it reads for the run after
  if(max_wd <= 0 || max_ht <= 0 || raw < 0 || dt_graph_run(g, s_graph_run_roi)) return 0;
  const dt_roi_t *roi = &g->module[raw].connector[0].roi;
  if(2 * max_wd > roi->full_wd || 2 * max_ht > roi->full_ht) return 0;
  for(int m = 0; m < g->num_modules && m < WASM_MODULES; m++) {
    dt_module_t *mod = g->module + m;
    if(mod->name != dt_token("demosaic")) continue;
    int32_t *p =
        (int32_t *)dt_module_param_int(mod, dt_module_get_param(mod->so, dt_token("method")));
    if(method) method[m] = p[0];
    p[0] = HALF_SIZE;
  }
  return 1;
}

// Reads each of the newline separated config lines in lines that start with prefix, or all of
// them without one.
static void _read_lines(dt_graph_t *g, const char *lines, const char *prefix) {
  char *copy = strdup(lines ? lines : "");
  char *save;
  for(char *l = strtok_r(copy, "\n", &save); l; l = strtok_r(NULL, "\n", &save)) {
    if(prefix && strncmp(l, prefix, strlen(prefix))) continue;
    if(dt_graph_read_config_line(g, l)) dt_log(s_log_pipe | s_log_err, "bad config line '%s'", l);
  }
  free(copy);
}

/**
 * Builds a graph from graph_cfg, as vkdt's dt_graph_export does for a graph that ends in a sink
 * named main, and runs it. extra holds config lines to apply over it, and disabled the
 * module:inst of modules to bypass, each newline separated.
 *
 * With reuse, g is already built from the same graph_cfg and keeps its modules, so i-raw keeps
 * the decoded raw and wvk keeps every result the change doesn't reach. Its params go back to
 * where building would have left them before extra applies.
 */
static VkResult _build(dt_graph_t *g, int reuse, const char *path, int max_wd, int max_ht,
                       const char *graph_cfg, const char *extra, const char *disabled,
                       int *halved, int32_t *method) {
  if(reuse) {
    for(int m = 0; m < g->num_modules; m++) dt_module_reset_params(g->module + m);
    _read_lines(g, graph_cfg, "param:");
  } else {
    // first, so that the caller can always clean up
    dt_graph_init(g, s_queue_compute);
    // vkdt reads graphs from files
    FILE *f = fopen("/graph.cfg", "wb");
    if(!f) return VK_ERROR_INITIALIZATION_FAILED;
    fputs(graph_cfg, f);
    fclose(f);
    if(dt_graph_read_config_ascii(g, "/graph.cfg")) return VK_INCOMPLETE;
    // resources such as the raw resolve against the image's directory
    dt_graph_set_searchpath(g, path);
  }
  const int raw = dt_module_get(g, dt_token("i-raw"), dt_token("main"));
  const char *slash = strrchr(path, '/');
  if(raw < 0 || dt_module_set_param_string(g->module + raw, dt_token("filename"),
                                           slash ? slash + 1 : path))
    return VK_INCOMPLETE;
  _read_lines(g, extra, NULL);
  for(int m = 0; m < g->num_modules; m++) g->module[m].disabled = 0;
  char *off = strdup(disabled ? disabled : "");
  char *save;
  for(char *l = strtok_r(off, "\n", &save); l; l = strtok_r(NULL, "\n", &save)) {
    char *inst = strchr(l, ':');
    if(!inst) continue;
    *inst++ = 0;
    const int m = dt_module_get(g, dt_token(l), dt_token(inst));
    if(m >= 0) g->module[m].disabled = 1;
  }
  free(off);
  for(int m = 0; m < g->num_modules; m++) dt_module_keyframe_post_update(g->module + m);
  const int out = dt_module_get(g, dt_token("o-wasm"), dt_token("main"));
  if(out < 0) return VK_INCOMPLETE;
  g->module[out].connector[0].max_wd = max_wd;
  g->module[out].connector[0].max_ht = max_ht;
  g->frame = 0;
  if(halved) *halved = _halve(g, method, max_wd, max_ht);
  return dt_graph_run(g, s_graph_run_all);
}

// Prepends the raw's noise profile to extra, since without one vkdt's denoise has nothing to go
// on (see noise.cc). Returns a string to free, or NULL when extra is fine as it is.
static char *_with_noise(const char *path, const char **extra) {
  static char profiled[1024];
  static float noise_a, noise_b;
  if(strstr(*extra, "param:i-raw:main:noise")) return NULL;
  if(strcmp(profiled, path) && !wasm_noise_profile(path, &noise_a, &noise_b))
    snprintf(profiled, sizeof(profiled), "%s", path);
  char *lines = NULL;
  if(!strcmp(profiled, path) &&
     asprintf(&lines, "param:i-raw:main:noise a:%g\nparam:i-raw:main:noise b:%g\n%s", noise_a,
              noise_b, *extra) >= 0)
    *extra = lines;
  return lines;
}

static int _same(const char *a, const char *b) { return a && b && !strcmp(a, b); }

/**
 * Opens the raw at path with graph_cfg, a vkdt graph whose input is i-raw:main and whose output
 * is o-wasm:main, and renders it to fit within max_wd by max_ht, 0 for full size. extra holds
 * newline separated vkdt config lines, such as "param:colour:01:exposure:1.0", applied over it,
 * and disabled the newline separated module:inst of modules to bypass. The graph stays open in
 * slot for wasm_set_params and the rest, which work on the slot last opened.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_open(int slot, const char *path, int max_wd, int max_ht, const char *graph_cfg,
             const char *extra, const char *disabled) {
  if(slot < 0 || slot >= WASM_SLOTS) return VK_ERROR_INITIALIZATION_FAILED;
  for(int i = 0; i < WASM_SLOTS; i++) {
    // a graph of another image only holds memory
    if(i == slot || !slots[i].open || _same(slots[i].path, path)) continue;
    dt_graph_cleanup(&slots[i].graph);
    slots[i].open = 0;
  }
  cur = slot;
  graph = &slots[slot].graph;
  const int reuse = slots[slot].open && _same(slots[slot].path, path)
      && _same(slots[slot].cfg, graph_cfg) && slots[slot].max_wd == max_wd
      && slots[slot].max_ht == max_ht;
  if(slots[slot].open && !reuse) dt_graph_cleanup(graph);
  slots[slot].open = 0;
  char *lines = _with_noise(path, &extra);
  const VkResult res = _build(graph, reuse, path, max_wd, max_ht, graph_cfg, extra, disabled,
                              &slots[slot].halved, slots[slot].method);
  free(lines);
  if(res != VK_SUCCESS) {
    dt_graph_cleanup(graph);
    return res;
  }
  slots[slot].open = 1;
  free(slots[slot].path);
  free(slots[slot].cfg);
  slots[slot].path = strdup(path);
  slots[slot].cfg = strdup(graph_cfg);
  slots[slot].max_wd = max_wd;
  slots[slot].max_ht = max_ht;
  return res;
}

/**
 * Renders the raw as wasm_open does but into a graph of its own, leaving the open ones alone. 0
 * for max_wd and max_ht renders at full size.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_render(const char *path, int max_wd, int max_ht, const char *graph_cfg,
               const char *extra, const char *disabled) {
  char *lines = _with_noise(path, &extra);
  dt_graph_t g;
  int halved;
  const VkResult res =
      _build(&g, 0, path, max_wd, max_ht, graph_cfg, extra, disabled, &halved, NULL);
  dt_graph_cleanup(&g);
  free(lines);
  return res;
}

// Sets a parameter on the open graph from comma separated values and adds the run flags vkdt's
// GUI would use for the change (see src/gui/dragkey.h) to flags. Unknown parameters are ignored.
static void _set_param(const char *module, const char *inst, const char *name, const char *values,
                      dt_graph_run_t *flags) {
  const int m = dt_module_get(graph, dt_token(module), dt_token(inst));
  if(m < 0) return;
  dt_module_t *mod = graph->module + m;
  const int p = dt_module_get_param(mod->so, dt_token(name));
  if(p < 0) return;
  const dt_ui_param_t *param = mod->so->param[p];

  float oldval = 0.0f;
  char *copy = strdup(values);
  int i = 0;
  for(char *v = strtok(copy, ","); v && i < param->cnt; v = strtok(NULL, ","), i++) {
    if(param->type == dt_token("float")) {
      float *f = (float *)(mod->param + param->offset);
      if(i == 0) oldval = f[0];
      f[i] = strtof(v, NULL);
    } else if(param->type == dt_token("int")) {
      int32_t *n = (int32_t *)(mod->param + param->offset);
      if(i == 0) oldval = n[0];
      n[i] = atoi(v);
    }
  }
  free(copy);

  *flags |= s_graph_run_record_cmd_buf;
  if(mod->so->check_params) *flags |= mod->so->check_params(mod, p, 0, &oldval);
  // These change the size of the image, so vkdt's GUI reruns everything for them. See
  // render_darkroom.h.
  const dt_token_t w = param->widget.type;
  if(w == dt_token("crop") || w == dt_token("pers") || w == dt_token("straight"))
    *flags |= s_graph_run_all;
}

static int _run(dt_graph_run_t flags) {
  return dt_graph_run(graph, flags | s_graph_run_record_cmd_buf | s_graph_run_download_sink
                                  | s_graph_run_wait_done);
}

/**
 * Sets several parameters and reruns the graph once. Each line of edits is
 * module:inst:name:values, with the values comma separated.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_set_params(const char *edits) {
  if(!graph_open) return 1;
  char *copy = strdup(edits);
  dt_graph_run_t flags = 0;
  char *save;
  for(char *line = strtok_r(copy, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
    char *fields[4] = {0};
    char *field_save;
    int n = 0;
    for(char *f = strtok_r(line, ":", &field_save); f && n < 4; f = strtok_r(NULL, ":", &field_save))
      fields[n++] = f;
    if(n < 4) continue;
    _set_param(fields[0], fields[1], fields[2], fields[3], &flags);
  }
  free(copy);
  if(slots[cur].halved) {
    // an edit to the method is the one to report, see wasm_state_json
    for(int m = 0; m < graph->num_modules && m < WASM_MODULES; m++) {
      dt_module_t *mod = graph->module + m;
      if(mod->name != dt_token("demosaic")) continue;
      int32_t *p =
          (int32_t *)dt_module_param_int(mod, dt_module_get_param(mod->so, dt_token("method")));
      if(p[0] != HALF_SIZE) slots[cur].method[m] = p[0];
      p[0] = HALF_SIZE;
    }
  }
  return _run(flags);
}

static void _json(const char *format, ...) {
  va_list args;
  va_start(args, format);
  char *s;
  if(vasprintf(&s, format, args) < 0) s = NULL;
  va_end(args);
  if(!s) return;
  const size_t n = strlen(s);
  if(json_len + n + 1 > json_cap) {
    json_cap = 2 * (json_len + n + 1);
    json = realloc(json, json_cap);
  }
  memcpy(json + json_len, s, n + 1);
  json_len += n;
  free(s);
}

static void _json_string(const char *s) {
  _json("\"");
  for(const char *c = s; *c; c++) {
    if(*c == '"' || *c == '\\') _json("\\%c", *c);
    else if((unsigned char)*c < 0x20) _json("\\u%04x", *c);
    else _json("%c", *c);
  }
  _json("\"");
}

/**
 * Describes every module in the open graph: its parameters, their widgets from params.ui, and
 * their current values.
 */
EMSCRIPTEN_KEEPALIVE
const char *wasm_state_json(void) {
  json_len = 0;
  _json("[");
  for(int m = 0; graph_open && m < graph->num_modules; m++) {
    dt_module_t *mod = graph->module + m;
    if(!mod->so || mod->name == 0) continue;
    _json("%s{\"module\":\"%" PRItkn "\",\"inst\":\"%" PRItkn "\",\"params\":[", m ? "," : "",
          dt_token_str(mod->name), dt_token_str(mod->inst));
    for(int p = 0; p < mod->so->num_params; p++) {
      const dt_ui_param_t *param = mod->so->param[p];
      _json("%s{\"name\":\"%" PRItkn "\",\"type\":\"%" PRItkn "\",\"cnt\":%d,", p ? "," : "",
            dt_token_str(param->name), dt_token_str(param->type), param->cnt);
      _json("\"widget\":\"%" PRItkn "\",\"min\":%g,\"max\":%g,", dt_token_str(param->widget.type),
            param->widget.min, param->widget.max);
      // the widget only applies while param grpid has value grp, and likewise for tabs, and shows
      // as many components as param cntid says
      _json("\"grpid\":%d,\"grp\":%d,\"tabid\":%d,\"tab\":%d,\"cntid\":%d,\"label\":",
            param->widget.grpid, param->widget.grp, param->widget.tabid, param->widget.tab,
            param->widget.cntid);
      _json_string(param->long_name ? param->long_name : "");
      _json(",\"options\":[");
      const dt_token_t w = param->widget.type;
      if((w == dt_token("combo") || w == dt_token("bitmask") || w == dt_token("btngrid")
          || w == dt_token("tabs")) && param->widget.data) {
        int k = 0;
        for(const char *o = param->widget.data; *o; o += strlen(o) + 1, k++) {
          if(k) _json(",");
          _json_string(o);
        }
      }
      _json("],\"value\":[");
      // Long arrays, such as i-brush's strokes or colour's rbmap, mostly end in unused zeros.
      int cnt = param->cnt;
      if(cnt > 16 && param->type != dt_token("string")) {
        const uint32_t *bits = (const uint32_t *)(mod->param + param->offset);
        while(cnt > 1 && !bits[cnt - 1]) cnt--;
      }
      // a halved graph's demosaic reports the method the look has, see _halve
      if(slots[cur].halved && mod->name == dt_token("demosaic") && param->name == dt_token("method")
         && m < WASM_MODULES) {
        _json("%d", slots[cur].method[m]);
        cnt = 0;
      }
      for(int i = 0; i < cnt; i++) {
        if(param->type == dt_token("float"))
          _json("%s%g", i ? "," : "", ((float *)(mod->param + param->offset))[i]);
        else if(param->type == dt_token("int"))
          _json("%s%d", i ? "," : "", ((int32_t *)(mod->param + param->offset))[i]);
      }
      _json("],\"default\":[");
      for(int i = 0; i < (param->cnt < 16 ? param->cnt : 16); i++) {
        if(param->type == dt_token("float")) _json("%s%g", i ? "," : "", param->val[i]);
        else if(param->type == dt_token("int")) _json("%s%d", i ? "," : "", param->vali[i]);
      }
      _json("]}");
    }
    _json("]}");
  }
  _json("]");
  return json;
}

/** The latest image from the output instance inst, such as main or hist. */
EMSCRIPTEN_KEEPALIVE
const uint8_t *wasm_output(const char *inst) { return outputs[_output(dt_token(inst))].pixels; }
EMSCRIPTEN_KEEPALIVE
int wasm_output_width(const char *inst) { return outputs[_output(dt_token(inst))].wd; }
EMSCRIPTEN_KEEPALIVE
int wasm_output_height(const char *inst) { return outputs[_output(dt_token(inst))].ht; }

/**
 * The size the open graph's main output would have at full resolution, as width in the low 16
 * bits and height in the high 16, or 0 when there's no open graph.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_full_size(void) {
  if(!graph_open) return 0;
  const int m = dt_module_get(graph, dt_token("o-wasm"), dt_token("main"));
  if(m < 0) return 0;
  const dt_roi_t *roi = &graph->module[m].connector[0].roi;
  // half size demosaic rounds up, so this can be a pixel more than the truth
  const int scale = slots[cur].halved ? 2 : 1;
  return ((roi->full_ht * scale) << 16) | (roi->full_wd * scale);
}

static void _mul3(const float m[9], const float v[3], float out[3]) {
  for(int j = 0; j < 3; j++) out[j] = m[3 * j] * v[0] + m[3 * j + 1] * v[1] + m[3 * j + 2] * v[2];
}

/**
 * Works out the white and exposure for colour:01 that balance white as pick:01's spot does, so
 * that the balance carries over to photos without that spot. out gets the white's red, green and
 * blue, then the exposure. Returns 0 on success.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_spot_white(float *out) {
  if(!graph_open) return 1;
  const int pm = dt_module_get(graph, dt_token("pick"), dt_token("01"));
  const int cm = dt_module_get(graph, dt_token("colour"), dt_token("01"));
  if(pm < 0 || cm < 0) return 2;
  dt_module_t *pick = graph->module + pm, *colour = graph->module + cm;

  // pick only writes what it measured into its params while grabbing
  int32_t *grab = (int32_t *)dt_module_param_int(pick, dt_module_get_param(pick->so, dt_token("grab")));
  *grab = 1;
  pick->flags |= s_module_request_write_sink;
  const int res = _run(s_graph_run_record_cmd_buf);
  *grab = 0;
  pick->flags = 0;
  if(res) return 3;

  // Spot white balance runs cat16(rgb, picked, mul) in colour's main-impl.glsl, where picked is
  // the spot in rec2020 with green at 1 and mul is colour's white with green at 1. CAT16 scales
  // cone responses by dst / src, with cones c(x) = M16 * rec2020_to_xyz * x, so the plain white
  // balance cat16(rgb, 1, w) matches when
  // c(w) / c(1) = c(mul) / c(picked)
  // => w = c^-1(c(1) * c(mul) / c(picked))
  // Colour normalises w to green 1, so the rest of w's green goes into exposure.
  const float *spot = dt_module_param_float(pick, dt_module_get_param(pick->so, dt_token("picked")));
  const float *white = dt_module_param_float(colour, dt_module_get_param(colour->so, dt_token("white")));
  const float exposure =
      dt_module_param_float(colour, dt_module_get_param(colour->so, dt_token("exposure")))[0];
  if(!(spot[1] > 0.0f) || !(white[1] > 0.0f)) return 4;

  const float m16[] = matrix_cat16_M, m16i[] = matrix_cat16_Mi;
  const float to_xyz[] = matrix_rec2020_to_xyz, from_xyz[] = matrix_xyz_to_rec2020;
  float picked[3], mul[3] = {white[0] / white[1], 1.0f, white[2] / white[1]}, one[3] = {1, 1, 1};
  // colour marks its own output as rec2020, so the camera's matrix is on its input
  const dt_image_params_t *input = dt_module_get_input_img_param(graph, colour, dt_token("input"));
  if(!input) return 6;
  _mul3(input->cam_to_rec2020, spot, picked);
  const float green = picked[1];
  if(!(green > 0.0f)) return 7;
  for(int k = 0; k < 3; k++) picked[k] /= green;

  float xyz[3], c_one[3], c_mul[3], c_picked[3], c_w[3], w[3];
  _mul3(to_xyz, one, xyz);
  _mul3(m16, xyz, c_one);
  _mul3(to_xyz, mul, xyz);
  _mul3(m16, xyz, c_mul);
  _mul3(to_xyz, picked, xyz);
  _mul3(m16, xyz, c_picked);
  for(int k = 0; k < 3; k++) c_w[k] = c_one[k] * c_mul[k] / c_picked[k];
  _mul3(m16i, c_w, xyz);
  _mul3(from_xyz, xyz, w);
  if(!(w[1] > 0.0f)) return 5;

  out[0] = w[0] / w[1];
  out[1] = 1.0f;
  out[2] = w[2] / w[1];
  out[3] = exposure + log2f(w[1]);
  return 0;
}

/**
 * Runs a module's ui_callback with arg, such as crop's auto crop with an aspect ratio. These
 * change params from the last run's state and don't rerun the graph.
 */
EMSCRIPTEN_KEEPALIVE
int wasm_ui_callback(const char *module, const char *inst, const char *param, float arg) {
  if(!graph_open) return 1;
  const int m = dt_module_get(graph, dt_token(module), dt_token(inst));
  if(m < 0 || !graph->module[m].so->ui_callback) return 2;
  graph->module[m].so->ui_callback(graph->module + m, dt_token(param), arg);
  return 0;
}

/** Dispatches that ran and were skipped since the last call, ran in the high 16 bits. */
EMSCRIPTEN_KEEPALIVE
int wasm_dispatch_stats(void) {
  uint32_t ran, skipped;
  wvk_dispatch_stats(&ran, &skipped);
  return (int)((ran << 16) | (skipped & 0xffff));
}
