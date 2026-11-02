#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wasm_vkdt_symbol_t {
  const char *name;
  void *address;
} wasm_vkdt_symbol_t;

typedef struct wasm_vkdt_module_t {
  const char *name;
  const wasm_vkdt_symbol_t *(*symbols)(void);
} wasm_vkdt_module_t;

// Every module linked into the binary, terminated by a zeroed entry. Generated from the modules
// listed in cc/BUILD.bazel.
extern const wasm_vkdt_module_t wasm_vkdt_modules[];

#ifdef __cplusplus
}
#endif
