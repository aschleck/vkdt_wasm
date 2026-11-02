// dlopen over the modules linked into this binary. vkdt opens modules/<name>/lib<name>.so and
// looks its callbacks up by name, see dt_module_so_load in src/pipe/global.c.

#include <dlfcn.h>
#include <string.h>

#include "wasm_vkdt_modules.h"

static const char *error = "";

void *dlopen(const char *filename, int flags) {
  if(!filename) return (void *)wasm_vkdt_modules;
  const char *base = strrchr(filename, '/');
  base = base ? base + 1 : filename;
  if(!strncmp(base, "lib", 3)) base += 3;
  const size_t length = strlen(base) > 3 ? strlen(base) - 3 : 0;  // drop ".so"
  for(const wasm_vkdt_module_t *m = wasm_vkdt_modules; m->name; m++)
    if(strlen(m->name) == length && !strncmp(m->name, base, length)) return (void *)m;
  error = "module not linked";
  return 0;
}

void *dlsym(void *handle, const char *symbol) {
  const wasm_vkdt_module_t *m = handle;
  if(m == wasm_vkdt_modules) return 0;
  for(const wasm_vkdt_symbol_t *s = m->symbols(); s->name; s++)
    if(!strcmp(s->name, symbol)) return s->address;
  return 0;
}

int dlclose(void *handle) { return 0; }

char *dlerror(void) { return (char *)error; }
