# vkdt_wasm

[vkdt](https://github.com/hanatos/vkdt)'s processing in the browser, compiled to WebAssembly and
running on WebGPU in a worker. [rawspeed](https://github.com/darktable-org/rawspeed) is used to
decode raw files. vkdt_wasm is not affiliated with or endorsed by hanatos or the vkdt project.

This repository is Claude-slop.

* vkdt's C sources and GLSL shaders are pulled in as `@vkdt`.
  * We apply a small patch `third_party/vkdt/vkdt.patch`
  * To avoid the GPL license: our archive import deletes GPL-licensed vkdt sources and replacements
    are written in `cc/replacements`
  * `cc/webgpu` implements the Vulkan subset vkdt uses on top of WebGPU.
  * `third_party/vkdt/shaders.bzl` transpiles each compute shader to WGSL with glslang, spirv-opt,
     and naga (with some rewrites in `glsl_fix.pl` and `spv_bindings.pl`.)
  * `third_party/vkdt/module_wrap.pl` lets vkdt's dlopen'ed modules link into one binary.
* For WASM, we add several modules to `cc/modules`:
  * `i-brush` for painted masks
  * `o-wasm`: the sink the worker reads images from
  * `heal`: spot removal, cloning, and healing. `inpaint` isn't WebGPU compatible.
* JS interoperability is achieved via `js/protocol.ts` and `js/engine_worker.ts`
  * A request carries a vkdt graph config, ending in `o-wasm:main`, and edits in vkdt's
  `param:<module>:<inst>:<name>:<values>` form.

## Building

```
bazel build //cc:engine //js:worker //third_party/licenses
```

Downstream JS/TS libraries can depend on `@vkdt_wasm//js`.

## Licensing

vkdt_wasm is BSD-2-Clause but has a LGPL 2.1 or later dependency on rawspeed. `licenses.txt` is
provided to maintain compliance.
