// WebGPU backend for the Vulkan subset in vulkan_webgpu.c. C owns the Vulkan semantics and hands
// us flat integer command lists; we own every WebGPU object, keyed by small integer ids.
//
// Storage textures in the transpiled shaders all say rgba16float (see glsl_fix.pl), so pipelines
// are specialized per combination of the formats actually bound.

addToLibrary({
  $WVK: {
    device: null,
    objects: [null],
    shaders: [null],
    pipelines: new Map(),
    layouts: new Map(),
    readbacks: [],
    pending: [],
    error: null,

    // A view for binding o, which is a GPUTexture or a layer of one (see wvk_js_layer), as a
    // 2d-array view if the shader declares an array.
    view(o, array) {
      const range = o.layer === undefined
          ? (array ? {} : {baseArrayLayer: 0, arrayLayerCount: 1})
          : {baseArrayLayer: o.layer, arrayLayerCount: 1};
      return (o.texture || o).createView(
          {dimension: array ? '2d-array' : '2d', baseMipLevel: 0, mipLevelCount: 1, ...range});
    },

    size(o) {
      const t = o.texture || o;
      return [t.width, t.height];
    },

    put(o) {
      WVK.objects.push(o);
      return WVK.objects.length - 1;
    },

    // Keep in sync with wvk_gpu_format_t in vulkan_webgpu.h.
    FORMATS: [
      'r32float', 'rg32float', 'rgba16float', 'rgba32float', 'rgba8unorm', 'r32uint',
      'rgba32uint',
    ],

    parseShader(text) {
      // @group(G) @binding(B) var<...> NAME: TYPE;
      const re = /@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+);/g;
      const decls = [];
      for (const m of text.matchAll(re)) {
        decls.push({group: +m[1], binding: +m[2], space: m[3] || '', name: m[4], type: m[5].trim()});
      }
      return decls;
    },

    layoutEntry(d, format) {
      const visibility = GPUShaderStage.COMPUTE;
      if (d.space.startsWith('uniform')) {
        return {binding: d.binding, visibility, buffer: {type: 'uniform'}};
      }
      if (d.space.startsWith('storage')) {
        const type = d.space.includes('read_write') ? 'storage' : 'read-only-storage';
        return {binding: d.binding, visibility, buffer: {type}};
      }
      if (d.type === 'sampler') {
        return {binding: d.binding, visibility, sampler: {type: 'filtering'}};
      }
      if (d.type.startsWith('texture_storage_2d')) {
        const access = /,\s*write\s*>/.test(d.type) ? 'write-only'
            : /,\s*read\s*>/.test(d.type) ? 'read-only' : 'read-write';
        const viewDimension = d.type.startsWith('texture_storage_2d_array') ? '2d-array' : '2d';
        return {binding: d.binding, visibility, storageTexture: {access, format, viewDimension}};
      }
      if (d.type.startsWith('texture_2d')) {
        const sampleType = d.type.includes('<u32>') ? 'uint'
            : d.type.includes('<i32>') ? 'sint' : 'float';
        const viewDimension = d.type.startsWith('texture_2d_array') ? '2d-array' : '2d';
        return {binding: d.binding, visibility, texture: {sampleType, viewDimension}};
      }
      throw new Error(`unsupported binding ${d.name}: ${d.type}`);
    },

    pipeline(shaderId, formats) {
      const key = `${shaderId}:${formats.join(',')}`;
      let p = WVK.pipelines.get(key);
      if (p) return p;
      const shader = WVK.shaders[shaderId];
      let code = shader.text;
      const groups = new Map();
      let fi = 0;
      for (const d of shader.decls) {
        let format;
        if (d.type.startsWith('texture_storage_2d')) {
          format = formats[fi++];
          code = code.replace(
              new RegExp(`(var\\s+${d.name}\\s*:\\s*texture_storage_2d(?:_array)?<)\\w+`), `$1${format}`);
        }
        if (!groups.has(d.group)) groups.set(d.group, []);
        groups.get(d.group).push(WVK.layoutEntry(d, format));
      }
      const device = WVK.device;
      const maxGroup = Math.max(-1, ...groups.keys());
      const bindGroupLayouts = [];
      for (let g = 0; g <= maxGroup; g++) {
        bindGroupLayouts.push(device.createBindGroupLayout({entries: groups.get(g) || []}));
      }
      device.pushErrorScope('validation');
      const module = device.createShaderModule({code, label: shader.label});
      const pipeline = device.createComputePipeline({
        label: shader.label,
        layout: device.createPipelineLayout({bindGroupLayouts}),
        compute: {module, entryPoint: 'main'},
      });
      device.popErrorScope().then(e => {
        if (e) console.error(`[wvk] ${shader.label}: ${e.message}`);
      });
      p = {pipeline, bindGroupLayouts};
      WVK.pipelines.set(key, p);
      return p;
    },

    // Kernels that move texels between tightly packed buffers in Vulkan's formats and the
    // textures we back them with. Keep in sync with wvk_pack_t in vulkan_webgpu.h.
    PACKS: [
      // [vulkan bytes per texel, channels, wgsl expression reading texel i from words w]
      [2, 1, 'vec4f(f32((w[(o + i) / 2u] >> (16u * ((o + i) % 2u))) & 0xffffu) / 65535.0, 0, 0, 1)'],
      [2, 1, 'vec4f(unpack2x16float(w[(o + i) / 2u])[(o + i) % 2u], 0, 0, 1)'],
      [4, 2, 'vec4f(unpack2x16float(w[o + i]), 0, 1)'],
      [8, 4, 'vec4f(unpack2x16float(w[2u * (o + i)]), unpack2x16float(w[2u * (o + i) + 1u]))'],
      [4, 1, 'vec4f(bitcast<f32>(w[o + i]), 0, 0, 1)'],
      [8, 2, 'vec4f(bitcast<f32>(w[2u * (o + i)]), bitcast<f32>(w[2u * (o + i) + 1u]), 0, 1)'],
      [16, 4, 'bitcast<vec4f>(vec4u(w[4u * (o + i)], w[4u * (o + i) + 1u], w[4u * (o + i) + 2u], w[4u * (o + i) + 3u]))'],
      [4, 4, 'unpack4x8unorm(w[o + i])'],
      [1, 1, 'vec4f(f32((w[(o + i) / 4u] >> (8u * ((o + i) % 4u))) & 0xffu) / 255.0, 0, 0, 1)'],
      [8, 4, 'vec4f(f32(w[2u * (o + i)] & 0xffffu), f32(w[2u * (o + i)] >> 16u), f32(w[2u * (o + i) + 1u] & 0xffffu), f32(w[2u * (o + i) + 1u] >> 16u)) / 65535.0'],
    ],

    packPipeline(kind, upload, format) {
      const key = `pack:${kind}:${upload}:${format}`;
      let p = WVK.pipelines.get(key);
      if (p) return p;
      const [bytes, channels, read] = WVK.PACKS[kind];
      const isUint = format.endsWith('uint');
      const common = `
struct P { wd: u32, ht: u32, row: u32, o: u32 }
@group(0) @binding(0) var<uniform> p: P;
`;
      // Unsigned formats hold whatever bits the shaders put there, such as floats from
      // uintBitsToFloat, so they move bit for bit.
      let code;
      if (upload) {
        code = `${common}
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var t: texture_storage_2d<${format}, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.wd || id.y >= p.ht) { return; }
  let o = p.o;
  let i = id.y * p.row + id.x;
  let v = ${read};
  textureStore(t, id.xy, ${isUint ? 'bitcast<vec4u>(v)' : 'v'});
}`;
      } else {
        // One invocation per output word, so formats narrower than a word stay race free.
        const perWord = Math.max(1, 4 / bytes);
        const wordsPerTexel = Math.max(1, bytes / 4);
        const toWords = {
          2: channels === 1 && kind === 0
              ? 'u32(clamp(v.x, 0.0, 1.0) * 65535.0 + 0.5)'
              : 'pack2x16float(vec2f(v.x, 0)) & 0xffffu',
          1: 'u32(clamp(v.x, 0.0, 1.0) * 255.0 + 0.5)',
        };
        let body;
        if (bytes < 4) {
          const bits = bytes * 8;
          body = `
  var word = 0u;
  for (var k = 0u; k < ${perWord}u; k++) {
    let x = id.x * ${perWord}u + k;
    if (x < p.wd) {
      let v = textureLoad(t, vec2u(x, id.y), 0);
      word |= (${toWords[bytes]}) << (${bits}u * k);
    }
  }
  w[(p.o + id.y * p.row + id.x * ${perWord}u) / ${perWord}u] = word;`;
        } else {
          const words = {
            4: channels === 4 ? ['pack4x8unorm(v)'] : channels === 2 ? ['pack2x16float(v.xy)'] : ['bitcast<u32>(v.x)'],
            8: channels === 4 ? ['pack2x16float(v.xy)', 'pack2x16float(v.zw)'] : ['bitcast<u32>(v.x)', 'bitcast<u32>(v.y)'],
            16: ['bitcast<u32>(v.x)', 'bitcast<u32>(v.y)', 'bitcast<u32>(v.z)', 'bitcast<u32>(v.w)'],
          }[bytes];
          body = `
  if (id.x >= p.wd) { return; }
  let v = ${isUint ? 'textureLoad(t, id.xy, 0)' : 'vec4f(textureLoad(t, id.xy, 0))'};
  let base = (p.o + id.y * p.row + id.x) * ${wordsPerTexel}u;
${words.map((e, k) => `  w[base + ${k}u] = ${e};`).join('\n')}`;
        }
        code = `${common}
@group(0) @binding(1) var<storage, read_write> w: array<u32>;
@group(0) @binding(2) var t: texture_2d<${isUint ? 'u32' : 'f32'}>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.y >= p.ht) { return; }${body}
}`;
      }
      const module = WVK.device.createShaderModule({code, label: key});
      const pipeline = WVK.device.createComputePipeline({
        label: key, layout: 'auto', compute: {module, entryPoint: 'main'},
      });
      p = {pipeline, perWord: upload ? 1 : Math.max(1, 4 / bytes)};
      WVK.pipelines.set(key, p);
      return p;
    },

    clearPipeline(format) {
      const key = `clear:${format}`;
      let p = WVK.pipelines.get(key);
      if (p) return p;
      const isUint = format.endsWith('uint');
      const code = `
@group(0) @binding(0) var<uniform> c: vec4f;
@group(0) @binding(1) var t: texture_storage_2d<${format}, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let d = textureDimensions(t);
  if (id.x >= d.x || id.y >= d.y) { return; }
  textureStore(t, id.xy, ${isUint ? 'vec4u(c)' : 'c'});
}`;
      const module = WVK.device.createShaderModule({code, label: key});
      p = WVK.device.createComputePipeline({
        label: key, layout: 'auto', compute: {module, entryPoint: 'main'},
      });
      WVK.pipelines.set(key, p);
      return p;
    },

    // Vulkan lets shaders declare bindings that nothing is bound to as long as they go unused,
    // which vkdt relies on for unconnected optional inputs. WebGPU needs something bound.
    dummies: new Map(),
    dummy(d, format) {
      const key = `${d.space}|${d.type}|${format}`;
      let r = WVK.dummies.get(key);
      if (r) return r;
      const device = WVK.device;
      if (d.space.startsWith('uniform') || d.space.startsWith('storage')) {
        const usage = d.space.startsWith('uniform') ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE;
        r = {buffer: device.createBuffer({size: 4096, usage})};
      } else if (d.type === 'sampler') {
        r = device.createSampler();
      } else {
        const storage = d.type.startsWith('texture_storage_2d');
        const texFormat = storage ? format
            : d.type.includes('<u32>') ? 'r32uint' : d.type.includes('<i32>') ? 'r32sint' : 'rgba16float';
        r = device.createTexture({
          size: [1, 1],
          format: texFormat,
          usage: GPUTextureUsage.TEXTURE_BINDING | (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
        }).createView();
      }
      WVK.dummies.set(key, r);
      return r;
    },

    // A buffer holding an r32uint image's size and texels, for atomics.
    mirror(o) {
      const t = o.texture || o;
      if (!o.wvkMirror) {
        const [wd, ht] = WVK.size(o);
        o.wvkMirror = WVK.device.createBuffer({
          size: 8 + wd * ht * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
      }
      return o.wvkMirror;
    },

    // Copies an r32uint image into its mirror, size first, or back out of it.
    syncMirror(encoder, o, buffer, toBuffer) {
      const key = toBuffer ? 'mirror:in' : 'mirror:out';
      let pipeline = WVK.pipelines.get(key);
      if (!pipeline) {
        const code = toBuffer ? `
@group(0) @binding(0) var t: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let d = textureDimensions(t);
  if (id.x == 0 && id.y == 0) { b[0] = d.x; b[1] = d.y; }
  if (id.x >= d.x || id.y >= d.y) { return; }
  b[2 + id.y * d.x + id.x] = textureLoad(t, id.xy, 0).x;
}` : `
@group(0) @binding(0) var t: texture_storage_2d<r32uint, write>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let d = textureDimensions(t);
  if (id.x >= d.x || id.y >= d.y) { return; }
  textureStore(t, id.xy, vec4u(b[2 + id.y * d.x + id.x], 0, 0, 1));
}`;
        const module = WVK.device.createShaderModule({code, label: key});
        pipeline = WVK.device.createComputePipeline({
          label: key, layout: 'auto', compute: {module, entryPoint: 'main'},
        });
        WVK.pipelines.set(key, pipeline);
      }
      const [wd, ht] = WVK.size(o);
      const pass = encoder.beginComputePass({label: key});
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, WVK.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{binding: 0, resource: WVK.view(o, false)}, {binding: 1, resource: {buffer}}],
      }));
      pass.dispatchWorkgroups(Math.ceil(wd / 8), Math.ceil(ht / 8));
      pass.end();
    },

    smallUniform(words) {
      const buffer = WVK.device.createBuffer({
        size: Math.max(16, words.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      WVK.device.queue.writeBuffer(buffer, 0, words);
      return buffer;
    },
  },

  wvk_js_init__deps: ['$WVK'],
  wvk_js_init__async: true,
  wvk_js_init: async function() {
    if (!navigator.gpu) {
      console.error('[wvk] WebGPU is unavailable');
      return 1;
    }
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) {
      console.error('[wvk] no WebGPU adapter');
      return 1;
    }
    const wanted = [
      'shader-f16', 'float32-filterable', 'texture-formats-tier1', 'texture-formats-tier2',
      'subgroups',
    ];
    const requiredFeatures = wanted.filter(f => adapter.features.has(f));
    const l = adapter.limits;
    WVK.device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxBufferSize: l.maxBufferSize,
        maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
        maxTextureDimension2D: l.maxTextureDimension2D,
        maxStorageTexturesPerShaderStage: l.maxStorageTexturesPerShaderStage,
        maxSampledTexturesPerShaderStage: l.maxSampledTexturesPerShaderStage,
        maxSamplersPerShaderStage: l.maxSamplersPerShaderStage,
        maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
        maxUniformBuffersPerShaderStage: l.maxUniformBuffersPerShaderStage,
        maxBindingsPerBindGroup: l.maxBindingsPerBindGroup,
        maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: l.maxComputeWorkgroupSizeY,
      },
    });
    WVK.device.lost.then(info => console.error(`[wvk] device lost: ${info.message}`));
    WVK.device.onuncapturederror = e => console.error(`[wvk] ${e.error.message}`);
    console.log(`[wvk] WebGPU device with ${requiredFeatures.join(', ') || 'no optional features'}`);
    return 0;
  },

  wvk_js_create_buffer__deps: ['$WVK'],
  wvk_js_create_buffer: function(size) {
    const u = GPUBufferUsage;
    return WVK.put(WVK.device.createBuffer({
      size: Math.max(16, Math.ceil(size / 4) * 4),
      usage: u.STORAGE | u.UNIFORM | u.COPY_SRC | u.COPY_DST,
    }));
  },

  wvk_js_create_texture__deps: ['$WVK'],
  wvk_js_create_texture: function(wd, ht, format, mips) {
    const u = GPUTextureUsage;
    return WVK.put(WVK.device.createTexture({
      size: [Math.max(1, wd), Math.max(1, ht)],
      format: WVK.FORMATS[format],
      mipLevelCount: Math.max(1, mips),
      usage: u.TEXTURE_BINDING | u.STORAGE_BINDING | u.COPY_SRC | u.COPY_DST,
    }));
  },

  wvk_js_create_sampler__deps: ['$WVK'],
  wvk_js_create_sampler: function(linear) {
    const filter = linear ? 'linear' : 'nearest';
    return WVK.put(WVK.device.createSampler({
      magFilter: filter, minFilter: filter, mipmapFilter: filter,
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    }));
  },

  wvk_js_create_array_texture__deps: ['$WVK'],
  wvk_js_create_array_texture: function(wd, ht, format, layers) {
    const u = GPUTextureUsage;
    const texture = WVK.device.createTexture({
      size: [Math.max(1, wd), Math.max(1, ht), layers],
      format: WVK.FORMATS[format],
      usage: u.TEXTURE_BINDING | u.STORAGE_BINDING | u.COPY_SRC | u.COPY_DST,
    });
    // Destroyed once every layer is released. The id stays valid for binding the whole array
    // while any layer lives.
    texture.wvkRefs = 0;
    return WVK.put(texture);
  },

  wvk_js_layer__deps: ['$WVK'],
  wvk_js_layer: function(array, layer) {
    const texture = WVK.objects[array];
    texture.wvkRefs++;
    return WVK.put({texture, layer});
  },

  wvk_js_destroy__deps: ['$WVK'],
  wvk_js_destroy: function(id) {
    const o = WVK.objects[id];
    WVK.objects[id] = null;
    if (!o) return;
    if (o.wvkMirror) o.wvkMirror.destroy();
    if (o.texture) {
      if (--o.texture.wvkRefs === 0) o.texture.destroy();
    } else if (o.destroy) {
      o.destroy();
    }
  },

  wvk_js_create_shader__deps: ['$WVK'],
  // writes gets one mask per bind group of the Vulkan bindings the shader may write
  wvk_js_create_shader: function(ptr, len, label, writes) {
    const text = new TextDecoder().decode(HEAPU8.subarray(ptr, ptr + len));
    const name = /^\/\/ vkdt (\S+)/.exec(text);
    const decls = WVK.parseShader(text);
    const masks = HEAPU32.subarray(writes >> 2, (writes >> 2) + 4);
    masks.fill(0);
    for (const d of decls) {
      const written = d.space.includes('read_write')
          || (d.type.startsWith('texture_storage_2d') && !/,\s*read\s*>/.test(d.type));
      // the bindings are doubled, see glsl_fix.pl
      if (written && d.group < 4) masks[d.group] |= 1 << (d.binding >> 1);
    }
    WVK.shaders.push({text, decls, label: name ? name[1] : UTF8ToString(label)});
    return WVK.shaders.length - 1;
  },

  wvk_js_write_buffer__deps: ['$WVK'],
  wvk_js_write_buffer: function(id, offset, ptr, size) {
    // writeBuffer wants multiples of four bytes
    const n = size & ~3;
    if (n > 0) WVK.device.queue.writeBuffer(WVK.objects[id], offset, HEAPU8, ptr, n);
  },

  // Replays one submission. Keep the op layout in sync with vulkan_webgpu.c.
  wvk_js_submit__deps: ['$WVK'],
  wvk_js_submit: function(ptr, count) {
    const device = WVK.device;
    const ops = HEAPU32.subarray(ptr >> 2, (ptr >> 2) + count);
    const encoder = device.createCommandEncoder();
    const temporaries = [];
    let pushBuffer = null;
    let i = 0;
    while (i < count) {
      const op = ops[i++];
      if (op === 8) { // push constants for this submit: pointer, size
        const ptr = ops[i++], size = ops[i++];
        if (size) {
          pushBuffer = device.createBuffer({
            size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(pushBuffer, 0, HEAPU8, ptr, size);
          temporaries.push(pushBuffer);
        }
      } else if (op === 1) { // dispatch: shader, gx, gy, gz, nbind, then nbind * [group, binding, kind, id, offset, size]
        const shaderId = ops[i++], gx = ops[i++], gy = ops[i++], gz = ops[i++], n = ops[i++];
        const byKey = new Map();
        for (let k = 0; k < n; k++, i += 6) {
          byKey.set(`${ops[i]}:${ops[i + 1]}`, ops.slice(i, i + 6));
        }
        const shader = WVK.shaders[shaderId];
        const formats = [];
        for (const d of shader.decls) {
          if (!d.type.startsWith('texture_storage_2d')) continue;
          const b = byKey.get(`${d.group}:${d.binding}`);
          formats.push(b ? WVK.FORMATS[b[5]] : 'rgba16float');
        }
        const {pipeline, bindGroupLayouts} = WVK.pipeline(shaderId, formats);
        // Images the shader updates atomically come in as storage buffers (see glsl_fix.pl), which
        // mirror the image across the dispatch.
        const mirrors = new Map();
        for (const d of shader.decls) {
          const b = byKey.get(`${d.group}:${d.binding}`);
          if (!b || b[2] !== 4 || !d.space.startsWith('storage')) continue;
          const texture = WVK.objects[b[3]];
          const buffer = WVK.mirror(texture);
          WVK.syncMirror(encoder, texture, buffer, true);
          mirrors.set(d, {texture, buffer});
        }
        const pass = encoder.beginComputePass({label: shader.label});
        pass.setPipeline(pipeline);
        for (let g = 0; g < bindGroupLayouts.length; g++) {
          const entries = [];
          let fi = 0;
          for (const d of shader.decls) {
            const format = d.type.startsWith('texture_storage_2d') ? formats[fi++] : undefined;
            if (d.group !== g) continue;
            const b = byKey.get(`${d.group}:${d.binding}`);
            if (!b) {
              entries.push({binding: d.binding, resource: WVK.dummy(d, format)});
              continue;
            }
            const [, , kind, id, offset, size] = b;
            const o = WVK.objects[id];
            let resource;
            if (mirrors.has(d)) resource = {buffer: mirrors.get(d).buffer};
            else if (kind === 5) resource = {buffer: pushBuffer, offset, size};
            else if (kind === 0 || kind === 1) resource = {buffer: o, offset, size};
            else if (kind === 2) resource = o;
            else resource = WVK.view(o, d.type.includes('_array'));
            entries.push({binding: d.binding, resource});
          }
          pass.setBindGroup(g, device.createBindGroup({layout: bindGroupLayouts[g], entries}));
        }
        pass.dispatchWorkgroups(gx, gy, gz);
        pass.end();
        for (const {texture, buffer} of mirrors.values()) {
          WVK.syncMirror(encoder, texture, buffer, false);
        }
      } else if (op === 2) { // copy buffer: src, srcOffset, dst, dstOffset, size
        const src = ops[i++], so = ops[i++], dst = ops[i++], dofs = ops[i++], size = ops[i++];
        encoder.copyBufferToBuffer(WVK.objects[src], so, WVK.objects[dst], dofs, size & ~3);
      } else if (op === 3 || op === 4) { // buffer <-> texture: buf, offset, rowTexels, tex, wd, ht, pack, format
        const buf = ops[i++], offset = ops[i++], row = ops[i++], tex = ops[i++], wd = ops[i++],
            ht = ops[i++], pack = ops[i++], format = WVK.FORMATS[ops[i++]];
        const upload = op === 3;
        const {pipeline, perWord} = WVK.packPipeline(pack, upload, format);
        const bytes = WVK.PACKS[pack][0];
        // the kernels index in texels for narrow formats and words otherwise
        const o = bytes < 4 ? offset / bytes : offset / 4 / Math.max(1, bytes / 4);
        const uniform = WVK.smallUniform(new Uint32Array([wd, ht, row, o]));
        temporaries.push(uniform);
        const texture = WVK.objects[tex];
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            {binding: 0, resource: {buffer: uniform}},
            {binding: 1, resource: {buffer: WVK.objects[buf]}},
            {binding: 2, resource: WVK.view(texture, false)},
          ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(wd / 8 / perWord), Math.ceil(ht / 8));
        pass.end();
      } else if (op === 5) { // clear texture: tex, format, r, g, b, a as float bits
        const tex = ops[i++], format = WVK.FORMATS[ops[i++]];
        const color = new Float32Array(ops.slice(i, i + 4).buffer.slice(0));
        i += 4;
        const pipeline = WVK.clearPipeline(format);
        const uniform = WVK.smallUniform(new Float32Array(color));
        temporaries.push(uniform);
        const texture = WVK.objects[tex];
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            {binding: 0, resource: {buffer: uniform}},
            {binding: 1, resource: WVK.view(texture, false)},
          ],
        }));
        const [wd, ht] = WVK.size(texture);
        pass.dispatchWorkgroups(Math.ceil(wd / 8), Math.ceil(ht / 8));
        pass.end();
      } else if (op === 6) { // fill buffer with zeros: buf, offset, size
        const buf = ops[i++], offset = ops[i++], size = ops[i++];
        const b = WVK.objects[buf];
        encoder.clearBuffer(b, offset, Math.min(size, b.size - offset) & ~3);
      } else if (op === 7) { // read back to host memory: buf, offset, size, host pointer
        const buf = ops[i++], offset = ops[i++], size = ops[i++], host = ops[i++];
        const n = Math.ceil(size / 4) * 4;
        const staging = device.createBuffer({
          size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        encoder.copyBufferToBuffer(WVK.objects[buf], offset, staging, 0, n);
        WVK.readbacks.push({staging, host, size});
      } else {
        throw new Error(`[wvk] unknown op ${op}`);
      }
    }
    device.queue.submit([encoder.finish()]);
    const done = device.queue.onSubmittedWorkDone();
    WVK.pending.push(done.then(() => temporaries.forEach(t => t.destroy())));
  },

  // Blocks the calling C code until all submitted work and readbacks are done.
  wvk_js_wait__deps: ['$WVK'],
  wvk_js_wait__async: true,
  wvk_js_wait: async function() {
    const pending = WVK.pending;
    const readbacks = WVK.readbacks;
    WVK.pending = [];
    WVK.readbacks = [];
    await Promise.all(pending);
    await Promise.all(readbacks.map(async r => {
      await r.staging.mapAsync(GPUMapMode.READ);
      HEAPU8.set(new Uint8Array(r.staging.getMappedRange(), 0, r.size), r.host);
      r.staging.unmap();
      r.staging.destroy();
    }));
    return 0;
  },
});
