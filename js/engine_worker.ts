import { rasterise } from './brush';
import { drawOriented } from './orientation';
import { Edit, editLine, fileName, ModuleState, Region, Request, Response, Slot } from './protocol';
import { drawWaveform, Waveform } from './waveform';

// wasm_open's numbers for the slots
const SLOTS: Slot[] = ['fit', 'full'];

// The subset of the Emscripten module built by //cc:engine that we call. Functions listed in
// JSPI_EXPORTS return promises because they wait on the GPU.
interface EngineModule {
  HEAPU8: Uint8Array;
  FS: {
    writeFile(path: string, data: Uint8Array): void;
    unlink(path: string): void;
    mkdir(path: string): void;
  };
  _wasm_init(): Promise<number>;
  _wasm_open(slot: number, path: number, maxWidth: number, maxHeight: number, graph: number,
            extra: number, disabled: number): Promise<number>;
  _wasm_render(path: number, maxWidth: number, maxHeight: number, graph: number, extra: number,
              disabled: number): Promise<number>;
  _wasm_set_params(edits: number): Promise<number>;
  _wasm_ui_callback(module: number, inst: number, param: number, arg: number): number;
  _wasm_state_json(): number;
  _wasm_output(inst: number): number;
  _wasm_output_width(inst: number): number;
  _wasm_output_height(inst: number): number;
  _wasm_full_size(): number;
  _wasm_dispatch_stats(): number;
  _wasm_spot_white(out: number): Promise<number>;
  _malloc(size: number): number;
  _free(ptr: number): void;
  stringToNewUTF8(s: string): number;
  UTF8ToString(ptr: number): string;
}

// i-brush's read_source calls this, see cc/modules/i-brush/main.c
(self as unknown as {wasmRasteriseBrush: typeof rasterise}).wasmRasteriseBrush = rasterise;

let engine: Promise<EngineModule>|undefined;
// the image currently in the engine's file system
let loaded: string|undefined;

function loadEngine(): Promise<EngineModule> {
  return (engine ??= (async () => {
    const glue = new URL('engine_cc.js', import.meta.url).href;
    const module = await (await import(/* webpackIgnore: true */ glue)).default() as EngineModule;
    const err = await module._wasm_init();
    if (err) {
      throw new Error(err === 2 ? 'WebGPU is unavailable' : `engine failed to start (${err})`);
    }
    module.FS.mkdir('/img');
    return module;
  })());
}

/** Calls fn with C strings that are freed afterwards. */
async function withStrings<T>(
    m: EngineModule, strings: string[], fn: (...ptrs: number[]) => Promise<T>): Promise<T> {
  const ptrs = strings.map(s => m.stringToNewUTF8(s));
  try {
    return await fn(...ptrs);
  } finally {
    ptrs.forEach(p => m._free(p));
  }
}

function pathFor(key: string): string {
  return `/img/${fileName(key)}`;
}

// the files other than the loaded image's in the engine's file system
let extras = new Set<string>();

/** The latest image from an output of the graph, main or hist. */
function output(m: EngineModule, inst = 'main'): ImageData|undefined {
  const s = m.stringToNewUTF8(inst);
  try {
    const wd = m._wasm_output_width(s);
    const ht = m._wasm_output_height(s);
    const ptr = m._wasm_output(s);
    if (!ptr || !wd || !ht) return undefined;
    // A view of the heap rather than a copy, since createImageBitmap copies anyway. Requests run
    // one at a time, so nothing changes the pixels before it does, and the heap is never shared
    // because the engine has no threads.
    const heap = m.HEAPU8.buffer as ArrayBuffer;
    return new ImageData(new Uint8ClampedArray(heap, ptr, wd * ht * 4), wd, ht);
  } finally {
    m._free(s);
  }
}

async function bitmaps(m: EngineModule, waveform?: Waveform, region?: Region):
    Promise<{bitmap: ImageBitmap, histogram?: ImageBitmap, region?: Region}> {
  const main = output(m);
  if (!main) throw new Error('vkdt produced no image');
  const part = region && clamp(region, main.width, main.height);
  return {
    // copying only part of a full resolution output saves most of the time it takes
    bitmap: await (part
        ? createImageBitmap(main, part.x, part.y, part.width, part.height)
        : createImageBitmap(main)),
    histogram: waveform ? await createImageBitmap(drawWaveform(main, waveform)) : undefined,
    region: part,
  };
}

function clamp(r: Region, width: number, height: number): Region {
  const x = Math.max(0, Math.min(width - 1, Math.floor(r.x)));
  const y = Math.max(0, Math.min(height - 1, Math.floor(r.y)));
  return {
    x,
    y,
    width: Math.max(1, Math.min(width - x, Math.ceil(r.width))),
    height: Math.max(1, Math.min(height - y, Math.ceil(r.height))),
  };
}

function fullSize(m: EngineModule): {fullWidth: number, fullHeight: number} {
  const packed = m._wasm_full_size();
  return {fullWidth: packed & 0xffff, fullHeight: packed >>> 16};
}

function state(m: EngineModule): ModuleState[] {
  return JSON.parse(m.UTF8ToString(m._wasm_state_json())) as ModuleState[];
}

function extra(edits: Edit[]): string {
  return edits.map(editLine).join('\n');
}

async function handle(request: Request): Promise<Response> {
  const m = await loadEngine();
  const start = performance.now();
  if (request.kind === 'open') {
    const path = pathFor(request.key);
    if (request.bytes) {
      const keep = new Set((request.extras ?? []).map(e => e.key));
      if (loaded && !keep.has(loaded)) m.FS.unlink(pathFor(loaded));
      for (const key of extras) {
        if (!keep.has(key) && key !== request.key) m.FS.unlink(pathFor(key));
      }
      m.FS.writeFile(path, new Uint8Array(request.bytes));
      loaded = request.key;
      extras = new Set();
    } else if (loaded !== request.key) {
      throw new Error('image is not loaded');
    }
    for (const {key, bytes} of request.extras ?? []) {
      if (!extras.has(key)) m.FS.writeFile(pathFor(key), new Uint8Array(bytes));
      extras.add(key);
    }
    const err = await withStrings(
        m, [path, request.graph, extra(request.edits), request.disabled.join('\n')],
        (p, g, e, d) => m._wasm_open(
            SLOTS.indexOf(request.slot), p, request.maxWidth, request.maxHeight, g, e, d));
    if (err) throw new Error(`vkdt failed to process the image (${err})`);
    return {
      id: request.id, ok: true, ...await bitmaps(m, request.waveform), ...fullSize(m), state: state(m),
      ms: performance.now() - start,
    };
  } else if (request.kind === 'set') {
    // module:inst:name:values with comma separated values, see wasm_set_params
    const lines = request.edits.map(e => `${e.module}:${e.inst}:${e.name}:${e.values.join(',')}`);
    const err = await withStrings(m, [lines.join('\n')], e => m._wasm_set_params(e));
    if (err) throw new Error(`vkdt failed to apply the edit (${err})`);
    const stats = m._wasm_dispatch_stats();
    console.debug(`[vkdt_wasm] ran ${stats >>> 16} dispatches and reused ${stats & 0xffff}`);
    return {
      id: request.id, ok: true, ...await bitmaps(m, request.waveform, request.region), ...fullSize(m),
      state: state(m), ms: performance.now() - start,
    };
  } else if (request.kind === 'waveform') {
    const main = output(m);
    if (!main) throw new Error('vkdt has no image yet');
    const histogram = await createImageBitmap(drawWaveform(main, request.waveform));
    return {id: request.id, ok: true, histogram, ms: performance.now() - start};
  } else if (request.kind === 'region') {
    const main = output(m);
    if (!main) throw new Error('vkdt has no image yet');
    const part = clamp(request.region, main.width, main.height);
    const bitmap = await createImageBitmap(main, part.x, part.y, part.width, part.height);
    return {id: request.id, ok: true, bitmap, region: part, ms: performance.now() - start};
  } else if (request.kind === 'callback') {
    const {module, inst, param, arg} = request;
    const err = await withStrings(
        m, [module, inst, param], (a, b, c) => Promise.resolve(m._wasm_ui_callback(a, b, c, arg)));
    if (err) throw new Error(`vkdt failed to run ${module}'s callback (${err})`);
    return {id: request.id, ok: true, state: state(m), ms: performance.now() - start};
  } else if (request.kind === 'spotWhite') {
    const out = m._malloc(16);
    try {
      const err = await m._wasm_spot_white(out);
      if (err) throw new Error(`vkdt couldn't measure the white balance spot (${err})`);
      const spotWhite = [...new Float32Array(m.HEAPU8.buffer, out, 4)];
      return {id: request.id, ok: true, spotWhite, ms: performance.now() - start};
    } finally {
      m._free(out);
    }
  } else if (request.kind === 'render') {
    if (loaded !== request.key) throw new Error('image is not loaded');
    const err = await withStrings(
        m, [pathFor(request.key), request.graph, extra(request.edits), request.disabled.join('\n')],
        (p, g, e, d) => m._wasm_render(p, request.maxWidth, request.maxHeight, g, e, d));
    if (err) throw new Error(`vkdt failed to render (${err})`);
    return {id: request.id, ok: true, ...await bitmaps(m), ms: performance.now() - start};
  } else {
    if (loaded !== request.key) throw new Error('image is not loaded');
    const err = await withStrings(
        m, [pathFor(request.key), request.graph, extra(request.edits), request.disabled.join('\n')],
        (p, g, e, d) => m._wasm_render(p, 0, 0, g, e, d));
    if (err) throw new Error(`vkdt failed to export (${err})`);
    const bitmap = await createImageBitmap(output(m)!);
    const canvas = drawOriented(bitmap, request.orientation);
    bitmap.close();
    const jpeg = await canvas.convertToBlob({type: 'image/jpeg', quality: request.quality});
    return {id: request.id, ok: true, jpeg, ms: performance.now() - start};
  }
}

// The engine isn't reentrant, so requests run one at a time.
let queue = Promise.resolve();
self.onmessage = (e: MessageEvent<Request>) => {
  queue = queue.then(async () => {
    let response: Response;
    try {
      response = await handle(e.data);
    } catch (err) {
      response = {id: e.data.id, ok: false, error: err instanceof Error ? err.message : String(err)};
    }
    const transfer = response.ok
        ? [response.bitmap, response.histogram].filter((b): b is ImageBitmap => !!b)
        : [];
    (self as unknown as Worker).postMessage(response, transfer);
  });
};
