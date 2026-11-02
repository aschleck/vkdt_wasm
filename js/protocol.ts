// Messages to and from engine_worker.ts.

import { Waveform } from './waveform';

export interface Param {
  name: string;
  type: 'float'|'int'|'string';
  cnt: number;
  // from the module's params.ui, such as slider or combo
  widget: string;
  min: number;
  max: number;
  // The widget only applies while the param at index grpid has the value grp (and likewise for
  // tabid and tab). -1 means always.
  grpid: number;
  grp: number;
  tabid: number;
  tab: number;
  // the index of the param whose value is how many of this one's components show, or -1 for all
  cntid: number;
  label: string;
  // combo entries, in value order
  options: string[];
  value: number[];
  default: number[];
  // decimal places to show, for params we make up such as the Basic panel's
  digits?: number;
}

export interface ModuleState {
  module: string;
  inst: string;
  params: Param[];
}

/** A rectangle of the output in pixels, in the sensor's orientation. */
export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One edit in vkdt's own config syntax, param:<module>:<inst>:<name>:<values>. */
export interface Edit {
  module: string;
  inst: string;
  name: string;
  values: number[];
}

/**
 * The engine keeps a graph of the open image that fits the screen and one at full resolution, so
 * that zooming past the first switches to the second rather than rebuilding either.
 */
export type Slot = 'fit'|'full';

export type Request = {
  id: number;
  kind: 'open';
  // which of the engine's graphs, see Slot
  slot: Slot;
  key: string;
  bytes: ArrayBuffer|undefined;
  // a vkdt graph config, whose sink must be o-wasm:main, see cc/modules/o-wasm
  graph: string;
  // module:inst of modules in graph to bypass
  disabled: string[];
  maxWidth: number;
  maxHeight: number;
  edits: Edit[];
  // what to draw a waveform of, if anything
  waveform?: Waveform;
  // more raws the graph reads, such as a stack's other frames
  extras?: Array<{key: string, bytes: ArrayBuffer}>;
}|{
  id: number;
  kind: 'set';
  edits: Edit[];
  // only this part of the output comes back, such as what shows at 1:1
  region?: Region;
  waveform?: Waveform;
}|{
  id: number;
  // the waveform of the latest output, without running anything
  kind: 'waveform';
  waveform: Waveform;
}|{
  id: number;
  // part of the latest output, without running anything
  kind: 'region';
  region: Region;
}|{
  id: number;
  // runs a module's ui_callback, such as crop's auto crop, and reports the params it changed
  kind: 'callback';
  module: string;
  inst: string;
  param: string;
  arg: number;
}|{
  id: number;
  // works out the plain white balance that matches the open graph's spot white balance
  kind: 'spotWhite';
}|{
  id: number;
  // renders a look of the loaded image in a graph of its own, leaving the open one alone
  kind: 'render';
  key: string;
  graph: string;
  disabled: string[];
  edits: Edit[];
  maxWidth: number;
  maxHeight: number;
}|{
  id: number;
  kind: 'export';
  key: string;
  graph: string;
  disabled: string[];
  edits: Edit[];
  quality: number;
  // EXIF orientation to bake into the JPEG, since vkdt leaves the sensor's orientation alone
  orientation: number;
};

export type Response = {
  id: number;
  ok: true;
  bitmap?: ImageBitmap;
  // the waveform the request asked for, see waveform.ts
  histogram?: ImageBitmap;
  // the part of the output bitmap holds, when it's not the whole
  region?: Region;
  jpeg?: Blob;
  // every module's params after the request, which modules may change beyond what was asked
  state?: ModuleState[];
  // the size of the open image at full resolution, in the sensor's orientation
  fullWidth?: number;
  fullHeight?: number;
  // colour's white red, green, blue and exposure, see wasm_spot_white
  spotWhite?: number[];
  ms: number;
}|{
  id: number;
  ok: false;
  error: string;
};

/** The name an engine key's file has in the engine's file system, under /img. */
export function fileName(key: string): string {
  return key.replace(/[^A-Za-z0-9_.-]/g, '_');
}

export function editLine(e: Edit): string {
  return `param:${e.module}:${e.inst}:${e.name}:${e.values.join(':')}`;
}
