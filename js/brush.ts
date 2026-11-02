// Brush strokes as a mask, for the i-brush module in cc/modules/i-brush.

// vertex flags, see i-brush/main.c
export const STROKE_START = 1;
export const STROKE_ERASE = 2;

/** A stroke's points, as fractions of the image in the sensor's orientation. */
export interface Stroke {
  points: Array<[number, number]>;
  // as a fraction of the image's longer side
  radius: number;
  erase: boolean;
}

// The vertices i-brush's strokes param holds. Keep in sync with its params file.
export const MAX_VERTICES = 2000;

/** Packs strokes into i-brush's strokes param, dropping whatever doesn't fit. */
export function packStrokes(strokes: Stroke[]): number[] {
  const values: number[] = [0];
  let count = 0;
  for (const s of strokes) {
    s.points.forEach(([x, y], i) => {
      if (count === MAX_VERTICES) return;
      const flags = (i === 0 ? STROKE_START : 0) | (s.erase ? STROKE_ERASE : 0);
      values.push(x, y, s.radius, flags);
      count++;
    });
  }
  values[0] = count;
  return values;
}

/** Unpacks i-brush's strokes param. */
export function unpackStrokes(values: number[]): Stroke[] {
  const strokes: Stroke[] = [];
  const count = Math.min(values[0] ?? 0, MAX_VERTICES);
  for (let i = 0; i < count; i++) {
    const [x, y, radius, flags] = values.slice(1 + 4 * i, 5 + 4 * i);
    if ((flags & STROKE_START) || strokes.length === 0) {
      strokes.push({points: [], radius, erase: (flags & STROKE_ERASE) !== 0});
    }
    strokes[strokes.length - 1].points.push([x, y]);
  }
  return strokes;
}

/**
 * Draws packed vertices into a wd by ht mask. Strokes paint by taking the maximum and erase by
 * taking the minimum, so overlaps never add up past the brush's strength.
 */
export function rasterise(
    vertices: Float32Array, wd: number, ht: number, hardness: number, opacity: number):
    Float32Array {
  const canvas = new OffscreenCanvas(wd, ht);
  const ctx = canvas.getContext('2d', {willReadFrequently: true})!;
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, wd, ht);
  const scale = Math.max(wd, ht);
  let previous: [number, number]|undefined;
  for (let i = 0; i + 3 < vertices.length; i += 4) {
    const x = vertices[i] * wd, y = vertices[i + 1] * ht, r = Math.max(0.5, vertices[i + 2] * scale);
    const flags = vertices[i + 3];
    const erase = (flags & STROKE_ERASE) !== 0;
    if (flags & STROKE_START) previous = undefined;
    ctx.globalCompositeOperation = erase ? 'darken' : 'lighten';
    // stamps close enough together that the stroke's edge looks smooth
    const [px, py] = previous ?? [x, y];
    const steps = Math.max(1, Math.ceil(Math.hypot(x - px, y - py) / (r * 0.2)));
    for (let k = previous ? 1 : 0; k <= steps; k++) {
      const cx = px + (x - px) * k / steps, cy = py + (y - py) * k / steps;
      const g = ctx.createRadialGradient(cx, cy, r * Math.min(0.99, hardness), cx, cy, r);
      g.addColorStop(0, erase ? 'black' : 'white');
      g.addColorStop(1, erase ? 'white' : 'black');
      ctx.fillStyle = g;
      ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);
    }
    previous = [x, y];
  }
  const pixels = ctx.getImageData(0, 0, wd, ht).data;
  const mask = new Float32Array(wd * ht);
  for (let p = 0; p < mask.length; p++) mask[p] = pixels[4 * p] / 255 * opacity;
  return mask;
}
