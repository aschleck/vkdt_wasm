// A waveform of the output: across, the image's columns as displayed, and up, each of red, green
// and blue's levels there, brighter for more pixels at that level.

import { displayPoint } from './orientation';

/** What a waveform covers: the image in its EXIF orientation, or a crop of it. */
export interface Waveform {
  orientation: number;
  // x0, x1, y0, y1 as fractions of the displayed image
  crop?: [number, number, number, number];
}

const WIDTH = 512;
const HEIGHT = 128;
// Enough samples to look the same as all of them, and few enough to draw for every edit.
const SAMPLES = 500_000;

/** Draws the waveform of an 8-bit RGBA image in the sensor's orientation. */
export function drawWaveform(image: ImageData, waveform: Waveform): ImageData {
  const {width, height, data} = image;
  const [x0, x1, y0, y1] = waveform.crop ?? [0, 1, 0, 1];
  const counts = new Uint32Array(WIDTH * HEIGHT * 3);
  const step = Math.max(1, Math.round(Math.sqrt(width * height * (x1 - x0) * (y1 - y0) / SAMPLES)));
  let samples = 0;
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const [dx, dy] = displayPoint(waveform.orientation, (x + 0.5) / width, (y + 0.5) / height);
      if (dx < x0 || dx >= x1 || dy < y0 || dy >= y1) continue;
      const column = Math.min(WIDTH - 1, Math.floor((dx - x0) / (x1 - x0) * WIDTH));
      const i = 4 * (y * width + x);
      for (let c = 0; c < 3; c++) {
        const row = HEIGHT - 1 - (data[i + c] * (HEIGHT - 1) / 255 | 0);
        counts[3 * (row * WIDTH + column) + c]++;
      }
      samples++;
    }
  }
  // a bin with this many is at full brightness, four times what an even spread would give it
  const full = Math.max(1, 4 * samples / (WIDTH * HEIGHT));
  const out = new ImageData(WIDTH, HEIGHT);
  for (let i = 0; i < WIDTH * HEIGHT; i++) {
    for (let c = 0; c < 3; c++) {
      // square root so that sparse levels still show
      out.data[4 * i + c] = 255 * Math.sqrt(Math.min(1, counts[3 * i + c] / full));
    }
    out.data[4 * i + 3] = 255;
  }
  return out;
}
