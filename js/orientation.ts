// Applying EXIF orientation, which neither embedded previews nor vkdt's output have baked in.

/** Draws a bitmap onto a new canvas with the EXIF orientation applied. */
export function drawOriented(bitmap: ImageBitmap, orientation: number): OffscreenCanvas {
  const swap = orientation >= 5;
  const w = swap ? bitmap.height : bitmap.width;
  const h = swap ? bitmap.width : bitmap.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('2d canvas unavailable');
  }
  applyOrientation(ctx, orientation, bitmap.width, bitmap.height);
  ctx.drawImage(bitmap, 0, 0);
  return canvas;
}

/** Sets a transform mapping an unrotated w by h image onto its EXIF-oriented canvas. */
export function applyOrientation(
    ctx: CanvasTransform, orientation: number, w: number, h: number): void {
  // https://magnushoff.com/articles/jpeg-orientation/
  switch (orientation) {
    case 2: ctx.setTransform(-1, 0, 0, 1, w, 0); break;
    case 3: ctx.setTransform(-1, 0, 0, -1, w, h); break;
    case 4: ctx.setTransform(1, 0, 0, -1, 0, h); break;
    case 5: ctx.setTransform(0, 1, 1, 0, 0, 0); break;
    case 6: ctx.setTransform(0, 1, -1, 0, h, 0); break;
    case 7: ctx.setTransform(0, -1, -1, 0, h, w); break;
    case 8: ctx.setTransform(0, -1, 1, 0, 0, w); break;
    default: ctx.setTransform(1, 0, 0, 1, 0, 0); break;
  }
}

/**
 * Maps a point on the displayed image, as fractions of its size, back through its EXIF
 * orientation to the sensor's orientation that vkdt works in. Inverts displayPoint.
 */
export function sensorPoint(orientation: number, u: number, v: number): [number, number] {
  switch (orientation) {
    case 2: return [1 - u, v];
    case 3: return [1 - u, 1 - v];
    case 4: return [u, 1 - v];
    case 5: return [v, u];
    case 6: return [v, 1 - u];
    case 7: return [1 - v, 1 - u];
    case 8: return [1 - v, u];
    default: return [u, v];
  }
}

/** Maps a point in the sensor's orientation to where it shows on the displayed image. */
export function displayPoint(orientation: number, x: number, y: number): [number, number] {
  switch (orientation) {
    case 2: return [1 - x, y];
    case 3: return [1 - x, 1 - y];
    case 4: return [x, 1 - y];
    case 5: return [y, x];
    case 6: return [1 - y, x];
    case 7: return [1 - y, 1 - x];
    case 8: return [y, 1 - x];
    default: return [x, y];
  }
}
