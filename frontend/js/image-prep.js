// Staging-time image preparation: cap the resolution, and hold BYTES rather
// than base64 until the moment of send.
//
// WHY. Measured through the real staging path: eight 3MB photos from a camera
// roll became a single 32MB POST (base64 is 1.33x, exactly). Sustained radio
// transmit is one of the largest discrete power draws on a phone, and the page
// also held three simultaneous copies of that base64 -- the retained data URL,
// the slice taken to build the content block, and the JSON string of the whole
// body. Nothing anywhere downscaled: the file that left the camera roll was the
// file that went on the wire, to be painted in a 260px box.
//
// WHAT THIS DOES NOT DECIDE. How much resolution a vision model can actually
// use is a MODEL question, not a transport one, and the answer is not uniform:
// the Qwen-VL family in this repo's modelzoo declares dynamic resolution
// (preprocessor_config.json: `size.longest_edge` is a PIXEL BUDGET of 16777216,
// ~4096x4096), so it consumes what it is given and pays for it in vision tokens
// and prefill; a fixed-input tower discards the surplus instead. MAX_EDGE_PX
// below is therefore a deliberate default, not a derived one -- it is the knob
// to turn if image detail matters more to you than upload size and prefill.

// Longest edge, in pixels, that a staged image is reduced to. 2048 keeps
// screenshot text legible and is far above what a fixed-input vision tower
// consumes, while taking a 12MP phone photo down by roughly an order of
// magnitude. Raise it if you feed a dynamic-resolution model images whose fine
// detail is the point.
export const MAX_EDGE_PX = 2048;

// JPEG quality for re-encoded photos. 0.85 is the usual "no visible loss at
// normal viewing" point and roughly halves the bytes versus 0.95.
const JPEG_QUALITY = 0.85;

// PNG in, PNG out: a re-encoded screenshot with text in it shows JPEG ringing,
// and flat UI colours are exactly what PNG compresses well. Everything else
// (JPEG, HEIC/HEIF off an iPhone, WebP) becomes JPEG.
const keepsPng = (type) => type === 'image/png';

// Decode a file to something drawable, honouring EXIF orientation.
//
// The orientation is load-bearing, not a nicety: phone cameras routinely store
// a landscape sensor read plus a rotation flag, and a canvas draw that ignores
// the flag silently rotates the user's photo. `createImageBitmap` takes an
// explicit option for it; the <img> fallback applies it by default.
async function decode(file) {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch { /* fall through -- e.g. a format this path cannot decode */ }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error('decode failed'));
      img.src = url;
    });
    // decode() resolves once the bitmap is actually ready to draw; without it
    // a draw can land on an undecoded image in some engines.
    if (img.decode) { try { await img.decode(); } catch { /* onload was enough */ } }
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

const bitmapWidth = (b) => b.width ?? b.naturalWidth;
const bitmapHeight = (b) => b.height ?? b.naturalHeight;

function toBlob(canvas, type, quality) {
  if (canvas.convertToBlob) return canvas.convertToBlob({ type, quality });
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('encode failed'))), type, quality);
  });
}

// Prepare one image file for staging.
//
// Returns { blob, mediaType, width, height, resized }. `blob` may BE the
// original file -- an image already within the cap is passed through untouched
// rather than re-encoded, because a lossy round-trip that saves nothing is
// strictly worse than doing nothing.
//
// Never throws: any failure to decode or encode returns the original file. A
// downscale bug must not be able to lose the user's picture, and an image this
// path cannot read may still be one the model can.
export async function prepareImage(file) {
  let bitmap = null;
  try {
    bitmap = await decode(file);
    const w = bitmapWidth(bitmap);
    const h = bitmapHeight(bitmap);
    if (!w || !h) return { blob: file, mediaType: file.type, width: 0, height: 0, resized: false };
    const longest = Math.max(w, h);
    if (longest <= MAX_EDGE_PX) {
      return { blob: file, mediaType: file.type, width: w, height: h, resized: false };
    }
    const scale = MAX_EDGE_PX / longest;
    const outW = Math.max(1, Math.round(w * scale));
    const outH = Math.max(1, Math.round(h * scale));
    const canvas = typeof OffscreenCanvas === 'function'
      ? new OffscreenCanvas(outW, outH)
      : Object.assign(document.createElement('canvas'), { width: outW, height: outH });
    const cx = canvas.getContext('2d');
    if (!cx) return { blob: file, mediaType: file.type, width: w, height: h, resized: false };
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(bitmap, 0, 0, outW, outH);
    const type = keepsPng(file.type) ? 'image/png' : 'image/jpeg';
    const blob = await toBlob(canvas, type, JPEG_QUALITY);
    // A "downscale" that produced MORE bytes than it started with is a loss on
    // both counts. Keep whichever is smaller.
    if (!blob || blob.size >= file.size) {
      return { blob: file, mediaType: file.type, width: w, height: h, resized: false };
    }
    return { blob, mediaType: type, width: outW, height: outH, resized: true };
  } catch {
    return { blob: file, mediaType: file.type, width: 0, height: 0, resized: false };
  } finally {
    bitmap?.close?.();
  }
}

// Blob -> bare base64 (no `data:` prefix), for the JSON wire.
//
// Called at SEND, one attachment at a time, so the base64 exists for as long as
// the request body is being built rather than for the whole time something sits
// staged in the composer.
export function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      resolve(url.slice(url.indexOf(',') + 1));
    };
    reader.onerror = () => reject(reader.error ?? new Error('read failed'));
    reader.readAsDataURL(blob);
  });
}
