/**
 * Snapshot capture utility for the Poincaré disc canvas.
 *
 * Captures the current state of the Three.js canvas as a base64 PNG string,
 * downscaling if the canvas exceeds maxSize×maxSize.
 *
 * Requirements: 3.1, 3.4
 */

/**
 * Keywords that suggest the user is referencing visual patterns on the disc.
 * When detected, the chat component auto-attaches a snapshot.
 * Requirements: 3.3
 */
const VISUAL_PATTERN_KEYWORDS = [
  "see", "look", "cluster", "pattern", "show",
  "visual", "color", "colours", "colors", "bright",
  "group", "clump", "spread", "dense", "sparse",
];

/**
 * Check if a message references visual patterns on the disc.
 * Returns true if the message likely benefits from a visual snapshot.
 * Requirements: 3.3
 */
export function messageReferencesVisualPatterns(message: string): boolean {
  const lower = message.toLowerCase();
  return VISUAL_PATTERN_KEYWORDS.some((kw) => lower.includes(kw));
}

/**
 * Capture the Poincaré disc canvas as a base64 PNG string.
 *
 * @param canvas - The HTMLCanvasElement rendered by Three.js
 * @param maxSize - Maximum dimension (width or height) in pixels. Default 512.
 * @returns base64 PNG string (including data URI prefix)
 */
export function capturePoincareSnapshot(
  canvas: HTMLCanvasElement,
  maxSize: number = 512,
): string {
  const { width, height } = canvas;

  // If canvas fits within maxSize, export directly
  if (width <= maxSize && height <= maxSize) {
    return canvas.toDataURL("image/png");
  }

  // Downscale: maintain aspect ratio, fit within maxSize×maxSize
  const scale = maxSize / Math.max(width, height);
  const targetWidth = Math.round(width * scale);
  const targetHeight = Math.round(height * scale);

  const offscreen = document.createElement("canvas");
  offscreen.width = targetWidth;
  offscreen.height = targetHeight;

  const ctx = offscreen.getContext("2d");
  if (!ctx) {
    // Fallback: return full-size if we can't get a 2D context
    return canvas.toDataURL("image/png");
  }

  ctx.drawImage(canvas, 0, 0, targetWidth, targetHeight);
  return offscreen.toDataURL("image/png");
}
