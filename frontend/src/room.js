export const ROOM = { width: 8, depth: 6, height: 2.8 };

// Positions are [x, y, z]: x/z the centre of the footprint, y the height of the model's lowest point
// above the floor (0 = on the floor). FurnitureModel shifts every model so its lowest point sits at y.
export function clampToRoom([x, y, z]) {
  const clamp = (v, max) => Math.min(Math.max(v, -max), max);
  return [clamp(x, ROOM.width / 2), Math.min(Math.max(y || 0, 0), ROOM.height), clamp(z, ROOM.depth / 2)];
}
