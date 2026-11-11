"""Furniture geometry: oriented footprints on the floor (X/Z plane, rotation around Y) extruded
over a height range.

Same conventions as the rest of the app: metres, room centred on (0, 0), rotation in radians,
and at rotation 0 `width` runs along X and `depth` along Z. `y` is the height of the BOTTOM of
the piece's bounding box above the floor (0 = standing on the floor), so a piece occupies
[y, y + height]. Footprints are oriented rectangles; floor overlaps use the Separating Axis
Theorem, and two pieces only intersect in 3D if their height ranges overlap as well. A piece
without a known height is treated as reaching from floor to ceiling (the former 2D behaviour).
"""
import math
from dataclasses import dataclass

Vec = tuple[float, float]
SIDES = ("back", "front", "left", "right")


@dataclass(frozen=True)
class Footprint:
    x: float
    z: float
    width: float
    depth: float
    rotation: float
    height: float | None = None
    y: float = 0.0

    @property
    def top(self) -> float | None:
        return None if self.height is None else self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.depth

    def contains(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Is the point (x, z) on the footprint (grown by `margin`)?"""
        (ux, uz), (vx, vz) = self.axes
        dx, dz = x - self.x, z - self.z
        return (abs(dx * ux + dz * uz) <= self.width / 2 + margin
                and abs(dx * vx + dz * vz) <= self.depth / 2 + margin)

    @property
    def axes(self) -> tuple[Vec, Vec]:
        """Unit vectors of the local width and depth directions in world X/Z."""
        c, s = math.cos(self.rotation), math.sin(self.rotation)
        return (c, -s), (s, c)

    @property
    def extent(self) -> Vec:
        """Half size of the axis-aligned box around the rotated footprint."""
        c, s = abs(math.cos(self.rotation)), abs(math.sin(self.rotation))
        return (self.width * c + self.depth * s) / 2, (self.width * s + self.depth * c) / 2

    def moved(self, x: float, z: float, y: float | None = None) -> "Footprint":
        return Footprint(x, z, self.width, self.depth, self.rotation, self.height, self.y if y is None else y)

    def corners(self) -> list[Vec]:
        (ux, uz), (vx, vz) = self.axes
        hw, hd = self.width / 2, self.depth / 2
        return [(self.x + sx * hw * ux + sz * hd * vx, self.z + sx * hw * uz + sz * hd * vz)
                for sx, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def _radius(fp: Footprint, axis: Vec) -> float:
    (ux, uz), (vx, vz) = fp.axes
    return fp.width / 2 * abs(ux * axis[0] + uz * axis[1]) + fp.depth / 2 * abs(vx * axis[0] + vz * axis[1])


def separation_axes(a: Footprint, b: Footprint) -> list[tuple[Vec, float]]:
    """For every SAT axis: (axis pointing from a to b, overlap along it). Overlap <= 0 means separated."""
    result = []
    dx, dz = b.x - a.x, b.z - a.z
    for axis in (*a.axes, *b.axes):
        distance = dx * axis[0] + dz * axis[1]
        sign = 1.0 if distance >= 0 else -1.0
        overlap = _radius(a, axis) + _radius(b, axis) - abs(distance)
        result.append(((axis[0] * sign, axis[1] * sign), overlap))
    return result


def overlap_depth(a: Footprint, b: Footprint) -> float:
    """Penetration depth of two footprints (smallest overlap over all SAT axes); <= 0 if they don't overlap."""
    return min(overlap for _, overlap in separation_axes(a, b))


def vertical_overlap(a: Footprint, b: Footprint) -> float:
    """How far the height ranges overlap (<= 0: one is above the other). Unknown height: infinite."""
    if a.top is None or b.top is None:
        return math.inf
    return min(a.top, b.top) - max(a.y, b.y)


@dataclass(frozen=True)
class RoomBounds:
    width: float
    depth: float
    height: float = math.inf
    walls: tuple[str, ...] = SIDES  # sides with a wall something can be mounted on

    def wall_gap(self, fp: Footprint) -> float:
        """Distance between the footprint and the nearest wall (negative if it sticks into it)."""
        ex, ez = fp.extent
        gaps = {"back": fp.z - ez + self.depth / 2, "front": self.depth / 2 - fp.z - ez,
                "left": fp.x - ex + self.width / 2, "right": self.width / 2 - fp.x - ex}
        return min((gaps[side] for side in self.walls), default=math.inf)

    def vertical_violation(self, fp: Footprint) -> float:
        """How far the piece reaches below the floor or above the ceiling (0 if neither)."""
        return max(0.0, -fp.y, (fp.top or fp.y) - self.height)

    def fits(self, fp: Footprint) -> bool:
        ex, ez = fp.extent
        return ex <= self.width / 2 and ez <= self.depth / 2

    def nearest_inside(self, fp: Footprint) -> Vec | None:
        """Closest centre that keeps the footprint inside the room, or None if it is too large."""
        ex, ez = fp.extent
        if not self.fits(fp):
            return None
        return (min(max(fp.x, -self.width / 2 + ex), self.width / 2 - ex),
                min(max(fp.z, -self.depth / 2 + ez), self.depth / 2 - ez))

    def violation(self, fp: Footprint) -> float:
        """How far the footprint sticks out of the room (0 if inside)."""
        ex, ez = fp.extent
        return max(0.0, abs(fp.x) + ex - self.width / 2, abs(fp.z) + ez - self.depth / 2)


def distance(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
