import * as THREE from 'three';

export const VIEW_MARGIN = 0.5; // metres around the room in the top view

// Orthographic camera straight above the room centre. Image right = +X, image up = -Z (back wall),
// so a point (x, z) lands at ((x / half + 1) / 2, (z / half + 1) / 2) of the image.
export function topCamera(room) {
  const half = Math.max(room.width, room.depth) / 2 + VIEW_MARGIN;
  const camera = new THREE.OrthographicCamera(-half, half, half, -half, 0.1, 100);
  camera.position.set(0, 30, 0);
  camera.up.set(0, 0, -1);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}

// Same direction as the app's default view (front right, elevated). The distance fits the
// room's bounding sphere into the field of view, so walls and furniture are never cut off.
const PERSPECTIVE_DIRECTION = new THREE.Vector3(6, 6, 8).normalize();

export function perspectiveCamera(room, fov = 40) {
  const radius = 0.5 * Math.hypot(room.width, room.height, room.depth);
  const distance = (radius / Math.sin(THREE.MathUtils.degToRad(fov / 2))) * 1.02;
  const target = new THREE.Vector3(0, room.height * 0.25, 0);
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, distance * 4);
  camera.position.copy(target).addScaledVector(PERSPECTIVE_DIRECTION, distance);
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}
