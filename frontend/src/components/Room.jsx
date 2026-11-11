import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Grid } from '@react-three/drei';
import { ROOM } from '../room.js';

const WALL = 0.1;

// Plain 1 m grid lines. Unlike drei's <Grid> it needs no per-frame camera updates,
// so it also works for the renderer, which draws single frames on demand.
function LineGrid({ width, depth }) {
  const geometry = useMemo(() => {
    const points = [];
    for (let x = -Math.floor(width / 2); x <= width / 2; x += 1) points.push(x, 0, -depth / 2, x, 0, depth / 2);
    for (let z = -Math.floor(depth / 2); z <= depth / 2; z += 1) points.push(-width / 2, 0, z, width / 2, 0, z);
    return new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  }, [width, depth]);
  return (
    <lineSegments geometry={geometry} position={[0, 0.002, 0]}>
      <lineBasicMaterial color="#9c876a" />
    </lineSegments>
  );
}

// Floor and walls. Used by the interactive scene and by the headless renderer.
export default function Room({ room = ROOM, placing = false, onFloorClick, grid = 'shader' }) {
  const cursor = useRef();
  const moveCursor = (e) => {
    if (cursor.current) cursor.current.position.set(e.point.x, 0.01, e.point.z);
  };
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow
        onClick={onFloorClick} onPointerMove={placing ? moveCursor : undefined}>
        <planeGeometry args={[room.width, room.depth]} />
        <meshStandardMaterial color="#d9c7ad" />
      </mesh>
      {grid === 'shader' && (
        <Grid args={[room.width, room.depth]} position={[0, 0.002, 0]} cellSize={0.5} sectionSize={1}
          cellColor="#b9a68b" sectionColor="#9c876a" fadeDistance={40} />
      )}
      {grid === 'lines' && <LineGrid width={room.width} depth={room.depth} />}
      <mesh position={[0, room.height / 2, -room.depth / 2 - WALL / 2]} receiveShadow>
        <boxGeometry args={[room.width + WALL * 2, room.height, WALL]} />
        <meshStandardMaterial color="#f1ede6" />
      </mesh>
      <mesh position={[-room.width / 2 - WALL / 2, room.height / 2, 0]} receiveShadow>
        <boxGeometry args={[WALL, room.height, room.depth]} />
        <meshStandardMaterial color="#ebe5dc" />
      </mesh>
      {placing && (
        <mesh ref={cursor} rotation={[-Math.PI / 2, 0, 0]} position={[0, -10, 0]} raycast={() => null}>
          <ringGeometry args={[0.25, 0.32, 40]} />
          <meshBasicMaterial color="#3b6ef5" />
        </mesh>
      )}
    </group>
  );
}
