import { Component, Suspense, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import FurnitureModel from './FurnitureModel.jsx';
import Room from './Room.jsx';
import { clampToRoom } from '../room.js';

// Stand-in for a model that failed to load; still selectable so it can be removed.
function BrokenModel({ instance, selected, onSelect }) {
  return (
    <mesh position={[instance.position[0], (instance.position[1] || 0) + 0.25, instance.position[2]]}
      onClick={(e) => { e.stopPropagation(); onSelect(instance.id); }}>
      <boxGeometry args={[0.5, 0.5, 0.5]} />
      <meshStandardMaterial color={selected ? '#ff7a59' : '#d64545'} wireframe />
    </mesh>
  );
}

class ModelBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) { this.props.onError(this.props.instance.assetId, error); }
  render() { return this.state.failed ? <BrokenModel {...this.props} /> : this.props.children; }
}

function LoadingBox({ position }) {
  return (
    <mesh position={[position[0], (position[1] || 0) + 0.25, position[2]]}>
      <boxGeometry args={[0.5, 0.5, 0.5]} />
      <meshStandardMaterial color="#7c8db5" transparent opacity={0.35} />
    </mesh>
  );
}

export default function RoomScene({
  instances, itemsById, selectedId, placing, mode,
  onPlace, onSelect, onChange, onDimensions, onModelError
}) {
  // Set while a TransformControls drag runs, so releasing the gizmo over the floor doesn't deselect.
  const dragging = useRef(false);

  const handleFloorClick = (e) => {
    if (e.delta > 4 || dragging.current) return;
    e.stopPropagation();
    if (placing) onPlace(clampToRoom([e.point.x, 0, e.point.z]));
    else onSelect(null);
  };

  return (
    <Canvas shadows camera={{ position: [6, 6, 8], fov: 45 }}>
      <color attach="background" args={['#e8ebf0']} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[4, 8, 5]} intensity={1.6} castShadow
        shadow-mapSize={[2048, 2048]} shadow-camera-left={-6} shadow-camera-right={6}
        shadow-camera-top={6} shadow-camera-bottom={-6} />
      <Room placing={placing} onFloorClick={handleFloorClick} />
      {instances.map((instance) => {
        const item = itemsById.get(instance.assetId);
        if (!item || item.missing) return null;
        const selected = instance.id === selectedId;
        return (
          <ModelBoundary key={`${instance.id}:${item.modelUrl}`} instance={instance} selected={selected}
            onSelect={onSelect} onError={onModelError}>
            <Suspense fallback={<LoadingBox position={instance.position} />}>
              <FurnitureModel instance={instance} item={item} selected={selected} placing={placing}
                mode={mode} dragging={dragging} onSelect={onSelect} onChange={onChange}
                onDimensions={onDimensions} onPlace={onPlace} />
            </Suspense>
          </ModelBoundary>
        );
      })}
      <OrbitControls makeDefault target={[0, 0.6, 0]} maxPolarAngle={Math.PI / 2.05} minDistance={2} maxDistance={25} />
    </Canvas>
  );
}
