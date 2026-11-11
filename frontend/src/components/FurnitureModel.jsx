import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { TransformControls, useGLTF } from '@react-three/drei';
import { clone as cloneSkeleton } from 'three/examples/jsm/utils/SkeletonUtils.js';
import { clampToRoom } from '../room.js';

const UP = new THREE.Vector3(0, 1, 0);

export default function FurnitureModel({
  instance, item, selected, placing, mode, dragging, onSelect, onChange, onDimensions, onPlace
}) {
  const { scene } = useGLTF(item.modelUrl);
  const [group, setGroup] = useState(null);

  // Each instance gets its own copy. The model is shifted so the bottom centre of its bounding box is the
  // instance origin: position y = height of the lowest point above the floor (0 stands on the floor).
  const { model, offset, size, outline } = useMemo(() => {
    const model = cloneSkeleton(scene);
    model.traverse((o) => {
      if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; }
    });
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const outline = new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z));
    return { model, offset: [-center.x, -box.min.y, -center.z], size, outline };
  }, [scene]);

  useEffect(() => () => outline.dispose(), [outline]);

  useEffect(() => {
    if (!item.dimensions && size.x > 0 && size.y > 0 && size.z > 0) {
      onDimensions(item.id, { width: size.x, height: size.y, depth: size.z });
    }
  }, [item.id, item.dimensions, size, onDimensions]);

  const commitTransform = () => {
    onChange(instance.id, {
      position: clampToRoom([group.position.x, group.position.y, group.position.z]),
      rotation: [0, group.rotation.y, 0]
    });
    setTimeout(() => { dragging.current = false; }, 0);
  };

  // In placement mode a click on the upper side of this piece puts the new piece on top of it;
  // a click on its side places on the floor below the click.
  const placeOnTop = (e) => {
    e.stopPropagation();
    const normal = e.face ? e.face.normal.clone().transformDirection(e.object.matrixWorld) : UP;
    onPlace?.(clampToRoom([e.point.x, normal.dot(UP) > 0.7 ? e.point.y : 0, e.point.z]));
  };

  return (
    <>
      <group ref={setGroup} position={instance.position} rotation={instance.rotation} scale={instance.scale}
        onClick={(e) => {
          if (e.delta > 4) return;
          if (placing) { placeOnTop(e); return; }
          e.stopPropagation();
          onSelect(instance.id);
        }}>
        <group position={offset}>
          <primitive object={model} />
        </group>
        {selected && (
          <lineSegments geometry={outline} position={[0, size.y / 2, 0]} raycast={() => null}>
            <lineBasicMaterial color="#3b6ef5" />
          </lineSegments>
        )}
      </group>
      {selected && group && (
        <TransformControls object={group} mode={mode} size={0.8}
          showX={mode === 'translate'} showY showZ={mode === 'translate'}
          onMouseDown={() => { dragging.current = true; }} onMouseUp={commitTransform} />
      )}
    </>
  );
}
