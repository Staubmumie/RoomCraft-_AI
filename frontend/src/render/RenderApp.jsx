// Headless renderer: draws a given room configuration (not the user's visible room) and returns
// a top-down and a perspective screenshot. Driven by the backend through Playwright:
//   await window.roomcraftRender(config)  ->  { images: { top, perspective }, stats } | { error, code }
import { Component, Suspense, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useThree } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import Room from '../components/Room.jsx';
import FurnitureModel from '../components/FurnitureModel.jsx';
import { perspectiveCamera, topCamera } from './cameras.js';

const noop = () => {};
const notDragging = { current: false };

class AssetBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) { this.props.onError(this.props.id, error); }
  render() { return this.state.failed ? null : this.props.children; }
}

class CanvasBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) { this.props.onError(error); }
  render() { return this.state.failed ? null : this.props.children; }
}

// Spread of luminance over a pixel sample; ~0 means an empty or single-colour image.
function contrast(ctx, size) {
  const { data } = ctx.getImageData(0, 0, size, size);
  let sum = 0;
  let sumSq = 0;
  let n = 0;
  for (let i = 0; i < data.length; i += 4 * 61) {
    const l = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    sum += l;
    sumSq += l * l;
    n += 1;
  }
  const mean = sum / n;
  return Math.sqrt(Math.max(sumSq / n - mean * mean, 0));
}

// Numbers 1..n on the top view, in the order of config.furniture, so the designer can
// match what it sees to the furniture ids. Markers of pieces at (nearly) the same spot, e.g. a lamp
// on a table, are moved sideways so both stay readable.
function drawLabels(ctx, camera, furniture, size) {
  ctx.font = 'bold 15px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const drawn = [];
  furniture.forEach((f, index) => {
    const p = new THREE.Vector3(f.x, 0, f.z).project(camera);
    let x = ((p.x + 1) / 2) * size;
    const y = ((1 - p.y) / 2) * size;
    while (drawn.some(([dx, dy]) => Math.hypot(dx - x, dy - y) < 26)) x += 28;
    drawn.push([x, y]);
    ctx.beginPath();
    ctx.arc(x, y, 13, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#1d2330';
    ctx.stroke();
    ctx.fillStyle = '#1d2330';
    ctx.fillText(String(index + 1), x, y + 1);
  });
}

function Capture({ job, failures }) {
  const { gl, scene } = useThree();
  useEffect(() => {
    let cancelled = false;
    const run = () => {
      if (cancelled) return;
      if (failures.current.length) {
        job.resolve({ code: 'asset_load_failed', error: 'Some furniture models could not be loaded.', failures: failures.current });
        return;
      }
      try {
        const { room, furniture, imageSize: size } = job.config;
        gl.setPixelRatio(1);
        gl.setSize(size, size, false);
        const images = {};
        const stats = {};
        for (const [view, camera] of [['top', topCamera(room)], ['perspective', perspectiveCamera(room)]]) {
          scene.updateMatrixWorld(true);
          gl.render(scene, camera);
          const canvas = document.createElement('canvas');
          canvas.width = size;
          canvas.height = size;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(gl.domElement, 0, 0, size, size);
          stats[view] = { contrast: contrast(ctx, size) };
          if (view === 'top') drawLabels(ctx, camera, furniture, size);
          images[view] = canvas.toDataURL('image/png');
        }
        job.resolve({ images, stats });
      } catch (error) {
        job.resolve({ code: 'capture_failed', error: String(error?.message || error) });
      }
    };
    // Two frames after everything has mounted, so models and materials are fully set up.
    requestAnimationFrame(() => requestAnimationFrame(run));
    return () => { cancelled = true; };
  }, [gl, scene, job, failures]);
  return null;
}

function Scene({ job }) {
  const failures = useRef([]);
  const { room, furniture } = job.config;
  const onAssetError = (id, error) => failures.current.push({ id, error: String(error?.message || error) });

  return (
    <>
      <color attach="background" args={['#f4f4f4']} />
      {/* Fixed neutral lighting, independent of anything in the user's view. */}
      <ambientLight intensity={0.8} />
      <hemisphereLight args={['#ffffff', '#d9c7ad', 0.6]} />
      <directionalLight position={[4, 8, 5]} intensity={1.4} />
      <Room room={room} grid="lines" />
      <Suspense fallback={null}>
        {furniture.map((f) => (
          <AssetBoundary key={f.id} id={f.id} onError={onAssetError}>
            <FurnitureModel
              instance={{ id: f.id, assetId: f.id, position: [f.x, f.y ?? 0, f.z], rotation: [0, f.rotation, 0], scale: [f.scale, f.scale, f.scale] }}
              item={{ id: f.id, modelUrl: f.modelUrl, dimensions: true }}
              selected={false} placing={false} mode="translate" dragging={notDragging}
              onSelect={noop} onChange={noop} onDimensions={noop} />
          </AssetBoundary>
        ))}
        {/* Inside the same Suspense boundary: mounts only once every model has loaded (or failed). */}
        <Capture job={job} failures={failures} />
      </Suspense>
    </>
  );
}

// Bounding box of a model in its own units, loaded the same way as useGLTF does in the app
// (Draco and Meshopt compression included). Used after a download to detect the model's scale.
async function measureModel(url) {
  try {
    const loader = new GLTFLoader();
    const draco = new DRACOLoader();
    draco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.5/');
    loader.setDRACOLoader(draco);
    loader.setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(url);
    gltf.scene.updateMatrixWorld(true);
    const size = new THREE.Box3().setFromObject(gltf.scene).getSize(new THREE.Vector3());
    return { size: { width: size.x, height: size.y, depth: size.z } };
  } catch (error) {
    return { error: String(error?.message || error) };
  }
}

function webglAvailable() {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

export default function RenderApp() {
  const [job, setJob] = useState(null);

  useEffect(() => {
    window.roomcraftMeasure = measureModel;
    window.roomcraftRender = (config) => new Promise((resolve) => {
      if (!webglAvailable()) {
        resolve({ code: 'webgl_unavailable', error: 'WebGL is not available in the rendering browser.' });
        return;
      }
      setJob({ id: Date.now(), config, resolve });
    });
    window.roomcraftReady = true;
  }, []);

  if (!job) return null;
  const size = job.config.imageSize;
  return (
    <div style={{ width: size, height: size }}>
      <CanvasBoundary onError={(error) => job.resolve({ code: 'webgl_unavailable', error: String(error?.message || error) })}>
        <Canvas key={job.id} frameloop="never" dpr={1} gl={{ preserveDrawingBuffer: true, antialias: true }}
          style={{ width: size, height: size }}>
          <Scene job={job} />
        </Canvas>
      </CanvasBoundary>
    </div>
  );
}
