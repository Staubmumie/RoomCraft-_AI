import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useGLTF } from '@react-three/drei';
import RoomScene from './components/RoomScene.jsx';
import AssetSidebar from './components/AssetSidebar.jsx';
import AiSidebar from './components/AiSidebar.jsx';
import { formatDimensions } from './components/common.jsx';
import { api } from './api.js';
import { ROOM } from './room.js';
import { useDesignRun } from './observability/useDesignRun.js';

// Storage key keeps the former product name so saved layouts survive.
const LAYOUT_KEY = 'roomforge.layout';
const ROTATION_STEP = Math.PI / 4;

function loadLayout() {
  try {
    const layout = JSON.parse(localStorage.getItem(LAYOUT_KEY));
    return Array.isArray(layout) ? layout : [];
  } catch {
    return [];
  }
}

const newInstanceId = () => `instance-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

function SelectionPanel({ instance, item, mode, onMode, onRotate, onScale, onRemove, onFloor }) {
  const elevation = instance.position[1] || 0;
  const scale = instance.scale[0];
  const dims = item?.dimensions && {
    width: item.dimensions.width * scale,
    height: item.dimensions.height * scale,
    depth: item.dimensions.depth * scale
  };
  return (
    <div className="selection-panel">
      <div>
        <strong>{item?.name ?? 'Unknown model'}</strong>
        {dims && <span className="muted small"> · {formatDimensions(dims)}</span>}
        {elevation > 0.005 && <span className="muted small"> · {elevation.toFixed(2)} m above the floor</span>}
      </div>
      <div className="toolbar">
        <div className="segmented">
          <button className={mode === 'translate' ? 'active' : ''} onClick={() => onMode('translate')}>Move</button>
          <button className={mode === 'rotate' ? 'active' : ''} onClick={() => onMode('rotate')}>Rotate</button>
        </div>
        <button onClick={onRotate} title="Rotate 45° (R)">↻ 45°</button>
        {elevation > 0.005 && <button onClick={onFloor} title="Put it back on the floor">↓ Floor</button>}
        <button onClick={() => onScale(1 / 1.25)} title="Smaller">−</button>
        <span className="small">{Math.round(scale * 100)} %</span>
        <button onClick={() => onScale(1.25)} title="Larger">+</button>
        <button className="danger" onClick={onRemove} title="Remove (Del)">Remove</button>
      </div>
    </div>
  );
}

export default function App() {
  const [inventory, setInventory] = useState([]);
  const [inventoryState, setInventoryState] = useState({ loading: true, error: '' });
  const [downloading, setDownloading] = useState(() => new Set());
  const [downloadErrors, setDownloadErrors] = useState({});
  const [instances, setInstances] = useState(loadLayout);
  const [selectedId, setSelectedId] = useState(null);
  const [placingId, setPlacingId] = useState(null);
  const [mode, setMode] = useState('translate');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [aiSidebarOpen, setAiSidebarOpen] = useState(true);
  const [layoutBeforeDesign, setLayoutBeforeDesign] = useState(null);
  const [tab, setTab] = useState('search');
  const [notice, setNotice] = useState('');
  const dimensionRequests = useRef(new Set());
  const designRun = useDesignRun();

  const itemsById = useMemo(() => new Map(inventory.map((item) => [item.id, item])), [inventory]);
  const inventoryAssetIds = useMemo(() => new Set(inventory.map((item) => item.assetId)), [inventory]);
  const placedCounts = useMemo(() => {
    const counts = new Map();
    instances.forEach((i) => counts.set(i.assetId, (counts.get(i.assetId) || 0) + 1));
    return counts;
  }, [instances]);
  const selected = instances.find((i) => i.id === selectedId) ?? null;
  const placingItem = placingId ? itemsById.get(placingId) : null;

  const loadInventory = useCallback(async () => {
    setInventoryState({ loading: true, error: '' });
    try {
      setInventory(await api.inventory());
      setInventoryState({ loading: false, error: '' });
    } catch (err) {
      setInventoryState({ loading: false, error: err.message });
    }
  }, []);

  useEffect(() => { loadInventory(); }, [loadInventory]);

  // Drop saved instances whose inventory item no longer exists (only once the inventory is known).
  useEffect(() => {
    if (inventoryState.loading || inventoryState.error) return;
    setInstances((list) => {
      const kept = list.filter((i) => itemsById.has(i.assetId));
      return kept.length === list.length ? list : kept;
    });
  }, [itemsById, inventoryState]);

  useEffect(() => {
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(instances)); } catch { /* storage unavailable */ }
  }, [instances]);

  const download = async (asset) => {
    if (downloading.has(asset.id) || inventoryAssetIds.has(asset.id)) return;
    setDownloading((s) => new Set(s).add(asset.id));
    setDownloadErrors(({ [asset.id]: _, ...rest }) => rest);
    try {
      const item = await api.download(asset.id);
      setInventory((list) => [item, ...list]);
    } catch (err) {
      setDownloadErrors((e) => ({ ...e, [asset.id]: err.message }));
    } finally {
      setDownloading((s) => {
        const next = new Set(s);
        next.delete(asset.id);
        return next;
      });
    }
  };

  const deleteItem = async (item) => {
    const placed = placedCounts.get(item.id) || 0;
    const question = placed
      ? `Delete “${item.name}”? ${placed} placed object(s) will be removed as well.`
      : `Delete “${item.name}” from the inventory?`;
    if (!window.confirm(question)) return;
    try {
      await api.remove(item.id);
      useGLTF.clear(item.modelUrl);
      setInventory((list) => list.filter((i) => i.id !== item.id));
      setInstances((list) => list.filter((i) => i.assetId !== item.id));
      if (placingId === item.id) setPlacingId(null);
    } catch (err) {
      setInventoryState((s) => ({ ...s, error: err.message }));
    }
  };

  const togglePlacing = (item) => {
    setSelectedId(null);
    setPlacingId((id) => (id === item.id ? null : item.id));
  };

  const place = (position) => {
    // Models not built in metres (e.g. centimetres) come with a unit scale from the download.
    const s = itemsById.get(placingId)?.unitScale ?? 1;
    const instance = { id: newInstanceId(), assetId: placingId, position, rotation: [0, 0, 0], scale: [s, s, s] };
    setInstances((list) => [...list, instance]);
    setPlacingId(null);
    setSelectedId(instance.id);
    setMode('translate');
  };

  const updateInstance = useCallback((id, patch) => {
    setInstances((list) => list.map((i) => (i.id === id ? { ...i, ...patch } : i)));
  }, []);

  const removeInstance = (id) => {
    setInstances((list) => list.filter((i) => i.id !== id));
    setSelectedId(null);
  };

  const rotateSelected = () => {
    if (selected) updateInstance(selected.id, { rotation: [0, selected.rotation[1] + ROTATION_STEP, 0] });
  };

  const scaleSelected = (factor) => {
    if (!selected) return;
    const s = Math.min(Math.max(selected.scale[0] * factor, 0.01), 100);
    updateInstance(selected.id, { scale: [s, s, s] });
  };

  // Snapshot of the room for the Interior Designer, in the backend's Room/Furniture format.
  const buildDesignInput = () => ({
    room: {
      width: ROOM.width,
      depth: ROOM.depth,
      height: ROOM.height,
      walls: [`back (z = ${-ROOM.depth / 2})`, `left (x = ${-ROOM.width / 2})`]
    },
    furniture: instances.map((instance) => {
      const item = itemsById.get(instance.assetId);
      const scale = instance.scale[0];
      const dims = item?.dimensions;
      return {
        id: instance.id,
        name: item?.name ?? 'Unknown piece',
        assetId: instance.assetId,
        width: dims ? dims.width * scale : null,
        depth: dims ? dims.depth * scale : null,
        height: dims ? dims.height * scale : null,
        scale,
        x: instance.position[0],
        y: instance.position[1] || 0,
        z: instance.position[2],
        rotation: instance.rotation[1]
      };
    })
  });

  // Applies a validated design through the same state as manual edits: existing pieces move,
  // removed pieces disappear, new pieces (ids "new-...", inventory assetId) become new instances.
  const applyDesign = useCallback((placements, removed) => {
    setInstances((list) => {
      const byId = new Map(placements.map((p) => [p.id, p]));
      const gone = new Set(removed);
      const known = new Set(list.map((i) => i.id));
      const kept = list.filter((i) => !gone.has(i.id)).map((i) => {
        const p = byId.get(i.id);
        return p ? { ...i, position: [p.x, p.y ?? 0, p.z], rotation: [0, p.rotation, 0] } : i;
      });
      const added = placements.filter((p) => !known.has(p.id)).map((p) => {
        const s = p.scale ?? 1;
        return {
          id: newInstanceId(), assetId: p.assetId, position: [p.x, p.y ?? 0, p.z], rotation: [0, p.rotation, 0], scale: [s, s, s]
        };
      });
      return [...kept, ...added];
    });
  }, []);

  const generateDesign = async ({ request, model, models, options }) => {
    const snapshot = instances;
    let result;
    try {
      // Furniture downloaded during the run shows up in the inventory right away.
      result = await designRun.start({ request, model, models, options, ...buildDesignInput() },
        { onDownloaded: loadInventory });
    } catch (err) {
      // Aborted or failed: the room stays as it is, but furniture downloaded meanwhile is already on the
      // server. Downloads that finished during an approval pause were never reported, so always reload.
      loadInventory();
      throw err;
    }
    // Furniture the run downloaded (directly or via the designer's delegated research) is new on the
    // server: load it before the layout is applied, so the new pieces show up in the inventory and the room.
    const unknownAsset = (result?.placements ?? []).some((p) => p.assetId && !itemsById.has(p.assetId));
    if (result?.downloads?.added?.length || unknownAsset) await loadInventory();
    // Other request kinds answer with a message and leave the room as it is.
    if (result?.kind !== 'design' || (!result.placements?.length && !result.removed?.length)) return result;
    const known = new Set(snapshot.map((i) => i.id));
    const valid = Array.isArray(result.placements) && result.placements.every((p) =>
      [p.x, p.y ?? 0, p.z, p.rotation].every(Number.isFinite) && (known.has(p.id) || p.assetId));
    if (!valid) throw new Error('The backend returned an invalid layout. The current layout was kept.');
    setLayoutBeforeDesign(snapshot);
    applyDesign(result.placements, result.removed ?? []);
    return result;
  };

  // Empties the room (the inventory stays). "Previous layout" in the Design tab brings it back.
  const removeAll = () => {
    if (!instances.length) return;
    if (!window.confirm(`Remove all ${instances.length} pieces from the room? The inventory is kept, and “Previous layout” in the Design tab restores the room.`)) return;
    setLayoutBeforeDesign(instances);
    setInstances([]);
    setSelectedId(null);
    setPlacingId(null);
  };

  // Restores the complete room as it was before the design (or "Remove all"), including removed and added pieces.
  const undoDesign = () => {
    if (!layoutBeforeDesign) return;
    setInstances(layoutBeforeDesign);
    setLayoutBeforeDesign(null);
  };

  // The real size is only known once Three.js has loaded the model, so the scene reports it back once.
  const saveDimensions = useCallback((itemId, dimensions) => {
    if (dimensionRequests.current.has(itemId)) return;
    dimensionRequests.current.add(itemId);
    api.setDimensions(itemId, dimensions)
      .then((updated) => {
        setInventory((list) => list.map((i) => (i.id === itemId ? updated : i)));
        // Pieces placed before the measurement used scale 1; a model in other units gets its unit scale now.
        const s = updated.unitScale ?? 1;
        if (s !== 1) {
          setInstances((list) => list.map((i) => (i.assetId === itemId && i.scale[0] === 1 ? { ...i, scale: [s, s, s] } : i)));
        }
      })
      .catch(() => dimensionRequests.current.delete(itemId));
  }, []);

  const reportModelError = useCallback((itemId, error) => {
    console.error(`Model ${itemId} failed to load`, error);
    setNotice('A model could not be loaded and is shown as a red box.');
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest?.('input, textarea')) return;
      if (e.key === 'Escape') {
        setPlacingId(null);
        setSelectedId(null);
      } else if (selected && (e.key === 'Delete' || e.key === 'Backspace')) {
        removeInstance(selected.id);
      } else if (selected && (e.key === 'r' || e.key === 'R')) {
        rotateSelected();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  return (
    <div className={`app ${placingItem ? 'is-placing' : ''}`}>
      <AiSidebar open={aiSidebarOpen} onToggle={() => setAiSidebarOpen((v) => !v)} designRun={designRun}
        designProps={{
          furnitureCount: instances.length,
          onGenerate: generateDesign,
          onCancel: designRun.cancel,
          cancelling: designRun.cancelling,
          canUndo: Boolean(layoutBeforeDesign),
          onUndo: undoDesign
        }} />

      <main className="scene">
        <RoomScene instances={instances} itemsById={itemsById} selectedId={selectedId}
          placing={Boolean(placingItem)} mode={mode}
          onPlace={place} onSelect={setSelectedId} onChange={updateInstance}
          onDimensions={saveDimensions} onModelError={reportModelError} />

        <header className="topbar">
          <h1>RoomCraft</h1>
          <span className="muted small">{instances.length} {instances.length === 1 ? 'object' : 'objects'} in the room</span>
          {instances.length > 0 && (
            <button className="danger topbar-action" onClick={removeAll} disabled={designRun.active}
              title={designRun.active ? 'Not available while the agents are working' : 'Remove every piece from the room (the inventory is kept)'}>
              Remove all
            </button>
          )}
        </header>

        {notice && (
          <div className="notice">{notice} <button className="link" onClick={() => setNotice('')}>OK</button></div>
        )}

        {placingItem && (
          <div className="hint">
            Click the floor, or the top of a piece of furniture, to place “{placingItem.name}”.
            <button onClick={() => setPlacingId(null)}>Cancel (Esc)</button>
          </div>
        )}

        {selected && !placingItem && (
          <SelectionPanel instance={selected} item={itemsById.get(selected.assetId)} mode={mode} onMode={setMode}
            onRotate={rotateSelected} onScale={scaleSelected} onRemove={() => removeInstance(selected.id)}
            onFloor={() => updateInstance(selected.id, { position: [selected.position[0], 0, selected.position[2]] })} />
        )}

        {!selected && !placingItem && instances.length === 0 && (
          <div className="hint subtle">Search for furniture on the right, add it to the inventory and place it in the room.</div>
        )}
      </main>

      <AssetSidebar open={sidebarOpen} onToggle={() => setSidebarOpen((v) => !v)} tab={tab} onTab={setTab}
        inventoryCount={inventory.length}
        searchProps={{ inventoryAssetIds, downloading, downloadErrors, onDownload: download }}
        inventoryProps={{
          items: inventory, loading: inventoryState.loading, error: inventoryState.error,
          placingId, placedCounts, onPlace: togglePlacing, onDelete: deleteItem, onReload: loadInventory
        }} />
    </div>
  );
}
