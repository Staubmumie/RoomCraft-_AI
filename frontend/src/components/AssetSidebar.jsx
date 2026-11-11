import AssetSearch from './AssetSearch.jsx';
import Inventory from './Inventory.jsx';

export default function AssetSidebar({ open, onToggle, tab, onTab, inventoryCount, searchProps, inventoryProps }) {
  if (!open) {
    return (
      <aside className="sidebar collapsed">
        <button className="icon-button" onClick={onToggle} title="Open sidebar" aria-label="Open sidebar">‹</button>
      </aside>
    );
  }
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'search'} className={tab === 'search' ? 'active' : ''}
            onClick={() => onTab('search')}>Search</button>
          <button role="tab" aria-selected={tab === 'inventory'} className={tab === 'inventory' ? 'active' : ''}
            onClick={() => onTab('inventory')}>Inventory <span className="count">{inventoryCount}</span></button>
        </div>
        <button className="icon-button" onClick={onToggle} title="Close sidebar" aria-label="Close sidebar">›</button>
      </div>
      {/* Both tabs stay mounted so search results survive switching tabs. */}
      <div className="sidebar-body" hidden={tab !== 'search'}><AssetSearch {...searchProps} /></div>
      <div className="sidebar-body" hidden={tab !== 'inventory'}><Inventory {...inventoryProps} /></div>
    </aside>
  );
}
