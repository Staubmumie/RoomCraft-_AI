import { useState } from 'react';
import DesignPanel from './DesignPanel.jsx';
import OverviewPanel from './overview/OverviewPanel.jsx';

export default function AiSidebar({ open, onToggle, designProps, designRun }) {
  const [tab, setTab] = useState('design');

  // Jump to the live view as soon as a design request is submitted.
  const onGenerate = (input) => {
    setTab('overview');
    return designProps.onGenerate(input);
  };

  if (!open) {
    return (
      <aside className="sidebar left collapsed">
        <button className="icon-button" onClick={onToggle} title="Open AI sidebar" aria-label="Open AI sidebar">›</button>
      </aside>
    );
  }
  return (
    <aside className="sidebar left">
      <div className="sidebar-header">
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'design'} className={tab === 'design' ? 'active' : ''}
            onClick={() => setTab('design')}>Design</button>
          <button role="tab" aria-selected={tab === 'overview'} className={tab === 'overview' ? 'active' : ''}
            onClick={() => setTab('overview')}>
            Overview {designRun.run.status === 'running' && <span className="spinner" />}
          </button>
        </div>
        <button className="icon-button" onClick={onToggle} title="Close AI sidebar" aria-label="Close AI sidebar">‹</button>
      </div>
      {/* Design stays mounted so its form and result survive tab switches; the graph needs a visible container. */}
      <div className="sidebar-body" hidden={tab !== 'design'}><DesignPanel {...designProps} onGenerate={onGenerate} /></div>
      {tab === 'overview' && <div className="sidebar-body"><OverviewPanel designRun={designRun} /></div>}
    </aside>
  );
}
