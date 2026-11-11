// Small expandable JSON viewer: objects/arrays fold open, leaves are coloured by type.

function Leaf({ value }) {
  if (value === null || value === undefined) return <span className="json-null">null</span>;
  if (typeof value === 'string') return <span className="json-string">"{value}"</span>;
  if (typeof value === 'number') return <span className="json-number">{value}</span>;
  if (typeof value === 'boolean') return <span className="json-bool">{String(value)}</span>;
  return <span>{String(value)}</span>;
}

function Node({ name, value, depth, openDepth }) {
  const isObject = value !== null && typeof value === 'object';
  const label = name !== undefined && <span className="json-key">{name}: </span>;
  if (!isObject) {
    return <div className="json-row">{label}<Leaf value={value} /></div>;
  }
  const entries = Array.isArray(value) ? value.map((v, i) => [i, v]) : Object.entries(value);
  const summary = Array.isArray(value) ? `[${entries.length}]` : `{${entries.length}}`;
  if (entries.length === 0) return <div className="json-row">{label}<span className="json-null">{summary}</span></div>;
  return (
    <details className="json-node" open={depth < openDepth}>
      <summary>{label}<span className="json-null">{summary}</span></summary>
      <div className="json-children">
        {entries.map(([k, v]) => <Node key={k} name={k} value={v} depth={depth + 1} openDepth={openDepth} />)}
      </div>
    </details>
  );
}

export default function JsonView({ data, openDepth = 1 }) {
  if (data === undefined || data === null) return <p className="muted small">No data.</p>;
  return <div className="json-view"><Node value={data} depth={0} openDepth={openDepth} /></div>;
}
