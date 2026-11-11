import { useState } from 'react';

export function formatBytes(bytes) {
  if (!bytes) return '';
  return bytes >= 1024 ** 2 ? `${(bytes / 1024 ** 2).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

export function formatDimensions(d) {
  if (!d) return null;
  return [d.width, d.depth, d.height].map((v) => v.toFixed(2)).join(' × ') + ' m';
}

export function Thumbnail({ src, alt }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) return <div className="thumb placeholder" aria-hidden="true">3D</div>;
  return <img className="thumb" src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />;
}

export function LicenseLabel({ license }) {
  if (!license?.name) return <span className="muted">Unknown license</span>;
  const note = license.attributionRequired ? ' · Attribution required' : '';
  const name = license.url
    ? <a href={license.url} target="_blank" rel="noreferrer">{license.name}</a>
    : license.name;
  return <span>{name}{note}</span>;
}
