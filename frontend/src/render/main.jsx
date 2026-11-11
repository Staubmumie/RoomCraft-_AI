import { createRoot } from 'react-dom/client';
import RenderApp from './RenderApp.jsx';

// No StrictMode: effects must run exactly once per render job.
createRoot(document.getElementById('root')).render(<RenderApp />);
