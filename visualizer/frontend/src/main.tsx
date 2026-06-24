import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import AppCockpit from './AppCockpit.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppCockpit />
  </StrictMode>,
);
