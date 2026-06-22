import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import AppDocked from './AppDocked.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppDocked />
  </StrictMode>,
);
