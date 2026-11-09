import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import './index.css';
import { CollectPage } from './pages/CollectPage';
import { EvalLogPage } from './pages/EvalLogPage';
import { HomePage } from './pages/HomePage';
import { HotspotEditorPage } from './pages/HotspotEditorPage';
import { QaidaLessonPage } from './pages/QaidaLessonPage';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/lesson/:pageId" element={<QaidaLessonPage />} />
        <Route path="/editor/:pageId" element={<HotspotEditorPage />} />
        <Route path="/logs" element={<EvalLogPage />} />
        <Route path="/collect" element={<CollectPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
