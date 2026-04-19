import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Home   from './pages/Home';
import Scan   from './pages/Scan';
import Result from './pages/Result';

const styles = {
  app: {
    maxWidth: 480,
    margin: '0 auto',
    minHeight: '100vh',
    background: '#1a1512',
    position: 'relative',
  },
};

export default function App() {
  return (
    <div style={styles.app}>
      <Routes>
        <Route path="/"              element={<Home />} />
        <Route path="/scan"          element={<Scan />} />
        <Route path="/result/:code"  element={<Result />} />
        <Route path="*"              element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
