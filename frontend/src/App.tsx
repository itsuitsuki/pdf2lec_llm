import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Display from './pages/Display';
import ConfigurePage from './pages/ConfigurePage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/configure/:pdfId" element={<ConfigurePage />} />
        <Route path="/display/:pdfId" element={<Display />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
