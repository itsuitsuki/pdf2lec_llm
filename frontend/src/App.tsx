import { useState } from "react";
import { Routes, Route, BrowserRouter } from "react-router-dom";
import "./App.css";
import Sidebar from "./components/Sidebar";
import Home from "./pages/Home";
import Starred from "./pages/Starred";
import Display from "./pages/Display";


function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />}></Route>
          <Route path="/starred" element={<Starred />}></Route>
          <Route path="/pdf/:pdfId" element={<Display />} />
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
