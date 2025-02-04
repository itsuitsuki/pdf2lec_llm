import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Display from "./pages/Display";
import ConfigurePage from "./pages/ConfigurePage";
import { useAuth } from "./contexts/AuthContext";
import ErrorPage from "./pages/ErrorPage";
import MyRecords from "./pages/myRecords";

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Home />
              </ProtectedRoute>
            }
          />
          <Route
            path="/myrecords"
            element={
              <ProtectedRoute>
                <MyRecords/>
              </ProtectedRoute>
            }
          />
          <Route
            path="/configure/:pdfId"
            element={
              <ProtectedRoute>
                <ConfigurePage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/display/:pdfId"
            element={
              <ProtectedRoute>
                <Display />
              </ProtectedRoute>
            }
          />
          <Route path="/error" element={<ErrorPage />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
