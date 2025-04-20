import { Routes, Route, Link } from "react-router-dom";
import Predict from "./pages/Predict";
import Result from "./pages/Result";
import Home from "./pages/Home"; // if you have a homepage
import NearbyDoctors from "./pages/NearbyDoctors";
import "./App.css";

function App() {
  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-brand">
          <Link to="/">Medi Care</Link>
        </div>
        <div className="nav-links">
          <Link to="/">Home</Link>
          <Link to="/predict">Predict</Link>
          <Link to="/nearby-doctors">Find Doctors</Link>
        </div>
      </nav>

      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/result" element={<Result />} />
          <Route path="/nearby-doctors" element={<NearbyDoctors />} />
        </Routes>
      </main>

      <footer className="footer">
        <p>&copy; 2025 Disease Predictor. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
