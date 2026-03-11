import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Analysis from "./pages/Analysis";
import Prediction from "./pages/Prediction";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Business Intelligence</h1>
          <p className="subtitle">Increase your sales</p>
        </div>
        <nav>
          <NavLink to="/" className={({ isActive }) => (isActive ? "active-nav" : "")}>Dashboard</NavLink>
          <NavLink to="/analysis" className={({ isActive }) => (isActive ? "active-nav" : "")}>Analysis</NavLink>
          <NavLink to="/prediction" className={({ isActive }) => (isActive ? "active-nav" : "")}>Prediction</NavLink>
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/prediction" element={<Prediction />} />
        </Routes>
      </main>
    </div>
  );
}
