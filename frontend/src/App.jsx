import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Analytics from "./pages/Analytics";
import Predictions from "./pages/Predictions";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Sales AI Platform</h1>
        <nav>
          <NavLink to="/" className={({ isActive }) => (isActive ? "active-nav" : "")}>Home</NavLink>
          <NavLink
            to="/analytics"
            className={({ isActive }) => (isActive ? "active-nav" : "")}
          >
            Analytics
          </NavLink>
          <NavLink
            to="/predictions"
            className={({ isActive }) => (isActive ? "active-nav" : "")}
          >
            Predictions
          </NavLink>
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/predictions" element={<Predictions />} />
        </Routes>
      </main>
    </div>
  );
}
