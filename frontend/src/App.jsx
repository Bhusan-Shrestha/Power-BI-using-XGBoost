import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>Sales Prediction</h1>
        <nav>
          <NavLink to="/" className={({ isActive }) => (isActive ? "active-nav" : "")}>Predict</NavLink>
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </main>
    </div>
  );
}
