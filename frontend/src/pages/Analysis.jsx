import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchAnalyticsOverview, parseApiError } from "../services/api";

const palette = ["#ff6b6b", "#ffd166", "#06d6a0", "#4cc9f0", "#4361ee", "#f72585", "#f97316"];

const currencyFmt = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const compactNumberFmt = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const VISUALIZATION_TYPES = [
  { id: "line", name: "Line Chart", icon: "📈", desc: "Show trends over time" },
  { id: "bar", name: "Column Chart", icon: "📊", desc: "Vertical bars for comparison" },
  { id: "pie", name: "Pie Chart", icon: "🥧", desc: "Show proportions" },
  { id: "donut", name: "Donut Chart", icon: "⭕", desc: "Pie with center" },
];

const DATA_FIELDS = [
  { id: "monthly_trend", label: "Monthly Sales Trend", sample: "12 months of sales and profit data" },
  { id: "segment_pie", label: "Segment Contribution", sample: "Revenue by business segment" },
  { id: "top_products", label: "Top Products", sample: "Best performing products" },
  { id: "discount_performance", label: "Discount Performance", sample: "Impact by discount band" },
];

export default function Analysis() {
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [canvas, setCanvas] = useState([]);
  const [nextId, setNextId] = useState(1);
  const [selectedChart, setSelectedChart] = useState(null);

  const loadOverview = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetchAnalyticsOverview();
      setOverview(response.data || {});
    } catch (requestError) {
      setError(parseApiError(requestError, "Failed to load analytics data."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOverview();
  }, []);

  const addChart = (type, dataField = DATA_FIELDS[0].id) => {
    const newChart = {
      id: nextId,
      type,
      dataField,
      title: `${VISUALIZATION_TYPES.find((v) => v.id === type)?.name || "Chart"}`,
    };
    setCanvas([...canvas, newChart]);
    setNextId(nextId + 1);
    setSelectedChart(newChart);
  };

  const removeChart = (id) => {
    setCanvas(canvas.filter((chart) => chart.id !== id));
    if (selectedChart?.id === id) setSelectedChart(null);
  };

  const updateChart = (id, updates) => {
    const updated = canvas.map((chart) => (chart.id === id ? { ...chart, ...updates } : chart));
    setCanvas(updated);
    if (selectedChart?.id === id) {
      setSelectedChart({ ...selectedChart, ...updates });
    }
  };

  const data = {
    monthly_trend: overview?.monthly_trend || [],
    segment_pie: overview?.segment_performance || [],
    top_products: overview?.top_products || [],
    discount_performance: overview?.discount_performance || [],
  };

  const renderChart = (chart) => {
    const chartData = data[chart.dataField] || [];
    const categoryKey =
      chartData.length > 0
        ? ["month", "product", "segment", "discount_band"].find((key) => key in chartData[0]) ||
          Object.keys(chartData[0]).find((key) => typeof chartData[0][key] === "string") ||
          Object.keys(chartData[0])[0]
        : "month";

    if (!chartData || chartData.length === 0) {
      return (
        <div style={{ padding: "2rem", textAlign: "center", color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
          <p>No data available for this visualization</p>
        </div>
      );
    }

    const commonProps = {
      data: chartData,
      margin: { top: 5, right: 30, left: 24, bottom: 50 },
    };

    switch (chart.type) {
      case "line":
        return (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
              <XAxis dataKey={categoryKey} angle={-18} textAnchor="end" height={80} stroke="#64748b" style={{ fontSize: "11px" }} />
              <YAxis
                stroke="#64748b"
                width={72}
                tickFormatter={(value) => compactNumberFmt.format(Number(value || 0))}
                style={{ fontSize: "12px" }}
              />
              <Tooltip
                formatter={(value) => currencyFmt.format(value)}
                contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #0891b2", borderRadius: "8px", color: "#fff" }}
              />
              <Legend />
              <Line type="monotone" dataKey="sales" stroke="#ff6b6b" strokeWidth={3} name="Sales" dot={{ r: 4, fill: "#ff6b6b" }} />
              <Line type="monotone" dataKey="profit" stroke="#4361ee" strokeWidth={3} name="Profit" dot={{ r: 4, fill: "#4361ee" }} />
            </LineChart>
          </ResponsiveContainer>
        );

      case "bar":
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart {...commonProps}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
              <XAxis dataKey={categoryKey} angle={-18} textAnchor="end" height={80} stroke="#64748b" style={{ fontSize: "11px" }} />
              <YAxis stroke="#64748b" style={{ fontSize: "12px" }} />
              <Tooltip
                formatter={(value) => currencyFmt.format(value)}
                contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #0891b2", borderRadius: "8px", color: "#fff" }}
              />
              <Legend />
              <Bar dataKey="sales" name="Sales" fill="#f97316" radius={[8, 8, 0, 0]} />
              <Bar dataKey="profit" name="Profit" fill="#06d6a0" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );

      case "pie":
      case "donut":
        const pieData = (chartData || []).map((item, index) => ({
          name:
            (item[categoryKey] != null && String(item[categoryKey]).trim()) ||
            (item.month != null && String(item.month).trim()) ||
            (item.segment != null && String(item.segment).trim()) ||
            (item.discount_band != null && String(item.discount_band).trim()) ||
            (item.product != null && String(item.product).trim()) ||
            `Item ${index + 1}`,
          value: Number(item.sales ?? item.profit ?? item.units_sold ?? item.value ?? 0),
          color: palette[index % palette.length],
        }));
        return (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                outerRadius={chart.type === "donut" ? 100 : 90}
                innerRadius={chart.type === "donut" ? 60 : 0}
                label={{ fill: "#1e293b", fontSize: 12 }}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => currencyFmt.format(value)}
                contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #0891b2", borderRadius: "8px", color: "#fff" }}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      default:
        return <div style={{ padding: "2rem", textAlign: "center", color: "#999" }}>Chart type not available</div>;
    }
  };

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", backgroundColor: "#f8fafc" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: "3rem", marginBottom: "1rem", animation: "spin 1s linear infinite" }}>⚙️</div>
          <p style={{ fontSize: "1.1rem", color: "#64748b" }}>Loading your analysis workspace...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", backgroundColor: "#f8fafc" }}>
        <div style={{ padding: "2rem", backgroundColor: "#fff", borderRadius: "12px", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}>
          <p style={{ fontSize: "1.1rem", color: "#dc2626", margin: 0 }}>⚠️ {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", height: "100vh", backgroundColor: "#f0f4f8" }}>
      {/* Center Canvas */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          backgroundColor: "#f0f4f8",
        }}
      >
        {/* Header */}
        <div style={{ backgroundColor: "#fff", borderBottom: "2px solid #e2e8f0", padding: "2rem" }}>
          <h1 style={{ margin: "0 0 0.5rem 0", fontSize: "2rem", fontWeight: 800, color: "#0f172a" }}>
            Analysis Builder
          </h1>
          <p style={{ margin: 0, fontSize: "0.95rem", color: "#64748b" }}>
            Create custom visualizations by selecting chart types and data sources
          </p>
        </div>

        {/* Canvas Area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "2rem", display: "flex", flexDirection: "column" }}>
          {canvas.length === 0 ? (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                textAlign: "center",
                color: "#94a3b8",
              }}
            >
              <div style={{ fontSize: "5rem", marginBottom: "1.5rem", opacity: 0.4 }}>📈</div>
              <h2 style={{ fontSize: "1.8rem", fontWeight: 700, margin: "0 0 0.5rem 0", color: "#475569" }}>
                Your canvas is empty
              </h2>
              <p style={{ fontSize: "1.05rem", margin: "0 0 2rem 0", maxWidth: "450px", color: "#64748b", lineHeight: "1.6" }}>
                Click a chart icon on the right panel to add a chart to your report
              </p>
              <div style={{ fontSize: "2.5rem", opacity: 0.5 }}>👉</div>
            </div>
          ) : (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(560px, 1fr))",
                gap: "1.5rem",
                height: "fit-content",
              }}
            >
              {canvas.map((chart) => (
                <div
                  key={chart.id}
                  onClick={() => setSelectedChart(chart)}
                  style={{
                    backgroundColor: "#fff",
                    border: selectedChart?.id === chart.id ? "2px solid #0891b2" : "1px solid #e2e8f0",
                    borderRadius: "12px",
                    padding: "1.5rem",
                    boxShadow:
                      selectedChart?.id === chart.id
                        ? "0 0 30px rgba(8, 145, 178, 0.2)"
                        : "0 2px 8px rgba(0, 0, 0, 0.08)",
                    cursor: "pointer",
                    minHeight: "460px",
                    display: "flex",
                    flexDirection: "column",
                    transition: "all 0.3s ease",
                    position: "relative",
                  }}
                  onMouseEnter={(e) => {
                    if (selectedChart?.id !== chart.id) {
                      e.currentTarget.style.borderColor = "#cbd5e1";
                      e.currentTarget.style.boxShadow = "0 8px 16px rgba(0, 0, 0, 0.12)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedChart?.id !== chart.id) {
                      e.currentTarget.style.borderColor = "#e2e8f0";
                      e.currentTarget.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.08)";
                    }
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                    <div>
                      <h3 style={{ margin: "0 0 0.25rem 0", fontSize: "1.1rem", fontWeight: 700, color: "#0f172a" }}>
                        {VISUALIZATION_TYPES.find((v) => v.id === chart.type)?.icon} {chart.title}
                      </h3>
                      <p style={{ margin: 0, fontSize: "0.8rem", color: "#64748b" }}>
                        {DATA_FIELDS.find((f) => f.id === chart.dataField)?.label}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeChart(chart.id);
                      }}
                      style={{
                        background: "none",
                        border: "none",
                        color: "#94a3b8",
                        cursor: "pointer",
                        fontSize: "1.5rem",
                        padding: 0,
                        transition: "color 0.2s",
                      }}
                      onMouseEnter={(e) => (e.target.style.color = "#ef4444")}
                      onMouseLeave={(e) => (e.target.style.color = "#94a3b8")}
                    >
                      ✕
                    </button>
                  </div>
                  <div style={{ flex: 1, minHeight: 0, minWidth: 0 }}>{renderChart(chart)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right Sidebar - Configuration */}
      <div
        style={{
          width: "280px",
          backgroundColor: "#fff",
          borderLeft: "2px solid #e2e8f0",
          overflowY: "auto",
          padding: "1.5rem",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
          boxShadow: "-2px 0 8px rgba(0,0,0,0.05)",
        }}
      >
        {/* Visualization Library */}
        <div>
          <h3 style={{ fontSize: "0.8rem", fontWeight: 800, margin: "0 0 1rem 0", color: "#0f172a", textTransform: "uppercase", letterSpacing: "1px" }}>
            📊 Chart Types
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "0.6rem" }}>
            {VISUALIZATION_TYPES.map((viz) => (
              <button
                key={viz.id}
                onClick={() => addChart(viz.id)}
                title={viz.name}
                aria-label={viz.name}
                style={{
                  width: "100%",
                  aspectRatio: "1",
                  backgroundColor: "#f8fafc",
                  border: "1px solid #e2e8f0",
                  borderRadius: "8px",
                  cursor: "pointer",
                  textAlign: "center",
                  fontSize: "1.25rem",
                  transition: "all 0.2s ease",
                  color: "#1e293b",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
                onMouseEnter={(e) => {
                  e.target.style.backgroundColor = "#0891b2";
                  e.target.style.color = "#fff";
                  e.target.style.borderColor = "#0891b2";
                }}
                onMouseLeave={(e) => {
                  e.target.style.backgroundColor = "#f8fafc";
                  e.target.style.color = "#1e293b";
                  e.target.style.borderColor = "#e2e8f0";
                }}
              >
                {viz.icon}
              </button>
            ))}
          </div>
        </div>

        {/* Chart Configuration */}
        {selectedChart ? (
          <div style={{ borderTop: "2px solid #e2e8f0", paddingTop: "1.5rem" }}>
            <h3 style={{ fontSize: "0.8rem", fontWeight: 800, margin: "0 0 1rem 0", color: "#0f172a", textTransform: "uppercase", letterSpacing: "1px" }}>
              ⚙️ Chart Settings
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 700, marginBottom: "0.5rem", color: "#475569", textTransform: "uppercase" }}>
                  Chart Title
                </label>
                <input
                  type="text"
                  value={selectedChart.title}
                  onChange={(e) => updateChart(selectedChart.id, { title: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    border: "1px solid #cbd5e1",
                    fontSize: "0.9rem",
                    boxSizing: "border-box",
                    transition: "border-color 0.2s",
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = "#0891b2";
                    e.target.style.boxShadow = "0 0 0 3px rgba(8, 145, 178, 0.1)";
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = "#cbd5e1";
                    e.target.style.boxShadow = "none";
                  }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 700, marginBottom: "0.5rem", color: "#475569", textTransform: "uppercase" }}>
                  Visualization
                </label>
                <select
                  value={selectedChart.type}
                  onChange={(e) => updateChart(selectedChart.id, { type: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    border: "1px solid #cbd5e1",
                    fontSize: "0.9rem",
                    boxSizing: "border-box",
                    backgroundColor: "#fff",
                    cursor: "pointer",
                  }}
                >
                  {VISUALIZATION_TYPES.map((viz) => (
                    <option key={viz.id} value={viz.id}>
                      {viz.icon} {viz.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ display: "block", fontSize: "0.8rem", fontWeight: 700, marginBottom: "0.5rem", color: "#475569", textTransform: "uppercase" }}>
                  Data Source
                </label>
                <select
                  value={selectedChart.dataField}
                  onChange={(e) => updateChart(selectedChart.id, { dataField: e.target.value })}
                  style={{
                    width: "100%",
                    padding: "0.75rem",
                    borderRadius: "8px",
                    border: "1px solid #cbd5e1",
                    fontSize: "0.9rem",
                    boxSizing: "border-box",
                    backgroundColor: "#fff",
                    cursor: "pointer",
                  }}
                >
                  {DATA_FIELDS.map((field) => (
                    <option key={field.id} value={field.id}>
                      {field.label}
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={() => removeChart(selectedChart.id)}
                style={{
                  padding: "0.75rem",
                  backgroundColor: "#fee2e2",
                  color: "#dc2626",
                  border: "1px solid #fca5a5",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontWeight: 600,
                  fontSize: "0.9rem",
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={(e) => {
                  e.target.style.backgroundColor = "#fcd4d4";
                  e.target.style.borderColor = "#f87171";
                }}
                onMouseLeave={(e) => {
                  e.target.style.backgroundColor = "#fee2e2";
                  e.target.style.borderColor = "#fca5a5";
                }}
              >
                Remove Chart
              </button>
            </div>
          </div>
        ) : (
          <div
            style={{
              borderTop: "2px solid #e2e8f0",
              paddingTop: "1.5rem",
              textAlign: "center",
              color: "#94a3b8",
              fontSize: "0.9rem",
            }}
          >
            <p style={{ margin: 0 }}>👈 Select a chart to edit</p>
          </div>
        )}
      </div>
    </div>
  );
}
