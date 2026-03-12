import { useEffect, useMemo, useState } from "react";
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
import { fetchAnalyticsOverview, parseApiError, uploadSalesFile } from "../services/api";

const palette = ["#0d9488", "#0284c7", "#f59e0b", "#be123c", "#7c3aed", "#16a34a"];

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const currencyFmt = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function KpiCard({ title, value, tone = "default" }) {
  return (
    <article className={`kpi-card ${tone}`}>
      <p>{title}</p>
      <strong>{value}</strong>
    </article>
  );
}

export default function Home() {
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [inputFile, setInputFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  const loadOverview = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetchAnalyticsOverview();
      setOverview(response.data || {});
    } catch (requestError) {
      setError(parseApiError(requestError, "Failed to load analytics dashboard."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOverview();
  }, []);

  const handleUploadInputData = async () => {
    if (!inputFile) {
      setUploadMessage("Please select an Excel file first.");
      return;
    }

    setUploading(true);
    setUploadMessage("");
    try {
      const response = await uploadSalesFile(inputFile);
      const rows = response.data?.sales_rows_inserted ?? 0;
      const apiMessage = response.data?.message;
      setUploadMessage(apiMessage || `Input file uploaded successfully. ${rows} rows added.`);
      if (response.data?.overview) {
        setOverview(response.data.overview);
      }
      await loadOverview();
    } catch (requestError) {
      setUploadMessage(parseApiError(requestError, "Failed to upload input file."));
    } finally {
      setUploading(false);
    }
  };

  const kpis = overview?.kpis || {};
  const monthly = overview?.monthly_trend || [];
  const segments = overview?.segment_performance || [];
  const discounts = overview?.discount_performance || [];
  const topProducts = overview?.top_products || [];
  const recentRows = overview?.recent_rows || [];

  const segmentPie = useMemo(
    () =>
      segments.map((item, index) => ({
        name: item.segment,
        value: Number(item.sales || 0),
        color: palette[index % palette.length],
      })),
    [segments]
  );

  if (loading) {
    return <section className="card"><p>Loading dashboard...</p></section>;
  }

  if (error) {
    return (
      <section className="card">
        <p className="error-text">{error}</p>
      </section>
    );
  }

  return (
    <div className="analysis-page">
      <section className="card hero-card">
        <h2>Dashboard</h2>
        <p className="muted-text">Upload input data here, then view detailed Analysis and use Prediction page to generate forecast.</p>

        <div className="inline-controls">
          <input
            type="file"
            accept=".xlsx,.xls"
            onChange={(event) => setInputFile(event.target.files?.[0] || null)}
          />
          <button onClick={handleUploadInputData} disabled={uploading}>
            {uploading ? "Uploading..." : "Upload Input Data"}
          </button>
        </div>
        {uploadMessage ? <p>{uploadMessage}</p> : null}
      </section>

      <section className="kpi-grid">
        <KpiCard title="Total Sales" value={currencyFmt.format(Number(kpis.total_sales || 0))} tone="teal" />
        <KpiCard title="Total Profit" value={currencyFmt.format(Number(kpis.total_profit || 0))} tone="blue" />
        <KpiCard title="Units Sold" value={numberFmt.format(Number(kpis.total_units || 0))} tone="amber" />
        <KpiCard title="Profit Margin" value={`${Number(kpis.avg_profit_margin || 0).toFixed(2)}%`} tone="rose" />
        <KpiCard title="Products" value={numberFmt.format(Number(kpis.products_count || 0))} />
        <KpiCard title="Forecast Total" value={currencyFmt.format(Number(kpis.predicted_sales_total || 0))} />
      </section>

      <section className="analysis-grid">
        <article className="card panel-large">
          <h3>Monthly Sales vs Profit</h3>
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={monthly}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="sales" stroke="#0f766e" strokeWidth={2} name="Sales" />
                <Line type="monotone" dataKey="profit" stroke="#0284c7" strokeWidth={2} name="Profit" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card">
          <h3>Segment Contribution</h3>
          <div className="chart-wrap small">
            <ResponsiveContainer>
              <PieChart>
                <Pie data={segmentPie} dataKey="value" nameKey="name" outerRadius={95} label>
                  {segmentPie.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card panel-large">
          <h3>Top Products by Sales</h3>
          <div className="chart-wrap">
            <ResponsiveContainer>
              <BarChart data={topProducts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="product" interval={0} angle={-18} textAnchor="end" height={70} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="sales" name="Sales" fill="#115e59" />
                <Bar dataKey="profit" name="Profit" fill="#075985" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card">
          <h3>Discount Band Performance</h3>
          <div className="chart-wrap small">
            <ResponsiveContainer>
              <BarChart data={discounts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="discount_band" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="sales" name="Sales" fill="#ca8a04" />
                <Bar dataKey="profit" name="Profit" fill="#be123c" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="card">
        <h3>Recent Transactions</h3>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Product</th>
                <th>Segment</th>
                <th>Month</th>
                <th>Discount Band</th>
                <th>Units</th>
                <th>Sales</th>
                <th>Profit</th>
              </tr>
            </thead>
            <tbody>
              {recentRows.map((row, index) => (
                <tr key={`${row.product}-${row.month}-${index}`}>
                  <td>{row.product}</td>
                  <td>{row.segment}</td>
                  <td>{row.month}</td>
                  <td>{row.discount_band}</td>
                  <td>{numberFmt.format(Number(row.units_sold || 0))}</td>
                  <td>{currencyFmt.format(Number(row.sales || 0))}</td>
                  <td>{currencyFmt.format(Number(row.profit || 0))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
