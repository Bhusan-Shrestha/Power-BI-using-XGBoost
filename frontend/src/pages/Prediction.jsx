import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchPredictions,
  parseApiError,
  triggerPredictionFromLatest,
} from "../services/api";

const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const currencyFmt = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

export default function Prediction() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");

  const loadPredictions = async () => {
    setLoading(true);
    try {
      const response = await fetchPredictions();
      setPredictions(response.data || []);
    } catch (error) {
      setMessage(parseApiError(error, "Failed to load prediction data."));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPredictions();
  }, []);

  const chartData = useMemo(
    () =>
      predictions.map((row) => ({
        product_id: row.product_id,
        predicted_sales: Number(row.predicted_sales || 0),
        predicted_unit_sales: Number(row.predicted_unit_sales || 0),
      })),
    [predictions]
  );

  const handleRunPrediction = async () => {
    setRunning(true);
    setMessage("");
    try {
      const response = await triggerPredictionFromLatest();
      const outputFileName = response.data?.output_file || "";
      const relativeDownloadUrl =
        response.data?.download_url ||
        (outputFileName ? `/download/${encodeURIComponent(outputFileName)}` : "");
      const apiBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
      setDownloadUrl(relativeDownloadUrl ? `${apiBase}${relativeDownloadUrl}` : "");
      setMessage(response.data?.message || "Prediction generated successfully.");

      const rows = (response.data?.predictions || []).map((item) => ({
        product_id: item.product,
        month: response.data?.month,
        predicted_sales: Number(item.predicted_sales || 0),
        predicted_unit_sales: Number(item.predicted_units || 0),
      }));
      setPredictions(rows);
    } catch (error) {
      setMessage(parseApiError(error, "Prediction generation failed."));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="prediction-page">
      <section className="card hero-card">
        <h2>Prediction Section</h2>
        <p className="muted-text">
          Generate forecast from the latest uploaded input data and download the predicted file.
        </p>

        <div className="inline-controls">
          <button onClick={handleRunPrediction} disabled={running}>
            {running ? "Generating..." : "Generate Prediction"}
          </button>
          <button onClick={loadPredictions} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh Data"}
          </button>
        </div>
        {message ? <p>{message}</p> : null}
        <p>
          {downloadUrl ? (
            <a href={downloadUrl} target="_blank" rel="noreferrer">
              Download Predicted File
            </a>
          ) : (
            <span className="muted-text">Run prediction to enable download.</span>
          )}
        </p>
      </section>

      <section className="card">
        <h3>Predicted Sales by Product</h3>
        <div className="chart-wrap">
          {loading ? (
            <p>Loading prediction chart...</p>
          ) : chartData.length === 0 ? (
            <p>No prediction data found. Generate prediction first.</p>
          ) : (
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="product_id" interval={0} angle={-20} textAnchor="end" height={70} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="predicted_unit_sales" name="Predicted Units" fill="#0f766e" />
                <Bar dataKey="predicted_sales" name="Predicted Sales" fill="#075985" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <section className="card">
        <h3>Prediction Data</h3>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Product</th>
                <th>Month</th>
                <th>Predicted Units</th>
                <th>Predicted Sales</th>
              </tr>
            </thead>
            <tbody>
              {predictions.length === 0 ? (
                <tr>
                  <td colSpan={4}>No prediction data available.</td>
                </tr>
              ) : (
                predictions.map((row) => (
                  <tr key={`${row.product_id}-${row.month}`}>
                    <td>{row.product_id}</td>
                    <td>{row.month}</td>
                    <td>{numberFmt.format(Number(row.predicted_unit_sales || 0))}</td>
                    <td>{currencyFmt.format(Number(row.predicted_sales || 0))}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
