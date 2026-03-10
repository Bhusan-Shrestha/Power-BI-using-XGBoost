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
import { fetchPredictions, parseApiError, triggerPredictionFromFile } from "../services/api";

export default function PredictionChart() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [message, setMessage] = useState("");
  const [file, setFile] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState("");

  const loadPredictions = async () => {
    const response = await fetchPredictions();
    setPredictions(response.data);
  };

  useEffect(() => {
    async function initialize() {
      setLoadingInitial(true);
      try {
        await loadPredictions();
      } catch (error) {
        setPredictions([]);
        setMessage(parseApiError(error, "Could not load predictions."));
      } finally {
        setLoadingInitial(false);
      }
    }

    initialize();
  }, []);

  const chartData = useMemo(
    () =>
      predictions.map((p) => ({
        product_id: p.product_id,
        predicted_unit_sales: Number(p.predicted_unit_sales ?? 0),
      })),
    [predictions]
  );

  const handleGenerate = async () => {
    if (!file) {
      setMessage("Please choose an input file first.");
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      const response = await triggerPredictionFromFile(file);

      const responsePredictions = (response.data?.predictions || []).map((item) => ({
        product_id: item.product,
        month: response.data?.month,
        predicted_sales: Number(item.predicted_sales || 0),
        predicted_unit_sales: Number(item.predicted_units || 0),
      }));

      const outputFileName = response.data?.output_file || "";
      const relativeDownloadUrl =
        response.data?.download_url ||
        (outputFileName ? `/download/${encodeURIComponent(outputFileName)}` : "");
      const apiBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
      setDownloadUrl(relativeDownloadUrl ? `${apiBase}${relativeDownloadUrl}` : "");

      setPredictions(responsePredictions);
      setMessage(response.data?.message || "Predictions generated successfully.");
    } catch (error) {
      setMessage(parseApiError(error, "Prediction generation failed."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Upload this month's sales data</h2>
      <p className="muted-text">
        Upload the latest sales file to generate product-wise predictions.
      </p>
      <label>
        Input File
        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
      </label>
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Predicting..." : "Generate Predictions"}
      </button>
      {message ? <p>{message}</p> : null}
      <p>
        {downloadUrl ? (
          <a href={downloadUrl} target="_blank" rel="noreferrer">
            Download Predicted File
          </a>
        ) : (
          <span className="muted-text">Generate predictions to enable file download.</span>
        )}
      </p>
      <div className="chart-wrap">
        {loadingInitial ? (
          <p>Loading predictions...</p>
        ) : chartData.length === 0 ? (
          <p>No predictions found. Generate predictions first.</p>
        ) : (
          <ResponsiveContainer>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="product_id"
                interval={0}
                angle={-25}
                textAnchor="end"
                height={70}
              />
              <YAxis label={{ value: "Unit Sales", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="predicted_unit_sales" fill="#2a9d8f" name="predicted_unit_sales" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
      {predictions.length > 0 ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Product ID</th>
                <th>Month</th>
                <th>Predicted Unit Sales</th>
                <th>Predicted Sales</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((row) => (
                <tr key={`${row.product_id}-${row.month}`}>
                  <td>{row.product_id}</td>
                  <td>{row.month}</td>
                  <td>{Number(row.predicted_unit_sales || 0).toFixed(0)}</td>
                  <td>{Number(row.predicted_sales).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
