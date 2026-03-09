import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  ComposedChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchPredictions, parseApiError, triggerPrediction } from "../services/api";

export default function PredictionChart() {
  const now = new Date();
  const [predictYear, setPredictYear] = useState(now.getFullYear());
  const [predictMonth, setPredictMonth] = useState(now.getMonth() + 1);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [message, setMessage] = useState("");

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
        month: p.month,
        predicted_sales: Number(p.predicted_sales),
        predicted_profit: Number(p.predicted_profit),
      })),
    [predictions]
  );

  const handleGenerate = async () => {
    if (predictMonth < 1 || predictMonth > 12) {
      setMessage("Month must be between 1 and 12.");
      return;
    }
    setLoading(true);
    setMessage("");
    try {
      const response = await triggerPrediction({
        predict_year: Number(predictYear),
        predict_month: Number(predictMonth),
      });
      await loadPredictions();
      setMessage(response.data?.message || "Predictions generated successfully.");
    } catch (error) {
      setMessage(parseApiError(error, "Prediction generation failed."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Future Sales & Profit Predictions</h2>
      <p className="muted-text">
        Generates one-month-ahead predictions for each product based on latest sales records.
      </p>
      <div className="prediction-controls">
        <label>
          Year
          <input
            type="number"
            min="2000"
            value={predictYear}
            onChange={(event) => setPredictYear(Number(event.target.value))}
          />
        </label>
        <label>
          Month
          <input
            type="number"
            min="1"
            max="12"
            value={predictMonth}
            onChange={(event) => setPredictMonth(Number(event.target.value))}
          />
        </label>
      </div>
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Predicting..." : "Generate Predictions"}
      </button>
      {message ? <p>{message}</p> : null}
      <div className="chart-wrap">
        {loadingInitial ? (
          <p>Loading predictions...</p>
        ) : chartData.length === 0 ? (
          <p>No predictions found. Generate predictions first.</p>
        ) : (
          <ResponsiveContainer>
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="predicted_sales" fill="#2a9d8f" />
              <Bar dataKey="predicted_profit" fill="#e9c46a" />
            </ComposedChart>
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
                <th>Predicted Sales</th>
                <th>Predicted Profit</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((row) => (
                <tr key={`${row.product_id}-${row.month}`}>
                  <td>{row.product_id}</td>
                  <td>{row.month}</td>
                  <td>{Number(row.predicted_sales).toFixed(2)}</td>
                  <td>{Number(row.predicted_profit).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
