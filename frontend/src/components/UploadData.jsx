import { useState } from "react";
import { parseApiError, uploadSalesFile } from "../services/api";

export default function UploadData() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("No file uploaded yet.");
  const [uploadStats, setUploadStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please choose an Excel file.");
      return;
    }

    setLoading(true);
    setUploadStats(null);
    try {
      const response = await uploadSalesFile(file);
      setMessage(response.data.message || "Upload complete.");
      setUploadStats({
        productsUpserted: response.data.products_upserted ?? 0,
        rowsInserted: response.data.sales_rows_inserted ?? 0,
      });
    } catch (error) {
      setMessage(parseApiError(error, "Upload failed."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>Upload Monthly Sales Data</h2>
      <p className="muted-text">Accepted format: `.xlsx` or `.xls` with required sales columns.</p>
      <input
        type="file"
        accept=".xlsx,.xls"
        onChange={(event) => setFile(event.target.files?.[0] || null)}
      />
      {file ? <p className="muted-text">Selected file: {file.name}</p> : null}
      <div style={{ marginTop: "0.8rem" }}>
        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Uploading..." : "Upload"}
        </button>
      </div>
      <p>{message}</p>
      {uploadStats ? (
        <div className="stats-grid compact">
          <article className="metric-card">
            <h4>Products Upserted</h4>
            <strong>{uploadStats.productsUpserted}</strong>
          </article>
          <article className="metric-card">
            <h4>Rows Inserted</h4>
            <strong>{uploadStats.rowsInserted}</strong>
          </article>
        </div>
      ) : null}
    </section>
  );
}
