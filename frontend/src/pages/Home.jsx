import UploadData from "../components/UploadData";

export default function Home() {
  return (
    <div>
      <h2>Home</h2>
      <p>Upload monthly Excel sales data to trigger ingestion and modeling.</p>
      <UploadData />
    </div>
  );
}
