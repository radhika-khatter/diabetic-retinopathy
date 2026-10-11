import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeImage = async () => {
    if (!file) {
      setError("Please select a fundus image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        "http://127.0.0.1:8000/api/analyze",
        formData
      );

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Could not connect to the RETINOVA backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>RETINOVA</h1>
      <h2>Explainable AI for Diabetic Retinopathy Screening</h2>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <br />
      <br />

      <button onClick={analyzeImage} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze Fundus Image"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <div>
          <h2>Screening Result</h2>

          <p>
            <strong>Grade:</strong> {result.grade} - {result.grade_name}
          </p>

          <p>
            <strong>Confidence:</strong>{" "}
            {(result.confidence * 100).toFixed(2)}%
          </p>

          <p>
            <strong>Action:</strong> {result.action}
          </p>

          <h3>Image Quality</h3>
          <p>
            Status: {result.quality.ok ? "Good" : "Needs Recapture"}
          </p>

          {result.quality.issues?.map((issue, index) => (
            <p key={index}>{issue}</p>
          ))}

          <h3>Grad-CAM</h3>
          {result.gradcam && (
            <img
              src={result.gradcam}
              alt="Grad-CAM"
              style={{ maxWidth: "500px" }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;