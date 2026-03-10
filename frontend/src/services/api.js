import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 30000,
});

export function parseApiError(error, fallbackMessage = "Request failed") {
  if (error?.response?.data?.detail) {
    return String(error.response.data.detail);
  }
  if (error?.message) {
    return String(error.message);
  }
  return fallbackMessage;
}

export const triggerPredictionFromFile = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const fetchPredictions = () => api.get("/predictions");

export default api;
