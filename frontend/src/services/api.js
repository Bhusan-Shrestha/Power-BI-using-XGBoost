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

function buildQueryString(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    query.append(key, String(value));
  });
  const text = query.toString();
  return text ? `?${text}` : "";
}

export const uploadSalesFile = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return api.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const fetchSales = () => api.get("/sales");
export const fetchPredictions = () => api.get("/predictions");
export const triggerPrediction = (payload) => api.post("/predict", payload);
export const fetchAnalyticsFilters = () => api.get("/analytics/filters");
export const fetchAnalyticsSummary = (filters = {}) =>
  api.get(`/analytics/summary${buildQueryString(filters)}`);
export const fetchRegionalAnalytics = (filters = {}) =>
  api.get(`/analytics/regional${buildQueryString(filters)}`);
export const fetchTrendAnalytics = (filters = {}) =>
  api.get(`/analytics/trend${buildQueryString(filters)}`);
export const fetchProductAnalytics = (filters = {}) =>
  api.get(`/analytics/products${buildQueryString(filters)}`);
export const fetchCategoryAnalytics = (filters = {}) =>
  api.get(`/analytics/category${buildQueryString(filters)}`);
export const fetchAnalyticsRecords = (filters = {}) =>
  api.get(`/analytics/records${buildQueryString(filters)}`);

export default api;
