import axios from 'axios';

const instance = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

instance.interceptors.request.use((config) => {
  try {
    const stored = localStorage.getItem('bi-auth');
    if (stored) {
      const parsed = JSON.parse(stored);
      const token = parsed?.state?.token;
      if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
      }
    }
  } catch (e) {}
  return config;
});

instance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('bi-auth');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default instance;