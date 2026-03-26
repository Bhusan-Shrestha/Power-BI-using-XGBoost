import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from '../api/axios';

const normalizeRole = (role) => (role === 'admin' ? 'admin' : 'user');

export const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      login: async (email, password) => {
  const params = new URLSearchParams();
  params.append('username', email);
  params.append('password', password);
  
  const res = await axios.post('/api/auth/login', params, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });
  
  const { access_token, user_id, role, full_name } = res.data;
  const normalizedRole = normalizeRole(role);
  axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
  set({
    token: access_token,
    user: { id: user_id, email, role: normalizedRole, full_name },
    isAuthenticated: true,
  });
  return res.data;
},

      logout: () => {
        delete axios.defaults.headers.common['Authorization'];
        set({ user: null, token: null, isAuthenticated: false });
      },

      // Role checkers
      isAdmin:   () => get().user?.role === 'admin',
      isUser:    () => ['admin', 'user'].includes(normalizeRole(get().user?.role)),

      normalizeCurrentUserRole: () => {
        const current = get().user;
        if (!current) return;
        const normalizedRole = normalizeRole(current.role);
        if (current.role !== normalizedRole) {
          set({ user: { ...current, role: normalizedRole } });
        }
      },
    }),
    { name: 'bi-auth' }
  )
);