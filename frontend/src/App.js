import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ConfigProvider, Result, Button } from 'antd';
import MainLayout from './layouts/MainLayout';
import Dashboard from './pages/Dashboard';
import Sales from './pages/Sales';
import Inventory from './pages/Inventory';
import Customers from './pages/Customers';
import Forecast from './pages/Forecast';
import Login from './pages/Login';
import Settings from './pages/Settings';
import { useAuthStore } from './store/authStore';

function RoleGuard({ children, allowedRoles }) {
  const { user } = useAuthStore();
  if (!allowedRoles.includes(user?.role)) {
    return (
      <Result
        status="403"
        title="Access Denied"
        subTitle="You do not have permission to view this page."
        extra={
          <Button type="primary" onClick={() => window.history.back()}>
            Go Back
          </Button>
        }
      />
    );
  }
  return children;
}

function App() {
  const { isAuthenticated, normalizeCurrentUserRole } = useAuthStore();

  useEffect(() => {
    normalizeCurrentUserRole();
  }, [normalizeCurrentUserRole]);

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1677ff',
          borderRadius: 8,
        },
      }}
    >
      <Routes>
        <Route path="/login" element={<Login />} />

        {isAuthenticated ? (
          <Route element={<MainLayout />}>
            <Route path="/" element={<Navigate to="/dashboard" />} />

            {/* All roles */}
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/sales"     element={<Sales />} />
            <Route path="/inventory" element={<Inventory />} />

            {/* Admin + User (analysis access) */}
            <Route path="/customers" element={
              <RoleGuard allowedRoles={['admin', 'user']}>
                <Customers />
              </RoleGuard>
            } />
            <Route path="/forecast" element={
              <RoleGuard allowedRoles={['admin', 'user']}>
                <Forecast />
              </RoleGuard>
            } />
            {/* Admin only */}
            <Route path="/settings" element={
              <RoleGuard allowedRoles={['admin']}>
                <Settings />
              </RoleGuard>
            } />

            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Route>
        ) : (
          <Route path="*" element={<Navigate to="/login" />} />
        )}
      </Routes>
    </ConfigProvider>
  );
}

export default App;