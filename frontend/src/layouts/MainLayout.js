import React, { useState } from 'react';
import { Layout, Menu, Avatar, Typography, Tag, Dropdown } from 'antd';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined, ShoppingCartOutlined, InboxOutlined,
  TeamOutlined, LineChartOutlined,
  LogoutOutlined, SettingOutlined,
  UserOutlined, CrownOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../store/authStore';
import FloatingChatbot from '../components/FloatingChatbot';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const ROLE_COLOR = {
  admin:   '#f5222d',
  user:    '#1677ff',
};

const ROLE_ICON = {
  admin:   <CrownOutlined />,
  user:    <UserOutlined />,
};

const ALL_MENU_ITEMS = [
  {
    key: '/dashboard',
    icon: <DashboardOutlined />,
    label: 'Dashboard',
    roles: ['admin', 'user'],
  },
  {
    key: '/sales',
    icon: <ShoppingCartOutlined />,
    label: 'Sales',
    roles: ['admin', 'user'],
  },
  {
    key: '/inventory',
    icon: <InboxOutlined />,
    label: 'Inventory',
    roles: ['admin', 'user'],
  },
  {
    key: '/customers',
    icon: <TeamOutlined />,
    label: 'Customers',
    roles: ['admin', 'user'],
  },
  {
    key: '/forecast',
    icon: <LineChartOutlined />,
    label: 'Forecast',
    roles: ['admin', 'user'],
  },
  {
    key: '/settings',
    icon: <SettingOutlined />,
    label: 'Settings',
    roles: ['admin'],
  },
];

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { user, logout } = useAuthStore();
  const siderWidth = collapsed ? 80 : 220;

  const role = user?.role || 'user';

  // Filter menu based on role
  const menuItems = ALL_MENU_ITEMS
    .filter(item => item.roles.includes(role))
    .map(item => ({
      key: item.key,
      icon: item.icon,
      label: item.label,
    }));

  return (
    <Layout style={{ height: '100vh', overflow: 'hidden' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        style={{
          background: '#001529',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          height: '100vh',
          overflow: 'auto',
        }}
        width={220}
      >
        {/* Logo */}
        <div style={{
          padding: '16px 12px',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          marginBottom: 8,
        }}>
          {!collapsed ? (
            <div>
              <Text style={{ color: '#1677ff', fontSize: 16, fontWeight: 700 }}>
                BI Platform
              </Text>
              <br />
              <Text style={{ color: 'rgba(255,255,255,0.45)', fontSize: 11 }}>
                AI-Powered Analytics
              </Text>
            </div>
          ) : (
            <Text style={{ color: '#1677ff', fontSize: 16, fontWeight: 700 }}>
              BI
            </Text>
          )}
        </div>

        {/* Menu */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>

      <Layout style={{ marginLeft: siderWidth, transition: 'margin-left 0.2s', minWidth: 0 }}>
        {/* Header */}
        <Header style={{
          background: '#fff',
          padding: '0 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid #f0f0f0',
          boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
        }}>
          <Text style={{ fontSize: 16, fontWeight: 600, color: '#333' }}>
            {ALL_MENU_ITEMS.find(m => m.key === pathname)?.label || 'BI Platform'}
          </Text>

          <Dropdown
            menu={{
              items: [
                {
                  type: 'group',
                  label: (
                    <div style={{ padding: '8px 0', textAlign: 'center' }}>
                      <div style={{ fontWeight: 600, color: '#333', marginBottom: 4 }}>
                        {user?.full_name}
                      </div>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {user?.email}
                      </Text>
                      <div style={{ marginTop: 8 }}>
                        <Tag icon={ROLE_ICON[role]} color={ROLE_COLOR[role]} style={{ fontSize: 11 }}>
                          {role.toUpperCase()}
                        </Tag>
                      </div>
                    </div>
                  ),
                },
                { type: 'divider' },
                {
                  key: 'logout',
                  icon: <LogoutOutlined />,
                  label: 'Logout',
                  danger: true,
                  onClick: logout,
                },
              ],
            }}
            placement="bottomRight"
            trigger={['click']}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                cursor: 'pointer',
                padding: '8px 12px',
                borderRadius: 6,
                transition: 'background 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#f5f5f5';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
              }}
            >
              <Avatar
                size={32}
                style={{
                  background: ROLE_COLOR[role],
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                {user?.full_name?.[0]?.toUpperCase() || 'U'}
              </Avatar>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <Text style={{ fontSize: 13, fontWeight: 600, color: '#333' }}>
                  {user?.full_name?.split(' ')[0] || 'User'}
                </Text>
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {role.charAt(0).toUpperCase() + role.slice(1)}
                </Text>
              </div>
            </div>
          </Dropdown>
        </Header>

        {/* Main content */}
        <Content style={{
          margin: '24px',
          background: '#f5f7fa',
          borderRadius: 8,
          padding: '24px',
          minHeight: 'calc(100vh - 112px)',
          overflow: 'auto',
        }}>
          <Outlet />
        </Content>
        <FloatingChatbot />
      </Layout>
    </Layout>
  );
}