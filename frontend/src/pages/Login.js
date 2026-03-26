import React, { useState } from 'react';
import {
  Form, Input, Button, Card, Typography,
  Alert, Checkbox, message,
} from 'antd';
import {
  UserOutlined, LockOutlined,
  EyeInvisibleOutlined, EyeTwoTone,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

const { Title, Text } = Typography;

export default function Login() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const login = useAuthStore(s => s.login);

  const onFinish = async ({ email, password }) => {
    setLoading(true);
    setError(null);
    try {
      await login(email, password);
      message.success('Welcome back!');
      navigate('/dashboard');
    } catch (e) {
      // Handle different error formats
      const detail = e.response?.data?.detail;
      if (Array.isArray(detail)) {
        // Validation error array from FastAPI
        setError(detail.map(d => d.msg).join(', '));
      } else if (typeof detail === 'string') {
        setError(detail);
      } else {
        setError('Invalid email or password. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #001529 0%, #003a70 50%, #1677ff 100%)',
      padding: '20px',
    }}>
      {/* Background pattern */}
      <div style={{
        position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none',
      }}>
        {[...Array(6)].map((_, i) => (
          <div key={i} style={{
            position: 'absolute',
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.03)',
            width: `${(i + 1) * 120}px`,
            height: `${(i + 1) * 120}px`,
            top: `${i * 15}%`,
            left: `${i % 2 === 0 ? i * 10 : 80 - i * 10}%`,
          }} />
        ))}
      </div>

      <div style={{ width: '100%', maxWidth: 420, position: 'relative', zIndex: 1 }}>
        {/* Logo / Brand */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            width: 64, height: 64, borderRadius: 16,
            background: 'linear-gradient(135deg, #1677ff, #0958d9)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 16px',
            boxShadow: '0 8px 32px rgba(22, 119, 255, 0.4)',
          }}>
            <SafetyCertificateOutlined style={{ fontSize: 32, color: '#fff' }} />
          </div>
          <Title level={2} style={{ color: '#fff', margin: 0, fontWeight: 700 }}>
            BI Platform
          </Title>
          <Text style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14 }}>
            AI-Powered Business Intelligence
          </Text>
        </div>

        {/* Login Card */}
        <Card style={{
          borderRadius: 16,
          boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
          border: 'none',
        }}>
          <Title level={4} style={{ marginBottom: 24, textAlign: 'center' }}>
            Sign in to your account
          </Title>

          {error && (
            <Alert
              type="error"
              message={error}
              showIcon
              closable
              onClose={() => setError(null)}
              style={{ marginBottom: 16, borderRadius: 8 }}
            />
          )}

          <Form
            name="login"
            onFinish={onFinish}
            layout="vertical"
            size="large"
            autoComplete="off"
          >
            <Form.Item
              name="email"
              label="Email Address"
              rules={[
                { required: true, message: 'Please enter your email' },
                { type: 'email', message: 'Please enter a valid email' },
              ]}
            >
              <Input
                prefix={<UserOutlined style={{ color: '#bbb' }} />}
                placeholder="Enter your email"
                autoComplete="email"
              />
            </Form.Item>

            <Form.Item
              name="password"
              label="Password"
              rules={[{ required: true, message: 'Please enter your password' }]}
            >
              <Input.Password
                prefix={<LockOutlined style={{ color: '#bbb' }} />}
                placeholder="Enter your password"
                iconRender={visible =>
                  visible ? <EyeTwoTone /> : <EyeInvisibleOutlined />
                }
                autoComplete="current-password"
              />
            </Form.Item>

            <Form.Item name="remember" valuePropName="checked">
              <Checkbox>Keep me signed in</Checkbox>
            </Form.Item>

            <Form.Item style={{ marginBottom: 0 }}>
              <Button
                type="primary"
                htmlType="submit"
                block
                loading={loading}
                style={{
                  height: 44,
                  borderRadius: 8,
                  fontSize: 15,
                  fontWeight: 600,
                }}
              >
                {loading ? 'Signing in...' : 'Sign In'}
              </Button>
            </Form.Item>
          </Form>
        </Card>

        {/* Footer */}
        <div style={{ textAlign: 'center', marginTop: 24 }}>
          <Text style={{ color: 'rgba(255,255,255,0.4)', fontSize: 12 }}>
            © 2024 BI Platform. All rights reserved.
          </Text>
        </div>
      </div>
    </div>
  );
}