import React, { useState, useEffect } from 'react';
import {
  Card, Table, Tag, Row, Col, Statistic,
  Progress, Alert, Button, Typography,
  Spin, Avatar, Modal,
  Form, Input, Select, InputNumber,
  Switch, message,
} from 'antd';
import {
  ReloadOutlined,
  WarningOutlined,
  EditOutlined,
} from '@ant-design/icons';
import {
  PieChart, Pie, Cell, Tooltip as RechartTooltip,
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, CartesianGrid,
} from 'recharts';
import axios from '../api/axios';

const { Text } = Typography;
const COLORS = ['#1677ff', '#52c41a', '#faad14', '#f5222d', '#722ed1'];

export default function Customers() {
  const [form] = Form.useForm();
  const [summary, setSummary]     = useState(null);
  const [segments, setSegments]   = useState([]);
  const [churn, setChurn]         = useState([]);
  const [acquisition, setAcquisition] = useState([]);
  const [records, setRecords]     = useState([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [saving, setSaving]       = useState(false);
  const [editingCustomer, setEditingCustomer] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sum, seg, ch, acq, rec] = await Promise.all([
        axios.get('/api/customers/summary'),
        axios.get('/api/customers/segments'),
        axios.get('/api/customers/churn-risk'),
        axios.get('/api/customers/acquisition'),
        axios.get('/api/customers/records'),
      ]);
      setSummary(sum.data);
      setSegments(seg.data);
      setChurn(ch.data);
      setAcquisition(acq.data);
      setRecords(rec.data);
    } catch (e) {
      setError('Failed to load customer data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const openEditModal = (customer) => {
    setEditingCustomer(customer);
    form.setFieldsValue({
      name: customer.name || '',
      email: customer.email || '',
      phone: customer.phone || '',
      country: customer.country || '',
      region: customer.region || '',
      segment: customer.segment || 'individual',
      lifetime_value: Number(customer.lifetime_value || 0),
      churn_risk_score: Number(customer.churn_risk_score || 0),
      acquisition_channel: customer.acquisition_channel || '',
      is_active: Boolean(customer.is_active),
    });
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
    setEditingCustomer(null);
    form.resetFields();
  };

  const handleSaveCustomer = async () => {
    try {
      const values = await form.validateFields();
      setSaving(true);

      if (!editingCustomer?.id) {
        message.error('Creating customers is disabled.');
        return;
      }

      await axios.put(`/api/customers/records/${editingCustomer.id}`, values);
      message.success('Customer updated successfully.');

      closeModal();
      fetchData();
    } catch (e) {
      if (e?.response?.data?.detail) {
        message.error(e.response.data.detail);
      } else if (e?.errorFields) {
        // Form validation error handled inline.
      } else {
        message.error('Failed to save customer.');
      }
    } finally {
      setSaving(false);
    }
  };

  const churnColumns = [
    {
      title: 'Customer', key: 'customer',
      render: (_, r) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Avatar
            style={{ background: '#1677ff20', color: '#1677ff' }}
            size={32}
          >
            {r.name?.[0]?.toUpperCase()}
          </Avatar>
          <div>
            <Text strong style={{ fontSize: 13 }}>{r.name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 11 }}>{r.email}</Text>
          </div>
        </div>
      ),
    },
    {
      title: 'Segment', dataIndex: 'segment', key: 'segment',
      render: s => (
        <Tag color={
          s === 'enterprise' ? 'blue' :
          s === 'smb' ? 'green' : 'orange'
        }>
          {s?.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Region', dataIndex: 'region', key: 'region',
      render: r => <Tag>{r}</Tag>,
    },
    {
      title: 'LTV', dataIndex: 'lifetime_value', key: 'ltv',
      render: v => (
        <Text strong style={{ color: '#1677ff' }}>
          NPR {Number(v).toLocaleString()}
        </Text>
      ),
      sorter: (a, b) => a.lifetime_value - b.lifetime_value,
    },
    {
      title: 'Churn Risk', dataIndex: 'churn_risk_score', key: 'churn',
      render: v => {
        const pct = Math.round(v * 100);
        return (
          <div style={{ minWidth: 120 }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: 4, fontSize: 12,
            }}>
              <Text type={pct > 80 ? 'danger' : 'warning'}>{pct}%</Text>
            </div>
            <Progress
              percent={pct}
              size="small"
              showInfo={false}
              strokeColor={pct > 80 ? '#ff4d4f' : pct > 60 ? '#faad14' : '#52c41a'}
            />
          </div>
        );
      },
      sorter: (a, b) => b.churn_risk_score - a.churn_risk_score,
      defaultSortOrder: 'descend',
    },
  ];

  const customerColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name, r) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>{r.email}</Text>
        </div>
      ),
    },
    {
      title: 'Segment',
      dataIndex: 'segment',
      key: 'segment',
      render: s => <Tag color="blue">{(s || 'individual').toUpperCase()}</Tag>,
    },
    {
      title: 'Region',
      dataIndex: 'region',
      key: 'region',
      render: r => r || '-',
    },
    {
      title: 'LTV',
      dataIndex: 'lifetime_value',
      key: 'lifetime_value',
      render: v => `NPR ${Number(v || 0).toLocaleString()}`,
      sorter: (a, b) => Number(a.lifetime_value || 0) - Number(b.lifetime_value || 0),
    },
    {
      title: 'Status',
      dataIndex: 'is_active',
      key: 'is_active',
      render: active => (
        <Tag color={active ? 'green' : 'red'}>
          {active ? 'ACTIVE' : 'INACTIVE'}
        </Tag>
      ),
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, r) => (
        <Button
          size="small"
          icon={<EditOutlined />}
          onClick={() => openEditModal(r)}
        >
          Edit
        </Button>
      ),
    },
  ];

  return (
    <div>
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 24,
      }}>
        <div>
          <h2 style={{ margin: 0 }}>Customer Analytics</h2>
          <Text type="secondary">
            Customer insights and churn analysis
          </Text>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchData}
            loading={loading}
          >
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Alert type="error" message={error} showIcon
          style={{ marginBottom: 24 }} />
      )}

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {[
          {
            title: 'Total Customers',
            value: Number(summary?.total || 0).toLocaleString(),
            color: '#1677ff',
          },
          {
            title: 'Active Customers',
            value: Number(summary?.active || 0).toLocaleString(),
            color: '#52c41a',
          },
          {
            title: 'Avg LTV',
            value: `NPR ${Number(summary?.avg_ltv || 0).toLocaleString()}`,
            color: '#faad14',
          },
          {
            title: 'Total LTV',
            value: `NPR ${Number(summary?.total_ltv || 0).toLocaleString()}`,
            color: '#722ed1',
          },
        ].map(item => (
          <Col xs={24} sm={12} lg={6} key={item.title}>
            <Card style={{
              borderRadius: 12,
              borderLeft: `4px solid ${item.color}`,
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
            }}>
              {loading ? <Spin size="small" /> : (
                <Statistic
                  title={<span style={{ color: '#888', fontSize: 13 }}>
                    {item.title}
                  </span>}
                  value={item.value}
                  valueStyle={{ color: item.color, fontSize: 18, fontWeight: 700 }}
                />
              )}
            </Card>
          </Col>
        ))}
      </Row>

      {/* Segments + Acquisition */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }} align="stretch">
        <Col xs={24} lg={12} style={{ display: 'flex' }}>
          <Card title="Customer Segments" style={{ borderRadius: 12, width: '100%', height: '100%' }}>
            {loading ? <Spin /> : (
              <>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={segments}
                      dataKey="count"
                      nameKey="segment"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ segment, percent }) =>
                        `${segment} ${(percent * 100).toFixed(0)}%`
                      }
                    >
                      {segments.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartTooltip />
                  </PieChart>
                </ResponsiveContainer>
                {segments.map((s, i) => (
                  <div key={s.segment} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '6px 0',
                    borderBottom: '1px solid #f0f0f0',
                    fontSize: 13,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{
                        width: 10, height: 10, borderRadius: 2,
                        background: COLORS[i % COLORS.length],
                      }} />
                      <Text>{s.segment?.toUpperCase()}</Text>
                    </div>
                    <div style={{ display: 'flex', gap: 16 }}>
                      <Text type="secondary">{s.count} customers</Text>
                      <Text strong style={{ color: COLORS[i % COLORS.length] }}>
                        NPR {Number(s.avg_ltv).toLocaleString()} avg LTV
                      </Text>
                    </div>
                  </div>
                ))}
              </>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={12} style={{ display: 'flex' }}>
          <Card title="Acquisition Channels" style={{ borderRadius: 12, width: '100%', height: '100%' }}>
            {loading ? <Spin /> : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={acquisition}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="acquisition_channel"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis
                    tick={{ fontSize: 11 }}
                  />
                  <RechartTooltip />
                  <Bar dataKey="customers" fill="#1677ff" radius={[4, 4, 0, 0]} name="Customers" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>
        </Col>
      </Row>

      {/* Churn Risk Table */}
      <Card
        title={
          <span>
            <WarningOutlined style={{ color: '#faad14', marginRight: 8 }} />
            High Churn Risk Customers
          </span>
        }
        style={{ borderRadius: 12 }}
        extra={
          <Tag color="red">{churn.length} at risk</Tag>
        }
      >
        <Table
          dataSource={churn}
          columns={churnColumns}
          rowKey="email"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="middle"
        />
      </Card>

      <Card
        title="Customer Directory"
        style={{ borderRadius: 12, marginTop: 16 }}
        extra={<Tag color="blue">{records.length} total</Tag>}
      >
        <Table
          dataSource={records}
          columns={customerColumns}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 8 }}
          size="middle"
        />
      </Card>

      <Modal
        title="Edit Customer"
        open={modalOpen}
        onOk={handleSaveCustomer}
        onCancel={closeModal}
        confirmLoading={saving}
        okText="Update"
        width={760}
      >
        <Form form={form} layout="vertical">
          <Row gutter={12}>
            <Col xs={24} md={12}>
              <Form.Item
                name="name"
                label="Name"
                rules={[{ required: true, message: 'Name is required' }]}
              >
                <Input placeholder="Customer name" />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                name="email"
                label="Email"
                rules={[
                  { required: true, message: 'Email is required' },
                  { type: 'email', message: 'Enter a valid email' },
                  {
                    pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                    message: 'Enter a correct email format',
                  },
                ]}
              >
                <Input placeholder="name@example.com" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col xs={24} md={12}>
              <Form.Item
                name="phone"
                label="Phone"
                rules={[
                  { required: true, message: 'Phone number is required' },
                  {
                    pattern: /^\d{10}$/,
                    message: 'Phone number must be exactly 10 digits',
                  },
                ]}
              >
                <Input placeholder="10-digit phone number" maxLength={10} />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item name="country" label="Country">
                <Input placeholder="Country" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col xs={24} md={12}>
              <Form.Item name="region" label="Region">
                <Input placeholder="Region" />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item name="segment" label="Segment">
                <Select
                  options={[
                    { label: 'Individual', value: 'individual' },
                    { label: 'SMB', value: 'smb' },
                    { label: 'Enterprise', value: 'enterprise' },
                  ]}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col xs={24} md={12}>
              <Form.Item name="acquisition_channel" label="Acquisition Channel">
                <Input placeholder="Web, Referral, Ads..." />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item name="lifetime_value" label="Lifetime Value">
                <InputNumber min={0} precision={2} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item name="churn_risk_score" label="Churn Risk (0-1)">
                <InputNumber min={0} max={1} step={0.01} precision={3} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="is_active" label="Active" valuePropName="checked">
            <Switch checkedChildren="Active" unCheckedChildren="Inactive" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}