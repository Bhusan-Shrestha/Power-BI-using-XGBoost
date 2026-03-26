import React, { useState, useEffect, useCallback } from 'react';
import {
  Row, Col, Card, Statistic, Table, Tag,
  Select, Button, Spin, Alert, Typography,
  Modal, Form, InputNumber, message,
} from 'antd';
import {
  ReloadOutlined,
  EditOutlined,
} from '@ant-design/icons';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie,
  Cell, Legend, LineChart, Line,
} from 'recharts';
import dayjs from 'dayjs';
import axios from '../api/axios';

const { Text } = Typography;
const { Option } = Select;

const COLORS = ['#1677ff','#52c41a','#faad14','#f5222d','#722ed1','#13c2c2'];

export default function Sales() {
  const [period, setPeriod]       = useState('30d');
  const [summary, setSummary]     = useState(null);
  const [byCategory, setByCategory] = useState([]);
  const [byRegion, setByRegion]   = useState([]);
  const [monthly, setMonthly]     = useState([]);
  const [topProducts, setTopProducts] = useState([]);
  const [records, setRecords]     = useState([]);
  const [products, setProducts]   = useState([]);
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading]     = useState(true);
  const [recordLoading, setRecordLoading] = useState(false);
  const [saving, setSaving]       = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingSale, setEditingSale] = useState(null);
  const [error, setError]         = useState(null);
  const [form] = Form.useForm();

  const fetchData = useCallback(async () => {
    setLoading(true);
    setRecordLoading(true);
    setError(null);
    try {
      const [s, c, r, m, p, rec, prod, cust] = await Promise.all([
        axios.get(`/api/sales/summary?period=${period}`),
        axios.get(`/api/sales/by-category?period=${period}`),
        axios.get(`/api/sales/by-region?period=${period}`),
        axios.get('/api/sales/monthly-trend'),
        axios.get(`/api/sales/top-products?period=${period}&limit=10`),
        axios.get('/api/sales/records?limit=100'),
        axios.get('/api/sales/products'),
        axios.get('/api/sales/customers'),
      ]);
      setSummary(s.data);
      setByCategory(c.data);
      setByRegion(r.data);
      setMonthly(m.data);
      setTopProducts(p.data);
      setRecords(rec.data);
      setProducts(prod.data);
      setCustomers(cust.data);
    } catch (e) {
      setError('Failed to load sales data.');
    } finally {
      setLoading(false);
      setRecordLoading(false);
    }
  }, [period]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const openEdit = (record) => {
    setEditingSale(record);
    form.setFieldsValue({
      product_id: record.product_id,
      customer_id: record.customer_id,
      quantity: record.quantity,
      unit_price: Number(record.unit_price),
      discount_pct: Number(record.discount || 0) * 100,
      region: record.region || 'Bagmati',
      channel: record.channel,
    });
    setModalOpen(true);
  };

  const handleProductChange = (productId) => {
    const selected = products.find(p => p.id === productId);
    if (!selected) return;
    const currentPrice = form.getFieldValue('unit_price');
    if (!currentPrice || Number(currentPrice) <= 0) {
      form.setFieldsValue({ unit_price: Number(selected.price) });
    }
  };

  const submitSale = async () => {
    try {
      const values = await form.validateFields();
      setSaving(true);

      const payload = {
        product_id: Number(values.product_id),
        customer_id: values.customer_id ? Number(values.customer_id) : null,
        quantity: Number(values.quantity),
        unit_price: values.unit_price ? Number(values.unit_price) : null,
        discount: Number(values.discount_pct || 0) / 100,
        region: values.region || null,
        channel: values.channel,
        sale_date: editingSale
          ? (editingSale.sale_date || dayjs().toISOString())
          : dayjs().toISOString(),
      };

      if (!editingSale) {
        message.error('Creating sales records is disabled.');
        return;
      }

      await axios.put(`/api/sales/records/${editingSale.id}`, payload);
      message.success('Sale updated successfully.');

      setModalOpen(false);
      form.resetFields();
      fetchData();
    } catch (e) {
      const detail = e?.response?.data?.detail;
      if (detail) {
        message.error(detail);
      } else if (!e?.errorFields) {
        message.error('Unable to save sale record.');
      }
    } finally {
      setSaving(false);
    }
  };

  const productColumns = [
    {
      title: '#', key: 'rank',
      render: (_, __, i) => (
        <Tag color={i < 3 ? 'gold' : 'default'}>{i + 1}</Tag>
      ),
      width: 50,
    },
    { title: 'Product',  dataIndex: 'name',     key: 'name'     },
    {
      title: 'Category', dataIndex: 'category', key: 'category',
      render: c => <Tag color="blue">{c}</Tag>,
    },
    {
      title: 'Revenue', dataIndex: 'revenue', key: 'revenue',
      render: v => (
        <Text strong style={{ color: '#1677ff' }}>
          NPR {Number(v).toLocaleString()}
        </Text>
      ),
      sorter: (a, b) => a.revenue - b.revenue,
      defaultSortOrder: 'descend',
    },
    {
      title: 'Units', dataIndex: 'units', key: 'units',
      render: v => <Tag color="green">{v}</Tag>,
    },
  ];

  const PERIOD_LABELS = {
    '7d': 'Last 7 Days',
    '30d': 'Last 30 Days',
    '90d': 'Last 90 Days',
    '1y': 'Last 1 Year',
  };

  const saleColumns = [
    {
      title: 'Date', dataIndex: 'sale_date', key: 'sale_date',
      render: v => dayjs(v).format('YYYY-MM-DD HH:mm'),
      sorter: (a, b) => dayjs(a.sale_date).valueOf() - dayjs(b.sale_date).valueOf(),
      defaultSortOrder: 'descend',
    },
    {
      title: 'Product', key: 'product',
      render: (_, r) => (
        <div>
          <Text strong>{r.product_name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 11 }}>{r.sku}</Text>
        </div>
      ),
    },
    {
      title: 'Qty', dataIndex: 'quantity', key: 'quantity',
      render: v => <Tag color="blue">{v}</Tag>,
    },
    {
      title: 'Channel', dataIndex: 'channel', key: 'channel',
      render: v => <Tag color="purple">{String(v || '').toUpperCase()}</Tag>,
    },
    {
      title: 'Region', dataIndex: 'region', key: 'region',
      render: v => v ? <Tag>{v}</Tag> : <Text type="secondary">N/A</Text>,
    },
    {
      title: 'Revenue', dataIndex: 'total_amount', key: 'total_amount',
      render: v => <Text strong style={{ color: '#1677ff' }}>NPR {Number(v).toLocaleString()}</Text>,
      sorter: (a, b) => Number(a.total_amount) - Number(b.total_amount),
    },
    {
      title: 'Margin %', dataIndex: 'margin_pct', key: 'margin_pct',
      render: v => (
        <Tag color={Number(v) >= 25 ? 'green' : Number(v) >= 10 ? 'orange' : 'red'}>
          {Number(v).toFixed(2)}%
        </Tag>
      ),
    },
    {
      title: 'Action', key: 'action',
      render: (_, r) => (
        <Button size="small" icon={<EditOutlined />} onClick={() => openEdit(r)}>
          Edit
        </Button>
      ),
    },
  ];

  return (
    <div>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 24,
      }}>
        <div>
          <h2 style={{ margin: 0 }}>Sales Analytics</h2>
          <Text type="secondary">{PERIOD_LABELS[period]}</Text>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Select
            value={period}
            onChange={setPeriod}
            style={{ width: 140 }}
          >
            <Option value="7d">Last 7 Days</Option>
            <Option value="30d">Last 30 Days</Option>
            <Option value="90d">Last 90 Days</Option>
            <Option value="1y">Last 1 Year</Option>
          </Select>
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
            title: 'Total Revenue',
            value: `NPR ${Number(summary?.revenue || 0).toLocaleString()}`,
            color: '#1677ff',
          },
          {
            title: 'Total Orders',
            value: Number(summary?.orders || 0).toLocaleString(),
            color: '#52c41a',
          },
          {
            title: 'Units Sold',
            value: Number(summary?.units_sold || 0).toLocaleString(),
            color: '#faad14',
          },
          {
            title: 'Avg Order Value',
            value: `NPR ${Number(summary?.avg_order_value || 0).toLocaleString()}`,
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
                  valueStyle={{ color: item.color, fontSize: 20, fontWeight: 700 }}
                />
              )}
            </Card>
          </Col>
        ))}
      </Row>

      {/* Monthly Trend */}
      <Card
        title="Monthly Revenue Trend"
        style={{ borderRadius: 12, marginBottom: 16 }}
        extra={<Tag color="blue">12 Months</Tag>}
      >
        {loading ? (
          <div style={{ textAlign: 'center', padding: 60 }}>
            <Spin size="large" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={monthly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={v => `${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip
                formatter={v => [`NPR ${Number(v).toLocaleString()}`, 'Revenue']}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="revenue"
                stroke="#1677ff"
                strokeWidth={2}
                dot={{ r: 4 }}
                name="Revenue"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>

      {/* Category + Region Charts */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }} align="stretch">
        <Col xs={24} lg={14} style={{ display: 'flex' }}>
          <Card
            title="Revenue by Category"
            style={{ borderRadius: 12, width: '100%', height: '100%' }}
          >
            {loading ? <Spin /> : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={byCategory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="category"
                    tick={{ fontSize: 11 }}
                    angle={-15}
                    textAnchor="end"
                    height={50}
                  />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    tickFormatter={v => `${(v / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    formatter={v => [`NPR ${Number(v).toLocaleString()}`, 'Revenue']}
                  />
                  <Bar
                    dataKey="revenue"
                    radius={[6, 6, 0, 0]}
                  >
                    {byCategory.map((_, i) => (
                      <Cell
                        key={i}
                        fill={COLORS[i % COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={10} style={{ display: 'flex' }}>
          <Card
            title="Sales by Region"
            style={{ borderRadius: 12, width: '100%', height: '100%' }}
          >
            {loading ? <Spin /> : (
              <>
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={byRegion}
                      dataKey="revenue"
                      nameKey="region"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ region, percent }) =>
                        `${region} ${(percent * 100).toFixed(0)}%`
                      }
                      labelLine={false}
                    >
                      {byRegion.map((_, i) => (
                        <Cell
                          key={i}
                          fill={COLORS[i % COLORS.length]}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={v => [`NPR ${Number(v).toLocaleString()}`, 'Revenue']}
                    />
                  </PieChart>
                </ResponsiveContainer>

                {/* Region legend */}
                <div style={{ marginTop: 8 }}>
                  {byRegion.map((r, i) => {
                    const total = byRegion.reduce((s, x) => s + x.revenue, 0);
                    const pct = total > 0
                      ? Math.round((r.revenue / total) * 100)
                      : 0;
                    return (
                      <div key={r.region} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '4px 0',
                        borderBottom: '1px solid #f0f0f0',
                        fontSize: 12,
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{
                            width: 10, height: 10, borderRadius: 2,
                            background: COLORS[i % COLORS.length],
                          }} />
                          <Text>{r.region}</Text>
                        </div>
                        <div style={{ display: 'flex', gap: 12 }}>
                          <Text type="secondary">{pct}%</Text>
                          <Text strong style={{ color: COLORS[i % COLORS.length] }}>
                            NPR {Number(r.revenue).toLocaleString()}
                          </Text>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </Card>
        </Col>
      </Row>

      {/* Top Products Table */}
      <Card
        title="Top Products by Revenue"
        style={{ borderRadius: 12 }}
        extra={<Tag color="gold">Top 10</Tag>}
      >
        <Table
          dataSource={topProducts}
          columns={productColumns}
          rowKey="sku"
          pagination={{ pageSize: 10 }}
          loading={loading}
          size="middle"
        />
      </Card>

      <Card
        title="Sales Operations"
        style={{ borderRadius: 12, marginTop: 16 }}
        extra={<Tag color="cyan">Latest 100</Tag>}
      >
        <Table
          dataSource={records}
          columns={saleColumns}
          rowKey="id"
          loading={recordLoading}
          pagination={{ pageSize: 10 }}
          size="middle"
        />
      </Card>

      <Modal
        title="Edit Sale Record"
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={submitSale}
        confirmLoading={saving}
        okText="Update Sale"
        width={760}
        destroyOnClose
      >
        <Form form={form} layout="vertical">
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="product_id" label="Product" rules={[{ required: true, message: 'Product is required' }]}>
                <Select
                  showSearch
                  placeholder="Select product"
                  optionFilterProp="label"
                  onChange={handleProductChange}
                  options={products.map(p => ({
                    value: p.id,
                    label: `${p.name} (${p.sku})`,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="customer_id" label="Customer (optional)">
                <Select
                  allowClear
                  showSearch
                  placeholder="Select customer by name or ID"
                  optionFilterProp="label"
                  options={customers.map(c => ({
                    value: c.id,
                    label: `${c.name} (ID: ${c.id})`,
                  }))}
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={8}>
              <Form.Item name="quantity" label="Quantity" rules={[{ required: true, message: 'Quantity is required' }]}>
                <InputNumber style={{ width: '100%' }} min={1} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="unit_price" label="Unit Price (NPR)" rules={[{ required: true, message: 'Unit price is required' }]}>
                <InputNumber style={{ width: '100%' }} min={0.01} precision={2} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="discount_pct" label="Discount %">
                <InputNumber style={{ width: '100%' }} min={0} max={100} precision={2} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="channel" label="Channel" rules={[{ required: true, message: 'Channel is required' }]}>
                <Select>
                  <Option value="online">Online</Option>
                  <Option value="retail">Retail</Option>
                  <Option value="wholesale">Wholesale</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="region" label="Region">
                <Select placeholder="Select region">
                  <Option value="Bagmati">Bagmati</Option>
                  <Option value="Koshi">Koshi</Option>
                  <Option value="Gandaki">Gandaki</Option>
                  <Option value="Lumbini">Lumbini</Option>
                  <Option value="Madhesh">Madhesh</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  );
}