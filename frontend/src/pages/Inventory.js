import React, { useState, useEffect } from 'react';
import {
  Card, Table, Tag, Badge, Row, Col,
  Statistic, Alert, Button, Input,
  Spin, Progress, Typography, Modal,
  Form, InputNumber, Select, Space, message, DatePicker,
} from 'antd';
import {
  SearchOutlined, ReloadOutlined,
  WarningOutlined, CheckCircleOutlined,
  CloseCircleOutlined,
  EditOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import axios from '../api/axios';

const { Text } = Typography;

const STATUS_CONFIG = {
  in_stock:     { color: 'success', icon: <CheckCircleOutlined />, label: 'In Stock',     tagColor: 'green'  },
  low_stock:    { color: 'warning', icon: <WarningOutlined />,     label: 'Low Stock',    tagColor: 'orange' },
  out_of_stock: { color: 'error',   icon: <CloseCircleOutlined />, label: 'Out of Stock', tagColor: 'red'    },
};

export default function Inventory() {
  const [data, setData]       = useState([]);
  const [records, setRecords] = useState([]);
  const [products, setProducts] = useState([]);
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [recordLoading, setRecordLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingRecord, setEditingRecord] = useState(null);
  const [search, setSearch]   = useState('');
  const [error, setError]     = useState(null);
  const [selectedProductStock, setSelectedProductStock] = useState(null);
  const [selectedProductReorderPoint, setSelectedProductReorderPoint] = useState(null);
  const [selectedProductSourceWarehouse, setSelectedProductSourceWarehouse] = useState('Kathmandu');
  const [form] = Form.useForm();
  const disablePastDate = (current) => current && current.startOf('day').isBefore(dayjs().startOf('day'));

  const fetchData = async () => {
    setLoading(true);
    setRecordLoading(true);
    setError(null);
    try {
      const [inv, sum, alt, rec, prod] = await Promise.all([
        axios.get('/api/inventory/status'),
        axios.get('/api/inventory/summary'),
        axios.get('/api/inventory/alerts'),
        axios.get('/api/inventory/records?limit=300'),
        axios.get('/api/inventory/products'),
      ]);
      setData(inv.data);
      setSummary(sum.data);
      setAlerts(alt.data.alerts || []);
      setRecords(rec.data);
      setProducts(prod.data);
    } catch (e) {
      setError('Failed to load inventory data.');
    } finally {
      setLoading(false);
      setRecordLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const openEdit = (record) => {
    setEditingRecord(record);
    setSelectedProductStock(Number(record.quantity_on_hand || 0));
    setSelectedProductReorderPoint(Number(record.reorder_point || 0));
    setSelectedProductSourceWarehouse(record.warehouse || 'Kathmandu');
    form.setFieldsValue({
      product_id: record.product_id,
      warehouse: record.warehouse,
      quantity_on_hand: record.quantity_on_hand,
      reorder_point: record.reorder_point,
      reorder_quantity: record.reorder_quantity,
      last_restocked: record.last_restocked ? dayjs(record.last_restocked) : dayjs(),
    });
    setModalOpen(true);
  };

  const handleProductChange = async (productId) => {
    try {
      const res = await axios.get(`/api/inventory/product-defaults/${productId}`);
      const currentStock = Number(res.data?.quantity_on_hand || 0);
      const currentReorderPoint = Number(res.data?.reorder_point || 50);
      const currentReorderQty = Number(res.data?.reorder_quantity || 200);
      const sourceWarehouse = res.data?.warehouse || 'Kathmandu';

      setSelectedProductStock(currentStock);
      setSelectedProductReorderPoint(currentReorderPoint);
      setSelectedProductSourceWarehouse(sourceWarehouse);
      form.setFieldsValue({
        quantity_on_hand: currentStock,
        reorder_point: currentReorderPoint,
        reorder_quantity: currentReorderQty,
      });
    } catch (e) {
      setSelectedProductStock(null);
      setSelectedProductReorderPoint(null);
      setSelectedProductSourceWarehouse('Kathmandu');
      message.error('Failed to load product inventory values from database.');
    }
  };

  const submitRecord = async () => {
    try {
      const values = await form.validateFields();
      setSaving(true);

      const payload = {
        product_id: Number(values.product_id),
        warehouse: values.warehouse,
        quantity_on_hand: Number(values.quantity_on_hand),
        reorder_point: Number(values.reorder_point),
        reorder_quantity: Number(values.reorder_quantity),
        last_restocked: values.last_restocked ? values.last_restocked.toISOString() : null,
      };

      if (!editingRecord) {
        message.error('Creating inventory records is disabled.');
        return;
      }

      await axios.put(`/api/inventory/records/${editingRecord.id}`, payload);
      message.success('Inventory record updated successfully.');

      setModalOpen(false);
      form.resetFields();
      fetchData();
    } catch (e) {
      const detail = e?.response?.data?.detail;
      if (detail) {
        message.error(detail);
      } else if (!e?.errorFields) {
        message.error('Unable to save inventory record.');
      }
    } finally {
      setSaving(false);
    }
  };

  const filtered = data.filter(item =>
    item.name?.toLowerCase().includes(search.toLowerCase()) ||
    item.sku?.toLowerCase().includes(search.toLowerCase()) ||
    item.category?.toLowerCase().includes(search.toLowerCase())
  );

  const columns = [
    {
      title: 'Product', key: 'product',
      render: (_, r) => (
        <div>
          <Text strong style={{ fontSize: 13 }}>{r.name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 11 }}>{r.sku}</Text>
        </div>
      ),
    },
    {
      title: 'Category', dataIndex: 'category', key: 'category',
      render: c => <Tag color="blue">{c}</Tag>,
      filters: [...new Set(data.map(d => d.category))].map(c => ({ text: c, value: c })),
      onFilter: (value, record) => record.category === value,
    },
    {
      title: 'Warehouse', dataIndex: 'warehouse', key: 'warehouse',
      render: w => <Tag color="purple">{w}</Tag>,
    },
    {
      title: 'On Hand', dataIndex: 'quantity_on_hand', key: 'qty',
      render: (v, r) => (
        <div>
          <Text strong style={{
            color: v <= 0 ? '#ff4d4f' : v <= r.reorder_point ? '#faad14' : '#52c41a',
          }}>
            {v}
          </Text>
          <Progress
            percent={Math.min(Math.round((v / (r.reorder_point * 3)) * 100), 100)}
            size="small"
            showInfo={false}
            strokeColor={v <= 0 ? '#ff4d4f' : v <= r.reorder_point ? '#faad14' : '#52c41a'}
            style={{ marginTop: 4 }}
          />
        </div>
      ),
      sorter: (a, b) => a.quantity_on_hand - b.quantity_on_hand,
    },
    {
      title: 'Reorder At', dataIndex: 'reorder_point', key: 'reorder',
      render: v => <Text type="secondary">{v} units</Text>,
    },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      render: s => {
        const cfg = STATUS_CONFIG[s] || STATUS_CONFIG.in_stock;
        return (
          <Badge
            status={cfg.color}
            text={
              <Tag color={cfg.tagColor} icon={cfg.icon}>
                {cfg.label}
              </Tag>
            }
          />
        );
      },
      filters: [
        { text: 'In Stock',     value: 'in_stock'     },
        { text: 'Low Stock',    value: 'low_stock'    },
        { text: 'Out of Stock', value: 'out_of_stock' },
      ],
      onFilter: (value, record) => record.status === value,
    },
  ];

  const operationColumns = [
    {
      title: 'Product', key: 'product',
      render: (_, r) => (
        <div>
          <Text strong>{r.name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 11 }}>{r.sku}</Text>
        </div>
      ),
    },
    {
      title: 'Category', dataIndex: 'category', key: 'category',
      render: c => <Tag color="blue">{c}</Tag>,
    },
    {
      title: 'Warehouse', dataIndex: 'warehouse', key: 'warehouse',
      render: w => <Tag color="purple">{w}</Tag>,
    },
    {
      title: 'On Hand', dataIndex: 'quantity_on_hand', key: 'quantity_on_hand',
      sorter: (a, b) => a.quantity_on_hand - b.quantity_on_hand,
    },
    {
      title: 'Reorder Point', dataIndex: 'reorder_point', key: 'reorder_point',
    },
    {
      title: 'Reorder Qty', dataIndex: 'reorder_quantity', key: 'reorder_quantity',
    },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      render: s => {
        const cfg = STATUS_CONFIG[s] || STATUS_CONFIG.in_stock;
        return <Tag color={cfg.tagColor}>{cfg.label}</Tag>;
      },
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
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: 24,
      }}>
        <div>
          <h2 style={{ margin: 0 }}>Inventory Management</h2>
          <Text type="secondary">
            {data.length} products across all warehouses
          </Text>
        </div>
        <Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchData}
            loading={loading}
          >
            Refresh
          </Button>
        </Space>
      </div>

      {error && (
        <Alert type="error" message={error} showIcon
          style={{ marginBottom: 24 }} />
      )}

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {[
          {
            title: 'Total Products', value: summary?.total_products || 0,
            color: '#1677ff',
          },
          {
            title: 'In Stock', value: summary?.in_stock || 0,
            color: '#52c41a',
          },
          {
            title: 'Low Stock', value: summary?.low_stock || 0,
            color: '#faad14',
          },
          {
            title: 'Out of Stock', value: summary?.out_of_stock || 0,
            color: '#ff4d4f',
          },
        ].map(item => (
          <Col xs={12} sm={6} key={item.title}>
            <Card style={{
              borderRadius: 12,
              borderLeft: `4px solid ${item.color}`,
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
            }}>
              {loading ? <Spin size="small" /> : (
                <Statistic
                  title={<span style={{ color: '#888', fontSize: 12 }}>
                    {item.title}
                  </span>}
                  value={item.value}
                  valueStyle={{ color: item.color, fontWeight: 700 }}
                />
              )}
            </Card>
          </Col>
        ))}
      </Row>

      {/* Alerts */}
      {alerts.length > 0 && (
        <Alert
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          message={`${alerts.length} products need reordering`}
          description={
            <div style={{ marginTop: 8 }}>
              {alerts.slice(0, 3).map(a => (
                <Tag key={a.sku} color="orange" style={{ marginBottom: 4 }}>
                  {a.name} — {a.quantity_on_hand} left
                </Tag>
              ))}
              {alerts.length > 3 && (
                <Text type="secondary"> +{alerts.length - 3} more</Text>
              )}
            </div>
          }
          style={{ marginBottom: 24 }}
        />
      )}

      {/* Search + Table */}
      <Card style={{ borderRadius: 12 }}>
        <Input
          prefix={<SearchOutlined />}
          placeholder="Search by product name, SKU or category..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{ marginBottom: 16, maxWidth: 400 }}
          allowClear
        />
        <Table
          dataSource={filtered}
          columns={columns}
          rowKey="sku"
          loading={loading}
          pagination={{ pageSize: 15, showSizeChanger: true }}
          rowClassName={r =>
            r.status === 'out_of_stock'
              ? 'row-danger'
              : r.status === 'low_stock'
              ? 'row-warning'
              : ''
          }
          size="middle"
        />
      </Card>

      <Card
        title="Inventory Operations"
        style={{ borderRadius: 12, marginTop: 16 }}
        extra={<Tag color="cyan">Create / Edit</Tag>}
      >
        <Table
          dataSource={records}
          columns={operationColumns}
          rowKey="id"
          loading={recordLoading}
          pagination={{ pageSize: 10 }}
          size="middle"
        />
      </Card>

      <Modal
        title="Edit Inventory Record"
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={submitRecord}
        confirmLoading={saving}
        okText="Update Inventory"
        width={720}
        destroyOnClose
      >
        <Form form={form} layout="vertical">
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                name="product_id"
                label="Product"
                rules={[{ required: true, message: 'Product is required' }]}
              >
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
              <Form.Item
                name="warehouse"
                label="Warehouse"
                rules={[{ required: true, message: 'Warehouse is required' }]}
              >
                <Select
                  placeholder="Select warehouse"
                  options={[{ value: 'Kathmandu', label: 'Kathmandu' }]}
                  disabled
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12} style={{ marginBottom: 8 }}>
            <Col span={24}>
              <Alert
                type="info"
                showIcon
                message={`Current Database Values (${selectedProductSourceWarehouse}) - Quantity On Hand: ${selectedProductStock ?? '-'} | Reorder Point: ${selectedProductReorderPoint ?? '-'}`}
              />
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={8}>
              <Form.Item
                name="last_restocked"
                label="Restock Date"
                rules={[{ required: true, message: 'Restock date is required' }]}
              >
                <DatePicker style={{ width: '100%' }} disabledDate={disablePastDate} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={8}>
              <Form.Item
                name="quantity_on_hand"
                label="Quantity On Hand"
                rules={[{ required: true, message: 'Quantity is required' }]}
              >
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="reorder_point"
                label="Reorder Point"
                rules={[{ required: true, message: 'Reorder point is required' }]}
              >
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="reorder_quantity"
                label="Reorder Quantity"
                rules={[{ required: true, message: 'Reorder quantity is required' }]}
              >
                <InputNumber min={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      <style>{`
        .row-danger td { background: #fff2f0 !important; }
        .row-warning td { background: #fffbe6 !important; }
      `}</style>
    </div>
  );
}