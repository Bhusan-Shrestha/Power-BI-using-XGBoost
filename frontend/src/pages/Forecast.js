import React, { useEffect, useState } from 'react';
import {
  Card, Alert,
  Row, Col, Statistic, Tag, Typography,
  Spin, Table,
} from 'antd';
import axios from '../api/axios';

const { Text } = Typography;

export default function Forecast() {
  const metric = 'sales';
  const periods = 30;
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const nextMonthLabel = data?.forecast_month
    ? new Date(`${data.forecast_month}T00:00:00`).toLocaleString('en-US', { month: 'long', year: 'numeric' })
    : new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1).toLocaleString('en-US', { month: 'long', year: 'numeric' });

  const fetchForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get('/api/forecast/latest');
      setData(res.data);
    } catch (e) {
      setError(
        e.response?.data?.detail ||
        'Forecast is not available right now.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchForecast();
  }, []);

  const predictions = data?.predictions || [];
  const summary = data?.summary || {};

  const predColumns = [
    {
      title: '#', dataIndex: 'rank', key: 'rank', width: 64,
      render: v => <Tag color="blue">{v}</Tag>,
    },
    {
      title: 'Product', dataIndex: 'product_name', key: 'product_name',
      render: v => <Text strong>{v}</Text>,
    },
    {
      title: 'Predicted Units (Next 30 Days)', dataIndex: 'predicted_units', key: 'predicted_units',
      render: v => (
        <Text strong style={{ color: '#1677ff' }}>
          {Math.round(Number(v || 0)).toLocaleString()} units
        </Text>
      ),
      sorter: (a, b) => Number(a.predicted_units || 0) - Number(b.predicted_units || 0),
      defaultSortOrder: 'descend',
    },
    {
      title: 'Lower Bound', dataIndex: 'lower_units', key: 'lower_units',
      render: v => (
        <Text type="secondary">{Math.round(Number(v || 0)).toLocaleString()} units</Text>
      ),
    },
    {
      title: 'Upper Bound', dataIndex: 'upper_units', key: 'upper_units',
      render: v => (
        <Text type="secondary">{Math.round(Number(v || 0)).toLocaleString()} units</Text>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-start',
          gap: 12,
          flexWrap: 'wrap',
        }}>
          <h2 style={{ margin: 0 }}>ML Forecasting</h2>
        </div>
        <Text type="secondary">
          Single-model AI forecast for next-month sales units by product
        </Text>
      </div>

      {error && (
        <Alert
          type="error"
          message={error}
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {loading && (
        <Card style={{ borderRadius: 12, textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
          <Text style={{ display: 'block', marginTop: 16, color: '#888' }}>
            Loading latest monthly Sales forecast...
          </Text>
        </Card>
      )}

      {data && !loading && (
        <>
          {data.note && (
            <Alert
              type="info"
              message={data.note}
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {/* Summary Stats */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            {[
              {
                title: 'Total Predicted Units',
                value: Math.round(Number(summary.total_predicted_units || 0)).toLocaleString(),
                color: '#1677ff',
              },
              {
                title: 'Products Forecasted',
                value: Number(summary.product_count || 0).toLocaleString(),
                color: '#52c41a',
              },
              {
                title: 'Top Product',
                value: summary.top_product || '-',
                color: '#faad14',
              },
              {
                title: 'Top Product Units',
                value: Math.round(Number(summary.top_predicted_units || 0)).toLocaleString(),
                color: '#722ed1',
              },
              {
                title: 'Forecast Month',
                value: nextMonthLabel,
                color: '#13c2c2',
              },
            ].map(item => (
              <Col xs={12} sm={8} lg={4} key={item.title}>
                <Card style={{
                  borderRadius: 12,
                  borderLeft: `4px solid ${item.color}`,
                }}>
                  <Statistic
                    title={<span style={{ color: '#888', fontSize: 12 }}>
                      {item.title}
                    </span>}
                    value={item.value}
                    valueStyle={{ color: item.color, fontSize: 16, fontWeight: 700 }}
                  />
                </Card>
              </Col>
            ))}
          </Row>

          {/* Predictions Table */}
          <Card
            title={`${metric.charAt(0).toUpperCase() + metric.slice(1)} Unit Forecast by Product (${periods} days)`}
            style={{ borderRadius: 12 }}
            extra={<Tag>{predictions.length} products</Tag>}
          >
            <Table
              dataSource={predictions}
              columns={predColumns}
              rowKey="product_id"
              pagination={{ pageSize: 10 }}
              size="small"
            />
          </Card>
        </>
      )}
    </div>
  );
}