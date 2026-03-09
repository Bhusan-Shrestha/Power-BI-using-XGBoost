import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchAnalyticsFilters,
  fetchAnalyticsRecords,
  fetchAnalyticsSummary,
  fetchCategoryAnalytics,
  fetchProductAnalytics,
  fetchRegionalAnalytics,
  fetchTrendAnalytics,
  parseApiError,
} from "../services/api";

const PIE_COLORS = ["#2a9d8f", "#f4a261", "#e76f51", "#264653", "#8ab17d", "#457b9d"];

export default function Dashboard() {
  const [filterOptions, setFilterOptions] = useState({
    regions: [],
    categories: [],
    products: [],
    min_date: null,
    max_date: null,
  });

  const [filters, setFilters] = useState({
    start_date: "",
    end_date: "",
    region: "",
    category: "",
    product_id: "",
    limit: 200,
  });

  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState([]);
  const [regional, setRegional] = useState([]);
  const [category, setCategory] = useState([]);
  const [products, setProducts] = useState([]);
  const [records, setRecords] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const normalizedFilters = useMemo(() => {
    const base = {
      start_date: filters.start_date,
      end_date: filters.end_date,
      region: filters.region,
      category: filters.category,
      product_id: filters.product_id,
    };
    return base;
  }, [filters]);

  const loadFilterOptions = async () => {
    const response = await fetchAnalyticsFilters();
    setFilterOptions(response.data);

    setFilters((prev) => ({
      ...prev,
      start_date: prev.start_date || response.data.min_date || "",
      end_date: prev.end_date || response.data.max_date || "",
    }));
  };

  const loadAnalytics = async (queryFilters) => {
    const recordsFilters = { ...queryFilters, limit: filters.limit };

    const [summaryRes, trendRes, regionalRes, productRes, categoryRes, recordsRes] = await Promise.all([
      fetchAnalyticsSummary(queryFilters),
      fetchTrendAnalytics(queryFilters),
      fetchRegionalAnalytics(queryFilters),
      fetchProductAnalytics(queryFilters),
      fetchCategoryAnalytics(queryFilters),
      fetchAnalyticsRecords(recordsFilters),
    ]);

    setSummary(summaryRes.data);
    setTrend(trendRes.data || []);
    setRegional(regionalRes.data || []);
    setProducts(productRes.data || []);
    setCategory(categoryRes.data || []);
    setRecords(recordsRes.data || []);
  };

  useEffect(() => {
    async function bootstrap() {
      setLoading(true);
      setError("");
      try {
        await loadFilterOptions();
      } catch (err) {
        setError(parseApiError(err, "Failed to load filter options."));
      } finally {
        setLoading(false);
      }
    }

    bootstrap();
  }, []);

  useEffect(() => {
    async function refresh() {
      if (!filters.start_date || !filters.end_date) {
        return;
      }
      setLoading(true);
      setError("");
      try {
        await loadAnalytics(normalizedFilters);
      } catch (err) {
        setError(parseApiError(err, "Failed to load analytics."));
      } finally {
        setLoading(false);
      }
    }

    refresh();
  }, [normalizedFilters, filters.limit]);

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const clearFilters = () => {
    setFilters((prev) => ({
      ...prev,
      region: "",
      category: "",
      product_id: "",
      start_date: filterOptions.min_date || prev.start_date,
      end_date: filterOptions.max_date || prev.end_date,
    }));
  };

  if (loading && !summary) {
    return (
      <section className="card">
        <h2>Sales Analytics</h2>
        <p>Loading business intelligence dashboard...</p>
      </section>
    );
  }

  return (
    <section className="card">
      <h2>Business Intelligence Dashboard</h2>
      {error ? <p className="error-text">{error}</p> : null}

      <div className="filters-panel">
        <h3>Filters</h3>
        <div className="filters-grid">
          <label>
            Start Date
            <input
              type="date"
              value={filters.start_date}
              onChange={(event) => handleFilterChange("start_date", event.target.value)}
            />
          </label>

          <label>
            End Date
            <input
              type="date"
              value={filters.end_date}
              onChange={(event) => handleFilterChange("end_date", event.target.value)}
            />
          </label>

          <label>
            Region
            <select
              value={filters.region}
              onChange={(event) => handleFilterChange("region", event.target.value)}
            >
              <option value="">All Regions</option>
              {filterOptions.regions.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label>
            Category
            <select
              value={filters.category}
              onChange={(event) => handleFilterChange("category", event.target.value)}
            >
              <option value="">All Categories</option>
              {filterOptions.categories.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label>
            Product
            <select
              value={filters.product_id}
              onChange={(event) => handleFilterChange("product_id", event.target.value)}
            >
              <option value="">All Products</option>
              {filterOptions.products.map((item) => (
                <option key={item.product_id} value={item.product_id}>
                  {item.product_name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Records Limit
            <input
              type="number"
              min="10"
              max="5000"
              value={filters.limit}
              onChange={(event) => handleFilterChange("limit", Number(event.target.value))}
            />
          </label>
        </div>
        <div className="filter-actions">
          <button type="button" onClick={clearFilters}>
            Reset Filters
          </button>
        </div>
      </div>

      {summary ? (
        <div className="stats-grid">
          <article className="metric-card">
            <h4>Total Sales</h4>
            <strong>{summary.total_sales.toFixed(2)}</strong>
          </article>
          <article className="metric-card">
            <h4>Total Profit</h4>
            <strong>{summary.total_profit.toFixed(2)}</strong>
          </article>
          <article className="metric-card">
            <h4>Profit Margin</h4>
            <strong>{summary.avg_margin.toFixed(2)}%</strong>
          </article>
          <article className="metric-card">
            <h4>Rows in Analysis</h4>
            <strong>{summary.record_count}</strong>
          </article>
        </div>
      ) : null}

      <h3>Sales vs Profit Trend</h3>
      <div className="chart-wrap">
        {trend.length === 0 ? (
          <p>No trend data for current filters.</p>
        ) : (
          <ResponsiveContainer>
            <AreaChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="sales" stroke="#2a9d8f" fill="#2a9d8f33" />
              <Area type="monotone" dataKey="profit" stroke="#f4a261" fill="#f4a26133" />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="two-col-charts">
        <div>
          <h3>Regional Performance</h3>
          <div className="chart-wrap small">
            {regional.length === 0 ? (
              <p>No regional data.</p>
            ) : (
              <ResponsiveContainer>
                <BarChart data={regional}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="sales" fill="#2a9d8f" />
                  <Bar dataKey="profit" fill="#f4a261" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div>
          <h3>Category Contribution</h3>
          <div className="chart-wrap small">
            {category.length === 0 ? (
              <p>No category data.</p>
            ) : (
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={category} dataKey="sales" nameKey="category" outerRadius={100} label>
                    {category.map((entry, index) => (
                      <Cell key={`${entry.category}-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      <h3>Top Products by Sales</h3>
      <div className="chart-wrap small">
        {products.length === 0 ? (
          <p>No product data available.</p>
        ) : (
          <ResponsiveContainer>
            <BarChart data={products} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="product_name" width={160} />
              <Tooltip />
              <Legend />
              <Bar dataKey="sales" fill="#457b9d" />
              <Bar dataKey="profit" fill="#e9c46a" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <h3>Detailed Sales Records</h3>
      <div className="table-wrap">
        {records.length === 0 ? (
          <p>No rows found for selected filters.</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Product</th>
                <th>Category</th>
                <th>Region</th>
                <th>Marketing Spend</th>
                <th>Sales</th>
                <th>Profit</th>
                <th>Margin %</th>
              </tr>
            </thead>
            <tbody>
              {records.map((row) => (
                <tr key={row.id}>
                  <td>{row.date}</td>
                  <td>{row.product_name}</td>
                  <td>{row.category}</td>
                  <td>{row.region}</td>
                  <td>{Number(row.marketing_spend).toFixed(2)}</td>
                  <td>{Number(row.sales).toFixed(2)}</td>
                  <td>{Number(row.profit).toFixed(2)}</td>
                  <td>{Number(row.margin_pct).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
