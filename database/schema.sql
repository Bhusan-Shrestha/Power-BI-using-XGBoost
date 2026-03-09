CREATE TABLE IF NOT EXISTS products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(120) NOT NULL,
    category VARCHAR(80) NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_sales (
    id SERIAL PRIMARY KEY,
    product_id INT NOT NULL REFERENCES products(product_id),
    date DATE NOT NULL,
    sales NUMERIC(14, 2) NOT NULL,
    profit NUMERIC(14, 2) NOT NULL,
    marketing_spend NUMERIC(14, 2) NOT NULL,
    region VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    product_id INT NOT NULL REFERENCES products(product_id),
    month VARCHAR(7) NOT NULL,
    predicted_sales NUMERIC(14, 2) NOT NULL,
    predicted_profit NUMERIC(14, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_monthly_sales_product_date
ON monthly_sales(product_id, date);

CREATE INDEX IF NOT EXISTS idx_predictions_product_month
ON predictions(product_id, month);
