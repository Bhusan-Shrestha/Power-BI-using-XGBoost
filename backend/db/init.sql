-- Initial schema for BI Platform
-- This runs automatically on first Docker startup

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The tables are created by SQLAlchemy/Alembic
-- This file is for extensions and custom functions

-- Function to calculate growth rate
CREATE OR REPLACE FUNCTION growth_rate(current_val FLOAT, previous_val FLOAT)
RETURNS FLOAT AS $$
BEGIN
  IF previous_val = 0 THEN RETURN 0; END IF;
  RETURN ROUND(((current_val - previous_val) / previous_val * 100)::NUMERIC, 2);
END;
$$ LANGUAGE plpgsql;
