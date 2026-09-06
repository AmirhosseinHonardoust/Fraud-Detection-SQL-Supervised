-- Point-in-time feature engineering: every aggregate for a transaction is
-- computed only from that user's *earlier* transactions (ROWS BETWEEN
-- UNBOUNDED PRECEDING AND 1 PRECEDING), so no feature ever sees the current
-- transaction or anything in the future. This avoids the look-ahead leakage
-- present in the previous version, which aggregated over the full dataset.
SELECT
  t.tx_id,
  t.user_id,
  t.date,
  t.region,
  t.merchant,
  t.amount,
  COALESCE(COUNT(t.amount) OVER user_hist, 0) AS tx_count,
  COALESCE(AVG(t.amount) OVER user_hist, 0.0) AS avg_amount,
  COALESCE(SUM(t.amount) OVER user_hist, 0.0) AS total_amount,
  COALESCE(COUNT(t.amount) OVER daily_hist, 0) AS daily_tx,
  COALESCE(SUM(t.amount) OVER daily_hist, 0.0) AS daily_amount,
  t.label
FROM transactions t
WINDOW
  user_hist AS (
    PARTITION BY t.user_id ORDER BY t.date, t.tx_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  ),
  daily_hist AS (
    PARTITION BY t.user_id, t.date ORDER BY t.tx_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  )
ORDER BY t.date, t.tx_id;
