-- Luật Bảo hiểm xã hội 41/2024/QH15 is in force. The source snapshot marked
-- it EXPIRED from 2026-01-01 because of an edge "74/2025/QH15 REPLACES
-- 41/2024/QH15", but 74/2025 (Luật Việc làm) only ends "Luật Việc làm số
-- 38/2013/QH13 đã được sửa đổi ... theo Luật số 41/2024/QH15".
-- See reports/expired_status_audit.csv.
BEGIN;

UPDATE effective_status
SET valid_to = NULL,
    metadata = metadata || '{"manual_fix": "not repealed by 74/2025/QH15"}'::jsonb,
    updated_at = now()
WHERE document_id = (SELECT id FROM documents WHERE external_id = '175027')
  AND status = 'EFFECTIVE'
  AND valid_from = DATE '2025-07-01';

DELETE FROM effective_status
WHERE document_id = (SELECT id FROM documents WHERE external_id = '175027')
  AND status = 'EXPIRED'
  AND valid_from = DATE '2026-01-01';

UPDATE documents
SET status = 'EFFECTIVE',
    expiry_date = NULL,
    metadata = metadata || '{"manual_fix": "not repealed by 74/2025/QH15", "previous_status": "EXPIRED"}'::jsonb,
    updated_at = now()
WHERE external_id = '175027'
  AND document_number = '41/2024/QH15';

-- Expect: EFFECTIVE | (null)
SELECT status, expiry_date FROM documents WHERE external_id = '175027';

COMMIT;
