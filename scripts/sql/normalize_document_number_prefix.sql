-- Normalize document numbers stored with a "Số:" / "số" prefix or spaces
-- around "/" (e.g. "Số: 15 /2017/QĐ-UBND" -> "15/2017/QĐ-UBND").
-- 761 documents (295 in force) on 2026-10-06. Exact-number seed resolution
-- (upper(replace(document_number,' ',''))) cannot match them, and the
-- verifier reports DOCUMENT_NUMBER_NOT_IN_EVIDENCE when a claim cites the
-- clean number (benchmark STT 6: "số: 34/2024/QH15").
--
-- Review, then run:  docker exec -i lawchat-postgres-1 psql -U lawchat -d lawchat < scripts/sql/normalize_document_number_prefix.sql
-- A full dump taken before any data change is in backups/20261006_data_fix/.

BEGIN;

CREATE TABLE IF NOT EXISTS document_number_prefix_backup_20261006 AS
SELECT id, external_id, document_number
FROM documents
WHERE document_number ~* '^\s*(số|so)\s*[:.]?\s*\d';

UPDATE documents
SET document_number = regexp_replace(
        regexp_replace(document_number, '^\s*(số|so)\s*[:.]?\s*', '', 'i'),
        '\s*/\s*', '/', 'g'),
    updated_at = now()
WHERE document_number ~* '^\s*(số|so)\s*[:.]?\s*\d';

SELECT count(*) AS remaining_prefixed
FROM documents
WHERE document_number ~* '^\s*(số|so)\s*[:.]?\s*\d';

COMMIT;

-- Revert:
-- UPDATE documents d SET document_number = b.document_number
-- FROM document_number_prefix_backup_20261006 b WHERE d.id = b.id;
