CREATE TABLE IF NOT EXISTS companies (
    company_id    TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    is_sme        INTEGER NOT NULL,
    region        TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    profile_json  TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    thread_id  TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    status     TEXT NOT NULL,          -- recommended / selected / done / failed
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendations (
    thread_id    TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    bidNtceNo    TEXT NOT NULL,
    bidNtceNm    TEXT,
    soft_score   REAL,
    reasons_json TEXT,
    source       TEXT,                 -- rag / rule
    PRIMARY KEY (thread_id, rank)
);

CREATE TABLE IF NOT EXISTS selections (
    thread_id   TEXT PRIMARY KEY,
    bidNtceNo   TEXT NOT NULL,
    selected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    thread_id     TEXT PRIMARY KEY,
    bidNtceNo     TEXT NOT NULL,
    criteria_json TEXT,
    proposal_html TEXT,
    created_at    TEXT NOT NULL
);
