-- Core schema. Embedding dimension (1024) must match EMBEDDING_DIM.
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Async backbone state ────────────────────────────────────────────────────────────────
CREATE TABLE tasks (
    id              text PRIMARY KEY,
    tenant          text NOT NULL,
    user_id         text NOT NULL,
    conversation_id text,
    query           text NOT NULL,
    status          text NOT NULL,
    result          jsonb,
    reply_to        jsonb,
    attempts        int NOT NULL DEFAULT 0,
    events          jsonb NOT NULL DEFAULT '[]',
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX tasks_tenant_created ON tasks (tenant, created_at DESC);
CREATE INDEX tasks_status ON tasks (status) WHERE status IN ('queued', 'running', 'awaiting_approval');

-- ── Human approvals for write tools ─────────────────────────────────────────────────────
CREATE TABLE approvals (
    id           text PRIMARY KEY,
    task_id      text NOT NULL,
    tenant       text NOT NULL,
    tool         text NOT NULL,
    args         jsonb NOT NULL,
    args_hash    text NOT NULL,
    status       text NOT NULL,
    requested_by text NOT NULL,
    decided_by   text,
    decided_at   timestamptz,
    reason       text,
    result       jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX approvals_tenant_status ON approvals (tenant, status);

-- ── Tamper-evident audit log (hash chained, append-only) ────────────────────────────────
CREATE TABLE audit_log (
    id         bigserial PRIMARY KEY,
    ts         timestamptz NOT NULL,
    actor      text NOT NULL,
    actor_type text NOT NULL,
    action     text NOT NULL,
    resource   text,
    decision   text,
    tenant     text,
    task_id    text,
    trace_id   text,
    details    jsonb NOT NULL DEFAULT '{}',
    prev_hash  text NOT NULL,
    hash       text NOT NULL UNIQUE
);
CREATE INDEX audit_task ON audit_log (task_id);
CREATE INDEX audit_ts ON audit_log (ts);

CREATE FUNCTION audit_log_immutable() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER audit_log_no_update BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();
CREATE TRIGGER audit_log_no_truncate BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_immutable();

-- ── Knowledge: documents, chunks (vector + FTS) ─────────────────────────────────────────
CREATE TABLE documents (
    id         text PRIMARY KEY,
    tenant     text NOT NULL,
    source     text NOT NULL,
    title      text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX documents_tenant ON documents (tenant);

CREATE TABLE chunks (
    id        text PRIMARY KEY,
    doc_id    text NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    tenant    text NOT NULL,
    ord       int NOT NULL,
    content   text NOT NULL,
    embedding vector(1024) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX chunks_tenant ON chunks (tenant);
CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);

-- ── GraphRAG: entities, relations, communities ──────────────────────────────────────────
CREATE TABLE graph_nodes (
    id          text PRIMARY KEY,
    tenant      text NOT NULL,
    name        text NOT NULL,
    type        text NOT NULL DEFAULT 'entity',
    description text NOT NULL DEFAULT '',
    UNIQUE (tenant, name)
);
CREATE TABLE graph_edges (
    id       bigserial PRIMARY KEY,
    tenant   text NOT NULL,
    src      text NOT NULL,
    dst      text NOT NULL,
    relation text NOT NULL,
    chunk_id text,
    UNIQUE (tenant, src, dst, relation)
);
CREATE INDEX graph_edges_src ON graph_edges (tenant, src);
CREATE INDEX graph_edges_dst ON graph_edges (tenant, dst);
CREATE TABLE graph_communities (
    id        text PRIMARY KEY,
    tenant    text NOT NULL,
    level     int NOT NULL DEFAULT 0,
    members   jsonb NOT NULL,
    summary   text NOT NULL,
    embedding vector(1024) NOT NULL
);
CREATE INDEX graph_communities_tenant ON graph_communities (tenant);

-- ── Long-term memory (per user, expiring) ───────────────────────────────────────────────
CREATE TABLE memories (
    id         text PRIMARY KEY,
    tenant     text NOT NULL,
    user_id    text NOT NULL,
    content    text NOT NULL,
    embedding  vector(1024) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz
);
CREATE INDEX memories_user ON memories (tenant, user_id);
CREATE INDEX memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);

-- ── Reference internal API (tickets) wrapped by the MCP tool server ─────────────────────
CREATE TABLE tickets (
    id          bigserial PRIMARY KEY,
    tenant      text NOT NULL,
    title       text NOT NULL,
    description text NOT NULL DEFAULT '',
    priority    text NOT NULL DEFAULT 'medium',
    status      text NOT NULL DEFAULT 'open',
    created_by  text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX tickets_tenant ON tickets (tenant, status);

-- ── Eval history (CI gate + online canary) ──────────────────────────────────────────────
CREATE TABLE eval_runs (
    id         bigserial PRIMARY KEY,
    git_sha    text,
    target     text NOT NULL,
    metrics    jsonb NOT NULL,
    passed     boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
