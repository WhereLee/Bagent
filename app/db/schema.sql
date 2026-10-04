-- Bagent RAG —— 目标库 schema（pgvector）
-- 幂等：可重复执行。M1 只做稠密向量检索，但为 M2/M3 预埋 parent_id、
-- metadata、doc_hash + is_deleted（增量与软删）等结构，避免后期大改表。

CREATE EXTENSION IF NOT EXISTS vector;

-- 源文档：一个文件/一条 URL 对应一行
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,           -- 文件路径 / URI（自然主键，用于 upsert/删除）
    doc_hash    TEXT        NOT NULL,           -- 内容哈希，用于变更检测（非唯一：允许历史软删行）
    media_type  TEXT,                            -- pdf / docx / md / html / txt
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

-- 文本块 + 向量
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id    BIGINT  REFERENCES chunks(id),          -- 父子块(small-to-big)：M2 启用
    chunk_index  INT     NOT NULL,                        -- 在文档内的顺序
    content      TEXT    NOT NULL,
    token_count  INT,
    -- 维度与 embedding 模型绑定；换模型需重建（M1: bge-small-zh=512）
    embedding    vector(512),
    metadata     JSONB   NOT NULL DEFAULT '{}'::jsonb,    -- 页码/标题层级等溯源信息
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- 向量近邻索引：HNSW（召回质量优于 IVFFlat，构建慢但查询快，适合本项目规模）
-- 用 cosine 距离，与 bge 归一化向量匹配。
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- M9b 记忆表：个人记忆(scope=personal, owner 私有) 与 知识记忆(scope=knowledge, 共享)
-- 分置避免“把私有偏好当共享知识”；双时态(valid_at/invalid_at)+trust 生命周期(draft/verified/curated)。
CREATE TABLE IF NOT EXISTS memories (
    id            BIGSERIAL PRIMARY KEY,
    scope         TEXT        NOT NULL,          -- personal | knowledge
    tenant_id     TEXT,                          -- ⑥ 租户隔离（数据面强制）
    owner_user_id TEXT,                          -- personal 必填；knowledge 为 NULL
    kind          TEXT        NOT NULL DEFAULT 'fact',  -- fact/preference/entity/claim
    level         TEXT        NOT NULL DEFAULT 'fact',   -- P3-A: fact(具体) | playbook(可迁移打法)
    tags          JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- P3-A: type/location/season/difficulty 等
    content       TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,          -- 幂等去重
    source_type   TEXT        NOT NULL DEFAULT 'research',  -- kb/web/research/user
    source_ref    TEXT,                          -- URL / doc source
    trust         TEXT        NOT NULL DEFAULT 'draft',      -- draft/verified/curated
    confidence    REAL        NOT NULL DEFAULT 0.5,
    support       INT         NOT NULL DEFAULT 1,            -- 佐证来源数(多源佐证→促升)
    embedding     vector(512),
    valid_at      TIMESTAMPTZ,                   -- 何时起为真(NULL=立即)
    invalid_at    TIMESTAMPTZ,                   -- 何时失效(NULL=有效中)
    is_deleted    BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memories_scope_owner ON memories(scope, owner_user_id);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops);

-- 幂等：为已存在的 memories 表补上 tenant_id（⑥ 增量）
ALTER TABLE memories ADD COLUMN IF NOT EXISTS tenant_id TEXT;
-- 幂等：P3-A 经验分层与标签
ALTER TABLE memories ADD COLUMN IF NOT EXISTS level TEXT NOT NULL DEFAULT 'fact';
ALTER TABLE memories ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '{}'::jsonb;

-- P3-B 事件/复盘实体：把一次活动的方案/决策/踩坑归成整份可检索可引用的经验档案
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    type        TEXT,                            -- 活动/年会/策划/复盘/事故...
    tags        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    summary     TEXT,
    created_by  TEXT,
    occurred_at TIMESTAMPTZ,
    is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS event_id BIGINT REFERENCES events(id);
