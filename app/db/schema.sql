-- Bagent RAG —— 目标库 schema（pgvector）
-- 幂等：可重复执行。M1 只做稠密向量检索，但为 M2/M3 预埋 parent_id、
-- metadata、doc_hash + is_deleted（增量与软删）等结构，避免后期大改表。

CREATE EXTENSION IF NOT EXISTS vector;

-- 源文档：一个文件/一条 URL 对应一行
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,           -- 文件路径 / URI
    doc_hash    TEXT        NOT NULL UNIQUE,    -- 内容哈希，用于去重与增量更新
    media_type  TEXT,                            -- pdf / docx / md / html / txt
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
