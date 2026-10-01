"""config / store 的纯逻辑单元测试（不连数据库、不发网络）。"""
from app.config import Settings
from app.retrieval.store import compute_doc_hash


def test_sqlalchemy_url_composition():
    s = Settings(
        pg_user="u", pg_password="p", pg_host="h", pg_port=1234, pg_database="d"
    )
    assert s.sqlalchemy_url == "postgresql+psycopg://u:p@h:1234/d"


def test_defaults_are_standalone_rag():
    s = Settings()
    assert s.mimo_model.startswith("mimo-v2.6")
    assert s.embedding_model_name.startswith("BAAI/")
    assert s.embedding_dim == 512


def test_doc_hash_is_deterministic_and_content_sensitive():
    h1 = compute_doc_hash("hello 世界")
    h2 = compute_doc_hash("hello 世界")
    h3 = compute_doc_hash("hello 世界!")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex
