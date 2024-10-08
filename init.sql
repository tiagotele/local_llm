CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items_test(
    id serial PRIMARY KEY,
    file_name text,
    embedding vector(384)
);
