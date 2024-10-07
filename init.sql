CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items_test(
    id serial PRIMARY KEY,
    description text,
    embedding vector(384)
);
