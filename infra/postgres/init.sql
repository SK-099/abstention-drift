-- runs once on the first start of the Postgres container
CREATE DATABASE langfuse;
\c app
CREATE EXTENSION IF NOT EXISTS vector;
