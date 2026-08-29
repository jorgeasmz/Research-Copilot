"""Engine and session factory, shared by the pipeline and the API."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ingest import config

engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
