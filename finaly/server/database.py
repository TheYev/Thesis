from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


POSTGRESQL_URL = "postgresql://root:secret@localhost:5433/finaly"

engine = create_engine(POSTGRESQL_URL)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()
