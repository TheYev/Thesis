from sqlalchemy import create_enginen
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


POSTGRESQL_URL = "postgresql://user:password@localhost:5432/"

engine = create_enginen(POSTGRESQL_URL)

session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()
