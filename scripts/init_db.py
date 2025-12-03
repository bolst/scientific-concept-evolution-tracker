import os, sys

# Add the src directory to the python path so we can import scet
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from scet.models import Base
from scet.db import engine

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    init_db()
