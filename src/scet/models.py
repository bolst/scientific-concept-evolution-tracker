from sqlalchemy import Column, Integer, String, Text, Date, Enum, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base


Base = declarative_base()


paper_authors = Table(
    'paper_authors',
    Base.metadata,
    Column('paper_id', String, ForeignKey('papers.arxiv_id'), primary_key=True),
    Column('author_id', Integer, ForeignKey('authors.author_id'), primary_key=True),
    Column('rank', Integer)
)


class Paper(Base):
    __tablename__ = 'papers'

    arxiv_id = Column(String, primary_key=True)
    title = Column(Text)
    abstract = Column(Text)
    primary_category = Column(String)
    published_date = Column(Date)
    last_updated_date = Column(Date)
    doi = Column(String)
    license = Column(String)
    processed_status = Column(Enum('pending', 'downloaded', 'parsed', 'embedded', 'failed', name='processing_status', native_enum=False), default='pending')
    
    authors = relationship('Author', secondary=paper_authors, back_populates='papers')
    # citations_as_source = relationship('Citation', foreign_keys='Citation.source_arxiv_id', back_populates='source_paper')
    # citations_as_target = relationship('Citation', foreign_keys='Citation.target_arxiv_id', back_populates='target_paper')


class Author(Base):
    __tablename__ = 'authors'

    author_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    original_string = Column(Text)

    papers = relationship('Paper', secondary=paper_authors, back_populates='authors')


class Citation(Base):
    __tablename__ = 'citations'

    source_arxiv_id = Column(String, ForeignKey('papers.arxiv_id'), primary_key=True)
    target_arxiv_id = Column(String, ForeignKey('papers.arxiv_id'), primary_key=True)
    context = Column(Text)

    # source_paper = relationship('Paper', foreign_keys=[source_arxiv_id], back_populates='citations_as_source')
    # target_paper = relationship('Paper', foreign_keys=[target_arxiv_id], back_populates='citations_as_target')
