from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from db.database import Base


class FactModel(Base):
    __tablename__ = "facts"

    id = Column(Integer, primary_key=True, index=True)
    material = Column(String, index=True)
    material_lower = Column(String, index=True)
    process = Column(String, index=True)
    process_lower = Column(String, index=True)
    geography = Column(String)
    year = Column(Integer, nullable=True)
    outcome = Column(String)
    outcome_lower = Column(String)
    confidence_level = Column(String)
    source_file = Column(String)

    parameters = relationship("ParameterModel", back_populates="fact", cascade="all, delete-orphan")


class ParameterModel(Base):
    __tablename__ = "parameters"

    id = Column(Integer, primary_key=True, index=True)
    fact_id = Column(Integer, ForeignKey("facts.id"))
    name = Column(String, index=True)
    name_lower = Column(String, index=True)
    value_min = Column(Float, nullable=True)
    value_max = Column(Float, nullable=True)
    unit = Column(String, nullable=True)

    fact = relationship("FactModel", back_populates="parameters")
