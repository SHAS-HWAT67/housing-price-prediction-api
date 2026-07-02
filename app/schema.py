from pydantic import BaseModel

class House(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int