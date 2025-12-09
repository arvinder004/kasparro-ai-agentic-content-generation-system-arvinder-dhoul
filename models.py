from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# Input
class ProductData(BaseModel):
    name: str
    concentration: Optional[str] = None
    skin_type: List[str]
    ingredients: List[str]
    benefits: List[str]
    how_to_use: str
    side_effects: Optional[str] = None
    price: float


#Intermediate
class UserQuestion(BaseModel):
    category: Literal['Informational', 'Safety', 'Usage', 'Purchase', 'Comparison']
    question_text: str
    answer_text: Optional[str] = None

class CompetitorProduct(BaseModel):
    name: str
    key_ingredients: List[str]
    benefits: List[str]
    price: float

class AgentState(BaseModel):
    primary_product: ProductData
    competitor_product: Optional[CompetitorProduct] = None
    generated_questions: List[UserQuestion] = []


# Output
class PageSection(BaseModel):
    heading: str
    content: str | List[Dict]

class PageOutput(BaseModel):
    page_type: str
    sections: List[PageSection]
    meta_tags: List[str]
