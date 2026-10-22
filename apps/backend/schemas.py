from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class UserCreate(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=100)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    display_name: str


class PreferenceCreate(BaseModel):
    preference_key: str = Field(min_length=1, max_length=100)
    preference_value: str = Field(min_length=1, max_length=5000)
    is_sensitive: bool = False


class PreferenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    preference_key: str
    preference_value: str
    is_sensitive: bool

class ServiceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    description: str
    active: bool

class RankedServiceRead(ServiceRead):
    relevance_score: float





class NegotiationCreate(BaseModel):
    service_id: int = Field(gt=0)
    budget_limit: float = Field(gt=0, allow_inf_nan=False)
    max_offers: int = Field(default=3, ge=1, le=10)


class NegotiationOfferRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_role: str
    amount: float
    message: str


class NegotiationRead(BaseModel):
    id: int
    service_id: int
    status: Literal[
        "PENDING",
        "NEGOTIATING",
        "ACCEPTED",
        "REJECTED",
        "FAILED",
    ]
    budget_limit: float
    selected_offer_id: int | None = None
    offers: list[NegotiationOfferRead] = Field(default_factory=list)

class NegotiationDecision(BaseModel):
    status: Literal["ACCEPTED", "REJECTED"]
    selected_offer_id: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_selected_offer(self):
        if self.status == "ACCEPTED" and self.selected_offer_id is None:
            raise ValueError("An accepted negotiation requires an offer ID")

        if self.status == "REJECTED" and self.selected_offer_id is not None:
            raise ValueError("A rejected negotiation cannot select an offer")

        return self
