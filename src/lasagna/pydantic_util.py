from .types import (
    ExtractionType,
)

from .util import get_name

from typing_extensions import is_typeddict

from typing import (
    Set, Dict, Any, Type,
    get_type_hints, cast,
)

from pydantic import BaseModel, Field, create_model


def create_pydantic_model_from_typeddict(
    typeddict_cls: Type[ExtractionType],
) -> Type[BaseModel]:
    annotations = get_type_hints(typeddict_cls)
    required_keys: Set = getattr(typeddict_cls, '__required_keys__', set())
    fields: Dict[str, Any] = {}
    for field_name, field_type in annotations.items():
        if field_name in required_keys:
            fields[field_name] = (field_type, Field(...))
        else:
            fields[field_name] = (field_type, Field(default=None))
    model_name = get_name(typeddict_cls)
    model: Type[BaseModel] = create_model(model_name, **fields)
    return model


def ensure_pydantic_model(
    extraction_type: Type[ExtractionType],
) -> Type[BaseModel]:
    if issubclass(extraction_type, BaseModel):
        return extraction_type
    elif is_typeddict(extraction_type):
        return create_pydantic_model_from_typeddict(extraction_type)
    else:
        raise ValueError(f'Cannot handle parsing data for type: {extraction_type}')


def build_and_validate(
    extraction_type: Type[ExtractionType],
    data: Any,
) -> ExtractionType:
    if issubclass(extraction_type, BaseModel):
        return cast(ExtractionType, extraction_type.model_validate(data))
    elif is_typeddict(extraction_type):
        model = create_pydantic_model_from_typeddict(extraction_type)
        obj = model.model_validate(data)
        return cast(ExtractionType, obj.model_dump())
    else:
        raise ValueError(f'Cannot handle parsing data for type: {extraction_type}')
