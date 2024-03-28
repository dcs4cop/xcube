# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections.abc
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Mapping, Sequence, Union, Tuple, Optional

from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true

# Make sure rfc3339-validator package is installed : jsonschema uses it for
# validating instances of JsonDateSchema and JsonDatetimeSchema.
# Use of __import__ avoids "unused package" warnings.
__import__("rfc3339_validator")

import jsonschema

from xcube.util.ipython import register_json_formatter
from xcube.util.undefined import UNDEFINED

Factory = Callable[..., Any]
Serializer = Callable[..., Any]

_TYPES_ENUM = {"null", "boolean", "integer", "number", "string", "array", "object"}
_NUMERIC_TYPES_ENUM = {"integer", "number"}


class JsonSchema(ABC):
    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        type: Union[str, Sequence[str]] = None,
        default: Any = UNDEFINED,
        const: Any = UNDEFINED,
        enum: Sequence[Any] = None,
        nullable: bool = None,
        title: str = None,
        description: str = None,
        examples: str = None,
        factory: Factory = None,
        serializer: Serializer = None,
    ):
        if type is not None and type not in _TYPES_ENUM:
            names = ", ".join(map(lambda t: f'"{t}"', sorted(list(_TYPES_ENUM))))
            raise ValueError(f"type must be one of {names}")
        if factory is not None and not callable(factory):
            raise ValueError("factory must be callable")
        if serializer is not None and not callable(serializer):
            raise ValueError("serializer must be callable")
        self.type: Optional[Union[str, Sequence[str]]] = type
        self.default: Any = default
        self.const: Any = const
        self.enum: Optional[Any] = list(enum) if enum is not None else None
        self.nullable: Optional[bool] = nullable
        self.title: Optional[str] = title
        self.description: Optional[str] = description
        self.examples: Optional[str] = examples
        self.factory = factory
        self.serializer = serializer

    def to_dict(self) -> Dict[str, Any]:
        d = dict()
        if self.type is not None:
            if self.nullable is True and self.type != "null":
                d.update(type=[self.type, "null"])
            else:
                d.update(type=self.type)
        elif self.nullable is True:
            d.update(type="null")
        if self.default != UNDEFINED:
            d.update(default=self.default)
        if self.const != UNDEFINED:
            d.update(const=self.const)
        if self.enum is not None:
            d.update(enum=self.enum)
        if self.title is not None:
            d.update(title=self.title)
        if self.description is not None:
            d.update(description=self.description)
        if self.examples is not None:
            d.update(examples=self.examples)
        return d

    def validate_instance(self, instance: Any):
        """Validate JSON value *instance*."""
        # As a base for our custom validator, we hard-code a validator
        # version; JSON metaschema versions are not guaranteed to be backward
        # compatible, so if we use jsonschema.validators.validator_for(
        # schema), our schema may become invalid when the default validator
        # changes. The metaschema version can also be pinned in the schema
        # itself with an entry like this:
        # '$schema': 'http://json-schema.org/draft-07/schema'
        # However, this makes it more work to keep unit tests up to date, and
        # it's fiddly to adapt the recursive to_dict code to make sure that
        # the metaschema key *only* appears in the outermost dictionary.
        base_validator = jsonschema.validators.Draft7Validator

        # By default, jsonschema only recognizes lists as arrays. Here we derive
        # and use a custom validator which recognizes both lists and tuples as
        # arrays.
        new_type_checker = base_validator.TYPE_CHECKER.redefine(
            "array", lambda checker, inst: isinstance(inst, (list, tuple))
        )
        custom_validator = jsonschema.validators.extend(
            base_validator, type_checker=new_type_checker
        )

        # jsconschema needs extra packages installed to validate some formats;
        # if they are missing, the format check will be skipped silently. For
        # date-time format, strict_rfc3339 or rfc3339-validator is required.
        jsonschema.validate(
            instance=instance,
            schema=self.to_dict(),
            cls=custom_validator,
            format_checker=jsonschema.draft7_format_checker,
        )

    def to_instance(self, value: Any) -> Any:
        """Convert Python object *value* into JSON value and return the validated result."""
        # TODO: support anyOf
        json_instance = self._to_unvalidated_instance(value)
        self.validate_instance(json_instance)
        return json_instance

    def from_instance(self, instance: Any) -> Any:
        """Validate JSON value *instance* and convert it into a Python object."""
        # TODO: support anyOf
        self.validate_instance(instance)
        return self._from_validated_instance(instance)

    @abstractmethod
    def _to_unvalidated_instance(self, value: Any) -> Any:
        """Turn Python object *value* into an unvalidated JSON value."""

    @abstractmethod
    def _from_validated_instance(self, instance: Any) -> Any:
        """Turn validated JSON value *instance* into a Python object."""


class JsonComplexSchema(JsonSchema):
    # TODO: implement JsonComplexTypeSchema more completely
    # For full support one_of, any_of, all_of should also be handled in from_instance and to_instance.

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        one_of: Sequence["JsonSchema"] = None,
        any_of: Sequence["JsonSchema"] = None,
        all_of: Sequence["JsonSchema"] = None,
        **kwargs,
    ):
        if len([x for x in (one_of, any_of, all_of) if bool(x)]) != 1:
            raise ValueError("exactly one of one_of, any_of, all_of must be given")
        super().__init__(**kwargs)
        self.one_of = one_of
        self.any_of = any_of
        self.all_of = all_of

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.one_of:
            d.update(oneOf=[schema.to_dict() for schema in self.one_of])
        if self.any_of:
            d.update(anyOf=[schema.to_dict() for schema in self.any_of])
        if self.all_of:
            d.update(allOf=[schema.to_dict() for schema in self.all_of])
        return d

    def _to_unvalidated_instance(self, value: Any) -> Any:
        return self.serializer(value) if self.serializer is not None else value

    def _from_validated_instance(self, instance: Any) -> Any:
        return self.factory(instance) if self.factory is not None else instance


class JsonSimpleSchema(JsonSchema):
    # noinspection PyShadowingBuiltins
    def __init__(self, type: str, **kwargs):
        super().__init__(type, **kwargs)
        if not isinstance(type, str):
            raise ValueError(f"illegal type: {type}")

    def _to_unvalidated_instance(self, value: Any) -> Any:
        return self.serializer(value) if self.serializer is not None else value

    def _from_validated_instance(self, instance: Any) -> Any:
        return self.factory(instance) if self.factory is not None else instance


class JsonNullSchema(JsonSimpleSchema):
    def __init__(self, **kwargs):
        super().__init__(type="null", **kwargs)


class JsonBooleanSchema(JsonSimpleSchema):
    def __init__(self, **kwargs):
        super().__init__(type="boolean", **kwargs)


class JsonStringSchema(JsonSimpleSchema):
    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        format: str = None,
        pattern: str = None,
        min_length: int = None,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__(type="string", **kwargs)
        self.format = format
        self.pattern = pattern
        self.min_length = min_length
        self.max_length = max_length

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.format is not None:
            d.update(format=self.format)
        if self.pattern is not None:
            d.update(pattern=self.pattern)
        if self.min_length is not None:
            d.update(minLength=self.min_length)
        if self.max_length is not None:
            d.update(maxLength=self.max_length)
        return d


class JsonDateAndTimeSchemaBase:
    # noinspection PyShadowingBuiltins
    @classmethod
    def _validate_value(cls, value: Optional[str], name: str, format: str):
        if value is not None and not cls._is_valid_value(value, format):
            raise ValueError(f'{name} must be formatted as a "{format}"')

    # noinspection PyShadowingBuiltins
    @classmethod
    def _is_valid_value(cls, value: str, format: str) -> bool:
        try:
            jsonschema.validate(
                value,
                dict(type="string", format=format),
                format_checker=jsonschema.draft7_format_checker,
            )
            return True
        except jsonschema.ValidationError:
            return False


class JsonDateSchema(JsonStringSchema, JsonDateAndTimeSchemaBase):
    """JSON schema for date instances.

    Args:
        min_date: optional minimum date.
        max_date: optional maximum date.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, min_date: str = None, max_date: str = None, **kwargs):
        super().__init__(**kwargs, format="date")
        self._validate_value(min_date, "min_date", "date")
        self._validate_value(max_date, "max_date", "date")
        self.min_date = min_date
        self.max_date = max_date

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.min_date is not None:
            d.update(minDate=self.min_date)
        if self.max_date is not None:
            d.update(maxDate=self.max_date)
        return d

    @classmethod
    def new_range(
        cls, min_date: str = None, max_date: str = None, nullable: bool = False
    ) -> "JsonArraySchema":
        """Return a schema for a date range.

        Args:
            min_date: optional minimum date.
            max_date: optional maximum date.
            nullable: whether the whole range as well as individual
                start and end may be None.

        Returns:
            a JsonArraySchema with two items.
        """
        return JsonArraySchema(
            items=[
                JsonDateSchema(min_date=min_date, max_date=max_date, nullable=nullable),
                JsonDateSchema(min_date=min_date, max_date=max_date, nullable=nullable),
            ],
            nullable=nullable,
        )


class JsonDatetimeSchema(JsonStringSchema, JsonDateAndTimeSchemaBase):
    """JSON schema for date-time instances.

    Args:
        min_datetime: optional minimum date-time.
        max_datetime: optional maximum date-time.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, min_datetime: str = None, max_datetime: str = None, **kwargs):
        super().__init__(**kwargs, format="date-time")
        self._validate_value(min_datetime, "min_datetime", "date-time")
        self._validate_value(max_datetime, "max_datetime", "date-time")
        self.min_datetime = min_datetime
        self.max_datetime = max_datetime

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.min_datetime is not None:
            d.update(minDatetime=self.min_datetime)
        if self.max_datetime is not None:
            d.update(maxDatetime=self.max_datetime)
        return d

    @classmethod
    def new_range(
        cls, min_datetime: str = None, max_datetime: str = None, nullable: bool = False
    ) -> "JsonArraySchema":
        """Return a schema for a date-time range.

        Args:
            min_datetime: optional minimum date.
            max_datetime: optional maximum date.
            nullable: whether the whole range as well as individual
                start and end dates may be None.

        Returns:
            a JsonArraySchema with two items.
        """
        return JsonArraySchema(
            items=[
                JsonDatetimeSchema(
                    min_datetime=min_datetime,
                    max_datetime=max_datetime,
                    nullable=nullable,
                ),
                JsonDatetimeSchema(
                    min_datetime=min_datetime,
                    max_datetime=max_datetime,
                    nullable=nullable,
                ),
            ],
            nullable=nullable,
        )


class JsonNumberSchema(JsonSimpleSchema):
    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        type: str = "number",
        minimum: Union[int, float] = None,
        exclusive_minimum: Union[int, float] = None,
        maximum: Union[int, float] = None,
        exclusive_maximum: Union[int, float] = None,
        multiple_of: Union[int, float] = None,
        **kwargs,
    ):
        if type not in _NUMERIC_TYPES_ENUM:
            names = ", ".join(sorted(map(lambda t: f'"{t}"', _NUMERIC_TYPES_ENUM)))
            raise ValueError(f"Type must be one of {names}")
        super().__init__(type, **kwargs)
        self.minimum = minimum
        self.exclusive_minimum = exclusive_minimum
        self.maximum = maximum
        self.exclusive_maximum = exclusive_maximum
        self.multiple_of = multiple_of

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.minimum is not None:
            d.update(minimum=self.minimum)
        if self.exclusive_minimum is not None:
            d.update(exclusiveMinimum=self.exclusive_minimum)
        if self.maximum is not None:
            d.update(maximum=self.maximum)
        if self.exclusive_maximum is not None:
            d.update(exclusiveMaximum=self.exclusive_maximum)
        if self.multiple_of is not None:
            d.update(multipleOf=self.multiple_of)
        return d


class JsonIntegerSchema(JsonNumberSchema):
    def __init__(self, **kwargs):
        super().__init__(type="integer", **kwargs)


class JsonArraySchema(JsonSchema):
    def __init__(
        self,
        items: Union[JsonSchema, Sequence[JsonSchema]] = None,
        min_items: int = None,
        max_items: int = None,
        unique_items: bool = None,
        **kwargs,
    ):
        super().__init__(type="array", **kwargs)
        self.items = items
        self.min_items = min_items
        self.max_items = max_items
        self.unique_items = unique_items

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.items is not None:
            if isinstance(self.items, JsonSchema):
                d.update(items=self.items.to_dict())
            else:
                d.update(items=[item.to_dict() for item in self.items])
        if self.min_items is not None:
            d.update(minItems=self.min_items)
        if self.max_items is not None:
            d.update(maxItems=self.max_items)
        if self.unique_items is not None:
            d.update(uniqueItems=self.unique_items)
        return d

    def _to_unvalidated_instance(
        self, value: Optional[Sequence[Any]]
    ) -> Optional[Sequence[Any]]:
        if self.serializer:
            return self.serializer(value)
        return self._convert_sequence(value, "_to_unvalidated_instance")

    def _from_validated_instance(self, instance: Sequence[Any]) -> Any:
        obj = self._convert_sequence(instance, "_from_validated_instance")
        return (
            self.factory(obj) if self.factory is not None and obj is not None else obj
        )

    def _convert_sequence(
        self, sequence: Optional[Sequence[Any]], method_name: str
    ) -> Optional[Sequence[Any]]:
        items_schema = self.items
        if sequence is None:
            # Sequence may be null too
            return None
        if isinstance(items_schema, JsonSchema):
            # Sequence turned into list with items_schema applying to all elements
            return [getattr(items_schema, method_name)(item) for item in sequence]
        if items_schema is not None:
            # Sequence turned into tuple with schema for every position
            return [
                getattr(items_schema[item_index], method_name)(sequence[item_index])
                for item_index in range(len(items_schema))
            ]
        # Sequence returned as-is, without schema
        return sequence


class JsonObjectSchema(JsonSchema):
    # TODO: also address property dependencies

    def __init__(
        self,
        properties: Mapping[str, JsonSchema] = None,
        additional_properties: Union[bool, JsonSchema] = None,
        min_properties: int = None,
        max_properties: int = None,
        required: Sequence[str] = None,
        dependencies: Mapping[str, Union[Sequence[str], JsonSchema]] = None,
        **kwargs,
    ):
        super().__init__(type="object", **kwargs)
        self.properties = dict(properties) if properties else dict()
        self.additional_properties = additional_properties
        self.min_properties = min_properties
        self.max_properties = max_properties
        self.required = list(required) if required else []
        self.dependencies = dict(dependencies) if dependencies else None

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.properties is not None:
            d.update(properties={k: v.to_dict() for k, v in self.properties.items()})
        if self.additional_properties is not None:
            d.update(
                additionalProperties=(
                    self.additional_properties.to_dict()
                    if isinstance(self.additional_properties, JsonSchema)
                    else self.additional_properties
                )
            )
        if self.min_properties is not None:
            d.update(minProperties=self.min_properties)
        if self.max_properties is not None:
            d.update(maxProperties=self.max_properties)
        if self.required:
            d.update(required=list(self.required))
        if self.dependencies:
            d.update(
                dependencies={
                    k: (v.to_dict() if isinstance(v, JsonSchema) else v)
                    for k, v in self.dependencies.items()
                }
            )
        return d

    # TODO: move away. this is a special-purpose utility
    def process_kwargs_subset(
        self, kwargs: Dict[str, Any], keywords: Sequence[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Utility that helps processing keyword-arguments. in *kwargs*:

        Pop every keyword in *keywords* contained in this object schema's properties
        from *kwargs* and put the keyword and value from *kwargs* into a new dictionary.

        Return a 2-element tuple, first *kwargs* with, second the and *kwargs* without the keywords from *keywords*.

        The original *kwargs* is not touched.

        Returns:
            a tuple of two new keyword-argument dictionaries
        """
        old_kwargs = dict(kwargs)
        new_kwargs = {}
        for k in keywords:
            if k in old_kwargs:
                new_kwargs[k] = old_kwargs.pop(k)
            elif k in self.properties:
                property_schema = self.properties[k]
                if property_schema.const != UNDEFINED:
                    new_kwargs[k] = property_schema.const
                elif property_schema.default != UNDEFINED and k in self.required:
                    if property_schema.default != UNDEFINED:
                        new_kwargs[k] = property_schema.default
        return new_kwargs, old_kwargs

    def _to_unvalidated_instance(self, value: Any) -> Optional[Mapping[str, Any]]:
        if self.serializer is not None:
            return self.serializer(value)
        return self._convert_mapping(value, "_to_unvalidated_instance")

    def _from_validated_instance(self, instance: Optional[Mapping[str, Any]]) -> Any:
        obj = self._convert_mapping(instance, "_from_validated_instance")
        return (
            self.factory(**obj) if self.factory is not None and obj is not None else obj
        )

    def _convert_mapping(
        self, mapping_or_obj: Optional[Mapping[str, Any]], method_name: str
    ) -> Optional[Mapping[str, Any]]:
        # TODO: recognise self.dependencies. if dependency is again a schema, compute effective schema
        #       by merging this and the dependency.

        if mapping_or_obj is None:
            return None

        # Fix for #432: call object's own to_dict() method but beware of infinite recursion
        if (
            hasattr(mapping_or_obj, "to_dict")
            and callable(mapping_or_obj.to_dict)
            # calling JsonSchema.to_dict() is always fine
            and (
                isinstance(mapping_or_obj, JsonSchema)
                # calling JsonObject.to_dict() would cause infinite recursion
                or not isinstance(mapping_or_obj, JsonObject)
            )
        ):
            # noinspection PyBroadException
            try:
                return mapping_or_obj.to_dict()
            except BaseException:
                pass

        required_set = set(self.required) if self.required else set()

        additional_properties = self.additional_properties
        if additional_properties is None:
            # As of JSON Schema, additional_properties defaults to True
            additional_properties = True

        additional_properties_schema = None
        if isinstance(additional_properties, JsonSchema):
            additional_properties_schema = additional_properties

        converted_mapping = dict()

        if isinstance(mapping_or_obj, (collections.abc.Mapping, dict)):
            # An object that is already a Mapping
            mapping = mapping_or_obj
        else:
            # An object that is not a Mapping: convert to mapping.
            mapping = {}
            for property_name in dir(mapping_or_obj):
                property_value = UNDEFINED
                property_ok = False
                if property_name in self.properties:
                    # If property_name is defined in properties, we fully trust it
                    property_value = getattr(mapping_or_obj, property_name)
                    property_ok = True
                elif additional_properties and not property_name.startswith("_"):
                    # If property_name is not defined in properties, but additional
                    # properties are allowed, filter out private variables and callables.
                    property_value = getattr(mapping_or_obj, property_name)
                    property_ok = not callable(property_value)
                property_ok = property_ok and (
                    property_name in required_set or property_value is not None
                )
                if property_ok and property_value != UNDEFINED:
                    mapping[property_name] = property_value

        # recursively convert mapping into converted_mapping according to schema

        # process defined properties
        for property_name, property_schema in self.properties.items():
            if property_name in mapping:
                property_value = mapping[property_name]
                converted_property_value = getattr(property_schema, method_name)(
                    property_value
                )
                converted_mapping[property_name] = converted_property_value
            else:
                property_value = property_schema.default
                if property_value != UNDEFINED:
                    converted_property_value = getattr(property_schema, method_name)(
                        property_value
                    )
                    if (
                        property_name in required_set
                        or converted_property_value is not None
                    ):
                        converted_mapping[property_name] = converted_property_value

        # process additional properties
        if additional_properties:
            for property_name, property_value in mapping.items():
                if (
                    property_name not in converted_mapping
                    and property_value != UNDEFINED
                ):
                    if additional_properties_schema:
                        property_value = getattr(
                            additional_properties_schema, method_name
                        )(property_value)
                    converted_mapping[property_name] = property_value

        return converted_mapping

    @classmethod
    def inject_attrs(cls, obj: object, attrs: Dict[str, Any]):
        for k, v in (attrs or {}).items():
            setattr(obj, k, v)


class JsonObject(ABC):
    """The abstract base class for objects

    * whose instances can be created from a JSON-serializable
      dictionary using their `from_dict()` class method;
    * whose instances can be converted into a JSON-serializable dictionary
      using their `to_dict()` instance method.

    Derived concrete classes must only implement the `get_schema()` class method
    that must return a `JsonObjectSchema`.

    Instances of this class have a JSON representation in Jupyter/IPython Notebooks.
    """

    @classmethod
    @abstractmethod
    def get_schema(cls) -> JsonObjectSchema:
        """Get JSON object schema."""

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "JsonObject":
        """Create instance from JSON-serializable dictionary *value*."""
        return cls.get_schema().from_instance(value)

    def to_dict(self) -> Dict[str, Any]:
        """Create JSON-serializable dictionary representation."""
        return self.get_schema().to_instance(self)

    def _inject_attrs(self, attrs: Dict[str, Any]):
        assert_instance(attrs, dict, name="attrs")
        schema = self.get_schema()
        assert_true(
            isinstance(schema, JsonObjectSchema),
            message="schema must be a JSON object schema",
        )
        all_attrs = {k: None for k in (schema.properties or {}).keys()}
        all_attrs.update(attrs)
        JsonObjectSchema.inject_attrs(self, all_attrs)


register_json_formatter(JsonSchema)
register_json_formatter(JsonObject)
