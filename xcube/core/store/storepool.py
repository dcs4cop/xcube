# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os.path
from typing import Any, Dict, Optional, List, Union, Mapping

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from .assertions import assert_valid_config
from .error import DataStoreError
from .store import DataStore
from .store import new_data_store
from ...util.config import load_json_or_yaml_config


def get_data_store_instance(
    store_id: str,
    store_params: Dict[str, Any] = None,
    store_pool: "DataStorePool" = None,
) -> "DataStoreInstance":
    """Get a data store instance for identifier *store_id*.

    If *store_id* is prefixed by a "@", it is an "instance identifier".
    In this case the store instance is retrieved from
    the expected *store_pool* argument. Otherwise a new store instance
    is created using optional *store_params*.

    Args:
        store_id: Store identifier, may be prefixed by a "@" to indicate
            a store instance identifier.
        store_params: Store parameters, only valid if *store_id* is not
            an instance identifier.
        store_pool: A pool of configured store instances used if
            *store_id* is an instance identifier.

    Returns:
        a DataStoreInstance object
    :raise: DataStoreError if a configured store does not exist
    """
    if store_id.startswith("@"):
        store_instance_id = store_id[1:]
        if store_pool is None:
            raise ValueError(
                f"store_pool must be given,"
                f' with store_id ("{store_id}")'
                f" referring to a configured store"
            )
        if store_params:
            raise ValueError(
                f"store_params cannot be given,"
                f' with store_id ("{store_id}")'
                f" referring to a configured store"
            )
        return store_pool.get_store_instance(store_instance_id)
    return DataStoreInstance(DataStoreConfig(store_id, store_params))


DATA_STORE_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        store_id=JsonStringSchema(min_length=1),
        store_params=JsonObjectSchema(additional_properties=True),
        title=JsonStringSchema(min_length=1),
        description=JsonStringSchema(min_length=1),
        cost_params=JsonObjectSchema(
            properties=dict(
                input_pixels_per_punit=JsonIntegerSchema(minimum=1),
                output_pixels_per_punit=JsonIntegerSchema(minimum=1),
                input_punits_weight=JsonNumberSchema(
                    exclusive_minimum=0.0, default=1.0
                ),
                output_punits_weight=JsonNumberSchema(
                    exclusive_minimum=0.0, default=1.0
                ),
            ),
            additional_properties=False,
            required=["input_pixels_per_punit", "output_pixels_per_punit"],
        ),
    ),
    required=["store_id"],
)

DATA_STORE_POOL_SCHEMA = JsonObjectSchema(
    additional_properties=DATA_STORE_CONFIG_SCHEMA,
)


class DataStoreConfig:
    """The configuration of a data store.
    The class is used by `DataStorePool` to instantiate
    stores in a deferred manner.

    Args:
        store_id: the data store identifier
        store_params: optional store parameters
        title: a human-readable title for the store instance
        description: a human-readable description of the store instance
        user_data: optional user-data
    """

    def __init__(
        self,
        store_id: str,
        store_params: Mapping[str, Any] = None,
        title: str = None,
        description: str = None,
        user_data: Any = None,
    ):
        assert_given(store_id, name="store_id")
        if store_params is not None:
            assert_instance(store_params, dict, name="store_params")
        self._store_id = store_id
        self._store_params = store_params
        self._title = title
        self._description = description
        self._user_data = user_data

    @property
    def store_id(self) -> Optional[str]:
        return self._store_id

    @property
    def store_params(self) -> Optional[Dict[str, Any]]:
        return self._store_params

    @property
    def title(self) -> Optional[str]:
        return self._title

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def user_data(self) -> Optional[Any]:
        return self._user_data

    @classmethod
    def from_dict(cls, data_store_config: Dict[str, Any]) -> "DataStoreConfig":
        assert_valid_config(
            data_store_config, name="data_store_config", schema=DATA_STORE_CONFIG_SCHEMA
        )
        return DataStoreConfig(
            data_store_config["store_id"],
            store_params=data_store_config.get("store_params"),
            title=data_store_config.get("title"),
            description=data_store_config.get("description"),
        )

    def to_dict(self) -> Dict[str, Any]:
        data_store_config = dict(store_id=self._store_id)
        if self._store_params:
            data_store_config.update(store_params=self._store_params)
        if self._title:
            data_store_config.update(name=self._title)
        if self._description:
            data_store_config.update(description=self._description)
        return data_store_config


class DataStoreInstance:
    """Internal class used by DataStorePool to maintain
    store configurations + instances.
    """

    def __init__(self, store_config: DataStoreConfig):
        assert_given(store_config, name="store_config")
        assert_instance(store_config, DataStoreConfig, name="store_config")
        self._store_config = store_config
        self._store: Optional[DataStore] = None

    @property
    def store_config(self) -> DataStoreConfig:
        return self._store_config

    @property
    def store(self) -> DataStore:
        if self._store is None:
            self._store = new_data_store(
                self._store_config.store_id, **(self._store_config.store_params or {})
            )
        return self._store

    def close(self):
        store = self._store
        if store is not None and hasattr(store, "close") and callable(store.close):
            store.close()


DataStoreConfigDict = Dict[str, DataStoreConfig]
DataStoreInstanceDict = Dict[str, DataStoreInstance]

DataStorePoolLike = Union[str, Dict[str, Any], "DataStorePool"]


class DataStorePool:
    """A pool of configured data store instances.

    Actual data store instantiation only takes place lazily.
    A pool is may be created using it `from_dict()` (or `from_file()`)
    which receives a (JSON) dictionary that maps store instance names to
    store configurations:

    ```
        {
            "<store_instance_id>": {
                "store_id": "<store_id>",
                "store_params": {
                    "<param_name>": <param_value>,
                    ...
                },
                "title": "<optional_human_readable_title>",
                "description": "<optional_human_readable_description>",
            },
            ...
        }
    ```

    Args:
        store_configs: A dictionary that maps store instance
            identifiers to store configurations.
    """

    def __init__(self, store_configs: DataStoreConfigDict = None):
        if store_configs is not None:
            assert_instance(store_configs, dict, name="stores_configs")
        else:
            store_configs = {}
        self._instances: DataStoreInstanceDict = {
            k: DataStoreInstance(v) for k, v in store_configs.items()
        }

    @property
    def is_empty(self) -> bool:
        return len(self._instances) == 0

    @property
    def store_instance_ids(self) -> List[str]:
        return sorted([k for k, v in self._instances.items()])

    @property
    def store_configs(self) -> List[DataStoreConfig]:
        return [v.store_config for k, v in self._instances.items()]

    def get_store_instance_id(
        self, store_config: DataStoreConfig, strict_check: bool = False
    ) -> Optional[str]:
        assert_instance(store_config, DataStoreConfig, "store_config")
        for store_instance_id, instance in self._instances.items():
            if strict_check:
                if instance.store_config == store_config:
                    return store_instance_id
            else:
                if (
                    instance.store_config.store_id == store_config.store_id
                    and instance.store_config.store_params == store_config.store_params
                ):
                    return store_instance_id
        return None

    def has_store_config(self, store_config: DataStoreConfig) -> bool:
        return self.get_store_instance_id(store_config) is not None

    def has_store_instance(self, store_instance_id: str) -> bool:
        assert_instance(store_instance_id, str, "store_instance_id")
        return store_instance_id in self._instances

    def add_store_config(self, store_instance_id: str, store_config: DataStoreConfig):
        assert_instance(store_instance_id, str, "store_instance_id")
        assert_instance(store_config, DataStoreConfig, "store_config")
        if store_instance_id in self._instances:
            self._instances[store_instance_id].close()
        self._instances[store_instance_id] = DataStoreInstance(store_config)

    def remove_store_config(self, store_instance_id: str):
        self._assert_valid_instance_id(store_instance_id)
        self._instances[store_instance_id].close()
        del self._instances[store_instance_id]

    def remove_all_store_configs(self):
        self._instances.clear()

    def get_store_config(self, store_instance_id: str) -> DataStoreConfig:
        self._assert_valid_instance_id(store_instance_id)
        return self._instances[store_instance_id].store_config

    def get_store(self, store_instance_id: str) -> DataStore:
        self._assert_valid_instance_id(store_instance_id)
        return self._instances[store_instance_id].store

    def get_store_instance(self, store_instance_id: str) -> DataStoreInstance:
        self._assert_valid_instance_id(store_instance_id)
        return self._instances[store_instance_id]

    def close_all_stores(self):
        for instance in self._instances.values():
            instance.close()

    @classmethod
    def normalize(cls, data_store_pool: DataStorePoolLike) -> "DataStorePool":
        """
        Normalize given *data_store_pool* to an instance of
        `DataStorePool`.

        If *data_store_pool* is already a DataStorePool it is returned as is.
        If it is a `str`, it is interpreted as a YAML or JSON file path
        and the request is read from file using `DataStorePool.from_file()`.
        If it is a `dict`, it is interpreted as a JSON object and the
        request is parsed using `DataStorePool.from_dict()`.

        Args:
            data_store_pool The data store pool instance,
                or data stores configuration file path, or data store pool
                JSON object.
        Raises:
            TypeError: if *data_store_pool* is not
                a `CubeGeneratorRequest`, `str`, or `dict`.
        """
        if isinstance(data_store_pool, DataStorePool):
            return data_store_pool
        if isinstance(data_store_pool, str):
            return DataStorePool.from_file(data_store_pool)
        if isinstance(data_store_pool, dict):
            return DataStorePool.from_dict(data_store_pool)
        raise TypeError(
            "data_store_pool must be a str, dict, " "or a DataStorePool instance"
        )

    @classmethod
    def from_file(cls, path: str) -> "DataStorePool":
        _, ext = os.path.splitext(path)
        store_configs = load_json_or_yaml_config(path)
        return cls.from_dict(store_configs or {})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataStorePool":
        DATA_STORE_POOL_SCHEMA.validate_instance(d)
        return cls({k: DataStoreConfig.from_dict(v) for k, v in d.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {
            instance_id: instance.store_config.to_dict()
            for instance_id, instance in self._instances.items()
        }

    def _assert_valid_instance_id(self, store_instance_id: str):
        assert_instance(store_instance_id, str, name="store_instance_id")
        if store_instance_id not in self._instances:
            raise DataStoreError(
                f"Configured data store instance" f' "{store_instance_id}" not found.'
            )
