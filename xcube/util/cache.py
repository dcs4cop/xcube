# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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


import os
import os.path
import sys
import time
from abc import ABCMeta, abstractmethod
from threading import RLock

from typing import Optional

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

from xcube.constants import LOG

_DEBUG_CACHE = False


class CacheStore(metaclass=ABCMeta):
    """Represents a store to which cached values can be stored
    into and restored from.
    """

    @abstractmethod
    def can_load_from_key(self, key) -> bool:
        """Test whether a stored value representation can be loaded
        from the given key.

        Args:
            key: the key
        Returns:
            True, if so
        """
        pass

    @abstractmethod
    def load_from_key(self, key):
        """Load a stored value representation of the value and its
        size from the given key.

        Args:
            key: the key
        Returns:
            a 2-element sequence containing the stored representation
            of the value and it's size
        """
        pass

    @abstractmethod
    def store_value(self, key, value):
        """Store a value and return it's stored representation and size in
        any unit, e.g. in bytes.

        Args:
            key: the key
            value: the value
        Returns:
            a 2-element sequence containing the stored representation
            of the value and it's size
        """
        pass

    @abstractmethod
    def restore_value(self, key, stored_value):
        """Restore a vale from its stored representation.

        Args:
            key: the key
            stored_value: the stored representation of the value
        Returns:
            the item
        """
        pass

    @abstractmethod
    def discard_value(self, key, stored_value):
        """Discard a value from it's storage.

        Args:
            key: the key
            stored_value: the stored representation of the value
        """
        pass


class MemoryCacheStore(CacheStore):
    """Simple memory store."""

    def can_load_from_key(self, key) -> bool:
        # This store type does not maintain key-value pairs on its own
        return False

    def load_from_key(self, key):
        raise NotImplementedError()

    def store_value(self, key, value):
        """Return (value, 1).

        Args:
            key: the key
            value: the original value

        Returns:
            the tuple (stored value, size) where stored value is the
            sequence [key, value].
        """
        return [key, value], _compute_object_size(value)

    def restore_value(self, key, stored_value):
        """Args:
            key: the key
            stored_value: the stored representation of the value

        Returns:
            the original value.
        """
        if key != stored_value[0]:
            raise ValueError("key does not match stored value")
        return stored_value[1]

    def discard_value(self, key, stored_value):
        """Clears the value in the given stored_value.

        Args:
            key: the key
            stored_value: the stored representation of the value
        """
        if key != stored_value[0]:
            raise ValueError("key does not match stored value")
        stored_value[1] = None


class FileCacheStore(CacheStore):
    """Simple file store for values which can be written
    and read as bytes, e.g. encoded PNG images."""

    def __init__(self, cache_dir: str, ext: str):
        self.cache_dir = cache_dir
        self.ext = ext

    def can_load_from_key(self, key) -> bool:
        path = self._key_to_path(key)
        return os.path.exists(path)

    def load_from_key(self, key):
        path = self._key_to_path(key)
        return path, os.path.getsize(path)

    def store_value(self, key, value):
        path = self._key_to_path(key)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as fp:
            fp.write(value)
        return path, os.path.getsize(path)

    def restore_value(self, key, stored_value):
        path = self._key_to_path(key)
        with open(path, "rb") as fp:
            return fp.read()

    def discard_value(self, key, stored_value):
        path = self._key_to_path(key)
        try:
            os.remove(path)
            # TODO (forman): also remove empty directories up to self.cache_dir
        except IOError:
            pass

    def _key_to_path(self, key):
        return os.path.join(self.cache_dir, str(key) + self.ext)


def _policy_lru(item):
    return item.access_time


def _policy_mru(item):
    return -item.access_time


def _policy_lfu(item):
    return item.access_count


def _policy_rr(item):
    return item.access_count % 2


#: Discard Least Recently Used items first
POLICY_LRU = _policy_lru
#: Discard Most Recently Used first
POLICY_MRU = _policy_mru
#: Discard Least Frequently Used first
POLICY_LFU = _policy_lfu
#: Discard items by Random Replacement
POLICY_RR = _policy_rr

_T0 = time.process_time()


class Cache:
    """An implementation of a cache.
    See https://en.wikipedia.org/wiki/Cache_algorithms
    """

    class Item:
        """Cache-private class representing an item in the cache."""

        def __init__(self):
            self.key = None
            self.stored_value = None
            self.stored_size = 0
            self.creation_time = 0
            self.access_time = 0
            self.access_count = 0

        @staticmethod
        def load_from_key(store, key):
            if not store.can_load_from_key(key):
                return None
            item = Cache.Item()
            item._load_from_key(store, key)
            return item

        def store(self, store, key, value):
            self.key = key
            self.access_count = 0
            self._access()
            stored_value, stored_size = store.store_value(key, value)
            self.stored_value = stored_value
            self.stored_size = stored_size

        def restore(self, store, key):
            self._access()
            return store.restore_value(key, self.stored_value)

        def discard(self, store, key):
            store.discard_value(key, self.stored_value)
            self.__init__()

        def _load_from_key(self, store, key):
            self.key = key
            self.access_count = 0
            self._access()
            stored_value, stored_size = store.load_from_key(key)
            self.stored_value = stored_value
            self.stored_size = stored_size

        def _access(self):
            self.access_time = time.process_time() - _T0
            self.access_count += 1

    def __init__(
        self,
        store=MemoryCacheStore(),
        capacity=1000,
        threshold=0.75,
        policy=POLICY_LRU,
        parent_cache=None,
    ):
        """Constructor.

        Args:
            store: the cache store, see CacheStore interface
            capacity: the size capacity in units used by the
                store's store() method
            threshold: a number greater than zero and less than one
            policy: cache replacement policy. This is a function
                that maps a `Cache.Item` to a numerical value.
                See `POLICY_LRU`, `POLICY_MRU`, `POLICY_LFU`, `POLICY_RR`
        """
        self._store = store
        self._capacity = capacity
        self._threshold = threshold
        self._policy = policy
        self._parent_cache = parent_cache
        self._size = 0
        self._max_size = self._capacity * self._threshold
        self._item_dict = {}
        self._item_list = []
        self._lock = RLock()

    @property
    def policy(self):
        return self._policy

    @property
    def store(self):
        return self._store

    @property
    def capacity(self):
        return self._capacity

    @property
    def threshold(self):
        return self._threshold

    @property
    def size(self):
        return self._size

    @property
    def max_size(self):
        return self._max_size

    def get_value(self, key):
        self._lock.acquire()
        item = self._item_dict.get(key)
        value = None
        restored = False
        if item:
            value = item.restore(self._store, key)
            restored = True
            if _DEBUG_CACHE:
                _debug_print('restored value for key "%s" from cache' % key)
        elif self._parent_cache:
            item = self._parent_cache.get_value(key)
            if item:
                value = item.restore(self._parent_cache.store, key)
                restored = True
                if _DEBUG_CACHE:
                    _debug_print('restored value for key "%s" from parent cache' % key)
        if not restored:
            item = Cache.Item.load_from_key(self._store, key)
            if item:
                self._add_item(item)
                value = item.restore(self._store, key)
                if _DEBUG_CACHE:
                    _debug_print('restored value for key "%s" from cache' % key)
        self._lock.release()
        return value

    def put_value(self, key, value):
        self._lock.acquire()
        if self._parent_cache:
            # remove value from parent cache, because this cache will now take over
            self._parent_cache.remove_value(key)
        item = self._item_dict.get(key)
        if item:
            self._remove_item(item)
            item.discard(self._store, key)
            if _DEBUG_CACHE:
                _debug_print('discarded value for key "%s" from cache' % key)
        else:
            item = Cache.Item()
        item.store(self._store, key, value)
        if _DEBUG_CACHE:
            _debug_print('stored value for key "%s" in cache' % key)
        self._add_item(item)
        self._lock.release()

    def remove_value(self, key):
        self._lock.acquire()
        if self._parent_cache:
            self._parent_cache.remove_value(key)
        item = self._item_dict.get(key)
        if item:
            self._remove_item(item)
            item.discard(self._store, key)
            if _DEBUG_CACHE:
                _debug_print(
                    'Cache: discarded value for key "%s" from parent cache' % key
                )
        self._lock.release()

    def _add_item(self, item):
        self._item_dict[item.key] = item
        self._item_list.append(item)
        if self._size + item.stored_size > self._max_size:
            self.trim(item.stored_size)
        self._size += item.stored_size

    def _remove_item(self, item):
        self._item_dict.pop(item.key)
        self._item_list.remove(item)
        self._size -= item.stored_size

    def trim(self, extra_size=0):
        if _DEBUG_CACHE:
            _debug_print("trimming...")
        self._lock.acquire()
        self._item_list.sort(key=self._policy)
        keys = []
        size = self._size
        max_size = self._max_size
        for item in self._item_list:
            if size + extra_size > max_size:
                keys.append(item.key)
                size -= item.stored_size
        self._lock.release()
        # release lock to give another thread a chance then require lock again
        self._lock.acquire()
        for key in keys:
            if self._parent_cache:
                # Before discarding item fully, put its value into the parent cache
                value = self.get_value(key)
                self.remove_value(key)
                if value:
                    self._parent_cache.put_value(key, value)
            else:
                self.remove_value(key)
        self._lock.release()

    def clear(self, clear_parent=True):
        self._lock.acquire()
        if self._parent_cache and clear_parent:
            self._parent_cache.clear(clear_parent)
        keys = list(self._item_dict.keys())
        self._lock.release()
        for key in keys:
            if self._parent_cache and not clear_parent:
                value = self.get_value(key)
                if value:
                    self._parent_cache.put_value(key, value)
            self.remove_value(key)


def _debug_print(msg):
    LOG.debug("Cache:", msg)


def _compute_object_size(obj):
    if hasattr(obj, "nbytes"):
        # A numpy ndarray instance
        return obj.nbytes
    elif hasattr(obj, "size") and hasattr(obj, "mode"):
        # A PIL Image instance
        w, h = obj.size
        m = obj.mode
        return (
            w
            * h
            * (
                4
                if m in ("RGBA", "RGBx", "I", "F")
                else (
                    3
                    if m in ("RGB", "YCbCr", "LAB", "HSV")
                    else 1.0 / 8.0 if m == "1" else 1
                )
            )
        )
    else:
        return sys.getsizeof(obj)


def parse_mem_size(mem_size_text: str) -> Optional[int]:
    mem_size_text = mem_size_text.upper()
    if mem_size_text != "" and mem_size_text not in ("OFF", "NONE", "NULL", "FALSE"):
        unit = mem_size_text[-1]
        factors = {"B": 10**0, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
        try:
            if unit in factors:
                capacity = int(mem_size_text[0:-1]) * factors[unit]
            else:
                capacity = int(mem_size_text)
        except ValueError:
            raise ValueError(f"invalid memory size: {mem_size_text!r}")
        if capacity > 0:
            return capacity
        elif capacity < 0:
            raise ValueError(f"negative memory size: {mem_size_text!r}")
    return None
