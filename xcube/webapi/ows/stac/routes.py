# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

from xcube.server.api import ApiHandler

from .api import api
from .context import StacContext
from .controllers import get_collection
from .controllers import get_collection_item
from .controllers import get_collection_items
from .controllers import get_collections
from .controllers import get_conformance
from .controllers import get_root
from .controllers import search


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog")
class CatalogRootHandler(ApiHandler[StacContext]):
    @api.operation(operation_id="getCatalogRoot",
                   summary="Get the STAC catalog's root.")
    async def get(self):
        return await self.finish(get_root(self.ctx.datasets_ctx,
                                          self.request.reverse_base_url))


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/conformance")
class CatalogConformanceHandler(ApiHandler[StacContext]):
    @api.operation(operation_id="getCatalogConformance",
                   summary="Get the STAC catalog's conformance.")
    async def get(self):
        return await self.finish(get_conformance(
            self.ctx.datasets_ctx
        ))


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections")
class CatalogCollectionsHandler(ApiHandler[StacContext]):
    @api.operation(operation_id="getCatalogCollections",
                   summary="Get the STAC catalog's collections.")
    async def get(self):
        return await self.finish(get_collections(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url
        ))


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections/{collectionId}")
class CatalogCollectionHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id="getCatalogCollection",
                   summary="Get a STAC catalog collection.")
    async def get(self, collectionId: str):
        return await self.finish(get_collection(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url,
            collectionId
        ))


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections/{collectionId}/items")
class CatalogCollectionItemsHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id="getCatalogCollectionItems",
                   summary="Get the items of a STAC catalog collection.")
    async def get(self, collectionId: str):
        return await self.finish(get_collection_items(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url,
            collectionId
        ))


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections/{collectionId}/items/{featureId}")
class CatalogCollectionItemHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id="getCatalogCollectionItem",
                   summary="Get an item of a STAC catalog collection.")
    async def get(self, collectionId: str, featureId: str):
        return await self.finish(get_collection_item(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url,
            collectionId,
            featureId
        ))


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/search")
class CatalogSearchHandler(ApiHandler[StacContext]):

    @api.operation(operation_id="searchCatalogByKeywords",
                   summary="Search the STAC catalog by keywords.")
    async def get(self):
        return await self.finish(search(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url
        ))

    @api.operation(operation_id="searchCatalogByJSON",
                   summary="Search the STAC catalog by JSON request body.")
    async def post(self):
        return await self.finish(search(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url
        ))