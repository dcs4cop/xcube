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

import urllib3.response

from ..helpers import RoutesTestCase


class S3RoutesTest(RoutesTestCase):

    def test_fetch_list_s3bucket(self):
        response = self.fetch('/s3bucket')
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket?delimiter=%2F')
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket?delimiter=%2F&prefix=demo%2F')
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket?delimiter=%2F&list-type=2')
        self.assertResponseOK(response)
        response = self.fetch(
            '/s3bucket?delimiter=%2F&prefix=demo%2F&list-type=2')
        self.assertResponseOK(response)

    def test_fetch_head_s3bucket_object(self):
        self._assert_fetch_head_s3bucket_object(method='HEAD')

    def test_fetch_get_s3bucket_object(self):
        self._assert_fetch_head_s3bucket_object(method='GET')

    def _assert_fetch_head_s3bucket_object(self, method):
        response = self.fetch('/s3bucket/demo', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/.zattrs', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/.zgroup', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/.zarray', method=method)
        self.assertResourceNotFoundResponse(response)
        response = self.fetch('/s3bucket/demo/time/.zattrs', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/time/.zarray', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/time/.zgroup', method=method)
        self.assertResourceNotFoundResponse(response)
        response = self.fetch('/s3bucket/demo/time/0', method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/conc_chl/.zattrs',
                              method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/conc_chl/.zarray',
                              method=method)
        self.assertResponseOK(response)
        response = self.fetch('/s3bucket/demo/conc_chl/.zgroup',
                              method=method)
        self.assertResourceNotFoundResponse(response)
        response = self.fetch('/s3bucket/demo/conc_chl/3.2.4', method=method)
        self.assertResponseOK(response)