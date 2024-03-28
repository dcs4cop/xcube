# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

import importlib
import inspect
import os.path
import sys
import warnings
from typing import Any, Union, Dict, Callable, Optional, Tuple

from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObject
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.temp import new_temp_dir
from .constants import DEFAULT_CALLABLE_NAME
from .constants import DEFAULT_MODULE_NAME
from .constants import TEMP_FILE_PREFIX
from .fileset import FileSet
from ...constants import LOG


class CodeConfig(JsonObject):
    """Code configuration object.

    Instances should always be created using one of the factory methods:

    * `from_callable()`
    * `from_code()`
    * `from_file_set()`

    Args:
        _callable: Optional function or class. Cannot be given if
            *inline_code* or *file_set* are given.
        callable_ref: Optional reference to the callable in the
            *file_set*. Must have form "<module-name>:<callable-name>".
        inline_code: Optional inline Python code string. Cannot be given
            if *_callable* or *file_set* are given.
        file_set: A file set that contains Python modules or packages.
            Must be of type `FileSet`. Cannot be given if
            *_callable* or *inline_code* are given.
        install_required: Whether *file_set* contains Python modules or
            packages that must be installed. Can only be True if
            *file_set* is given.
        callable_params: Optional parameters that are supposed to be
            passed as keyword-arguments to the callable. Stored as part
            of the configuration, but this class does not apply them.
            Must be a dictionary, if given.
    """

    def __init__(
        self,
        _callable: Callable = None,
        callable_ref: str = None,
        callable_params: Dict[str, Any] = None,
        inline_code: str = None,
        file_set: FileSet = None,
        install_required: bool = None,
    ):
        if callable_ref is not None:
            # Will validate callable_ref
            _normalize_callable_ref(callable_ref)
        if _callable is not None:
            assert_true(callable(_callable), "_callable must be a function or class")
        if callable_params is not None:
            assert_instance(callable_params, dict, "callable_params")
        if inline_code is not None:
            assert_instance(inline_code, str, "inline_code")
        if file_set is not None:
            assert_instance(file_set, FileSet, "file_set")
        assert_true(
            sum((1 if v is not None else 0) for v in (_callable, inline_code, file_set))
            == 1,
            "_callable, inline_code, file_set "
            "are mutually exclusive and one must be given",
        )
        assert_true(
            not install_required or file_set is not None,
            "install_required can only be used if file_set is given",
        )
        self._callable = _callable
        self.callable_ref = callable_ref
        self.callable_params = callable_params
        self.inline_code = inline_code
        self.file_set = file_set
        self.install_required = install_required

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema for CodeConfig objects."""
        return JsonObjectSchema(
            properties=dict(
                callable_ref=JsonStringSchema(min_length=1),
                callable_params=JsonObjectSchema(additional_properties=True),
                inline_code=JsonStringSchema(min_length=1),
                file_set=FileSet.get_schema(),
                install_required=JsonBooleanSchema(),
            ),
            additional_properties=False,
            factory=cls,
        )

    @classmethod
    def from_code(
        cls,
        *code: Union[str, Callable],
        callable_name: str = None,
        module_name: str = None,
        callable_params: Dict[str, Any] = None,
    ) -> "CodeConfig":
        """Create a code configuration from the given *code* which may be
        given as one or more plain text strings or callables.

        This will create a configuration that uses an inline
        `code_string` which contains the source code.

        Args:
            *code: The code.
            callable_name: The callable name. If not given, will be
                inferred from first callable. Otherwise, it defaults to
                "process_dataset".
            module_name: The module name. If not given, defaults to
                "user_code".
            callable_params: Optional parameters that are supposed to be
                passed as keyword-arguments to the callable. Stored as
                part of the configuration, but this class does not apply
                them. Must be a dictionary, if given.

        Returns:
            A new code configuration.
        """
        assert_given(code, "code")
        inline_code, callable_ref = _normalize_inline_code(
            code, module_name=module_name, callable_name=callable_name
        )
        return CodeConfig(
            callable_ref=callable_ref,
            inline_code=inline_code,
            callable_params=callable_params,
        )

    @classmethod
    def from_callable(
        cls, _callable: Callable, callable_params: Optional[Dict[str, Any]] = None
    ) -> "CodeConfig":
        """Create a code configuration from the callable *_callable*.

        Note, the resulting code configuration is only
        valid in a local context. It cannot be JSON-serialized.

        To pass such configurations to a service, convert it first
        using the `to_service()` method.

        Args:
            _callable: A function or class
            callable_params: Optional parameters that are supposed to be
                passed as keyword-arguments to the callable. Stored as
                part of the configuration, but this class does not apply
                them. Must be a dictionary, if given.

        Returns:
            A new code configuration.
        """
        assert_given(_callable, "_callable")
        return CodeConfig(_callable=_callable, callable_params=callable_params)

    @classmethod
    def from_file_set(
        cls,
        file_set: Union[FileSet, str, Any],
        callable_ref: str,
        callable_params: Optional[Dict[str, Any]] = None,
        install_required: Optional[bool] = None,
    ) -> "CodeConfig":
        """Create a code configuration from a file set.

        Args:
            file_set: The file set. Can be a path or a `FileSet`
                instance.
            callable_ref: Reference to the callable in the *file_set*,
                must have form "<module-name>:<callable-name>"
            install_required: Whether the *file_set* is a package that
                must be installed.
            callable_params: Optional parameters that are supposed to be
                passed as keyword-arguments to the callable. Stored as
                part of the configuration, but this class does not apply
                them. Must be a dictionary, if given.

        Returns:
            A new code configuration.
        """
        assert_given(file_set, "file_set")
        assert_given(callable_ref, "callable_ref")
        return CodeConfig(
            callable_ref=callable_ref,
            file_set=_normalize_file_set(file_set),
            install_required=install_required,
            callable_params=callable_params,
        )

    @classmethod
    def from_github_archive(
        cls,
        gh_org: str,
        gh_repo: str,
        gh_tag: str,
        gh_release: str,
        callable_ref: str,
        callable_params: Optional[Dict[str, Any]] = None,
        gh_username: Optional[str] = None,
        gh_token: Optional[str] = None,
    ):
        """Create a code configuration from a GitHub archive.

        Args:
            gh_org: GitHub organisation name or username
            gh_repo: GitHub repository name
            gh_tag: GitHub release tag
            gh_release: The name of a GitHub release. It is used to form
                the sub-path into the archive. The sub-path has the form
                "<gh_repo>-<gh_release>".
            callable_ref: Reference to the callable in the *file_set*,
                must have form "<module-name>:<callable-name>"
            callable_params: Optional parameters that are supposed to be
                passed as keyword-arguments to the callable. Stored as
                part of the configuration, but this class does not apply
                them. Must be a dictionary, if given.
            gh_username: Optional GitHub username.
            gh_token: Optional GitHub username.

        Returns:

        """
        assert_given(gh_org, "gh_org")
        assert_given(gh_org, "gh_repo")
        assert_given(gh_org, "gh_tag")
        assert_given(gh_org, "gh_release")
        gh_url = f"https://github.com/{gh_org}/{gh_repo}/archive/{gh_tag}.zip"
        gh_sub_path = f"{gh_repo}-{gh_release}"
        gh_params = None
        if gh_token is not None:
            gh_params = dict(token=gh_token, username=gh_username or gh_org)
        elif gh_username is not None:
            assert_given(gh_token, "gh_token")  # fails always
        return CodeConfig(
            file_set=FileSet(gh_url, sub_path=gh_sub_path, storage_params=gh_params),
            callable_ref=callable_ref,
            callable_params=callable_params,
        )

    def for_service(self) -> "CodeConfig":
        """Convert this code configuration so can be used by the
        generator service.

        The returned code configuration defines either
        inline code or a file set that represents a local
        ZIP archive.
        """
        return _for_service(self)

    def for_local(self) -> "CodeConfig":
        """Convert this code configuration so can be used by the
        local generator. This means, the returned configuration
        can be used to load executable code. i.e.
        `get_callable()` method will return a callable.

        The returned code configuration defines either
        a callable (member _callable) or a file set that represents
        a local directory. This directory may be used without
        package installation (i.e. added to Python's sys.path) or
        used after installation (i.e. by executing
        `python setup.py install`).
        """
        return _for_local(self)

    def get_callable(self) -> Callable:
        """Get the callable specified by this configuration.

        In the common case, this will require importing the callable.

        Returns:
            A callable
        :raise ImportError if the callable can not be imported
        """
        if self._callable is None:
            self.set_callable(self._load_callable())
        return self._callable

    def set_callable(self, func_or_class: Callable):
        """Set the callable that is represented by this configuration.

        Args:
            func_or_class: A callable
        """
        assert_true(callable(func_or_class), f"func_or_class must be callable")
        self._callable = func_or_class

    def _load_callable(self) -> Callable:
        """Load the callable specified by this configuration.

        Returns:
            A callable
        :raise ImportError if the callable can not be imported
        """
        code_config = self.for_local()
        return _load_callable(
            code_config.file_set.path,
            code_config.callable_ref,
            code_config.install_required,
        )


def _for_service(code_config: CodeConfig) -> "CodeConfig":
    if code_config.inline_code is not None:
        # If we have inline code,
        # there is no need to do anything else.
        return code_config

    if code_config.file_set is not None and not code_config.file_set.is_local():
        # For service requests, remote file set will
        # simply turn into JSON. for_local() will download
        # them when request is executed.
        # So we are done here.
        return code_config

    # noinspection PyProtectedMember
    if (
        code_config._callable is not None
        and code_config.inline_code is None
        and code_config.file_set is None
    ):
        # If we have callable that was not loaded
        # from inline string or files set,
        # turn into a local Python module.
        # noinspection PyProtectedMember
        code_config = _callable_to_module(
            code_config._callable, code_config.callable_params
        )

    if code_config.file_set is not None and code_config.file_set.is_local_dir():
        # If we have a file set that is a local
        # directory, turn it into a local ZIP Archive.
        file_set = code_config.file_set.to_local_zip()
        code_config = code_config.from_file_set(
            file_set=file_set,
            callable_ref=code_config.callable_ref,
            callable_params=code_config.callable_params,
            install_required=code_config.install_required,
        )

    if code_config.file_set is not None and code_config.file_set.is_local_zip():
        # Local ZIP archives will become file-attachments
        # to HTTP service requests. In the request's JSON we will
        # still have the local path.
        # So we are done here.
        return code_config

    if code_config.file_set is not None and code_config.file_set.is_local_zip():
        # At this point, code_config.file_set
        # is always a local ZIP archive.
        return code_config

    raise RuntimeError("for_service() failed due to an " "invalid CodeConfig state")


def _for_local(code_config: CodeConfig) -> "CodeConfig":
    # noinspection PyProtectedMember
    if code_config._callable is not None:
        # If the callable is already defined,
        # there is no need to do anything else.
        return code_config

    if code_config.inline_code:
        # Turn inline code into a local Python module.
        code_config = _inline_code_to_module(
            code_config.inline_code,
            code_config.callable_ref,
            code_config.callable_params,
        )
    file_set = code_config.file_set
    if file_set is None:
        raise RuntimeError(
            "CodeConfig.for_local() failed" " due to an invalid internal state"
        )

    if file_set.is_local_dir() or (file_set.is_local_zip() and not file_set.sub_path):
        # Either a directory or Zip file that can be added
        # to sys.path as is
        return code_config

    # If the file set is not a local directory,
    # turn it into one.
    return code_config.from_file_set(
        file_set=file_set.to_local_dir(),
        callable_ref=code_config.callable_ref,
        install_required=code_config.install_required,
        callable_params=code_config.callable_params,
    )


def _load_callable(
    dir_path: str, callable_ref: str, install_required: bool
) -> Callable:
    """Load the callable from given *dir_path* using the
    callable reference *callable_red*.

    Args:
        dir_path: A local directory path (local ZIPs should work too).
        callable_ref: A callable reference of form "module:callable"
        install_required: Whether source code in *dir_path* must be
            installed using setup.

    Returns:
        The loaded callable
    """
    module_name, callable_name = _normalize_callable_ref(callable_ref)
    if install_required:
        warnings.warn(
            f"This user-code configuration requires "
            f"package installation, "
            f"but this is not supported yet"
        )
    if dir_path not in sys.path:
        # Ok, we need to change global state here.
        # This cannot be done temporarily (e.g., by using a context manager)
        # as additional code might be dynamically loaded later
        # from local imports.
        LOG.warn(f"Python sys.path prepended by {dir_path}")
        sys.path = [dir_path] + sys.path
    # Now we should be able to import the module...
    # (import_module() will raise an ImportError otherwise).
    module = importlib.import_module(module_name)
    # ...and find the callable.
    _callable = getattr(module, callable_name, None)
    if _callable is None:
        raise ImportError(
            f"callable {callable_name!r}" f" not found in module {module_name!r}"
        )
    if not callable(_callable):
        raise ImportError(
            f"{callable_name} of module {callable_name}" f" is not callable"
        )
    LOG.info(f"Imported {callable_ref!r} from {dir_path}")
    return _callable


_user_module_counter = 0


def _next_user_module_name() -> str:
    global _user_module_counter
    _user_module_counter += 1
    return f"{DEFAULT_MODULE_NAME}_{_user_module_counter}"


def _callable_to_module(
    _callable: Callable, callable_params: Dict[str, Any] = None
) -> CodeConfig:
    """Create a code configuration from the callable *_callable*.

    This will create a configuration that uses a `file_set`
    which contains the source code for the *func_or_class*.

    Args:
        _callable: A function or class
        callable_params: The parameters passed as keyword-arguments to
            the callable.

    Returns:
        A new code configuration.
    """
    callable_name = _callable.__name__
    if not callable_name:
        raise ValueError(f"cannot detect name " f"for func_or_class")

    module = inspect.getmodule(_callable)
    module_name = module.__name__ if module is not None else None
    if not module_name:
        raise ValueError(f"cannot detect module " f"for callable {callable_name!r}")

    source_file = inspect.getabsfile(_callable)
    if source_file is None:
        raise ValueError(
            f"cannot detect source file " f"for func_or_class {callable_name!r}"
        )

    module_path, ext = os.path.splitext(os.path.normpath(source_file))
    if not module_path.replace(os.path.sep, "/").endswith(
        module_name.replace(".", "/")
    ):
        raise ValueError(
            f"cannot detect module path " f"for func_or_class {callable_name!r}"
        )

    module_path = os.path.normpath(module_path[0 : -len(module_name)])

    code_config = CodeConfig.from_file_set(
        file_set=FileSet(module_path, includes=["*.py"]),
        callable_ref=f"{module_name}:{callable_name}",
        callable_params=callable_params,
    )
    code_config.set_callable(_callable)
    return code_config


def _inline_code_to_module(
    inline_code: str, callable_ref: str, callable_params: Dict[str, Any] = None
):
    module_name, callable_name = _normalize_callable_ref(callable_ref)
    dir_path = new_temp_dir(prefix=TEMP_FILE_PREFIX)
    with open(os.path.join(dir_path, f"{module_name}.py"), "w") as stream:
        stream.write(inline_code)
    return CodeConfig.from_file_set(
        file_set=FileSet(dir_path),
        callable_ref=callable_ref,
        callable_params=callable_params,
    )


def _normalize_file_set(file_set: Union[FileSet, str, Any]) -> FileSet:
    if isinstance(file_set, FileSet):
        return file_set
    return FileSet(str(file_set))


def _normalize_inline_code(
    code: Tuple[Union[str, Callable], ...],
    callable_name: str = None,
    module_name: str = None,
) -> Tuple[str, str]:
    if not callable_name:
        first_callable = next(filter(callable, code), None)
        if first_callable is not None:
            callable_name = first_callable.__name__
    callable_name = callable_name or DEFAULT_CALLABLE_NAME
    module_name = module_name or _next_user_module_name()
    if all(map(callable, code)):
        warnings.warn(
            "inline code may not be executable due to missing " "import statements"
        )
    inline_code = "\n\n".join(
        [c if isinstance(c, str) else inspect.getsource(c) for c in code]
    )
    return inline_code, f"{module_name}:{callable_name}"


def _normalize_callable_ref(callable_ref: str) -> Tuple[Optional[str], str]:
    """Normalize a callable reference *callable_ref* of the form
    "[module_name:]callable_name" into a pair.
    """
    assert_instance(callable_ref, str, "callable_ref")

    splits = callable_ref.split(":")
    if len(splits) == 2:
        module_name, callable_name = [s.strip() for s in splits]
    else:
        module_name, callable_name = "", ""

    # Verify components

    assert_true(
        module_name and callable_name,
        message=f"illegal callable_ref {callable_ref!r}: "
        f'expected format "<module>:<callable>"',
    )
    assert_true(
        all(s.isidentifier() for s in module_name.split(".")),
        message=f"illegal callable_ref {callable_ref!r}: "
        f"{module_name!r} is not a valid Python module name",
    )
    assert_true(
        callable_name.isidentifier(),
        message=f"illegal callable_ref {callable_ref!r}: "
        f"{callable_name!r} is not a valid Python identifier",
    )

    return module_name, callable_name
