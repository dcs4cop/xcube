import logging
import sys
from typing import Dict, Any, Optional, Union, Sequence, Type, Tuple

import click

from xcube.constants import (
    GENERAL_LOG_FORMAT,
    XCUBE_LOG_FORMAT,
    LOG,
    LOG_LEVEL_OFF_NAME,
    LOG_LEVEL_OFF,
    LOG_LEVEL_DETAIL,
    LOG_LEVEL_TRACE,
)


def new_cli_ctx_obj():
    return {
        "traceback": False,
        "scheduler": None,
    }


def cli_option_traceback(func):
    """Decorator for adding a reusable CLI option `--traceback`."""

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: bool):
        ctx_obj = ctx.ensure_object(dict)
        ctx_obj["traceback"] = value
        return value

    return click.option(
        "--traceback",
        is_flag=True,
        help="Enable tracing back errors by dumping the Python call stack. "
        "Pass as very first option to also trace back error during command-line validation.",
        callback=_callback,
    )(func)


def cli_option_quiet(func):
    """Decorator for adding a reusable CLI option `--quiet`/'-q'."""

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: bool):
        ctx_obj = ctx.ensure_object(dict)
        ctx_obj["quiet"] = value
        return value

    return click.option(
        "--quiet",
        "-q",
        is_flag=True,
        help="Disable output of log messages to the console entirely."
        " Note, this will also suppress error and warning messages.",
        callback=_callback,
    )(func)


def cli_option_verbosity(func):
    """Decorator for adding a reusable CLI option `--verbose`/'-v' that
    can be used multiple times.
    The related kwarg is named `verbosity` and is of type int (= count).
    """

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: int):
        ctx_obj = ctx.ensure_object(dict)
        ctx_obj["verbose"] = value
        ctx_obj["verbosity"] = value
        return value

    return click.option(
        "--verbose",
        "-v",
        "verbosity",
        count=True,
        help="Enable output of log messages to the console."
        " Has no effect if --quiet/-q is used."
        " May be given multiple times to control the level"
        " of log messages, i.e.,"
        " -v refers to level INFO, -vv to DETAIL, -vvv to DEBUG,"
        " -vvvv to TRACE."
        " If omitted, the log level of the console is WARNING.",
        callback=_callback,
    )(func)


def cli_option_dry_run(func):
    """Decorator for adding a reusable CLI option `--dry-run`/'-d'.
    The related kwarg is named `dry_run` and is of type bool.
    """

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: bool):
        ctx_obj = ctx.ensure_object(dict)
        ctx_obj["dry_run"] = value
        return value

    return click.option(
        "--dry-run",
        "-d",
        "dry_run",
        is_flag=True,
        help="Do not change any data," " just report what would have been changed.",
        callback=_callback,
    )(func)


def cli_option_scheduler(func):
    """Decorator for adding a pre-defined, reusable CLI option `--scheduler`."""

    # noinspection PyUnusedLocal
    def _callback(ctx: click.Context, param: click.Option, value: Optional[str]):
        if not value:
            return

        address_and_kwargs = value.split("?", 2)
        if len(address_and_kwargs) == 2:
            address, kwargs_string = address_and_kwargs
            kwargs = parse_cli_kwargs(kwargs_string, metavar="SCHEDULER")
        else:
            (address,) = address_and_kwargs
            kwargs = dict()

        try:
            # The Dask Client registers itself as the default Dask scheduler, and so runs dask.array used by xarray
            import distributed

            scheduler_client = distributed.Client(address, **kwargs)
            ctx_obj = ctx.ensure_object(dict)
            ctx_obj["scheduler"] = scheduler_client
            return scheduler_client
        except ValueError as e:
            raise click.BadParameter(
                f"Failed to create Dask scheduler client: {e}"
            ) from e

    return click.option(
        "--scheduler",
        metavar="SCHEDULER",
        help="Enable distributed computing using the Dask scheduler identified by SCHEDULER. "
        "SCHEDULER can have the form <address>?<keyword>=<value>,... where <address> "
        "is <host> or <host>:<port> and specifies the scheduler's address in your network. "
        "For more information on distributed computing "
        "using Dask, refer to http://distributed.dask.org/. "
        "Pairs of <keyword>=<value> are passed to the Dask client. "
        "Refer to http://distributed.dask.org/en/latest/api.html#distributed.Client",
        callback=_callback,
    )(func)


def parse_cli_kwargs(value: str, metavar: str = None) -> Dict[str, Any]:
    """Parse a string value of the form [<kw>=<arg>{,<kw>=<arg>}] into a dictionary.
    <kw> must be a valid Python identifier, <arg> must be a Python literal.

    Args:
        value: A string value.
        metavar: Name of a meta-variable used in CLI.

    Returns:
        a dictionary of keyword-arguments
    """
    if value:
        try:
            return eval(f"dict({value})", {}, {})
        except Exception:
            if metavar:
                message = f"Invalid value for {metavar}: {value!r}"
            else:
                message = f"Invalid value: {value!r}"
            raise click.ClickException(message)
    else:
        return dict()


def parse_cli_sequence(
    seq_value: Union[None, str, Sequence[Any]],
    metavar: str = "parameter",
    item_parser=None,
    item_validator=None,
    allow_none: bool = True,
    allow_empty_items: bool = False,
    strip_items: bool = True,
    item_plural_name: str = "items",
    num_items: int = None,
    num_items_min: int = None,
    num_items_max: int = None,
    separator: str = ",",
    error_type: Type[Exception] = click.ClickException,
) -> Optional[Tuple[Any, ...]]:
    """Parse a CLI argument that is supposed to be a sequence.

    Args:
        seq_value: A string, a string sequence, or None.
        metavar: CLI metavar name
        item_parser: an optional function that takes a string and parses
            it
        item_validator: a optional function that takes a parsed value
            and parses it
        allow_none: whether it is ok for *seq_value* to be None
        allow_empty_items: whether it is ok to have empty values in
            *seq_value*
        strip_items: whether to strip values in *seq_value*
        item_plural_name: a name for multiple items
        num_items: expected number of items
        num_items_min: expected minimum number of items
        num_items_max: expected maximum number of items
        separator: expected separator if *seq_value* is a string,
            default is ','.
        error_type: value error to be raised in case, defaults to
            `click.ClickException`

    Returns:
        parsed and validated *seq_value* as a tuple of values
    """
    if seq_value is None:
        if allow_none:
            return None
        raise error_type(f"{metavar} must be given")
    if isinstance(seq_value, str):
        if "," in seq_value:
            items = seq_value.split(",")
        elif num_items is not None:
            items = num_items * (seq_value,)
        else:
            items = (seq_value,)
    else:
        items = seq_value
    item_count = len(items)
    if num_items is not None and item_count != num_items:
        raise error_type(
            f"{metavar} must have {num_items} {item_plural_name} separated by {separator!r}"
        )
    if num_items_min is not None and item_count < num_items_min:
        raise error_type(
            f"{metavar} must have at least {num_items_min} {item_plural_name} separated by {separator!r}"
        )
    if num_items_max is not None and item_count > num_items_max:
        raise error_type(
            f"{metavar} must have no more than {num_items_max} {item_plural_name} separated by {separator!r}"
        )
    if strip_items:
        items = tuple(item.strip() for item in items)
    if not allow_empty_items:
        for item in items:
            if not item:
                raise error_type(f"{item_plural_name} in {metavar} must not be empty")
    if item_parser:
        try:
            items = tuple(map(item_parser, items))
        except ValueError as e:
            raise error_type(f"Invalid {item_plural_name} in {metavar} found: {e}")
    if item_validator:
        try:
            for item in items:
                item_validator(item)
        except ValueError as e:
            raise error_type(f"Invalid {item_plural_name} in {metavar} found: {e}")
    return tuple(items)


def assert_positive_int_item(item: int):
    """A validator for positive integer number sequences.
    Frequently used to validate counts or image and tile sizes passes as args to the CLI.
    """
    if item <= 0:
        raise ValueError("all items must be positive integer numbers")


def handle_cli_exception(
    e: BaseException, exit_code: int = None, traceback_mode: bool = False
) -> int:
    exc_info = traceback_mode and e
    if isinstance(e, click.Abort):
        LOG.error("Aborted.", exc_info=exc_info)
        exit_code = exit_code or 1
    elif isinstance(e, click.ClickException):
        LOG.error("%s", e, exc_info=exc_info)
        exit_code = exit_code or e.exit_code
    elif isinstance(e, OSError):
        LOG.error("OS error: %s", e, exc_info=exc_info)
        exit_code = exit_code or 2
    else:
        LOG.error(f"Internal error: %s", e, exc_info=exc_info)
        exit_code = exit_code or 3
    LOG.debug("Exit with code %d", exit_code)
    return exit_code


def configure_warnings(enable_warnings: bool):
    # In normal operation, it's not necessary to explicitly set the default
    # filter when --warnings is omitted, but it can be needed during
    # unit testing if a previous test has caused the filter to be changed.
    import warnings

    warnings.simplefilter(
        "default" if enable_warnings else "ignore", category=DeprecationWarning
    )
    warnings.simplefilter(
        "default" if enable_warnings else "ignore", category=RuntimeWarning
    )


_general_handler: Union[
    logging.NullHandler, logging.FileHandler, logging.StreamHandler
] = logging.NullHandler()


def configure_logging(
    log_file: Optional[str],
    log_level: Optional[str],
    logger: logging.Logger = logging.getLogger(),
):
    remove_log_handlers(logger)
    if log_level == LOG_LEVEL_OFF_NAME:
        logger.setLevel(LOG_LEVEL_OFF)
    else:
        logger.setLevel(log_level)
        formatter = logging.Formatter(GENERAL_LOG_FORMAT)
        if log_file:
            handler = logging.FileHandler(log_file, "a", encoding="utf8")
        else:
            handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        global _general_handler
        _general_handler = handler


def configure_cli_output(
    quiet: Optional[bool] = None,
    verbosity: Optional[Union[bool, int]] = None,
    logger: logging.Logger = LOG,
):
    remove_log_handlers(logger)

    if quiet:
        level = LOG_LEVEL_OFF
    elif not verbosity:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity == 2:
        level = LOG_LEVEL_DETAIL
    elif verbosity == 3:
        level = logging.DEBUG
    else:
        level = LOG_LEVEL_TRACE

    logger.setLevel(level)

    if isinstance(_general_handler, (logging.FileHandler, logging.NullHandler)):
        # Only if we do not already redirect output of general logging
        # to stderr, install a new handler with a simple message format
        # for the console.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(ConsoleMessageFormatter())
        logger.addHandler(handler)


def remove_log_handlers(logger: logging.Logger):
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


class ConsoleMessageFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(XCUBE_LOG_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if logging.DEBUG < record.levelno < logging.WARNING:
            # Just return plain message and skip formatting level
            msg = str(record.msg)
            return msg % record.args if record.args else msg
        else:
            return super().format(record)
