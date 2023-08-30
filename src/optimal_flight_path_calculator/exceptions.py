import functools
import json
import sys
import time
import traceback
from typing import Callable, TypeVar

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


__all__ = ["ErrorHandler"]


class ErrorHandler:
    # float, int, str, list, dict, tuple

    def time_this(self, func: Callable[P, T]) -> Callable[P, T]:
        """A decorator that runs supplied function & times it"""

        @functools.wraps(func)
        def func_executor(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter_ns()
            result = func(*args, **kwargs)
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            # signature = ", ".join(args_repr + kwargs_repr)
            signature = ""
            time_taken = round((time.perf_counter_ns() - start_time) * 1.0e-9, 4)
            print(
                f"\nFunction:{func.__name__} Args:{signature} Time taken:{time_taken} seconds\n"
            )
            return result

        return func_executor

    def run_this(self, func: Callable[P, T]) -> Callable[P, T]:
        """A decorator that runs supplied function & handles the Error"""

        @functools.wraps(func)
        def func_executor(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                error_message = {"error_message": repr(ex)}
                print(json.dumps(error_message, indent=2))
                print(traceback.format_exc())
                sys.exit(1)

        return func_executor
