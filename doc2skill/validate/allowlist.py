from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doc2skill.validate.extract import Identifier

STDLIB_MODULES = {
    "abc",
    "argparse",
    "asyncio",
    "base64",
    "codecs",
    "collections",
    "concurrent",
    "contextlib",
    "contextvars",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "email",
    "enum",
    "functools",
    "hashlib",
    "html",
    "http",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "logging",
    "math",
    "mimetypes",
    "os",
    "pathlib",
    "pprint",
    "random",
    "re",
    "shutil",
    "socket",
    "ssl",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "xml",
}

PYTHON_BUILTINS = {
    "AttributeError",
    "ConnectionError",
    "Exception",
    "KeyError",
    "OSError",
    "RuntimeError",
    "TimeoutError",
    "TypeError",
    "ValueError",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "dict",
    "dir",
    "enumerate",
    "float",
    "format",
    "getattr",
    "hasattr",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "max",
    "min",
    "next",
    "object",
    "open",
    "print",
    "range",
    "repr",
    "set",
    "setattr",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
}

KNOWN_THIRD_PARTY = {
    "aiohttp",
    "anyio",
    "django",
    "fastapi",
    "flask",
    "pytest",
    "requests",
    "rich",
    "starlette",
    "tqdm",
    "trio",
    "urllib3",
}

COMMON_ENGLISH_COMPOUNDS = {
    "back_end",
    "edge_case",
    "file_name",
    "front_end",
    "key_error",
    "pass_word",
    "path_name",
    "time_out",
    "use_case",
    "user_name",
}


def is_known_external(identifier: "Identifier") -> bool:
    """Return True if identifier belongs to Python stdlib, builtins, known 3p libs, or English compounds."""
    text = identifier.text
    if text in COMMON_ENGLISH_COMPOUNDS:
        return True

    # Method calls extracted as bare method names (e.g. response.json() -> "json")
    # must be verified against the source corpus rather than shadowed by stdlib names.
    if identifier.kind == "method":
        return False

    if "." in text:
        root = text.split(".")[0]
        return (
            root in STDLIB_MODULES
            or root in PYTHON_BUILTINS
            or root in KNOWN_THIRD_PARTY
        )

    return (
        text in STDLIB_MODULES
        or text in PYTHON_BUILTINS
        or text in KNOWN_THIRD_PARTY
    )

