# Installation

HTTPX requires Python 3.8 or later. You can install HTTPX using `pip`:

```shell
$ pip install httpx
$ python -m pip install httpx
```

## Optional Dependencies

If you want HTTP/2 support, install the optional `http2` extras:

```shell
$ pip install httpx[http2]
```

If you want Brotli response decoding support:

```shell
$ pip install httpx[brotli]
```

If you plan to use the command line client directly, install the `cli` extras:

```shell
$ pip install httpx[cli]
```

To verify your installation in a virtual environment:

```shell
$ python -m venv .venv
$ source .venv/bin/activate
$ python -c "import httpx; print(httpx.__version__)"
```

