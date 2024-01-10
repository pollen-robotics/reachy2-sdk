# Python SDK for Reachy 2

## Install

```console
$ pip install -e .[pollen,dev]
```

*[dev]* contains the tools for developers.
*[pollen]* has the custom pollen robotics repos. It is mandatory but not in the default install because github actions cannot fetch private repo directly.

## Logs

The SDK relies on the [python logging system](https://docs.python.org/3/howto/logging.html). Set the desired debug level to see messages from the SDK.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```