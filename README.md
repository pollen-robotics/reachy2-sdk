# Python SDK for Reachy 2

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![linter](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/lint.yml/badge.svg) ![pytest](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/unit_tests.yml/badge.svg) ![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FabienDanieau/92452aca5c894f95fb934dc2a7a6815f/raw/covbadge.json)

## Install

Use the following command to install:

```console
$ pip install -e .[pollen,dev]
```

The *[dev]* option includes tools for developers. The *[pollen]* option contains custom Pollen Robotics repositories. It is essential but not included in the default installation because GitHub Actions cannot directly fetch from private repositories.

## Usage

Check out the [examples](src/examples/) folder for jupyter notebooks and example scripts.

## Documentation

Documentation is generated via pdoc. It can be generated locally with:
```console
pdoc reachy2_sdk --output-dir docs --logo "https://www.pollen-robotics.com/img/company/logo/pollen_logo_square_black.svg"
```

## Unit tests

To ensure everything is functioning correctly, run the unit tests. There are two groups of tests: offline and online. Offline tests check internal functions using mock objects. Online tests require a connection to a simulated robot (e.g., in rviz), and the virtual robot should exhibit movement during these tests.

To execute the tests, use pytest with an optional category:

```console
$ pytest [-m offline|online]
```

Note that only **offline tests** are executed by the Continuous Integration/Continuous Deployment (CI/CD) pipeline, as they don't require a gRPC connection.

### Camera tests

Camera tests have their own marks because it requires the camera to be plugged to the sdk server (*sr_camera* and/or *teleop_camera*). 

```console
$ pytest -m [sr_camera | teleop_camera]
```

## Logs

The SDK relies on the [python logging system](https://docs.python.org/3/howto/logging.html). Set the desired debug level to see messages from the SDK.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

