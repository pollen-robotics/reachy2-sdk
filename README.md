# Python SDK for Reachy 2

[![Licence](https://img.shields.io/badge/licence-Apache%202.0-blue)](LICENSE) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
![linter](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/unit_tests.yml/badge.svg) 
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/FabienDanieau/92452aca5c894f95fb934dc2a7a6815f/raw/covbadge.json)
![Docs](https://github.com/pollen-robotics/python-template/actions/workflows/docs.yml/badge.svg)

## Install

You can install the library either from Pypi, or by cloning the entire repository : 

<details> 
<summary>From PyPi</summary>

```bash
pip install reachy2-sdk -e .
```
</details>

<details>
<summary>From the source</summary>

```bash
git clone https://github.com/pollen-robotics/reachy2-sdk.git
cd reachy2-sdk
pip install -e .[dev]
```

The *[dev]* option includes tools for developers.

</details>


## Usage

Check out the [examples](src/examples/) folder for jupyter notebooks and example scripts.
You can also take a look on the [Reachy2 Documentation](https://pollen-robotics.github.io/reachy2-docs/developing-with-reachy-2/getting-started-sdk/), which provides a more detailed overview of the SDK's various functions.

## Documentation

SDK Documentation is generated via pdoc, and it's available at [https://pollen-robotics.github.io/reachy2-sdk/reachy2_sdk.html](https://pollen-robotics.github.io/reachy2-sdk/reachy2_sdk.html)


It can be generated locally with:
```console
pdoc reachy2_sdk --output-dir docs --logo "https://pollen-robotics.github.io/reachy2-sdk/pollen_logo.png" --logo-link "https://www.pollen-robotics.com" --docformat google
```

*You maybe have to install pdoc first by running <code>pip install pdoc</code>.*

The documentation relies on the provided docstrings with the google style. pydocstyle is used to enforced this style.
```console
pydocstyle src/ --convention google --count
```

## Unit tests

To ensure everything is functioning correctly, run the unit tests. There are two main groups of tests: offline and online. Offline tests check internal functions using mock objects. Online tests require a connection to a simulated robot (e.g., in rviz), and the virtual robot should exhibit movement during these tests.

To execute the tests, use pytest with an optional category:

```console
$ pytest [-m offline|online]
```

Note that only **offline tests** are executed by the Continuous Integration/Continuous Deployment (CI/CD) pipeline, as they don't require a gRPC connection.

### Camera tests

Camera tests have their own marks because it requires the cameras to be plugged to the sdk server 

```console
$ pytest -m cameras
```

### Audio tests

Audio tests have their own marks.

```console
$ pytest -m audio
```

### Mobile_base tests

Mobile base tests have their own marks as it used to be not supported in Rviz. The tests are based on the Gazebo simulation, so the robot must be launched in Gazebo mode when executing the tests.

```console
$ pytest -m mobile_base
```

## Logs

The SDK relies on the [python logging system](https://docs.python.org/3/howto/logging.html). Set the desired debug level to see messages from the SDK.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

