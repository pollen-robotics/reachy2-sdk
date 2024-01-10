# Python SDK for Reachy 2

## Install

Use the following command to install:

```console
$ pip install -e .[pollen,dev]
```

The *[dev]* option includes tools for developers. The *[pollen]* option contains custom Pollen Robotics repositories. It is essential but not included in the default installation because GitHub Actions cannot directly fetch from private repositories.

## Unit tests

To ensure everything is functioning correctly, run the unit tests. There are two groups of tests: offline and online. Offline tests check internal functions using mock objects. Online tests require a connection to a simulated robot (e.g., in rviz), and the virtual robot should exhibit movement during these tests.

To execute the tests, use pytest with an optional category:

```console
$ pytest [-m offline|online]
```

Note that only **offline tests** are executed by the Continuous Integration/Continuous Deployment (CI/CD) pipeline, as they don't require a gRPC connection.