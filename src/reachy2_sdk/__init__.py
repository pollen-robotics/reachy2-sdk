"""ReachySDK package.

This package provides remote access (via socket) to a Reachy robot.
It automatically handles the synchronization with the robot.
In particular, you can easily get an always up-to-date robot state (joint positions, sensors value).
You can also send joint commands, compute forward or inverse kinematics.

Simply do
```python
from reachy2_sdk.reachy_sdk import ReachySDK
reachy = ReachySDK(host="ip_address")
```

And you're ready to use Reachy!

*Examples are available [here](https://github.com/pollen-robotics/reachy2-sdk/tree/develop/src/examples)
 and tutorials [there](https://github.com/pollen-robotics/reachy2-tutorials) !*

"""
import configparser
import os
from importlib.metadata import PackageNotFoundError, version
from typing import List

import reachy2_sdk_api
from packaging.requirements import Requirement
from packaging.version import parse

from .reachy_sdk import ReachySDK  # noqa: F401

__version__ = "1.0.12"


def get_dependencies_from_setup_cfg() -> List[str]:
    """Get dependencies from setup.cfg file."""
    setup_cfg_path = os.path.abspath(os.path.join(__file__, "../../..", "setup.cfg"))

    config = configparser.ConfigParser()
    config.read(setup_cfg_path)

    if "options" in config and "install_requires" in config["options"]:
        dependencies = config["options"]["install_requires"].strip().splitlines()
        return [dep.strip() for dep in dependencies if dep.strip()]

    return []


def check_reachy2_sdk_api_dependency(requirement: str) -> None:
    """Check if the installed version of reachy2-sdk-api is compatible with the required one.

    Also check if the used version of reachy2-sdk-api is higher than the minimal required version.
    """
    api_requirement = Requirement(requirement)

    try:
        installed_version = reachy2_sdk_api.__version__
    except AttributeError:
        try:
            installed_version = version("reachy2-sdk-api")
        except PackageNotFoundError:
            raise ImportError("❌ reachy2-sdk-api is NOT installed!")

    installed_parsed = parse(installed_version)

    if installed_parsed in api_requirement.specifier:
        min_required_version = None
        for spec in api_requirement.specifier:
            if spec.operator in (">=", "=="):
                min_required_version = parse(spec.version)
                break

        if min_required_version is None:
            raise ValueError(f"❌ No valid minimum version found in '{api_requirement}'")

        if installed_parsed > min_required_version:
            print(
                f"Installed version of reachy2-sdk-api {installed_version} is higher than"
                f" the minimal requirement {min_required_version},"
                " a newer version of reachy2-sdk may be available."
            )
    else:
        raise ValueError(
            f"⚠️  Version conflict for reachy2-sdk-api:"
            f"\n\tInstalled {installed_version},"
            f"\n\tRequired {api_requirement.specifier}"
        )


def check_dependencies() -> None:
    """Check if the installed dependencies are compatible with the required ones."""
    dependencies = get_dependencies_from_setup_cfg()
    for requirement in dependencies:
        try:
            if requirement.startswith("reachy2-sdk-api"):
                check_reachy2_sdk_api_dependency(requirement)
            else:
                req = Requirement(requirement)
                installed_version = version(req.name)
                if parse(installed_version) not in req.specifier:
                    raise ValueError(
                        f"⚠️  Version conflict for {req.name}: \n\tInstalled {installed_version}, \n\tRequired {req.specifier}"
                    )
        except PackageNotFoundError:
            print(f"❌ Missing dependency: {requirement}")
        except ValueError as e:
            print(e)


check_dependencies()
