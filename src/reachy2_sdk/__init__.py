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
from typing import List

import pkg_resources

from .reachy_sdk import ReachySDK  # noqa: F401

__version__ = "1.0.10"


def get_dependencies_from_setup_cfg() -> List[str]:
    """Get dependencies from setup.cfg file."""
    setup_cfg_path = os.path.abspath(os.path.join(__file__, "../../..", "setup.cfg"))

    config = configparser.ConfigParser()
    config.read(setup_cfg_path)

    if "options" in config and "install_requires" in config["options"]:
        dependencies = config["options"]["install_requires"].strip().splitlines()
        return [dep.strip() for dep in dependencies if dep.strip()]

    return []


def check_dependencies() -> None:
    """Check if the installed dependencies are compatible with the required ones.

    Also check is the installed version of reachy2-sdk-api is higher than the minimal required version.
    """
    dependencies = get_dependencies_from_setup_cfg()

    for requirement in dependencies:
        try:
            pkg_resources.require(requirement)
            if requirement.startswith("reachy2-sdk-api"):
                api_requirement = pkg_resources.Requirement.parse(requirement)
                installed_version = pkg_resources.get_distribution("reachy2-sdk-api").version
                min_required_version = None
                for spec in api_requirement.specifier:
                    if spec.operator in (">=", "=="):
                        min_required_version = spec.version
                        break
                if min_required_version is None:
                    raise ValueError(f"❌ No valid minimum version found in '{api_requirement}'")

                installed_parsed = pkg_resources.parse_version(installed_version)
                min_parsed = pkg_resources.parse_version(min_required_version)
                if installed_parsed > min_parsed:
                    print(
                        f"Installed version of reachy2-sdk-api {installed_version} is higher than"
                        f" the minimal requirements {min_required_version},"
                        " a newer version of reachy2-sdk is probably available."
                    )
        except pkg_resources.VersionConflict as e:
            print(
                f"⚠️ Version conflict for {e.dist.key}: Installed {e.dist.version}, "
                f" Required {requirement.split(e.dist.key)[-1]}"
            )
        except pkg_resources.DistributionNotFound as e:
            print(f"❌ Missing dependency : {e.req.name}")


check_dependencies()
