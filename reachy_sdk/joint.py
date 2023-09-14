# from .register import Register, MetaRegister
# import numpy as np

# from google.protobuf.wrappers_pb2 import FloatValue


# def _to_position(internal_pos: float) -> float:
#     result: float
#     result = round(np.rad2deg(internal_pos), 2)
#     return result


# def _to_internal_position(pos: float) -> float:
#     result: float
#     result = np.deg2rad(pos)
#     return result


# class Joint(metaclass=MetaRegister):
#     present_position = Register(readonly=True, type=FloatValue, conversion=(_to_internal_position, _to_position))
#     goal_position = Register(
#         readonly=False,
#         type=FloatValue,
#         conversion=(_to_internal_position, _to_position),
#     )

#     def __init__(self) -> None:
#         pass
