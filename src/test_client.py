import asyncio
import numpy as np
import time

from aiortc import RTCPeerConnection

from reachy_sdk_api.arm_kinematics_pb2 import ArmSide
from reachy_sdk_api.joint_pb2 import JointId, JointCommand, JointsCommand, JointsState
from reachy_sdk_api.fullbody_cartesian_command_pb2 import FullBodyCartesianCommand


from google.protobuf.wrappers_pb2 import BoolValue

from soulac.client import ReachyWebRTCClient

from scipy.spatial.transform import Rotation


SIGNALING_URL = 'http://10.0.0.88:8080/ws'
UID = 'reachy_xprize_finals'


async def flush_output(channel):
    try:
        while channel.bufferedAmount > 0:
            await channel.transport._data_channel_flush()
            await channel.transport._transmit()
            await asyncio.sleep(1 / 1000)
    except ConnectionError:
        pass


async def stream_joint_commands(channel):
    while channel.readyState == 'open':
        state = JointsCommand(
            commands=[
                JointCommand(
                    id=JointId(name='neck_roll'),
                    compliant=BoolValue(value=False),
                )
            ],
        ) 
        print(f'SEND {state}')
        msg = state.SerializeToString()
        channel.send(msg)
        await flush_output(channel)
        await asyncio.sleep(2)


        state = JointsCommand(
            commands=[
                JointCommand(
                    id=JointId(name='neck_roll'),
                    compliant=BoolValue(value=True),
                )
            ],
        )        
        print(f'SEND {state}')
        msg = state.SerializeToString()
        channel.send(msg)
        await flush_output(channel)
        await asyncio.sleep(2)


async def stream_body_commands(channel):
    while channel.readyState == 'open':
        t = time.time()
        yaw = np.sin(2 * np.pi * 0.5 * t) * 0.3
        q = Rotation.from_euler('xyz', [0, 0, yaw]).as_quat()

        state = FullBodyCartesianCommand()
        # state.neck.q.x = q[0]
        # state.neck.q.y = q[1]
        # state.neck.q.z = q[2]
        # state.neck.q.w = q[3]

        # state.left_arm.target.side = ArmSide.LEFT
        # print(f'Left arm target: {state.left_arm.target}')
        # state.left_arm.target.pose.data.extend([
        #     1, 0, 0, 0.2,
        #     0, 1, 0, 0.1,
        #     0, 0, 1, 0.5,
        #     0, 0, 0, 1,
        # ])

        state.right_arm.target.side = ArmSide.RIGHT
        state.right_arm.target.pose.data.extend([
            0.2, 0.8, 0, 0.2,
            0, 1, 0, -0.4,
            0.2, 0, 0.8, 0.3,
            0, 0, 0, 1,
        ])

        print(f'SEND {state}')

        msg = state.SerializeToString()
        channel.send(msg)

        await flush_output(channel)
        await asyncio.sleep(1 / 100)


def main():
    async def serve4ever():
        signaling = ReachyWebRTCClient(
            signaling_server_url=SIGNALING_URL,
            role='operator',
            room=f'body_control_{UID}',
            uid=UID,
        )

        @signaling.on('ready')
        async def call(pc: RTCPeerConnection):
            print('Call ready!')

            @pc.on("datachannel")
            def on_datachannel(channel):
                if channel.label == 'joint':
                    @channel.on('message')
                    def on_joints_msg(msg):
                        state = JointsState()
                        state.ParseFromString(msg)

                    # asyncio.ensure_future(stream_joint_commands(channel))           

                if channel.label == 'body_control':
                    asyncio.ensure_future(stream_body_commands(channel))

        await signaling.connect()
        await signaling.consume()


    asyncio.run(serve4ever())


if __name__ == '__main__':
    main()