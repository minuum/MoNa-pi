"""
MoNa-pi ROS2 런치 파일

실행:
    ros2 launch robot/launch/mona_pi.launch.py \
        instruction:="장애물을 피해 직진" \
        ckpt:="checkpoints/best"
"""

import subprocess
import sys
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    LogInfo,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── 런치 인수 선언 ─────────────────────────────────────────────
    instruction_arg = DeclareLaunchArgument(
        "instruction",
        default_value="Navigate to the goal",
        description="자연어 주행 지시어",
    )
    ckpt_arg = DeclareLaunchArgument(
        "ckpt",
        default_value="checkpoints/best",
        description="MoNa-pi 체크포인트 경로",
    )
    server_url_arg = DeclareLaunchArgument(
        "server_url",
        default_value="http://localhost:8080",
        description="추론 서버 URL",
    )
    control_hz_arg = DeclareLaunchArgument(
        "control_hz",
        default_value="10.0",
        description="cmd_vel 발행 주파수 (Hz)",
    )

    # ── 추론 서버 프로세스 (Python subprocess) ──────────────────────
    inference_server = ExecuteProcess(
        cmd=[
            sys.executable,
            "inference/server.py",
            "--ckpt",   LaunchConfiguration("ckpt"),
            "--host",   "0.0.0.0",
            "--port",   "8080",
            "--solver", "heun",
            "--steps",  "5",
        ],
        output="screen",
        name="mona_pi_inference_server",
    )

    # ── ROS2 컨트롤러 노드 ─────────────────────────────────────────
    controller_node = Node(
        package="mona_pi",       # ros2 패키지 이름 (setup.py에서 정의)
        executable="ros2_controller",
        name="mona_pi_controller",
        output="screen",
        parameters=[{
            "inference_server_url": LaunchConfiguration("server_url"),
            "instruction":          LaunchConfiguration("instruction"),
            "control_hz":           LaunchConfiguration("control_hz"),
            "max_linear_vel":       1.15,
            "max_angular_vel":      1.15,
        }],
    )

    # ── 서버 종료 시 전체 종료 ────────────────────────────────────
    on_server_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=inference_server,
            on_exit=[LogInfo(msg="추론 서버가 종료되었습니다.")],
        )
    )

    return LaunchDescription([
        instruction_arg,
        ckpt_arg,
        server_url_arg,
        control_hz_arg,
        inference_server,
        controller_node,
        on_server_exit,
    ])
