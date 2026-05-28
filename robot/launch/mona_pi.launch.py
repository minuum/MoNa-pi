"""
MoNa-pi ROS2 런치 파일

실행:
    # 전체 스택 (카메라 + 추론서버 + 컨트롤러 + 키보드)
    ros2 launch robot/launch/mona_pi.launch.py

    # VLA 자동 모드
    ros2 launch robot/launch/mona_pi.launch.py mode:=vla instruction:="직진해"

    # 추론 서버 없이 수동만 (테스트)
    ros2 launch robot/launch/mona_pi.launch.py inference:=false mode:=manual
"""

import sys
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    LogInfo,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # ── 런치 인수 ────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("instruction", default_value="Navigate to the goal"),
        DeclareLaunchArgument("config",      default_value="configs/serbot2.yaml",
                              description="모델/배포 설정 YAML"),
        DeclareLaunchArgument("ckpt",        default_value="checkpoints/best"),
        DeclareLaunchArgument("server_url",  default_value="http://localhost:8080"),
        DeclareLaunchArgument("control_hz",  default_value="10.0"),
        DeclareLaunchArgument("mode",        default_value="hybrid",
                              description="manual | vla | hybrid"),
        DeclareLaunchArgument("camera_backend", default_value="gstreamer",
                              description="gstreamer | usb | dummy"),
        # 선택 활성화 플래그
        DeclareLaunchArgument("inference",   default_value="true"),
        DeclareLaunchArgument("camera",      default_value="true"),
        DeclareLaunchArgument("keyboard",    default_value="true"),
    ]

    # ── 추론 서버 (GX10 또는 온보드) ────────────────────────────────
    inference_server = ExecuteProcess(
        cmd=[
            sys.executable,
            "inference/server.py",
            "--config", LaunchConfiguration("config"),
            "--ckpt",   LaunchConfiguration("ckpt"),
            "--host",   "0.0.0.0",
            "--port",   "8080",
            "--solver", "heun",
            "--steps",  "5",
        ],
        output="screen",
        name="mona_pi_inference_server",
        condition=IfCondition(LaunchConfiguration("inference")),
    )

    # ── 카메라 노드 ──────────────────────────────────────────────────
    camera_node = ExecuteProcess(
        cmd=[
            sys.executable,
            "robot/camera_node.py",
            "--backend", LaunchConfiguration("camera_backend"),
            "--fps", "30",
            "--compressed",
        ],
        output="screen",
        name="mona_pi_camera",
        condition=IfCondition(LaunchConfiguration("camera")),
    )

    # ── ROS2 컨트롤러 노드 ──────────────────────────────────────────
    controller_node = Node(
        package="mona_pi",
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

    # ── 키보드 컨트롤러 ─────────────────────────────────────────────
    keyboard_node = ExecuteProcess(
        cmd=[
            sys.executable,
            "robot/keyboard_controller.py",
            "--mode", LaunchConfiguration("mode"),
            "--throttle", "50",
        ],
        output="screen",
        name="mona_pi_keyboard",
        condition=IfCondition(LaunchConfiguration("keyboard")),
    )

    # ── 추론 서버 종료 시 알림 ───────────────────────────────────────
    on_server_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=inference_server,
            on_exit=[LogInfo(msg="[MoNa-pi] 추론 서버가 종료되었습니다.")],
        )
    )

    return LaunchDescription(
        args + [
            inference_server,
            camera_node,
            controller_node,
            keyboard_node,
            on_server_exit,
        ]
    )
