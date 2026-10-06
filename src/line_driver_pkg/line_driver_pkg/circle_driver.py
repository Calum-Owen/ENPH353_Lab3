#!/usr/bin/env python3

from dataclasses import dataclass
import math

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data


@dataclass
class LineObservation:
    """Camera-derived cross-track error, heading error, and detection confidence."""

    lateral_error_m: float
    heading_error_rad: float
    confidence: float
    valid: bool
    line_center_px: int | None = None


class CameraLineDetector:
    """Find the mean x position of dark pixels on the image's bottom row."""

    def __init__(self, threshold: int = 85, lateral_scale_m: float = 0.15):
        self.threshold = threshold
        self.lateral_scale_m = lateral_scale_m

    def process_frame(self, frame: np.ndarray) -> LineObservation:
        """Apply bottom-row grayscale thresholding and return the detected center."""

        if frame.size == 0 or frame.ndim not in (2, 3):
            return LineObservation(0.0, 0.0, 0.0, False)

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3
            else frame
        )
        height, width = gray.shape
        road_values = np.where(gray[-1, :] < self.threshold)[0]
        if road_values.size == 0:
            return LineObservation(0.0, 0.0, 0.0, False)

        middle_road = int(np.mean(road_values))
        lateral_error = (
            (middle_road - (width - 1) / 2.0) / (width / 2.0)
        ) * self.lateral_scale_m
        return LineObservation(
            lateral_error,
            0.0,
            1.0,
            True,
            middle_road,
        )



class LineFollowingModel:
    """Forward-Euler model of lateral and heading error on a straight path."""

    def __init__(self, forward_speed: float = 0.15, timestep: float = 0.1):
        self.forward_speed = -abs(forward_speed)
        self.timestep = timestep

    def predict(
        self,
        observation: LineObservation,
        command: Twist,
    ) -> LineObservation:
        """Predict one model step using e_dot=v sin(psi) and psi_dot=omega."""

        lateral_error = observation.lateral_error_m + (
            self.forward_speed
            * math.sin(observation.heading_error_rad)
            * self.timestep
        )
        heading_error = observation.heading_error_rad + (
            command.angular.z * self.timestep
        )
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        return LineObservation(
            lateral_error,
            heading_error,
            observation.confidence,
            observation.valid,
        )


class MBCController:
    """Choose a bounded yaw rate by predicting each candidate over a short horizon."""

    def __init__(
        self,
        max_angular_speed: float = 5.0,
        lateral_scale_m: float = 0.15,
        horizon_steps: int = 10,
        heading_lookahead_m: float = 0.25,
        control_penalty: float = 0.015,
    ):
        self.max_angular_speed = max_angular_speed
        self.lateral_scale_m = lateral_scale_m
        self.horizon_steps = horizon_steps
        self.heading_lookahead_m = heading_lookahead_m
        self.control_penalty = control_penalty
        self.candidate_turn_rates = np.linspace(
            -max_angular_speed,
            max_angular_speed,
            17,
        )

    def compute_command(
        self,
        observation: LineObservation,
        model: LineFollowingModel,
    ) -> Twist:
        """Select the constant candidate turn rate with the lowest predicted cost."""

        best_turn_rate = 0.0
        best_cost = math.inf
        for turn_rate in self.candidate_turn_rates:
            candidate = Twist()
            candidate.angular.z = float(turn_rate)
            predicted = observation
            total_cost = 0.0

            for _ in range(self.horizon_steps):
                predicted = model.predict(predicted, candidate)
                normalized_lateral = (
                    predicted.lateral_error_m / self.lateral_scale_m
                )
                normalized_heading = (
                    predicted.heading_error_rad
                    * self.heading_lookahead_m
                    / self.lateral_scale_m
                )
                total_cost += (
                    normalized_lateral**2
                    + normalized_heading**2
                    + self.control_penalty
                    * (turn_rate / self.max_angular_speed) ** 2
                )

            if total_cost < best_cost:
                best_cost = total_cost
                best_turn_rate = float(turn_rate)

        command = Twist()
        heading_speed_scale = max(
            0.35,
            math.cos(min(abs(observation.heading_error_rad), math.pi / 2.0)),
        )
        command.linear.x = model.forward_speed * heading_speed_scale
        command.angular.z = best_turn_rate
        return command



class LineFollowerNode(Node):
    """Connects camera images to line detection, model-based control, and cmd_vel."""

    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('line_threshold', 85)
        self.declare_parameter('lateral_scale_m', 0.15)
        self.declare_parameter('forward_speed', 0.15)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('prediction_timestep', 0.1)
        self.declare_parameter('prediction_horizon_steps', 10)

        lateral_scale_m = float(self.get_parameter('lateral_scale_m').value)
        self.detector = CameraLineDetector(
            threshold=int(self.get_parameter('line_threshold').value),
            lateral_scale_m=lateral_scale_m,
        )
        self.model = LineFollowingModel(
            forward_speed=float(self.get_parameter('forward_speed').value),
            timestep=float(self.get_parameter('prediction_timestep').value),
        )
        self.controller = MBCController(
            max_angular_speed=float(
                self.get_parameter('max_angular_speed').value
            ),
            lateral_scale_m=lateral_scale_m,
            horizon_steps=int(
                self.get_parameter('prediction_horizon_steps').value
            ),
        )
        self.last_lost_line_warning_ns = 0
        self.image_sub = self.create_subscription(
            Image,
            str(self.get_parameter('image_topic').value),
            self.callback,
            qos_profile_sensor_data,
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('cmd_vel_topic').value),
            1,
        )
        self.processed_image_pub = self.create_publisher(
            Image,
            '/camera/processed_image',
            1,
        )

    def callback(self, data: Image) -> None:
        """Convert a camera message and publish a safe line-following command."""

        try:
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Camera image conversion failed: {error}')
            self.publish_stop()
            return

        observation = self.detector.process_frame(frame)
        if observation.valid and observation.line_center_px is not None:
            height = frame.shape[0]
            cv2.circle(
                frame,
                (observation.line_center_px, max(height - 10, 0)),
                10,
                (0, 0, 255),
                -1,
            )

        if not observation.valid:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_lost_line_warning_ns > 2_000_000_000:
                self.get_logger().warning('Line not detected; turning left to search.')
                self.last_lost_line_warning_ns = now_ns
            self.publish_search_turn()
        else:
            command = self.controller.compute_command(observation, self.model)
            command.linear.x *= max(0.35, observation.confidence)
            self.cmd_vel_pub.publish(command)

        try:
            processed_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Processed image conversion failed: {error}')
            return
        processed_image.header = data.header
        self.processed_image_pub.publish(processed_image)

    def publish_stop(self) -> None:
        """Publish a zero velocity command when vision is unavailable."""

        self.cmd_vel_pub.publish(Twist())

    def publish_search_turn(self) -> None:
        """Rotate slowly left in place while searching for a lost line."""

        command = Twist()
        command.angular.z = min(0.7, self.controller.max_angular_speed)
        self.cmd_vel_pub.publish(command)


def main(args=None):
    """Initialize ROS, run the line-follower node, and clean up on exit."""

    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()