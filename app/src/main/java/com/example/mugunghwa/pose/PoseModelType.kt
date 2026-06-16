package com.example.mugunghwa.pose

enum class PoseModelType(
    val label: String
) {
    MEDIAPIPE("MediaPipe"),
    RTMPOSE("RTMPose"),
    YOLO26("YOLO26")
}
