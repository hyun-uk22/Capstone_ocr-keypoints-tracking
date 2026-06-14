# 사용 방법

## 1. 준비물

- Android Studio 또는 Android SDK가 설치된 Windows PC
- 카메라가 있는 Android 스마트폰
- Android 스마트폰

## 2. 모델 파일 확인

현재 저장소에는 앱 실행에 필요한 모델 파일이 포함되어 있습니다.

```text
app/src/main/assets/pose_landmarker_heavy.task
app/src/main/assets/pose_landmarker_full.task
app/src/main/assets/efficientdet_lite0.tflite
app/src/main/assets/rtmpose.onnx
app/src/main/assets/yolo26n-pose.onnx
```

앱 설정에서 MediaPipe, RTMPose, YOLO26 중 포즈 모델을 선택할 수 있습니다. MediaPipe는 `pose_landmarker_heavy.task`를 우선 사용하고, 이 파일이 없으면 `pose_landmarker_full.task`, `pose_landmarker_lite.task` 순서로 fallback을 시도합니다. 선택한 pose 모델 파일이 없으면 앱은 실행되더라도 해당 포즈 인식 기능은 정상 동작하지 않습니다.
OCR 모델은 앱에 번들되어 설치되므로 최초 실행 시점의 모델 다운로드가 필요하지 않습니다.

## 3. APK 빌드

프로젝트 루트에서 실행합니다.

```powershell
.\gradlew.bat assembleDebug
```

빌드 결과 APK 위치:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 4. 스마트폰에 설치

USB 디버깅을 켠 스마트폰을 연결한 뒤 실행합니다.

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

또는 Android Studio에서 `app` 실행 버튼으로 설치해도 됩니다.

## 5. 앱 사용 순서

1. 앱을 실행합니다.
2. 카메라 권한을 허용합니다.
3. 카메라 프리뷰가 보이는지 확인합니다.
4. `Start` 버튼을 눌러 게임을 시작합니다.
5. `Green/Red` 버튼으로 `GREEN_LIGHT`와 `RED_LIGHT`를 전환합니다.
6. 참가자들이 가만히 서 있는 상태에서 `Calibrate`를 눌러 움직임 기준값을 보정합니다.
7. `RED_LIGHT` 상태에서 움직임이 기준값을 3프레임 연속 넘으면 해당 참가자가 `OUT` 처리됩니다.
8. `Reset` 버튼으로 게임 상태와 트래킹 정보를 초기화합니다.

## 6. 화면 표시 의미

- `State`: 현재 게임 상태
- `Active`: 탈락하지 않은 참가자 수 / 전체 추적 수
- `OUT`: 탈락자 수
- `Th`: 현재 movement threshold
- 초록 박스: 정상 추적 중인 참가자
- 빨간 박스 또는 `OUT`: 탈락 처리된 참가자
- `#1`, `#2`: 임시 trackId
- OCR로 번호가 3번 이상 반복 인식되면 trackId 대신 해당 번호가 label로 고정됩니다.

## 7. 테스트 체크리스트

1. 앱 실행 시 카메라 권한 요청이 뜨는가?
2. 카메라 프리뷰가 보이는가?
3. 사람이 화면에 들어오면 skeleton이 보이는가?
4. 여러 명이 보이면 각각 trackId가 붙는가?
5. `RED_LIGHT` 상태에서 움직이면 movement score가 증가하는가?
6. threshold 초과가 3프레임 연속 발생하면 `OUT` 처리되는가?
7. 번호표가 보이면 OCR label이 잠기는가?
8. `Reset` 버튼으로 상태가 초기화되는가?
9. pose model `.task` 파일이 없을 때 앱이 죽지 않고 안내 메시지를 보여주는가?

## 8. 현재 한계

- 사람 ID는 단순 centroid tracking이라 사람이 겹치거나 교차하면 바뀔 수 있습니다.
- OCR은 bbox crop 기반 MVP라 번호표 위치와 조명에 민감합니다.
- 자동 Green/Red 진행과 탈락자 음성 안내는 구현되어 있습니다.
- 탈락자 저장 DB는 아직 없습니다.
