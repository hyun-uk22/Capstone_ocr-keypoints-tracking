# Freeze!

Android 스마트폰 카메라로 사람의 자세를 추정하고, "무궁화 꽃이 피었습니다" 게임의 영희 역할을 수행하는 온디바이스 앱입니다. 초록불에서는 움직임을 허용하고, 빨간불에서는 MediaPipe Pose Landmarker 기반 움직임 점수를 계산해 기준치를 넘은 참가자를 탈락 처리합니다.

## 주요 기능

- Android Native Kotlin + Jetpack Compose UI
- CameraX 기반 실시간 카메라 프리뷰 및 프레임 분석
- MediaPipe Tasks Vision Pose Landmarker 기반 인체 키포인트 추정
- 번들형 ML Kit Korean Text Recognition 번호표/OCR 라벨 보조 인식
- 빨간불 시작 시점 자세를 baseline으로 저장하고 이후 자세 변화량으로 움직임 판정
- 정지 상태 캘리브레이션을 통한 movement threshold 보정
- 자동 Green/Red 진행, 수동 전환, 카메라 렌즈 모드 선택
- 탈락자 번호 음성 안내 및 배경 음원 재생
- 사용자가 선택한 경우 게임 시작부터 정지 시점까지 카메라 영상을 휴대폰에 저장

## 기술 스택

- Kotlin
- Android Gradle Plugin 8.5.2
- Gradle Wrapper 8.7
- Jetpack Compose
- CameraX
- Google AI Edge MediaPipe Tasks Vision
- Bundled ML Kit Korean Text Recognition
- Kotlin Coroutines

## 프로젝트 구조

```text
.
├── app/
│   └── src/main/
│       ├── assets/        # MediaPipe / ONNX / TFLite 모델 파일
│       ├── java/          # 앱 소스 코드
│       └── res/           # UI 리소스, 음성 파일, 아이콘
├── gradle/wrapper/        # Gradle Wrapper
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

## 사전 준비

다음 중 하나의 개발 환경이 필요합니다.

- Android Studio가 설치된 Windows/macOS/Linux PC
- 또는 Android SDK, JDK 17, Android Debug Bridge(`adb`)가 설치된 PC

Android 스마트폰에 설치해서 실행하려면 다음도 필요합니다.

- 카메라가 있는 Android 기기
- USB 케이블
- 스마트폰의 개발자 옵션 및 USB 디버깅 활성화

## Clone

```powershell
git clone https://github.com/hyun-uk22/Capstone_ocr-keypoints-tracking.git
cd Capstone_ocr-keypoints-tracking
```

macOS/Linux에서도 같은 명령을 사용할 수 있습니다.

```bash
git clone https://github.com/hyun-uk22/Capstone_ocr-keypoints-tracking.git
cd Capstone_ocr-keypoints-tracking
```

## Android Studio로 실행

1. Android Studio를 실행합니다.
2. `Open`을 눌러 clone한 프로젝트 폴더를 엽니다.
3. Gradle Sync가 끝날 때까지 기다립니다.
4. 스마트폰을 USB로 연결하고 USB 디버깅 권한을 허용합니다.
5. 상단 실행 대상에서 연결된 기기를 선택합니다.
6. `Run app`을 실행합니다.
7. 앱이 설치되면 카메라 권한을 허용합니다.

## CLI로 APK 빌드

Windows PowerShell:

```powershell
.\gradlew.bat assembleDebug
```

macOS/Linux:

```bash
./gradlew assembleDebug
```

빌드가 성공하면 debug APK가 생성됩니다.

```text
app/build/outputs/apk/debug/app-debug.apk
```

## 스마트폰에 APK 설치

먼저 스마트폰에서 개발자 옵션과 USB 디버깅을 켭니다.

1. 스마트폰 설정에서 `휴대전화 정보`로 이동합니다.
2. `빌드 번호`를 여러 번 눌러 개발자 옵션을 활성화합니다.
3. 개발자 옵션에서 `USB 디버깅`을 켭니다.
4. 스마트폰을 PC에 USB로 연결합니다.
5. 스마트폰 화면에 뜨는 USB 디버깅 허용 팝업을 승인합니다.

PC에서 연결 상태를 확인합니다.

```powershell
adb devices
```

기기가 `device` 상태로 보이면 APK를 설치합니다.

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

macOS/Linux:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

설치 후 스마트폰 앱 목록에서 `Freeze!`를 실행합니다.

## 앱 사용 방법

1. 앱을 실행하고 카메라 권한을 허용합니다.
2. 화면 오른쪽의 메뉴 버튼을 눌러 설정 패널을 엽니다.
3. 영상을 저장하려면 `REC OFF` 버튼을 눌러 `REC ON` 상태로 바꿉니다.
4. `START`로 자동 게임 진행을 시작합니다.
5. `REC ON` 상태에서 게임을 시작하면 녹화가 시작되고, `STOP`을 누르면 녹화가 종료됩니다.
6. 저장된 영상은 휴대폰 갤러리 또는 파일 앱의 `Movies/Freeze`에서 확인할 수 있습니다.
7. 참가자들이 가만히 서 있는 상태에서 Calibrate를 실행하면 정지 기준값이 보정됩니다.
8. 빨간불 상태에서 움직임 기준값을 넘으면 해당 참가자가 `OUT` 처리됩니다.
9. 번호표가 OCR로 반복 인식되면 track ID 대신 해당 번호가 라벨로 고정됩니다.
10. `STOP`으로 게임 상태를 초기화합니다.

## 모델 파일

현재 저장소에는 앱 실행에 필요한 모델 파일이 포함되어 있습니다.

```text
app/src/main/assets/pose_landmarker_full.task
app/src/main/assets/pose_landmarker_heavy.task
app/src/main/assets/efficientdet_lite0.tflite
app/src/main/assets/rtmpose.onnx
app/src/main/assets/yolo26n-pose.onnx
```

모델 파일이 누락되면 앱은 실행되더라도 포즈 추정 기능이 정상 동작하지 않을 수 있습니다.

OCR 모델은 앱에 번들되어 설치되므로 최초 실행 시점의 모델 다운로드가 필요하지 않습니다.

## 빌드 문제 해결

`SDK location not found` 오류가 나면 Android Studio로 프로젝트를 한 번 열어 Gradle Sync를 수행하거나, 프로젝트 루트에 `local.properties`를 만들고 본인 PC의 Android SDK 경로를 지정합니다.

Windows 예시:

```properties
sdk.dir=C\:\\Users\\사용자명\\AppData\\Local\\Android\\Sdk
```

macOS 예시:

```properties
sdk.dir=/Users/사용자명/Library/Android/sdk
```

`local.properties`는 PC마다 경로가 달라지는 로컬 설정 파일이므로 Git에 올리지 않습니다.

JDK 관련 오류가 나면 JDK 17이 설치되어 있는지 확인합니다.

```powershell
java -version
```

## 현재 한계

- 사람 ID 추적은 centroid 기반이라 참가자가 겹치거나 교차하면 ID가 바뀔 수 있습니다.
- OCR은 번호표 크기, 조명, 거리, 카메라 흔들림에 민감합니다.
- 움직임 판정 기준은 기기 성능, 카메라 거리, 조명 환경에 따라 캘리브레이션이 필요합니다.
- debug APK 기준 설치 방법을 제공합니다. 배포용 release signing 설정은 포함되어 있지 않습니다.
- 녹화 영상은 카메라 입력만 저장합니다. 앱에서 재생되는 배경음/탈락 음성은 녹화 파일에 포함되지 않습니다.

## 확인된 빌드

다음 명령으로 debug 빌드를 확인했습니다.

```powershell
.\gradlew.bat assembleDebug
```

생성 결과:

```text
app/build/outputs/apk/debug/app-debug.apk
```
