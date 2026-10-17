\# Intelligent Face Tracker with Auto-Registration and Visitor Counting



\## Project Overview



This project implements a computer-vision-based face tracking system that detects faces in a video, generates face embeddings, assigns visitor IDs, and records entry and exit events. It stores visitor metadata and event history in an SQLite database and saves an annotated output video.



\## Features



\- Face detection using YOLO.

\- Face embeddings and similarity matching using InsightFace.

\- Automatic visitor ID registration.

\- Visitor entry and exit event logging.

\- SQLite database for visitor and event records.

\- Cropped face image storage.

\- Annotated output video with bounding boxes and visitor IDs.

\- Configurable frame processing and similarity thresholds.



\## Technology Stack



\- Python

\- OpenCV

\- Ultralytics YOLO

\- InsightFace / ArcFace embeddings

\- NumPy

\- SQLite

\- ONNX Runtime



\## Project Structure



```text

FaceTracker/

├── config.json

├── main.py

├── main\_backup.py

├── models/

│   └── yolov8n-face.pt

├── videos/

│   └── video\_sample1.mp4

├── database/

│   └── visitors.db

└── logs/

&#x20;   ├── events.log

&#x20;   ├── annotated\_output.mp4

&#x20;   └── visitor\_faces/

```



Generated database files, logs, and output videos are produced when the application runs.



\## Setup



1\. Install Python 3.12.

2\. Create and activate a virtual environment.

3\. Install the project dependencies.

4\. Place the sample video in the `videos` directory.

5\. Ensure the YOLO face model is available in `models`.

6\. Configure the input video and thresholds in `config.json`.

7\. Run the application:



```powershell

python main.py

```



\## Configuration



The `config.json` file controls:



\- `video\_source`: input video path.

\- `yolo\_model`: YOLO face detection model path.

\- `frame\_skip`: number of frames skipped between detection runs.

\- `confidence\_threshold`: minimum YOLO confidence.

\- `similarity\_threshold`: face embedding matching threshold.

\- `max\_lost\_frames`: frame gap used to determine an exit.

\- `database\_path`: SQLite database location.

\- `face\_images\_dir`: saved visitor face images.

\- `events\_log`: event log path.



\## Outputs



After processing, the application generates:



\- `logs/annotated\_output.mp4` — processed video with visual annotations.

\- `logs/events.log` — timestamped entry and exit records.

\- `logs/visitor\_faces/` — saved face crops.

\- `database/visitors.db` — visitor identities and event records.



\## Architecture



```text

Input Video

&#x20;   |

&#x20;   v

YOLO Face Detection

&#x20;   |

&#x20;   v

Face Crop

&#x20;   |

&#x20;   v

InsightFace Embedding

&#x20;   |

&#x20;   v

Similarity Matching

&#x20;   |

&#x20;   +---- Existing Visitor

&#x20;   |

&#x20;   +---- Register New Visitor

&#x20;   |

&#x20;   v

SQLite Database + Event Logs

&#x20;   |

&#x20;   v

Annotated Output Video

```



\## Assumptions and Limitations



\- The current implementation is developed and tested using a sample video.

\- Face matching depends on image quality, lighting, face angle, and the configured similarity threshold.

\- Similarity matching may incorrectly merge or separate identities.

\- Entry and exit events are inferred from how long a face is missing from analysed frames; they are not verified physical door crossings.

\- A live RTSP camera feed requires a suitable camera stream and additional testing.

\- GPU acceleration depends on compatible hardware and installed runtime components.

\- Face images and embeddings are biometric data and should be handled securely with appropriate permission.



\## Testing



The sample video was processed successfully. The application generated an annotated video, stored visitor records in SQLite, saved face crops, and wrote event logs.



Exact visitor-count accuracy and entry/exit accuracy require further validation against labelled video.



\## Future Improvements



\- Improve multi-frame tracking and identity consistency.

\- Reduce duplicate visitor registration.

\- Add live RTSP camera support.

\- Add a dashboard for visitor statistics.

\- Add tests for database events and identity matching.

\- Improve privacy controls and data retention.



\## Hardware and Performance



Development was performed on a Windows laptop with an NVIDIA GeForce RTX 4050 Laptop GPU. Processing speed depends on video resolution, model configuration, and runtime setup. Performance benchmarks should be measured and added after testing.



\## Demos



Add your recorded project demonstration video link here.



\## Hackathon



This project is a part of a hackathon run by https://katomaran.com



