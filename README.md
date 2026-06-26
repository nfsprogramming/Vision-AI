# Smart Attendance System (Lite Version)


## Features
- **Live Face Detection**: Continuously monitors the webcam.
- **Fast Recognition**: Uses simple geometric matching (Euclidean Distance).
- **Zero Heavy Dependencies**: Runs with just `opencv-python` and `numpy`.
- **Automatic Attendance**: Marks `attendance.csv` correctly.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install opencv-python --no-deps pandas numpy
   ```
   *(We use `--no-deps` to avoid version conflicts with your newer Python/Numpy).*

2. **Add Images**:
   - Put clear face images in the `images/` folder.
   - Name them normally (e.g., `Mohamed.jpg`).

3. **Run the System**:
   ```bash
   python simple_attendance.py
   ```

## Troubleshooting
- If you see `ModuleNotFoundError`, ask the assistant to help reinstall `opencv-python`.
- For the "Pro" version (using Deep Learning/dlib), please downgrade to **Python 3.12** and install **Visual Studio C++ Build Tools**.

## How it works
This version detects faces using standard Haar Cascades and recognizes them by direct image comparison (Mean Squared Error). It is less robust than Deep Learning but works great for demonstrations and simple projects!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
