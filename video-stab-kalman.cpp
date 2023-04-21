//imported from google bard

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

using namespace cv;

int main() {
  // Load the video.
  VideoCapture cap("input.mp4");
  if (!cap.isOpened()) {
    cout << "Could not open the video file." << endl;
    return -1;
  }

  // Create a video writer to write the stabilized video.
  VideoWriter writer("output.mp4", VideoWriter::fourcc('M', 'J', 'P', 'G'), cap.get(CAP_PROP_FPS), cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
  if (!writer.isOpened()) {
    cout << "Could not open the video writer." << endl;
    return -1;
  }

  // Create a Kalman filter to track the motion of the camera.
  KalmanFilter kf(6, 6, 0);
  kf.transitionMatrix = (Mat_<float>(6, 6) << 1, 0, 0, 0, 1, 0,
                           0, 1, 0, 0, 0, 1,
                           0, 0, 1, 0, 0, 0,
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 1);
  kf.measurementMatrix = (Mat_<float>(2, 6) << 1, 0, 0, 0, 0, 0,
                                         0, 1, 0, 0, 0, 0);
  kf.processNoiseCov = (Mat_<float>(6, 6) << 1e-4, 0, 0, 0, 0, 0,
                                         0, 1e-4, 0, 0, 0, 0,
                                         0, 0, 1e-4, 0, 0, 0,
                                         0, 0, 0, 1e-4, 0, 0,
                                         0, 0, 0, 0, 1e-4, 0,
                                         0, 0, 0, 0, 0, 1e-4);
  kf.measurementNoiseCov = (Mat_<float>(2, 2) << 1e-2, 0,
                                            0, 1e-2);

  // Create a buffer to store the previous frames.
  Mat prevFrame(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT), CV_8UC3);

  // Iterate over the frames of the video.
  for (;;) {
    // Read the next frame.
    Mat frame;
    if (!cap.read(frame)) {
      break;
    }

    // Convert the frame to grayscale.
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Estimate the motion of the camera using the Kalman filter.
    Mat motion = kf.predict();

    // Warp the previous frame to the current frame using the estimated motion.
    Mat warpedFrame;
    warpAffine(prevFrame, warpedFrame, motion, frame.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    // Write the stabilized frame to the video writer.
    writer << warpedFrame;

    // Update the previous frame.
    prevFrame = frame;
  }

  // Close the video writer.
  writer.release();

  // Close the video capture object.
  cap.release();

  return 0;
}