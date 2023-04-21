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

    // Calculate the optical flow between the previous frame and the current frame.
    Mat flow = calcOpticalFlowFarneback(prevFrame, frame, 0.5, 3, 15, 3, 5, 1.2, 0);

    // Create a transformation matrix that maps the pixels from the previous frame to the current frame.
    Mat transform = getAffineTransform(flow);

    // Warp the previous frame to the current frame using the transformation matrix.
    Mat warpedFrame;
    warpAffine(prevFrame, warpedFrame, transform, frame.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

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