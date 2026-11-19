package com.emotion;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/** The application window: shows the webcam video, face rectangles and detected emotions. */
public class CameraWindow extends JFrame {

    private static final int CAMERA_INDEX = 0;   // 0 = default webcam. Try 1 if it does not open.
    private static final Scalar GREEN = new Scalar(0, 255, 0);

    private final JLabel videoLabel = new JLabel();
    private final JLabel statusLabel = new JLabel("Starting camera...", SwingConstants.CENTER);
    private final FaceDetector detector;
    private final EmotionClassifier classifier;
    private volatile boolean running = false;

    public CameraWindow(FaceDetector detector, EmotionClassifier classifier) {
        super("Emotion Recognition System");
        this.detector = detector;
        this.classifier = classifier;

        videoLabel.setPreferredSize(new Dimension(640, 480));
        videoLabel.setHorizontalAlignment(SwingConstants.CENTER);
        statusLabel.setFont(new Font("SansSerif", Font.BOLD, 22));
        statusLabel.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JButton stopButton = new JButton("Stop Camera & Exit");
        stopButton.addActionListener(e -> stopCamera());

        setLayout(new BorderLayout());
        add(statusLabel, BorderLayout.NORTH);
        add(videoLabel, BorderLayout.CENTER);
        add(stopButton, BorderLayout.SOUTH);

        // Closing the window with the X button also stops the camera safely.
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                stopCamera();
            }
        });
        pack();
        setLocationRelativeTo(null);
    }

    /** Shows the window and starts the camera thread. */
    public void start() {
        setVisible(true);
        running = true;
        Thread worker = new Thread(this::captureLoop, "camera-thread");
        worker.start();
    }

    /** Asks the camera thread to stop. The thread cleans up and closes the app. */
    private void stopCamera() {
        running = false;
        statusLabel.setText("Stopping...");
    }

    private void captureLoop() {
        VideoCapture camera = new VideoCapture(CAMERA_INDEX);
        try {
            if (!camera.isOpened()) {
                showError("Cannot open the webcam.\nClose other apps that use the camera, "
                        + "check camera permission, or change CAMERA_INDEX.");
                return;
            }

            Mat frame = new Mat();
            Mat gray = new Mat();
            Mat equalized = new Mat();
            int failedReads = 0;

            while (running) {
                if (!camera.read(frame) || frame.empty()) {
                    if (++failedReads > 100) {
                        showError("The webcam stopped sending images.");
                        return;
                    }
                    Thread.sleep(30);
                    continue;
                }
                failedReads = 0;

                Core.flip(frame, frame, 1);                           // mirror view
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
                Imgproc.equalizeHist(gray, equalized);                // helps detection in poor light

                Rect[] faces = detector.detect(equalized);

                // Find the biggest face (the main person).
                Rect largest = null;
                for (Rect f : faces) {
                    if (largest == null || f.area() > largest.area()) largest = f;
                }

                String headline = "No face detected";
                for (Rect f : faces) {
                    Mat faceGray = new Mat(gray, f);                  // crop the face
                    Prediction p = classifier.predict(faceGray);
                    faceGray.release();

                    String text = String.format("%s %.0f%%", p.label(), p.confidence() * 100);
                    Imgproc.rectangle(frame, f.tl(), f.br(), GREEN, 2);
                    Imgproc.putText(frame, text, new Point(f.x, Math.max(f.y - 10, 25)),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2);
                    if (f == largest) headline = "Emotion: " + text;
                }

                BufferedImage image = toBufferedImage(frame);
                final String status = headline;
                SwingUtilities.invokeLater(() -> {
                    videoLabel.setIcon(new ImageIcon(image));
                    statusLabel.setText(status);
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
            showError("Unexpected error:\n" + e);
        } finally {
            camera.release();          // free the webcam
            classifier.close();        // free the model
            SwingUtilities.invokeLater(() -> {
                dispose();
                System.exit(0);
            });
        }
    }

    /** Converts an OpenCV BGR image into a Java image. */
    private static BufferedImage toBufferedImage(Mat bgr) {
        BufferedImage image = new BufferedImage(bgr.cols(), bgr.rows(), BufferedImage.TYPE_3BYTE_BGR);
        byte[] target = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        bgr.get(0, 0, target);
        return image;
    }

    /** Shows an error box and waits until the user clicks OK (called from the camera thread). */
    private void showError(String message) {
        System.err.println(message);
        running = false;
        try {
            SwingUtilities.invokeAndWait(() ->
                    JOptionPane.showMessageDialog(this, message, "Error", JOptionPane.ERROR_MESSAGE));
        } catch (Exception ignored) {
            // nothing more we can do
        }
    }
}
