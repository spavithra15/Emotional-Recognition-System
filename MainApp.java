package com.emotion;

import javax.swing.*;
import java.io.File;

/** Program entry point: checks files, loads libraries, opens the window. */
public class MainApp {

    private static final String MODEL_PATH = "models/emotion-ferplus-8.onnx";
    private static final String CASCADE_PATH = "models/haarcascade_frontalface_default.xml";

    public static void main(String[] args) {
        // Step 1: load the OpenCV native library (included in the Maven dependency).
        try {
            nu.pattern.OpenCV.loadLocally();
        } catch (Throwable t) {
            fatal("Could not load the OpenCV native library:\n" + t);
            return;
        }

        // Step 2: make sure the two downloaded files exist.
        if (!new File(MODEL_PATH).exists()) {
            fatal("Model file not found:\n" + new File(MODEL_PATH).getAbsolutePath()
                    + "\n\nDownload emotion-ferplus-8.onnx into the 'models' folder.");
            return;
        }
        if (!new File(CASCADE_PATH).exists()) {
            fatal("Face detector file not found:\n" + new File(CASCADE_PATH).getAbsolutePath()
                    + "\n\nDownload haarcascade_frontalface_default.xml into the 'models' folder.");
            return;
        }

        // Step 3: create the detector and the CNN, then open the window.
        try {
            FaceDetector detector = new FaceDetector(CASCADE_PATH);
            EmotionClassifier classifier = new EmotionClassifier(MODEL_PATH);
            SwingUtilities.invokeLater(() -> new CameraWindow(detector, classifier).start());
        } catch (Exception e) {
            e.printStackTrace();
            fatal("Could not start the application:\n" + e);
        }
    }

    private static void fatal(String message) {
        System.err.println(message);
        JOptionPane.showMessageDialog(null, message, "Emotion Recognition - Error",
                JOptionPane.ERROR_MESSAGE);
    }
}
