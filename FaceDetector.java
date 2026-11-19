package com.emotion;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;

/** Finds human faces in an image using OpenCV's Haar Cascade detector. */
public class FaceDetector {

    private final CascadeClassifier cascade;

    public FaceDetector(String cascadePath) {
        if (!new File(cascadePath).exists()) {
            throw new IllegalStateException("Haar cascade file not found: " + cascadePath);
        }
        cascade = new CascadeClassifier(cascadePath);
        if (cascade.empty()) {
            throw new IllegalStateException("Could not load Haar cascade: " + cascadePath);
        }
    }

    /**
     * @param grayImage a grayscale image (ideally histogram-equalized)
     * @return rectangles, one per face found (empty array if none)
     */
    public Rect[] detect(Mat grayImage) {
        MatOfRect faces = new MatOfRect();
        // scaleFactor 1.1 = look at several sizes; minNeighbors 5 = fewer false alarms;
        // minSize 80x80 = ignore tiny detections.
        cascade.detectMultiScale(grayImage, faces, 1.1, 5, 0, new Size(80, 80), new Size());
        Rect[] result = faces.toArray();
        faces.release();
        return result;
    }
}
