package com.emotion;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Map;

/**
 * Runs the FER+ CNN (emotion-ferplus-8.onnx) with ONNX Runtime.
 *
 * Model facts:
 *  - Input : float tensor [1, 1, 64, 64] = one grayscale face, pixel values 0..255 (no scaling).
 *  - Output: 8 scores in this order:
 *            neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.
 *  We keep only the first 7 (the emotions you asked for) and ignore "contempt".
 */
public class EmotionClassifier implements AutoCloseable {

    public static final String[] LABELS =
            {"Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear"};

    private static final int SIZE = 64;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public EmotionClassifier(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
        inputName = session.getInputNames().iterator().next();
    }

    /** @param grayFace a grayscale (8-bit, 1 channel) crop of one face, any size. */
    public Prediction predict(Mat grayFace) throws OrtException {
        // 1. Resize the face to 64x64.
        Mat resized = new Mat();
        Imgproc.resize(grayFace, resized, new Size(SIZE, SIZE));

        // 2. Copy the pixels into a float array (values stay 0..255).
        byte[] bytes = new byte[SIZE * SIZE];
        resized.get(0, 0, bytes);
        resized.release();
        float[] pixels = new float[SIZE * SIZE];
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = bytes[i] & 0xFF;
        }

        // 3. Run the model.
        long[] shape = {1, 1, SIZE, SIZE};
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(pixels), shape);
             OrtSession.Result result = session.run(Map.of(inputName, input))) {

            float[][] output = (float[][]) result.get(0).getValue();
            float[] scores = output[0];

            // 4. Softmax over the first 7 scores turns them into probabilities.
            float[] probs = softmax(scores, LABELS.length);
            int best = 0;
            for (int i = 1; i < probs.length; i++) {
                if (probs[i] > probs[best]) best = i;
            }
            return new Prediction(LABELS[best], probs[best]);
        }
    }

    private static float[] softmax(float[] scores, int count) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) max = Math.max(max, scores[i]);
        float sum = 0f;
        float[] result = new float[count];
        for (int i = 0; i < count; i++) {
            result[i] = (float) Math.exp(scores[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < count; i++) result[i] /= sum;
        return result;
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            System.err.println("Could not close ONNX session: " + e.getMessage());
        }
    }
}
