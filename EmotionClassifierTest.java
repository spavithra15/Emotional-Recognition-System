package com.emotion;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EmotionClassifierTest {

    @BeforeAll
    static void loadOpenCv() {
        nu.pattern.OpenCV.loadLocally();
    }

    @Test
    void modelReturnsOneOfTheSevenEmotions() throws Exception {
        File model = new File("models/emotion-ferplus-8.onnx");
        Assumptions.assumeTrue(model.exists(), "Model not downloaded yet - test skipped");

        try (EmotionClassifier classifier = new EmotionClassifier(model.getPath())) {
            Mat fakeFace = Mat.zeros(120, 120, CvType.CV_8UC1);   // plain black image
            Prediction p = classifier.predict(fakeFace);

            assertTrue(Arrays.asList(EmotionClassifier.LABELS).contains(p.label()));
            assertTrue(p.confidence() > 0f && p.confidence() <= 1f);
        }
    }
}
