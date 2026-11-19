package com.emotion;

/** Result of one emotion prediction: the emotion name and how sure the model is (0.0 to 1.0). */
public record Prediction(String label, float confidence) {
}
