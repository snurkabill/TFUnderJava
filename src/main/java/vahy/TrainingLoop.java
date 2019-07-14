package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import vahy.timer.SimpleTimer;

import java.util.Arrays;

public class TrainingLoop {

    private static final Logger logger = LoggerFactory.getLogger(SpeedTesting.class.getName());

    public static void trainingLoop(double[][] inputData, double[][] targetData, TFModel model, double keepProbability, double learningRate) {
        SimpleTimer timer = new SimpleTimer();
        double[][] outputData = new double[inputData.length][];

        long nanoStart = System.nanoTime();
        timer.startTimer();
        for (int j = 0; j < inputData.length; j++) {
            double[] prediction = model.predict(inputData[j]);
            outputData[j] = new double[prediction.length];
            System.arraycopy(prediction, 0, outputData[j], 0, targetData[0].length);
        }
        long nanoEnd = System.nanoTime();
        timer.stopTimer();
        logger.info("Predicting [{}] samples by one took: [{}] ms. Per sample: [{}] ms. ", inputData.length, timer.getTotalTimeInNanos() / (1000.0 * 1000.0), timer.getTotalTimeInNanos() / (1000.0 * 1000.0 * inputData.length));
        logger.info("Precise: [{}] nanos per sample", (nanoEnd - nanoStart) / (double) inputData.length);

        timer.startTimer();
        double[][] outputData2 = model.predict(inputData);
        timer.stopTimer();
        logger.info("Predicting [{}] samples in batch took: [{}] ms. Per sample: [{}] ms. ", inputData.length, timer.getTotalTimeInNanos() / (1000.0 * 1000.0), timer.getTotalTimeInNanos() / (1000.0 * 1000.0 * inputData.length));

        checkPredictionDifference(targetData[0].length, outputData, outputData2);
        printFirstPredictions(outputData, 10);
        model.fit(inputData, targetData, learningRate, keepProbability);
    }

    private static void printFirstPredictions(double[][] outputData, int predictionCount) {
        for (int j = 0; j < (predictionCount < outputData.length ? predictionCount : outputData.length); j++) {
            logger.info("Prediction: [{}]", Arrays.toString(outputData[j]));
        }
    }

    private static void checkPredictionDifference(int outputDim, double[][] outputData, double[][] outputData2) {
        for (int j = 0; j < outputData.length; j++) {
            for (int k = 0; k < outputDim; k++) {
                if(Math.abs(outputData2[j][k] - outputData[j][k]) > Math.pow(10, -10)) {
                    throw new IllegalStateException("Predictions differ: [" + Arrays.toString(outputData[j]) + "] and [" +  Arrays.toString(outputData2[j]) + "]");
                }
            }
        }
    }
}
