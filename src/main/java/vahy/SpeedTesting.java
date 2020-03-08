package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.SplittableRandom;

public class SpeedTesting {

    private static final Logger logger = LoggerFactory.getLogger(SpeedTesting.class.getName());

    public static void main(String[] args) {

        SplittableRandom random = new SplittableRandom(0);

        int instanceCount = 100_000;
        int batchSize = 1024;
        int inputDim = 20;
        int outputDim = 5;
        int trainingIterations = 10;

        double[][] inputData = new double[instanceCount][];
        double[][] targetData = new double[instanceCount][];

        for (int i = 0; i < instanceCount; i++) {
            double[] input = new double[inputDim];
            double[] target = new double[outputDim];
            for (int j = 0; j < inputDim; j++) {
                input[j] = random.nextDouble() - 0.5;
            }
            for (int j = 0; j < outputDim; j++) {
                target[j] = random.nextDouble();
            }
            inputData[i] = input;
            targetData[i] = target;
        }

        try(TFModelImproved model = new TFModelImproved(inputDim, outputDim, batchSize, trainingIterations, SpeedTesting.class.getClassLoader().getResourceAsStream("tfModel/graph_FastTF.pb").readAllBytes(), 1, random))
        {
            for (int i = 0; i < 100; i++) {
                TrainingLoop.trainingLoop(inputData, targetData, model, 1, 0.01);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
