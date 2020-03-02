package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.SplittableRandom;

public class SpeedTesting {

    private static final Logger logger = LoggerFactory.getLogger(SpeedTesting.class.getName());

    public static void main(String[] args) {

        SplittableRandom random = new SplittableRandom(0);

        int instanceCount = 10000;
        int batchSize = instanceCount / 100;
        int inputDim = 20;
        int outputDim = 5;

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

        try(TFModel model = new TFModel(inputDim, outputDim, 1, batchSize, SpeedTesting.class.getClassLoader().getResourceAsStream("tfModel/graph_FastTF.pb").readAllBytes(), 1, random))
        {
            TrainingLoop.trainingLoop(inputData, targetData, model, 1, 0.01);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
