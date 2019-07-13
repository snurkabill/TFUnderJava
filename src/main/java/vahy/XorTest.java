package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.SplittableRandom;

public class XorTest {

    private static final Logger logger = LoggerFactory.getLogger(XorTest.class.getName());

    public static void main(String[] args) {

        SplittableRandom random = new SplittableRandom(0);

        int batchSize = 1;
        int inputDim = 2;
        int outputDim = 1;

        double[][] inputData = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        double[][] targetData =  { {-1}, {1}, {1}, {-1}};

        try(TFModel model = new TFModel(inputDim, outputDim, 1, batchSize, XorTest.class.getClassLoader().getResourceAsStream("tfModel/graph_XorNetwork.pb").readAllBytes(), random))
        {
            for (int i = 0; i < 100; i++) {
                TrainingLoop.trainingLoop(inputData, targetData, model, 0.5, 0.01);
            }
            double[][] output = model.predict(inputData);
            if(output[0][0] >= 0 || output[1][0] <= 0 || output[2][0] <= 0 || output[3][0] >= 0) {
                logger.error("Xor test failed");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
