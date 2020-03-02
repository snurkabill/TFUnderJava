package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.Closeable;
import java.nio.DoubleBuffer;

public class TFWrapper implements Closeable {

    private static final Logger logger = LoggerFactory.getLogger(TFWrapper.class.getName());

    private final int inputDimension;
    private final int outputDimension;
    private final Session sess;

    private double[] outputVector;
    private DoubleBuffer doubleBuffer;
    private double[][] inputMatrixForOneVector;
    private Tensor<Double> inferenceKeepProbability = Tensors.create(1.0);

    public TFWrapper(int inputDimension, int outputDimension, Session sess) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;

        this.outputVector = new double[outputDimension];
        this.doubleBuffer = DoubleBuffer.wrap(outputVector);
        this.inputMatrixForOneVector = new double[1][inputDimension];

        this.sess = sess;
//        this.sess = new Session(graph);
//        this.sess.runner().addTarget("init").run();

        logger.info("Initialized model based on TensorFlow backend.");
        logger.debug("Model with input dimension: [{}] and output dimension: [{}].", inputDimension, outputDimension);
    }



    public double[] predict(double[] input) {
        System.arraycopy(input, 0, inputMatrixForOneVector[0], 0, inputDimension);
        try (Tensor<Double> tfInput = Tensors.create(inputMatrixForOneVector)) {
            Tensor<?> output = sess
                .runner()
                .feed("input_node", tfInput)
//                .feed("keep_prob_node", inferenceKeepProbability)
                .fetch("prediction_node")
                .run()
                .get(0);

            output.writeTo(doubleBuffer);
            doubleBuffer.position(0);
            output.close();  // needed?
            return outputVector;
        }
    }

    public double[][] predict(double[][] input) {
        try (Tensor<Double> tfInput = Tensors.create(input)) {
            double[] outputBuffer = new double[outputDimension * input.length];
            DoubleBuffer doubleBuffer = DoubleBuffer.wrap(outputBuffer);
            sess
                .runner()
                .feed("input_node", tfInput)
//                .feed("keep_prob_node", inferenceKeepProbability)
                .fetch("prediction_node")
                .run()
                .forEach(x -> {
                    x.writeTo(doubleBuffer);
                    x.close(); // needed?
                });
            double[][] outputMatrix = new double[input.length][];
            for (int i = 0; i < outputMatrix.length; i++) {
                outputMatrix[i] = new double[outputDimension];
                System.arraycopy(outputBuffer, i * outputDimension, outputMatrix[i], 0, outputDimension);
            }
            return outputMatrix;
        }
    }

    public int getInputDimension() {
        return inputDimension;
    }

    public int getOutputDimension() {
        return outputDimension;
    }

    @Override
    public void close() {
        logger.trace("Finalizing TF model resources");
        sess.close();
        inferenceKeepProbability.close();
        logger.debug("TF resources closed");
    }
}
