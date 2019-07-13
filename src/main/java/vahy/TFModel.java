package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import vahy.timer.SimpleTimer;

import java.io.Closeable;
import java.nio.DoubleBuffer;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

public class TFModel implements Closeable {

    private static final Logger logger = LoggerFactory.getLogger(TFModel.class.getName());

    private final int inputDimension;
    private final int outputDimension;
    private final int trainingIterations;
    private final int batchSize;
    private final SplittableRandom random;
    private final Session sess;
    private final double[][] trainInputBatch;
    private final double[][] trainTargetBatch;
    private final SimpleTimer timer = new SimpleTimer();

    private double[] outputVector;
    private DoubleBuffer doubleBuffer;
    private double[][] inputMatrixForOneVector;
    private Tensor<Double> inferenceKeepProbability = Tensors.create(1.0);

    public TFModel(int inputDimension, int outputDimension, int trainingIterations, int batchSize, byte[] bytes, SplittableRandom random) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.trainingIterations = trainingIterations;
        this.batchSize = batchSize;
        this.random = random;
        this.trainInputBatch = new double[batchSize][];
        this.trainTargetBatch = new double[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            trainInputBatch[i] = new double[inputDimension];
            trainTargetBatch[i] = new double[outputDimension];
        }

        this.outputVector = new double[outputDimension];
        this.doubleBuffer = DoubleBuffer.wrap(outputVector);
        this.inputMatrixForOneVector = new double[1][inputDimension];

        Graph graph = new Graph();
        this.sess = new Session(graph);
        graph.importGraphDef(bytes);
        this.sess.runner().addTarget("init").run();

        logger.info("Initialized model based on TensorFlow backend.");
        logger.debug("Model with input dimension: [{}] and output dimension: [{}]. Batch size of model set to: [{}]", inputDimension, outputDimension, batchSize);
    }

    private void fillBatch(int batchesDone, int[] order, double[][] input, double[][] target) {
        logger.trace("Filling data batch. Already done batches: [{}]", batchesDone);
        for (int i = 0; i < batchSize; i++) {
            int index = batchesDone * batchSize + i;
            if(index >= order.length) {
                break; // leaving part of batch from previous iteration.
            }
            System.arraycopy(input[order[index]], 0, trainInputBatch[i], 0, inputDimension);
            System.arraycopy(target[order[index]], 0, trainTargetBatch[i], 0, outputDimension);
        }
    }

    public void fit(double[][] input, double[][] target, double learningRate, double keepProbability) {
        if(input.length != target.length) {
            throw new IllegalArgumentException("Input and target lengths differ");
        }
        logger.debug("Partially fitting TF model on [{}] inputs.", input.length);
        timer.startTimer();
        int[] order = IntStream.range(0, input.length).toArray();
        for (int i = 0; i < trainingIterations; i++) {
            shuffleArray(order, new Random(random.nextInt()));
            for (int j = 0; j < (target.length / batchSize) + 1; j++) {
                fillBatch(j, order, input, target);
                try (
                    Tensor<Double> tfInput = Tensors.create(this.trainInputBatch);
                    Tensor<Double> tfTarget = Tensors.create(this.trainTargetBatch);
                    Tensor<Double> tfLearningRate = Tensors.create(learningRate);
                    Tensor<Double> tfKeepProbability = Tensors.create(keepProbability);
                ) {
                    sess
                        .runner()
                        .feed("input_node", tfInput)
                        .feed("target_node", tfTarget)
                        .feed("learning_rate_node", tfLearningRate)
                        .feed("keep_prob_node", tfKeepProbability)
                        .addTarget("optimize_node")
                        .run();
                }
            }
        }
        timer.stopTimer();
        logger.debug("Training of [{}] inputs with minibatch size [{}] took [{}] milliseconds. Samples per sec: [{}]",
            input.length, batchSize, timer.getTotalTimeInMillis() / 1000.0, timer.samplesPerSec(input.length));
    }

    private static void shuffleArray(int[] array, Random rng) {
        for(int i = array.length - 1; i > 0; --i) {
            int j = rng.nextInt(i + 1);
            int temp = array[j];
            array[j] = array[i];
            array[i] = temp;
        }
    }

    public double[] predict(double[] input) {
        System.arraycopy(input, 0, inputMatrixForOneVector[0], 0, inputDimension);
        try (Tensor<Double> tfInput = Tensors.create(inputMatrixForOneVector)) {
            Tensor<?> output = sess
                .runner()
                .feed("input_node", tfInput)
                .feed("keep_prob_node", inferenceKeepProbability)
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
                .feed("keep_prob_node", inferenceKeepProbability)
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
