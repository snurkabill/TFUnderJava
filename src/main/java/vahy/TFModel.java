package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import vahy.timer.SimpleTimer;

import java.util.SplittableRandom;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.stream.IntStream;

public class TFModel implements Model, AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(TFModel.class.getName());

    private final BlockingQueue<TFWrapper> pool;

    private final SimpleTimer timer = new SimpleTimer();
    private final SplittableRandom random;
    private final int inputDimension;
    private final int outputDimension;
    private final int trainingIterations;
    private final int batchSize;
    private final double[][] trainInputBatch;
    private final double[][] trainTargetBatch;
    private final Graph commonGraph;
    private final Session trainingSession;

    public TFModel(int inputDimension, int outputDimension, int batchSize, int trainingIterations, byte[] bytes, int poolSize, SplittableRandom random) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.batchSize = batchSize;
        this.random = random;
        this.trainingIterations = trainingIterations;
        this.trainInputBatch = new double[batchSize][];
        this.trainTargetBatch = new double[batchSize][];
        for (int i = 0; i < batchSize; i++) {
            trainInputBatch[i] = new double[inputDimension];
            trainTargetBatch[i] = new double[outputDimension];
        }

        this.commonGraph = new Graph();
        this.commonGraph.importGraphDef(bytes);
        this.trainingSession = new Session(commonGraph);
        this.trainingSession.runner().addTarget("init").run();

        this.pool = new ArrayBlockingQueue<>(poolSize, true);
        for (int i = 0; i < poolSize; i++) {
            this.pool.add(new TFWrapper(inputDimension, outputDimension, trainingSession));
        }

    }

    @Override
    public double[] predict(double[] input) {
        try {
            var tfWrapper = pool.take();
            double[] prediction = tfWrapper.predict(input);
            pool.add(tfWrapper);
            return prediction;
        } catch (InterruptedException e) {
            throw new IllegalStateException("Model prediction was interrupted.", e);
        }
    }

    @Override
    public double[][] predict(double[][] input) {
        try {
            var tfWrapper = pool.take();
            double[][] prediction = tfWrapper.predict(input);
            pool.add(tfWrapper);
            return prediction;
        } catch (InterruptedException e) {
            throw new IllegalStateException("Model prediction was interrupted.", e);
        }
    }

    @Override
    public void fit(double[][] input, double[][] target, double learningRate, double keepProbability) {
        if(input.length != target.length) {
            throw new IllegalArgumentException("Input and target lengths differ");
        }
        logger.debug("Partially fitting TF model on [{}] inputs.", input.length);
        timer.startTimer();
        int[] order = IntStream.range(0, input.length).toArray();
        for (int i = 0; i < trainingIterations; i++) {
            shuffleArray(order, random);
            for (int j = 0; j < (target.length / batchSize) + 1; j++) {
                fillBatch(j, order, input, target);
                try (
                    Tensor<Double> tfInput = Tensors.create(this.trainInputBatch);
                    Tensor<Double> tfTarget = Tensors.create(this.trainTargetBatch)
                ) {
                    trainingSession
                        .runner()
                        .feed("input_node", tfInput)
                        .feed("target_node", tfTarget)
                        .addTarget("optimize_node")
                        .run();
                }
            }
        }
        timer.stopTimer();
        logger.debug("Training of [{}] inputs with minibatch size [{}] took [{}] milliseconds. Samples per sec: [{}]",
            input.length, batchSize, timer.getTotalTimeInMillis() / 1000.0, timer.samplesPerSec(input.length));
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

    private static void shuffleArray(int[] array, SplittableRandom rng) {
        for(int i = array.length - 1; i > 0; --i) {
            int j = rng.nextInt(i + 1);
            int temp = array[j];
            array[j] = array[i];
            array[i] = temp;
        }
    }

    @Override
    public void close() throws Exception {
        for (int i = 0; i < pool.size(); i++) {
            pool.take().close();
        }
        this.trainingSession.close();
    }
}
