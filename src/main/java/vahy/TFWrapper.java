package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import vahy.utils.ImmutableTuple;

import java.io.Closeable;
import java.nio.DoubleBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TFWrapper implements Closeable {

    private static final Logger logger = LoggerFactory.getLogger(TFWrapper.class.getName());

    private final int inputDimension;
    private final int outputDimension;
    private final Session sess;

    private final double[] singleOutputArray;
    private final DoubleBuffer singleOutputDoubleBuffer;

    private final double[][] inputMatrixForOneVector;
    private final Tensor<Double> tfSingleInput;

    private final Map<Integer, ImmutableTuple<ImmutableTuple<double[][], Tensor<Double>>, ImmutableTuple<double[], DoubleBuffer>>> inputBufferMap = new HashMap<>();

    private Tensor<Double> inferenceKeepProbability = Tensors.create(1.0);

    public TFWrapper(int inputDimension, int outputDimension, Session sess) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;

        this.singleOutputArray = new double[outputDimension];
        this.singleOutputDoubleBuffer = DoubleBuffer.wrap(singleOutputArray);
        this.inputMatrixForOneVector = new double[1][inputDimension];
        this.tfSingleInput = Tensors.create(inputMatrixForOneVector);

        this.sess = sess;

        logger.info("Initialized model based on TensorFlow backend.");
        logger.debug("Model with input dimension: [{}] and output dimension: [{}].", inputDimension, outputDimension);
    }

    public double[] predict(double[] input) {
        System.arraycopy(input, 0, inputMatrixForOneVector[0], 0, inputDimension);
        Tensor<?> output = sess
            .runner()
            .feed("input_node", tfSingleInput)
            .fetch("prediction_node")
            .run()
            .get(0);

        output.writeTo(singleOutputDoubleBuffer);
        singleOutputDoubleBuffer.position(0);
        output.close();
        return singleOutputArray;
    }

    public double[][] predict(double[][] input) {

        var length = input.length;
        if (!inputBufferMap.containsKey(length)) {
            var inputArray = new double[input.length][inputDimension];
            var outputArray = new double[outputDimension * input.length];
            var inputTensor = Tensors.create(inputArray);
            var buffer = DoubleBuffer.wrap(outputArray);
            var inputTuple = new ImmutableTuple<>(inputArray, inputTensor);
            var outputTuple = new ImmutableTuple<>(outputArray, buffer);
            inputBufferMap.put(length, new ImmutableTuple<>(inputTuple, outputTuple));
        }
        var buffers = inputBufferMap.get(length);
        var inputArray = buffers.getFirst().getFirst();
        var inputTensor = buffers.getFirst().getSecond();
        var outputArray = buffers.getSecond().getFirst();
        var outputBuffer = buffers.getSecond().getSecond();

        for (int i = 0; i < input.length; i++) {
            System.arraycopy(input[i], 0, inputArray[i], 0, inputDimension);
        }

        Session.Runner runner = sess.runner();
        runner = runner.feed("input_node", Tensor.create(inputArray));
        runner = runner.fetch("prediction_node");
        List<Tensor<?>> output = runner.run();

        for (Tensor<?> tensor : output) {
            tensor.writeTo(outputBuffer);
            tensor.close();
        }

//
//        sess
//            .runner()
//            .feed("input_node", inputTensor)
////                .feed("keep_prob_node", inferenceKeepProbability)
//            .fetch("prediction_node")
//            .run()
//            .forEach(x -> {
//                x.writeTo(outputBuffer);
//                x.close();
//            });
        double[][] outputMatrix = new double[input.length][];
        for (int i = 0; i < outputMatrix.length; i++) {
            outputMatrix[i] = new double[outputDimension];
            System.arraycopy(outputArray, i * outputDimension, outputMatrix[i], 0, outputDimension);
        }
        outputBuffer.clear();
        return outputMatrix;
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
