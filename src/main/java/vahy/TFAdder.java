package vahy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.Closeable;
import java.nio.DoubleBuffer;


public class TFAdder implements Closeable {

    private static final Logger logger = LoggerFactory.getLogger(TFWrapper.class.getName());
    private final Session sess;

    public TFAdder(Session sess) {
        this.sess = sess;
        logger.info("Initialized model based on TensorFlow backend.");
    }

    public double[] predict(double[] a, double[] b) {
        if(a.length != b.length) {
            throw new IllegalArgumentException("Different length");
        }
        try (
            Tensor<Double> aInput = Tensors.create(a);
            Tensor<Double> bInput = Tensors.create(b);
        ) {
            double[] outputBuffer = new double[a.length];
            DoubleBuffer doubleBuffer = DoubleBuffer.wrap(outputBuffer);

            Tensor<?> output = sess
                .runner()
                .feed("A_node", aInput)
                .feed("B_node", bInput)
                .fetch("prediction_node")
                .run()
                .get(0);

            output.writeTo(doubleBuffer);
            output.close();  // needed?
            return outputBuffer;
        }
    }

    @Override
    public void close() {
        logger.trace("Finalizing TF model resources");
        sess.close();
        logger.debug("TF resources closed");
    }
}
