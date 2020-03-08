package vahy;

import org.tensorflow.Tensor;

import java.nio.FloatBuffer;
import java.util.Random;

public class TestTensorCreation {

    public static void test() {
        Random r = new Random();
        int imageSize = 224 * 224 * 3;
        int batch = 128;
        long[] shape = new long[] {batch, imageSize};

        float[] array = new float[batch * imageSize];

        for (int i = 0; i < imageSize * batch; ++i) {
            array[i] = r.nextFloat();
        }
        FloatBuffer buf = FloatBuffer.allocate(imageSize * batch);
        buf.put(array);
        buf.flip();

        long start = System.nanoTime();
        Tensor.create(shape, buf);
        long end = System.nanoTime();
        System.out.println("Took: " + (end - start));
    }
}
