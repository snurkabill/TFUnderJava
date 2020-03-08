package vahy.speedBenchmark;

import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.RunnerException;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import vahy.TFAdder;

import java.io.IOException;
import java.util.Map;
import java.util.Properties;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = {"-Xms4G", "-Xmx4G"})
@Warmup(iterations = 2)
@Measurement(iterations = 5)
public class AddedBenchmark {

    @Param({"100"})
    private int instanceCount;

    private double[] A_DATA;
    private double[] B_DATA;

    private TFAdder adder;

    public static void main(String[] args) throws RunnerException {

//        myTest();
        runTests();

//        Options opt = new OptionsBuilder()
//            .include(AddedBenchmark.class.getSimpleName())
//            .build();
//
//        new Runner(opt).run();
    }

    public static void runTests() throws RunnerException {
        for (int i = 1; i < 100_000_000; i*= 2) {
            myTest(i);
        }
    }

    public static void myTest2(int count) {
//        var count = 10_000_000;
        var A = new double[count];
        var B = new double[count];
        var D = new double[count];
        for (int i = 0; i < count; i++) {
            A[i] = i;
            B[i] = i;
        }
        long start2 = System.nanoTime();
        for (int i = 0; i < count; i++) {
            D[i] = A[i] + B[i];
        }
        long end2 = System.nanoTime();
        long total2 = end2 - start2;
        System.out.println("Total nanoseconds: [" + total2 + "].  Samples: [" + count + "]. Per sample: [" + total2 / (double) count + "]");
    }

    public static void myTest(int count) throws RunnerException {

//        var count = 10_000_000;
        byte[] bytes = new byte[0];
        try {
            bytes = PredictionLatencyBenchmark.class.getClassLoader().getResourceAsStream("tfModel/graph_AddingTwoNumbers4.pb").readAllBytes();
        } catch (IOException e) {
            throw new RunnerException(e);
        }

        var commonGraph = new Graph();
        commonGraph.importGraphDef(bytes);
        var trainingSession = new Session(commonGraph);
        trainingSession.runner().addTarget("init").run();
//        trainingSession.runner().addTarget("finalize").run();
        var model = new TFAdder(trainingSession);


        var A = new double[count];
        var B = new double[count];
        var C = new double[count];
        var D = new double[count];

        for (int i = 0; i < count; i++) {
            A[i] = i;
            B[i] = i;
        }

        var bufferSize = Math.min((int)Math.pow(2, 15), count);
//        var bufferSize = Math.min(1, count);
        var A_buffer = new double[bufferSize];
        var B_buffer = new double[bufferSize];
        var C_buffer = new double[bufferSize];

        long start = System.nanoTime();

        long justCalls = 0;
        int callCount = 0;
        for (int i = 0; i < count; i += bufferSize) {
            System.arraycopy(A, i, A_buffer, 0, bufferSize);
            System.arraycopy(B, i, B_buffer, 0, bufferSize);
            long startCall = System.currentTimeMillis();
            C_buffer = model.predict(A_buffer, B_buffer);
            long endCall = System.currentTimeMillis();
            callCount++;
            justCalls += endCall - startCall;
            System.arraycopy(C_buffer, 0, C, i, bufferSize);
        }

        long end = System.nanoTime();
        long total = end - start;
        long start2 = System.nanoTime();
        for (int i = 0; i < count; i++) {
            D[i] = A[i] + B[i];
        }
        long end2 = System.nanoTime();
        long total2 = end2 - start2;
        System.out.println("Samples: [" + count + "]. Naive per sample: [" + total2 / (double) count + "] in ns. TF per sample: [" + total / (double) count + "]. Buffer size: [" + bufferSize + "]. Per call: [" + justCalls / (double) callCount + "] in ms");
        for (int i = 0; i < count; i++) {
            if(Math.abs(C[i] - D[i]) > Math.pow(10, -15)) {
                throw new IllegalStateException("Different by: [" + Math.abs(C[i] - D[i]) + "]");
            }
        }

    }


    @Setup
    public void setup() throws IOException {
        SplittableRandom random = new SplittableRandom(0);
        A_DATA = createInputData(random);
        B_DATA = createInputData(random);

        var bytes = PredictionLatencyBenchmark.class.getClassLoader().getResourceAsStream("tfModel/graph_AddingTwoNumbers.pb").readAllBytes();

        var commonGraph = new Graph();
        commonGraph.importGraphDef(bytes);
        var trainingSession = new Session(commonGraph);
        trainingSession.runner().addTarget("init").run();
        adder = new TFAdder(trainingSession);
    }

//    @Benchmark
//    public void benchmark(Blackhole bh) {
//        for (int i = 0; i < A_DATA.length; i++) {
////            System.out.println(adder.predict(A_DATA[i], B_DATA[i]));
//            bh.consume(adder.predict(A_DATA[i], B_DATA[i]));
//        }
//    }

    private double[] createInputData(SplittableRandom random) {
        double[] inputData = new double[instanceCount];
        for (int i = 0; i < instanceCount; i++) {
            inputData[i] = random.nextDouble() - 0.5;
//            inputData[i] = i;
        }
        return inputData;
    }

}
