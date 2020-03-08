package vahy.speedBenchmark;


import org.openjdk.jmh.annotations.Benchmark;
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
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import vahy.Model;
import vahy.TFModelImproved;
import vahy.TFModelWithArgs;

import java.io.IOException;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(value = 3, jvmArgs = {"-Xms4G", "-Xmx4G"})
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class PredictionLatencyBenchmark {

    @Param({"10"})
    private int instanceCount;

    @Param({"WithoutArgs", "WithArgs"})
    private String implementation;

    private double[][] INPUT_DATA;
    private double[][] TARGET_DATA;
    private Model model;

    public static void main(String[] args) throws RunnerException {

        Options opt = new OptionsBuilder()
            .include(PredictionLatencyBenchmark.class.getSimpleName())
            .build();

        new Runner(opt).run();
    }

    @Setup
    public void setup() throws IOException {
        SplittableRandom random = new SplittableRandom(0);
        INPUT_DATA = createInputData(random);
        TARGET_DATA = createTargetData(random);
        model = implementation.equals("WithArgs") ?
            new TFModelWithArgs(1, 3, 1, 1, PredictionLatencyBenchmark.class.getClassLoader().getResourceAsStream("tfModel/graph_MinimalNetworkForCallTestSpeed_withArgs.pb").readAllBytes(), random) :
            new TFModelImproved(1, 3, 1, 1, PredictionLatencyBenchmark.class.getClassLoader().getResourceAsStream("tfModel/graph_MinimalNetworkForCallTestSpeed.pb").readAllBytes(), 1, random);
        model.fit(INPUT_DATA, TARGET_DATA, 0.01, 1.0);
    }

    @Benchmark
    public void benchmark(Blackhole bh) {
        for (int i = 0; i < INPUT_DATA.length; i++) {
            bh.consume(model.predict(INPUT_DATA[i]));
        }
    }

    private double[][] createInputData(SplittableRandom random) {
        int inputDim = 1;
        double[][] inputData = new double[instanceCount][];
        for (int i = 0; i < instanceCount; i++) {
            double[] input = new double[inputDim];
            for (int j = 0; j < inputDim; j++) {
                input[j] = random.nextDouble() - 0.5;
            }
            inputData[i] = input;
        }
        return inputData;
    }


    private double[][] createTargetData(SplittableRandom random) {
        int outputDim = 3;
        double[][] targetData = new double[instanceCount][];
        for (int i = 0; i < instanceCount; i++) {
            double[] target = new double[outputDim];
            for (int j = 0; j < outputDim; j++) {
                target[j] = random.nextDouble();
            }
            targetData[i] = target;
        }
        return targetData;
    }
}
