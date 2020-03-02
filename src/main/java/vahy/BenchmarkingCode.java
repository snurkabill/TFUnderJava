package vahy;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import vahy.speedBenchmark.AddedBenchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OperationsPerInvocation(BenchmarkingCode.N)
@Warmup(iterations = 5, time = 5)
@Measurement(iterations = 5, time = 5)
public class BenchmarkingCode {
    public static final int N = 10;

    static List<Integer> sourceList = new ArrayList<>();

    static Queue<Integer> queue = new LinkedList<>();
    static List<Integer> newElementList = new ArrayList<>();

    @Setup
    public static void setup() {
        for (int i = 0; i < N; i++) {
            queue.add(i);
            newElementList.add(N + i);
        }
        Collections.shuffle(newElementList, new Random(0));
    }

    @TearDown
    public static void tearDown() {
        for (int i = 0; i < N; i++) {
            queue.add(i);
            newElementList.add(N + i);
        }
        Collections.shuffle(newElementList, new Random(0));
    }

//    @Setup
//    public static void setup() {
//        for (int i = 0; i < N; i++) {
//            sourceList.add(i);
//        }
//    }

    @Benchmark
    public void iterator(Blackhole bh) {
        for (Integer integer : newElementList) {
            queue.add(integer);
        }
        System.out.println(queue.size());
    }

//    @Benchmark
//    public void streamForEach(Blackhole bh) {
////        List<Double> result = new ArrayList<>(sourceList.size() / 2 + 1);
////        for (Integer i : sourceList) {
////            if (i % 2 == 0){
////                result.add(Math.sqrt(i));
////            }
////        }
////        bh.consume(result);
//        newElementList.stream().filter(x -> x % 2 == 0).forEach(queue::add);
//    }
//
//    @Benchmark
//    public void streamForEachOrdered(Blackhole bh) {
//        newElementList.stream().filter(x -> x % 2 == 0).forEachOrdered(queue::add);
//    }
//
//    @Benchmark
//    public void streamCollect(Blackhole bh) {
//        bh.consume(queue.addAll(newElementList.stream().filter(x -> x % 2 == 0).collect(Collectors.toList())));
////        bh.consume(sourceList.stream()
////            .filter(i -> i % 2 == 0)
////            .map(Math::sqrt)
////            .collect(Collectors.toCollection(
////                () -> new ArrayList<>(sourceList.size() / 2 + 1))));
//    }
}