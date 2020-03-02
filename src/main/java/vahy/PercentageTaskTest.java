package vahy;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class PercentageTaskTest {

    public static void main(String[] args) {

        var pool = Executors.newFixedThreadPool(1);
        var taskCount = 100;
        var doneRatio = 1.0;
        var callableList = new ArrayList<Callable<String>>();
        var nextTryMillis = 100;

        for (int i = 1; i <= taskCount; i++) {
            int finalI = i;
            callableList.add(() -> {
                long millis = 1000 * finalI;
                int counter = 0;
                for (long j = 0; j < millis * 100_000; j++) {
                    counter++;
                }
                return millis + "_" + counter;
            });
        }

        var futureList = callableList.stream().map(pool::submit).collect(Collectors.toList());

        while((futureList.stream().filter(Future::isDone).count() / (double) taskCount) < doneRatio) {
            try {
                Thread.sleep(nextTryMillis);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        pool.shutdownNow();
        try {
            pool.awaitTermination(1, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        System.out.println("Tasks done: ");
        futureList.stream().filter(Future::isDone).forEach(x -> {
            try {
                System.out.println(x.get());
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        });


    }

}
