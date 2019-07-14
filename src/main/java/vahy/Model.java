package vahy;

public interface Model {

    double[] predict(double[] input);

    double[][] predict(double[][] input);

    void fit(double[][] input, double[][] target, double learningRate, double keepProbability);

}
