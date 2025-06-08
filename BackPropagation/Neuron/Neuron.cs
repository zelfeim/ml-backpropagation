using BackPropagation.ActivationFunction;

namespace BackPropagation.Neuron;

public class Neuron : INeuron
{
    public Neuron(int weightCount)
    {
        Weights = new List<double>(weightCount);

        InitializeWeights();
    }

    public Neuron(List<double> weights, double bias = 0)
    {
        Weights = weights;
        Bias = bias;
    }

    public double WeightedInput { get; set; }
    public List<double> Weights { get; set; }
    public double Bias { get; set; }
    public double Delta { get; set; }
    public double Activation { get; set; }

    public double Calculate(List<double> inputs, IActivationFunction activationFunction)
    {
        var result = inputs.Select((t, i) => t * Weights[i]).Sum() + Bias;
        WeightedInput = result;
        Activation = activationFunction.Function(result);

        return Activation;
    }

    private void InitializeWeights()
    {
        var random = new Random();

        for (var i = 0; i < Weights.Count; i++)
            Weights[i] = random.NextDouble() * 1;

        Bias = random.NextDouble() * 1;
    }
}
