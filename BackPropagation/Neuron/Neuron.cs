using BackPropagation.ActivationFunction;

namespace BackPropagation.Neuron;

public class Neuron : INeuron
{
    private double _bias;

    public Neuron(int weightCount)
    {
        Weights = new List<double>(weightCount);

        InitializeWeights();
    }

    public Neuron(List<double> weights, double bias = 0)
    {
        Weights = weights;
        _bias = bias;
    }

    public List<double> Weights { get; set; }
    public double Activation { get; set; }

    public double Calculate(List<double> inputs, IActivationFunction _activationFunction)
    {
        var result = inputs.Select((t, i) => t * Weights[i]).Sum() + _bias;
        Activation = _activationFunction.Function(result);
        return Activation;
    }

    private void InitializeWeights()
    {
        var random = new Random();

        for (var i = 0; i < Weights.Count; i++)
            Weights[i] = random.NextDouble() * 1;

        _bias = random.NextDouble() * 1;
    }
}
