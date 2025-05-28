using BackPropagation.ActivationFunction;

namespace BackPropagation.NeuralNetwork;

public class NeuralNetwork : INeuralNetwork
{
    private readonly IActivationFunction _activationFunction;

    public NeuralNetwork(List<int> networkStructure, IActivationFunction activationFunction)
    {
        _activationFunction = activationFunction;

        InitializeLayers(networkStructure);
    }

    public NeuralNetwork(List<Layer> layers, IActivationFunction activationFunction)
    {
        Layers = layers;
        _activationFunction = activationFunction;
    }

    private List<Layer> Layers { get; } = [];

    private void InitializeLayers(List<int> networkStructure)
    {
        foreach (var i in networkStructure)
            Layers.Add(new Layer(i));
    }

    public void Train(List<double> inputs, List<double> targets)
    {
        var results = Evaluate(inputs).ActivationValues;

        var errorSum = results.Select((t, i) => Math.Pow(targets[i] - t, 2)).Sum();
        Console.WriteLine($"Error: {errorSum}");
    }

    public void Backpropagate(List<double> inputs, List<double> targets)
    {
        var activationFunctionDerivative = (double x) =>
            _activationFunction.Function(x) * (1 - _activationFunction.Function(x));

        Evaluate(inputs);

        // Output layer

        var outputLayer = Layers.Last();

        var outputLayerDelta = outputLayer
            .Neurons.Select(
                (n, i) =>
                {
                    var error = targets[i] - n.Activation;
                    return error * activationFunctionDerivative.Invoke(n.WeightedInput);
                }
            )
            .ToList();

        // Hidden layers

        var nextLayer = outputLayer;
        foreach (var layer in Layers.Skip(1).SkipLast(1).Reverse())
        {
            layer.Neurons.Select(
                (n, i) =>
                {
                    var error = nextLayer.ActivationValues[i] * outputLayerDelta[i];
                    return error * activationFunctionDerivative.Invoke(n.WeightedInput);
                }
            );

            nextLayer = layer;
        }
    }

    public Layer Evaluate(List<double> inputs)
    {
        Layers.First().SetActivationValues(inputs);
        var previousLayer = Layers.First();

        foreach (var currentLayer in Layers.Skip(1))
        {
            currentLayer.Evaluate(previousLayer, _activationFunction);
            previousLayer = currentLayer;
        }

        return Layers.Last();
    }
}
