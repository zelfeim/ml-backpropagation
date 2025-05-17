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

    public void Train(List<double> inputs, List<double> expectedOutputs)
    {
        var results = Evaluate(inputs).ActivationValues;

        var errorSum = results.Select((t, i) => Math.Pow(expectedOutputs[i] - t, 2)).Sum();
        Console.WriteLine($"Error: {errorSum}");
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
