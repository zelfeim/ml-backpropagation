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

    public void Train(
        List<double> inputs,
        List<double> targets,
        int iterationCount,
        double learningRate = 0.01
    )
    {
        for (var i = 0; i < iterationCount; ++i)
        {
            Console.WriteLine($"Iteration: {i}");

            var results = Evaluate(inputs).ActivationValues;
            var errorSum = results.Select((t, i) => Math.Pow(targets[i] - t, 2)).Sum();
            Console.WriteLine($"Error: {errorSum}");

            Backpropagate(targets, learningRate);

            i++;
        }
    }

    private void Backpropagate(List<double> targets, double learningRate)
    {
        CalculateDeltas(targets, ActivationFunctionDerivative);
        AdjustNeurons(learningRate);
        return;

        double ActivationFunctionDerivative(double x)
        {
            return _activationFunction.Function(x) * (1 - _activationFunction.Function(x));
        }
    }

    private void CalculateDeltas(
        List<double> targets,
        Func<double, double> activationFunctionDerivative
    )
    {
        // Output layer
        var outputLayer = Layers.Last();
        var i = 0;
        outputLayer.Neurons.ForEach(n =>
        {
            var error = targets[i] - n.Activation;
            n.Delta = error * activationFunctionDerivative.Invoke(n.WeightedInput);
            i++;
        });

        // Hidden layers
        var previousLayer = outputLayer;
        foreach (var layer in Layers.Skip(1).SkipLast(1).Reverse())
        {
            layer.Neurons.ForEach(n =>
            {
                var weightedError = previousLayer.Neurons.Aggregate(
                    0.0,
                    (current, neuron) => current + neuron.Activation * neuron.Delta
                );

                n.Delta = weightedError * activationFunctionDerivative.Invoke(n.WeightedInput);
            });

            previousLayer = layer;
        }
    }

    private void AdjustNeurons(double learningRate)
    {
        for (var i = 1; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var previousLayer = Layers[i - 1];

            layer.Neurons.ForEach(n =>
            {
                n.Bias += learningRate * n.Delta;
                for (
                    var previousNeuronIndex = 0;
                    previousNeuronIndex < previousLayer.Neurons.Count;
                    previousNeuronIndex++
                )
                    n.Weights[previousNeuronIndex] +=
                        learningRate
                        * n.Delta
                        * previousLayer.ActivationValues[previousNeuronIndex];
            });
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
