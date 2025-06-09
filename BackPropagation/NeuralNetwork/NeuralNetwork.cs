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

    public List<double> Errors { get; } = [];

    private List<Layer> Layers { get; } = [];

    private void InitializeLayers(List<int> networkStructure)
    {
        Layers.Add(new Layer(networkStructure[0]));

        for (var i = 1; i < networkStructure.Count; i++)
            Layers.Add(new Layer(networkStructure[i], networkStructure[i - 1]));
    }

    public void Train(
        List<List<double>> inputs,
        List<List<double>> targets,
        int epochs,
        double learningRate = 0.3
    )
    {
        if (inputs.Count != targets.Count)
            throw new ArgumentException("Inputs and targets count must be equal");

        if (inputs[0].Count != Layers.First().Neurons.Count)
            throw new ArgumentException("Inputs count must be equal to input layer size");

        if (targets[0].Count != Layers.Last().Neurons.Count)
            throw new ArgumentException("Targets count must be equal to output layer size");

        for (var i = 0; i < epochs; ++i)
        {
            var random = new Random();

            var randomIndex = random.Next(0, inputs.Count);
            var randomInput = inputs[randomIndex];
            var randomTarget = targets[randomIndex];

            var results = Evaluate(randomInput).ActivationValues;
            var error = results.Select((t, j) => Math.Pow(randomTarget[j] - t, 2)).Sum();
            Errors.Add(error);
            if (i % 1000 == 0)
                Console.WriteLine($"Epoch: {i}, Error: {error}");

            Backpropagate(randomTarget, learningRate);
        }
    }

    private void Backpropagate(List<double> targets, double learningRate)
    {
        CalculateDeltas(targets);
        AdjustNeurons(learningRate);
    }

    private void CalculateDeltas(List<double> targets)
    {
        // Output layer
        var outputLayer = Layers.Last();

        for (var i = 0; i < outputLayer.Neurons.Count; i++)
        {
            var error = targets[i] - outputLayer.Neurons[i].Activation;
            outputLayer.Neurons[i].Delta =
                error * _activationFunction.Derivative.Invoke(outputLayer.Neurons[i].WeightedInput);
        }

        // Hidden layers
        var previousLayer = outputLayer;
        foreach (var layer in Layers.Skip(1).SkipLast(1).Reverse())
        {
            for (var i = 0; i < layer.Neurons.Count; i++)
            {
                var weightedError = previousLayer.Neurons.Aggregate(
                    0.0,
                    (current, neuron) => current + neuron.Weights[i] * neuron.Delta
                );

                layer.Neurons[i].Delta =
                    weightedError
                    * _activationFunction.Derivative.Invoke(layer.Neurons[i].WeightedInput);
            }

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
