using BackPropagation.ActivationFunction;
using BackPropagation.Neuron;

namespace BackPropagation.NeuralNetwork;

public class Layer
{
    public List<INeuron> Neurons;

    public Layer(int layerSize, int previousLayerSize = 0)
    {
        Neurons = new List<INeuron>(layerSize);

        for (var i = 0; i < layerSize; i++)
            Neurons.Add(
                previousLayerSize != 0
                    ? new Neuron.Neuron(previousLayerSize)
                    : new Neuron.Neuron(layerSize)
            );
    }

    public Layer(List<INeuron> neurons)
    {
        Neurons = neurons;
    }

    public List<double> ActivationValues => Neurons.Select(n => n.Activation).ToList();

    public void SetActivationValues(List<double> values)
    {
        for (var i = 0; i < values.Count; i++)
            Neurons[i].Activation = values[i];
    }

    public List<double> Evaluate(Layer previousLayer, IActivationFunction _activationFunction)
    {
        return Neurons
            .Select(n => n.Calculate(previousLayer.ActivationValues, _activationFunction))
            .ToList();
    }
}
