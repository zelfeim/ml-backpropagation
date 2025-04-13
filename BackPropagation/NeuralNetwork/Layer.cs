using BackPropagation.Neuron;

namespace BackPropagation.NeuralNetwork;

public class Layer
{
    public List<INeuron> Neurons;

    public Layer(int previousLayerSize, int layerSize)
    {
        // Are there more neuron types?
        for (int i = 0; i < layerSize; ++i)
        {
        }
    }

    public List<double> Evaluate(List<double> inputs)
    {
        return Neurons.Select(n => n.Calculate(inputs)).ToList();
    }
}