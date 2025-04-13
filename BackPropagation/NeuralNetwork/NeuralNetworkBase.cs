using BackPropagation.Neuron;

namespace BackPropagation.NeuralNetwork;

public class NeuralNetworkBase : INeuralNetwork
{
    public List<int> NetworkStructure { get; init; }
    public List<Layer> Layers { get; init; }
    public IActivationFunction ActivationFunction { get; init; }

    public NeuralNetworkBase(List<int> networkStructure)
    {
        var inputSize = networkStructure.First();
        var resultSize = networkStructure.Last();

        Layers = new List<Layer>(networkStructure.Count);
        for (var i = 1; i < networkStructure.Count - 1; i++)
        {
            Layers.Add(new Layer(networkStructure[i - 1], networkStructure[i]));  
        }
    }
    
    public double Evaluate(IParameters inputs)
    {
        throw new NotImplementedException();
    }
}