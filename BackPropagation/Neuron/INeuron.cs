using BackPropagation.ActivationFunction;

namespace BackPropagation.Neuron;

public interface INeuron
{
    // Neuron weight count depends on amount of inputs from previous layer, it's equal to input count + 1
    // Should be probably initialized during network creation
    public List<double> Weights { get; set; }
    public double Bias { get; set; }
    public double WeightedInput { get; set; }
    public double Delta { get; set; }
    double Activation { get; set; }

    // Result can also be complex
    // Input count is dynamic, depends on previous layer count
    public double Calculate(List<double> parameters, IActivationFunction activationFunction);
}
