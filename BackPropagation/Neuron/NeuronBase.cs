namespace BackPropagation.Neuron;

public class NeuronBase : INeuron
{
    public List<double> Weights { get; set; }
    private double Bias => Weights.Last();

    public NeuronBase(int weightCount)
    {
        Weights = new List<double>(weightCount);

        InitializeWeights();
    }
    
    public double Calculate(List<double> inputs)
    {
        var result = inputs.Select((t, i) => t * Weights[i]).Sum();
        result += Bias;
        return result;    
    }

    private void InitializeWeights()
    {
        var random = new Random();

        // TODO: Value range should be taken from different place?
        for (var i = 0; i < Weights.Count; i++)
        {
            Weights[i] = random.NextDouble() * 1;
        }
    }
}