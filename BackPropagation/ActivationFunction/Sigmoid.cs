namespace BackPropagation.ActivationFunction;

public class Sigmoid : IActivationFunction
{
    public Func<double, double> Function { get; set; } 
        = (value) => (1 / (1 + Math.Exp(-value)));
}