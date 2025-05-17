namespace BackPropagation.ActivationFunction;

public class Sigmoid : IActivationFunction
{
    public Func<double, double> Function { get; set; } =
        value => 1.0f / (1.0f + (float)Math.Exp(-value));
}
