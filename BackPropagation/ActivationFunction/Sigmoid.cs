namespace BackPropagation.ActivationFunction;

public class Sigmoid : IActivationFunction
{
    public Sigmoid()
    {
        Function = value => 1.0f / (1.0f + (float)Math.Exp(-value));
        Derivative = value => Function.Invoke(value) * (1.0f - Function.Invoke(value));
    }

    public Func<double, double> Function { get; set; }

    public Func<double, double> Derivative { get; set; }
}
