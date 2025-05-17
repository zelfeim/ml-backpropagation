namespace BackPropagation.ActivationFunction;

public interface IActivationFunction
{
    Func<double, double> Function { get; set; }
}