namespace BackPropagation.ActivationFunction;

public interface IActivationFunction
{
    Func<double, double> Function { get; set; }
    Func<double, double> Derivative { get; set; }
}
