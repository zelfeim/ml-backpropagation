namespace BackPropagation;

public interface IActivationFunction
{
    Func<double, double> Function { get; set; }
}