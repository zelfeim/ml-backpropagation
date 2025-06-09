// See https://aka.ms/new-console-template for more information

using BackPropagation.ActivationFunction;
using BackPropagation.NeuralNetwork;

Console.WriteLine("Hello, World!");

var neuralNetwork = new NeuralNetwork([2, 2, 1], new Sigmoid());

List<List<double>> inputs =
[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];

List<List<double>> targets =
[
    [0],
    [1],
    [1],
    [0],
];

neuralNetwork.Train(inputs, targets, 50000);
