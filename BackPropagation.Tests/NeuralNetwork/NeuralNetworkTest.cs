using System.Collections.Generic;
using System.Linq;
using BackPropagation.ActivationFunction;
using BackPropagation.NeuralNetwork;
using BackPropagation.Neuron;
using FluentAssertions;
using Xunit;

namespace BackPropagation.Tests.NeuralNetwork;

public class NeuralNetworkTest
{
    public static TheoryData<int[], int> XorData()
    {
        return new TheoryData<int[], int>
        {
            { [0, 0], 0 },
            { [0, 1], 1 },
            { [1, 0], 1 },
            { [1, 1], 0 },
        };
    }

    /// <summary>
    ///     Test checking if a tuned neural network returns the expected result for the XOR problem.
    ///     Neuron weights were generated with GeneticAlgorithm XOR solution.
    ///     Calculated fitness for these weights was 0.00025.
    /// </summary>
    [Theory]
    [MemberData(nameof(XorData))]
    public void NeuralNetwork_Should_Return_Expected_XOR_Result(int[] inputs, int expectedOutput)
    {
        // Arrange
        var sigmoidFunction = new Sigmoid();

        var tunedLayers = new List<Layer>();

        var inputNeurons = new List<INeuron> { new Neuron.Neuron([0]), new Neuron.Neuron([0]) };
        var hiddenNeurons = new List<INeuron>
        {
            new Neuron.Neuron([10, -10], -6),
            new Neuron.Neuron([-10, 10], -6),
        };
        var outputNeurons = new List<INeuron> { new Neuron.Neuron([10, 10], -4.666) };

        tunedLayers.Add(new Layer(inputNeurons));
        tunedLayers.Add(new Layer(hiddenNeurons));
        tunedLayers.Add(new Layer(outputNeurons));

        var neuralNetwork = new BackPropagation.NeuralNetwork.NeuralNetwork(
            tunedLayers,
            sigmoidFunction
        );

        // Act
        var outputLayer = neuralNetwork.Evaluate(inputs.Select(i => (double)i).ToList());

        // Assert
        outputLayer.ActivationValues.First().Should().BeApproximately(expectedOutput, 0.01);
    }
}
