package neuralnetwork;

public class FeedForward {

	private Matrix[] layerWeights;
	private Matrix[] layerBiases;
	private ActivationFunction activationFunction;
	private final int numberOfHiddenLayers;
	private final int numberOfInputNodes;
	private final int numberOfOutputNodes;
	private final int numberOfHiddenNodes;
	private double biasLearningRate;
	private double weightLearningRate;

	private FeedForward(final ActivationFunction activationFunction, final int numberOfHiddenLayers,
			final int numberOfInputNodes, final int numberOfOutputNodes, final int numberOfHiddenNodes,
			final double biasLearningRate, final double weightLearningRate) {

		this.layerWeights = new Matrix[numberOfHiddenLayers + 1];
		this.layerBiases = new Matrix[numberOfHiddenLayers + 1];
		this.activationFunction = activationFunction;
		this.numberOfHiddenLayers = numberOfHiddenLayers;
		this.numberOfInputNodes = numberOfInputNodes;
		this.numberOfOutputNodes = numberOfOutputNodes;
		this.numberOfHiddenNodes = numberOfHiddenNodes;
		this.biasLearningRate = biasLearningRate;
		this.weightLearningRate = weightLearningRate;

		// Instantiate the weights and biases matrices

		for (int i = 0; i < numberOfHiddenLayers + 1; i++) {

			// first connection connects between the input and the hidden layers
			if (i == 0) {
				layerWeights[i] = new Matrix.Builder().setRows(this.numberOfHiddenNodes)

						.setColumns(this.numberOfInputNodes).build();
			}

			// last connection connects between the hidden and the output layers
			else if (i == numberOfHiddenLayers) {

				layerWeights[i] = new Matrix.Builder().setRows(this.numberOfOutputNodes)
						.setColumns(this.numberOfHiddenNodes).build();
			}

			else

			{
				layerWeights[i] = new Matrix.Builder().setRows(this.numberOfHiddenNodes)
						.setColumns(this.numberOfHiddenNodes).build();
			}

			layerWeights[i].setToRandomValues();
		}

		// Instantiate the biases matrices

		for (int i = 0; i < numberOfHiddenLayers + 1; i++) {
			// first layer connects between the input and the hidden layers

			if (i == numberOfHiddenLayers) {

				layerBiases[i] = new Matrix.Builder().setRows(this.numberOfOutputNodes).setColumns(1).build();
			} else

				layerBiases[i] = new Matrix.Builder().setRows(this.numberOfHiddenNodes).setColumns(1).build();

			layerBiases[i].setToRandomValues();
		}

	}

// Builder design pattern
	public static class Builder {

		private ActivationFunction activationFunction;
		private int numberOfHiddenLayers;
		private int numberOfInputNodes;
		private int numberOfOutputNodes;
		private int numberOfHiddenNodes;
		private double biasLearningRate;
		private double weightLearningRate;

		public Builder() {
		}

		public Builder setActivationFunction(final ActivationFunction activationFunction) {
			this.activationFunction = activationFunction;
			return this;
		}

		public Builder setNumberOfHiddenLayers(final int numberOfHiddenLayers) {
			this.numberOfHiddenLayers = numberOfHiddenLayers;
			return this;
		}

		public Builder setNumberOfInputNodes(final int numberOfInputNodes) {
			this.numberOfInputNodes = numberOfInputNodes;
			return this;
		}

		public Builder setNumberOfOutputNodes(final int numberOfOutputNodes) {
			this.numberOfOutputNodes = numberOfOutputNodes;
			return this;
		}

		public Builder setNumberOfHiddenNodes(final int numberOfHiddenNodes) {
			this.numberOfHiddenNodes = numberOfHiddenNodes;
			return this;
		}

		public Builder setBiasLearningRate(final double biasLearningRate) {
			this.biasLearningRate = biasLearningRate;
			return this;
		}

		public Builder setWeightLearningRate(final double weightLearningRate) {
			this.weightLearningRate = weightLearningRate;
			return this;
		}

		public FeedForward Build() {
			return new FeedForward(activationFunction, numberOfHiddenLayers, numberOfInputNodes, numberOfOutputNodes,
					numberOfHiddenNodes, biasLearningRate, weightLearningRate);
		}

	}

	public Matrix getOutputFromRNN(final Matrix input) {
		if (input.getRows() != numberOfInputNodes)
			throw new IllegalArgumentException(
					"Input matrix number of columns doesn't match with the number of input nodes");

		if (input.getColumns() != 1)
			throw new IllegalArgumentException(
					"Input matrix number of rows doesn't match with the number of input nodes");

		Matrix output = input;

		for (int i = 0; i < numberOfHiddenLayers + 1; i++)

		{
			output = layerWeights[i].multiply(output).add(layerBiases[i]).applyActivationFunction(activationFunction);

		}

		return output;
	}

	public void train(Matrix input, Matrix correctOutput) {

		Matrix[] layers = new Matrix[numberOfHiddenLayers + 2];

		layers[0] = input;

		Matrix output = input;

		for (int i = 1; i < numberOfHiddenLayers + 2; i++)

		{
			output = (layerWeights[i - 1].multiply(output).add(layerBiases[i - 1]))
					.applyActivationFunction(activationFunction);
			layers[i] = output;

		}

		Matrix currentLayer = correctOutput;

		for (int i = numberOfHiddenLayers + 1; i > 0; i--) {

			Matrix error = currentLayer.subtract(layers[i]);

			Matrix gradient = layers[i].applyActivationFunctionDerivative(activationFunction);

			gradient = gradient.elementWiseMultiply(error);

			Matrix differenceInBiases = gradient.constantElementMultiply(biasLearningRate);

			Matrix differenceInWeights = gradient
					.multiply(layers[i - 1].transpose().constantElementMultiply(weightLearningRate));

			layerBiases[i - 1] = layerBiases[i - 1].add(differenceInBiases);

			layerWeights[i - 1] = layerWeights[i - 1].add(differenceInWeights);

			currentLayer = layerWeights[i - 1].transpose().multiply(error).add(layers[i - 1]);

		}

	}
}
