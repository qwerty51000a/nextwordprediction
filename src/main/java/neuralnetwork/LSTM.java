package neuralnetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * A long short-term memory recurrent neural network implementation from scratch. 
 * 
 * @author Ibrahim
 *
 */
public class LSTM {

	// input gate
	private Matrix Wih;
	private Matrix Wix;
	private Matrix bi;

	// forget gate
	private Matrix Wfh;
	private Matrix Wfx;
	private Matrix bf;

	// output gate
	private Matrix Woh;
	private Matrix Wox;
	private Matrix bo;

	// cell gate
	private Matrix Wch;
	private Matrix Wcx;
	private Matrix bc;

	// output
	private Matrix Wy;
	private Matrix by;

	// list to save input to lstm
	private List<Matrix> listOfTimestepsOfX = new ArrayList<Matrix>();

	// lists to save internal lstm states
	private List<Matrix> listOfTimestepsOfI = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfO = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfH = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfF = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfC = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfCBar = new ArrayList<Matrix>();

	//
	//
	// Remember to change THIS (CHat)
	//
	//

	private List<Matrix> listOfTimestepsOfdI = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfdO = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfdH = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfdF = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfdC = new ArrayList<Matrix>();
	private List<Matrix> listOfTimestepsOfdCBar = new ArrayList<Matrix>();

	private Matrix totalGradientOfWih;
	private Matrix totalGradientOfWfh;
	private Matrix totalGradientOfWoh;
	private Matrix totalGradientOfWch;

	private Matrix totalGradientOfBi;
	private Matrix totalGradientOfBf;
	private Matrix totalGradientOfBo;
	private Matrix totalGradientOfBc;

	private Matrix totalGradientOfWix;
	private Matrix totalGradientOfWfx;
	private Matrix totalGradientOfWox;
	private Matrix totalGradientOfWcx;

	private int timeStep = 0;

	public final int inputSize;
	public final int outputSize;
	public final double learningRate;

	public LSTM(final int inputSize, final int outputSize, final double learningRate)

	{
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.learningRate = learningRate;

		this.Wih = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		this.Wix = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		this.bi = new Matrix.Builder().setRows(outputSize).setColumns(1).build();

		this.Wfh = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		this.Wfx = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		this.bf = new Matrix.Builder().setRows(outputSize).setColumns(1).build();

		this.Woh = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		this.Wox = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		this.bo = new Matrix.Builder().setRows(outputSize).setColumns(1).build();

		this.Wch = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		this.Wcx = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		this.bc = new Matrix.Builder().setRows(outputSize).setColumns(1).build();

		this.Wy = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		this.by = new Matrix.Builder().setRows(outputSize).setColumns(1).build();

		initForward();
		randomizeWeights();

	}

	/**
	 * 
	 * method for assigning random values to weights
	 */

	public void randomizeWeights() {

		this.Wih.setToRandomValues();
		this.Wix.setToRandomValues();
		this.bi.setToRandomValues();

		this.Wfh.setToRandomValues();
		this.Wfx.setToRandomValues();
		this.bf.setToRandomValues();

		this.Woh.setToRandomValues();
		this.Wox.setToRandomValues();
		this.bo.setToRandomValues();

		this.Wch.setToRandomValues();
		this.Wcx.setToRandomValues();
		this.bc.setToRandomValues();

		this.Wy.setToRandomValues();
		this.by.setToRandomValues();
	}

	public Matrix getWih() {
		return Wih;
	}

	public Matrix getWix() {
		return Wix;
	}

	public Matrix getBi() {
		return bi;
	}

	public Matrix getWfh() {
		return Wfh;
	}

	public Matrix getWfx() {
		return Wfx;
	}

	public Matrix getBf() {
		return bf;
	}

	public Matrix getWoh() {
		return Woh;
	}

	public Matrix getWox() {
		return Wox;
	}

	public Matrix getBo() {
		return bo;
	}

	public Matrix getWch() {
		return Wch;
	}

	public Matrix getWcx() {
		return Wcx;
	}

	public Matrix getBc() {
		return bc;
	}

	public Matrix getWy() {
		return Wy;
	}

	public Matrix getBy() {
		return by;
	}

	/**
	 * Initialize the lists. Run this method before the running the forward(Matrix
	 * x) method.
	 */

	public void initForward() {

		this.timeStep = 0;

		listOfTimestepsOfX.clear();

		listOfTimestepsOfI.clear();
		listOfTimestepsOfO.clear();
		listOfTimestepsOfH.clear();
		listOfTimestepsOfF.clear();
		listOfTimestepsOfC.clear();
		listOfTimestepsOfCBar.clear();

		listOfTimestepsOfX.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfI.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfO.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfH.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfF.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfC.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfCBar.add(new Matrix.Builder().setRows(this.outputSize).setColumns(1).build());

		listOfTimestepsOfdI.clear();
		listOfTimestepsOfdO.clear();
		listOfTimestepsOfdH.clear();
		listOfTimestepsOfdF.clear();
		listOfTimestepsOfdC.clear();
		listOfTimestepsOfdCBar.clear();

		totalGradientOfWih = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		totalGradientOfWfh = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		totalGradientOfWoh = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();
		totalGradientOfWch = new Matrix.Builder().setRows(outputSize).setColumns(outputSize).build();

		totalGradientOfBi = new Matrix.Builder().setRows(outputSize).setColumns(1).build();
		totalGradientOfBf = new Matrix.Builder().setRows(outputSize).setColumns(1).build();
		totalGradientOfBo = new Matrix.Builder().setRows(outputSize).setColumns(1).build();
		totalGradientOfBc = new Matrix.Builder().setRows(outputSize).setColumns(1).build();

		totalGradientOfWix = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		totalGradientOfWfx = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		totalGradientOfWox = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();
		totalGradientOfWcx = new Matrix.Builder().setRows(outputSize).setColumns(inputSize).build();

	}

	/**
	 * Returns a matrix that is the output of the lstm after giving inputMatrices as
	 * input.
	 * 
	 * @param inputMatrices
	 * @return
	 */
	public Matrix getOutput(List<Matrix> inputMatrices) {

		initForward();

		for (Matrix x : inputMatrices)
			forward(x);

		Matrix output = listOfTimestepsOfH.get(listOfTimestepsOfH.size() - 1);

		initForward();

		return output;

	}

	/**
	 * method for running the forward part of the lstm. for each input x (which is
	 * sometimes denoted as x_t ), run the method once. All of the input matrices
	 * are stored automatically.
	 * 
	 * @param x
	 */
	private void forward(Matrix x) {

		if (x.getRows() != inputSize || x.getColumns() != 1) {
			throw new IllegalArgumentException("x.getRows(): " + x.getRows() + " " + " x.getColumns(): "
					+ x.getColumns() + "   and  x.getRows() != inputSize || x.getColumns() != 1 ");
		}

		this.timeStep++;
		listOfTimestepsOfX.add(x);

		// Get previous C and H

		Matrix prevC = this.listOfTimestepsOfC.get(timeStep - 1);
		Matrix prevH = this.listOfTimestepsOfH.get(timeStep - 1);

		Matrix f = ((Wfx.multiply(x)).add(Wfh.multiply(prevH)).add(bf))
				.applyActivationFunction(ActivationFunction.SIGMOID);
		listOfTimestepsOfF.add(f);

		Matrix i = ((Wix.multiply(x)).add(Wih.multiply(prevH)).add(bi))
				.applyActivationFunction(ActivationFunction.SIGMOID);
		listOfTimestepsOfI.add(i);

		Matrix cBar = ((Wcx.multiply(x)).add(Wch.multiply(prevH)).add(bi))
				.applyActivationFunction(ActivationFunction.TANH);
		listOfTimestepsOfCBar.add(cBar);

		Matrix c = f.elementWiseMultiply(prevC).add(i.elementWiseMultiply(cBar));
		listOfTimestepsOfC.add(c);

		Matrix o = ((Wox.multiply(x)).add(Woh.multiply(prevH)).add(bo))
				.applyActivationFunction(ActivationFunction.SIGMOID);
		listOfTimestepsOfO.add(o);

		Matrix h = (o.elementWiseMultiply(c.applyActivationFunction(ActivationFunction.TANH)));
		listOfTimestepsOfH.add(h);

		Matrix y = (Wy.multiply(h).add(by)).applyActivationFunction(ActivationFunction.SIGMOID);
		listOfTimestepsOfdF.add(y);
		// CHANGE THIS

	}

	/**
	 * Calculate the backward part of the lstm training.
	 * 
	 * @param lastExpectedOutput the expected output Matrix given the input matrix (
	 *                           or matrices) already given in the forward(Matrix x)
	 *                           method.
	 */
	private void backward(Matrix lastExpectedOutput) {

		// dH = realOutput - exptectedOutput
		Matrix dH = this.listOfTimestepsOfH.get(listOfTimestepsOfH.size() - 1).subtract(lastExpectedOutput);

		for (int i = 0; i < timeStep + 1; i++) {
			listOfTimestepsOfdI.add(null);
			listOfTimestepsOfdO.add(null);
			listOfTimestepsOfdH.add(null);
			listOfTimestepsOfdF.add(null);
			listOfTimestepsOfdC.add(null);
			listOfTimestepsOfdCBar.add(null);

		}

		listOfTimestepsOfdH.set(listOfTimestepsOfdH.size() - 1, dH);

		// backpropagation
		for (int i = timeStep; i > 0; i--) {

			// Get the current values from memory
			Matrix currentI = listOfTimestepsOfI.get(i);
			Matrix currentO = listOfTimestepsOfO.get(i);
			Matrix currentH = listOfTimestepsOfH.get(i);
			Matrix currentF = listOfTimestepsOfF.get(i);
			Matrix currentC = listOfTimestepsOfC.get(i);
			Matrix currentCBar = listOfTimestepsOfCBar.get(i);
			Matrix prevC = listOfTimestepsOfC.get(i - 1);

			Matrix currentDH = listOfTimestepsOfdH.get(i);

			// output gate
			Matrix currentDO = currentDH.elementWiseMultiply((currentC.applyActivationFunction(ActivationFunction.TANH))
					.elementWiseMultiply(currentO.applyActivationFunctionDerivative(ActivationFunction.SIGMOID)));

			// input gate
			Matrix currentDI = currentDH
					.elementWiseMultiply((currentC.applyActivationFunctionDerivative(ActivationFunction.TANH))
							.elementWiseMultiply(currentI.applyActivationFunctionDerivative(ActivationFunction.SIGMOID)
									.elementWiseMultiply(currentO).elementWiseMultiply(currentCBar)));

			// forget gate
			Matrix currentDF = currentDH
					.elementWiseMultiply((currentC.applyActivationFunctionDerivative(ActivationFunction.TANH))
							.elementWiseMultiply(currentF.applyActivationFunctionDerivative(ActivationFunction.SIGMOID)
									.elementWiseMultiply(currentO).elementWiseMultiply(prevC)));

			Matrix currentDCBar = currentDH
					.elementWiseMultiply((currentC.applyActivationFunctionDerivative(ActivationFunction.TANH))
							.elementWiseMultiply(currentCBar.applyActivationFunctionDerivative(ActivationFunction.TANH)
									.elementWiseMultiply(currentO).elementWiseMultiply(currentI)));

			Matrix prevDH = ((currentDI.transpose().multiply(Wih)).add(currentDO.transpose().multiply(Woh))
					.add(currentDF.transpose().multiply(Wfh)).add(currentDCBar.transpose().multiply(Wch))).transpose();

			// propagate the error
			listOfTimestepsOfdH.set(i - 1, prevDH);
			listOfTimestepsOfdI.set(i, currentDI);
			listOfTimestepsOfdO.set(i, currentDO);
			listOfTimestepsOfdF.set(i, currentDF);
			listOfTimestepsOfdCBar.set(i, currentDCBar);

		}

		// calculating gradient in preperation to update the parameters

		for (int i = this.timeStep; i > 0; i--) {
			Matrix prevH = listOfTimestepsOfH.get(i - 1);

			Matrix currentGradientOfWih = listOfTimestepsOfdI.get(i).multiply(prevH.transpose());
			Matrix currentGradientOfWfh = listOfTimestepsOfdF.get(i).multiply(prevH.transpose());
			Matrix currentGradientOfWoh = listOfTimestepsOfdO.get(i).multiply(prevH.transpose());
			Matrix currentGradientOfWch = listOfTimestepsOfdCBar.get(i).multiply(prevH.transpose());

			Matrix currentGradientOfBi = listOfTimestepsOfdI.get(i);
			Matrix currentGradientOfBf = listOfTimestepsOfdF.get(i);
			Matrix currentGradientOfBo = listOfTimestepsOfdO.get(i);
			Matrix currentGradientOfBc = listOfTimestepsOfdCBar.get(i);

			Matrix currentX = listOfTimestepsOfX.get(i);

			Matrix currentGradientOfWix = listOfTimestepsOfdI.get(i).multiply(currentX.transpose());
			Matrix currentGradientOfWfx = listOfTimestepsOfdF.get(i).multiply(currentX.transpose());
			Matrix currentGradientOfWox = listOfTimestepsOfdO.get(i).multiply(currentX.transpose());
			Matrix currentGradientOfWcx = listOfTimestepsOfdCBar.get(i).multiply(currentX.transpose());

			totalGradientOfWih = totalGradientOfWih.add(currentGradientOfWih);
			totalGradientOfWfh = totalGradientOfWfh.add(currentGradientOfWfh);
			totalGradientOfWoh = totalGradientOfWoh.add(currentGradientOfWoh);
			totalGradientOfWch = totalGradientOfWch.add(currentGradientOfWch);

			totalGradientOfBi = totalGradientOfBi.add(currentGradientOfBi);
			totalGradientOfBf = totalGradientOfBf.add(currentGradientOfBf);
			totalGradientOfBo = totalGradientOfBo.add(currentGradientOfBo);
			totalGradientOfBc = totalGradientOfBc.add(currentGradientOfBc);

			totalGradientOfWix = totalGradientOfWix.add(currentGradientOfWix);
			totalGradientOfWfx = totalGradientOfWfx.add(currentGradientOfWfx);
			totalGradientOfWox = totalGradientOfWox.add(currentGradientOfWox);
			totalGradientOfWcx = totalGradientOfWcx.add(currentGradientOfWcx);

		}

	}

	private void clipGradients(final double lower, final double upper) {

		totalGradientOfWih = totalGradientOfWih.clip(lower, upper);
		totalGradientOfWfh = totalGradientOfWfh.clip(lower, upper);
		totalGradientOfWoh = totalGradientOfWoh.clip(lower, upper);
		totalGradientOfWch = totalGradientOfWch.clip(lower, upper);

		totalGradientOfBi = totalGradientOfBi.clip(lower, upper);
		totalGradientOfBf = totalGradientOfBf.clip(lower, upper);
		totalGradientOfBo = totalGradientOfBo.clip(lower, upper);
		totalGradientOfBc = totalGradientOfBc.clip(lower, upper);

		totalGradientOfWix = totalGradientOfWix.clip(lower, upper);
		totalGradientOfWfx = totalGradientOfWfx.clip(lower, upper);
		totalGradientOfWox = totalGradientOfWox.clip(lower, upper);
		totalGradientOfWcx = totalGradientOfWcx.clip(lower, upper);

	}

	private void updateWeights(final double lower, final double upper) {

		clipGradients(lower, upper);

		Wih = Wih.subtract(totalGradientOfWih.constantElementMultiply(learningRate));
		Wfh = Wfh.subtract(totalGradientOfWfh.constantElementMultiply(learningRate));
		Woh = Woh.subtract(totalGradientOfWoh.constantElementMultiply(learningRate));
		Wch = Wch.subtract(totalGradientOfWch.constantElementMultiply(learningRate));

		bi = bi.subtract(totalGradientOfBi.constantElementMultiply(learningRate));
		bf = bf.subtract(totalGradientOfBf.constantElementMultiply(learningRate));
		bo = bo.subtract(totalGradientOfBo.constantElementMultiply(learningRate));
		bc = bc.subtract(totalGradientOfBc.constantElementMultiply(learningRate));

		Wix = Wix.subtract(totalGradientOfWix.constantElementMultiply(learningRate));
		Wfx = Wfx.subtract(totalGradientOfWfx.constantElementMultiply(learningRate));
		Wox = Wox.subtract(totalGradientOfWox.constantElementMultiply(learningRate));
		Wcx = Wcx.subtract(totalGradientOfWcx.constantElementMultiply(learningRate));

	}

	/**
	 * Trains the lstm by doing one iteration. The assumption is many-to-one lstm.
	 * 
	 * @param listOfInputMatrices list of all the input matrices (x_t) through time
	 * @param expectedOutput      The output that is expected after running all the
	 *                            input matrices
	 * 
	 * @param lower               lower value used for clipping the gradient
	 * @param upper               upper value used for clipping the gradient
	 */
	private void trainOneIteration(final List<Matrix> listOfInputMatrices, final Matrix expectedOutput,
			final double lower, final double upper) {

		for (int i = 0; i < listOfInputMatrices.size(); i++) {
			// calculate the forward part for all the input matrices
			forward(listOfInputMatrices.get(i));
		}

		// calculate the backward part
		backward(expectedOutput);

		updateWeights(lower, upper);

		initForward();

	}

	/**
	 * trains the lstm by running inputs and expected outputs. The assumption is
	 * many-to-one lstm.
	 * 
	 * Takes a list of list of input matrices and a corresponding list of output
	 * matrices. For Example, let's assume that lstm is to expected to have the
	 * following input and output values.
	 * 
	 * input | input | input | output matrix1 matrix2 matrix3 matrix4 matrix5
	 * matrix6 matrix7 matrix8
	 * 
	 * The list of inputs will be stored in the list listOfListOfInputMatrices. The
	 * list of corresponding outputs will be stored in listOfExpectedOutputMatrices.
	 * 
	 * List<List<Matrix>> listOfListOfInputMatrices = List.of(List.of(matrix1,
	 * matrix2, matrix3), List.of(matrix5, matrix6, matrix7)); List<Matrix>
	 * listOfExpectedOutputMatrices = List.of(matrix4, matrix8);
	 * 
	 * if the number of iterations is 500, the method would be called like this:
	 * train(listOfListOfInputMatrices, listOfExpectedOutputMatrices, 500);
	 * 
	 * 
	 * @param listOfListOfInputMatrices    a list of of list of input matrices
	 * @param listOfExpectedOutputMatrices a list of expected matrices
	 * @param numberOfIterations           number of iterations
	 * @param lower                        lower value used for clipping the
	 *                                     gradient
	 * @param upper                        upper value used for clipping the
	 *                                     gradient
	 */
	public void train(final List<List<Matrix>> listOfListOfInputMatrices,
			final List<Matrix> listOfExpectedOutputMatrices, final int numberOfIterations, final double lower,
			final double upper) {

		if (listOfListOfInputMatrices.size() != listOfExpectedOutputMatrices.size()) {
			throw new IllegalArgumentException(
					"listOfListOfInputMatrices must have the same size as listOfExpectedOutputMatrices");
		}

		for (int iteration = 0; iteration < numberOfIterations; iteration++) {

			for (int i = 0; i < listOfListOfInputMatrices.size(); i++) {
				trainOneIteration(listOfListOfInputMatrices.get(i), listOfExpectedOutputMatrices.get(i), lower, upper);

			}

		}

	}

}
