package neuralnetwork;

import java.util.Random;

/**
 * A class that represents 2-dimensional matrix. Multiple
 * operations (e.g, addition, subtraction, multiplication, element-wise
 * multiplication, applying an activation function) can be used on the matrix.
 * 
 * @author Ibrahim
 *
 */
public class Matrix {

	private final int rows;
	private final int columns;
	private final double[][] internalArray;

	private Matrix(final int rows, final int columns) {

		this.rows = rows;
		this.columns = columns;
		internalArray = new double[rows][columns];

	}

	public int getRows() {
		return rows;
	}

	public int getColumns() {
		return columns;
	}

	public double[][] getMatrix() {
		return internalArray;
	}

	public double getValue(final int i, final int j) {
		return this.internalArray[i][j];
	}

	public void setValue(final int i, final int j, double value) {
		this.internalArray[i][j] = value;
	}

	// Builder design pattern for building a matrix
	public static class Builder {

		private int rows = 0;
		private int columns = 0;

		public Builder() {

		}

		public Builder setRows(final int rows) {
			this.rows = rows;
			return this;
		}

		public Builder setColumns(final int columns) {
			this.columns = columns;
			return this;
		}

		public Matrix build() {
			return new Matrix(rows, columns);
		}
	}

	public void setToRandomValues() {
		Random random = new Random();

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {

				this.internalArray[i][j] = (random.nextDouble() - 0.5) * Math.exp(-4);
			}
		}
	}

	public Matrix transpose() {

		Matrix resultMatrix = new Matrix.Builder().setRows(this.columns).setColumns(this.rows).build();

		for (int i = 0; i < resultMatrix.rows; i++) {
			for (int j = 0; j < resultMatrix.columns; j++) {

				resultMatrix.internalArray[i][j] = this.internalArray[j][i];
			}
		}

		return resultMatrix;
	}

	public Matrix add(final Matrix other) {
		if (other == null)
			throw new IllegalArgumentException("other Matrix cannot be null");

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < resultMatrix.rows; i++) {
			for (int j = 0; j < resultMatrix.columns; j++) {

				resultMatrix.internalArray[i][j] = this.internalArray[i][j] + other.internalArray[i][j];
			}
		}

		return resultMatrix;
	}

	public Matrix subtract(final Matrix other) {
		if (other == null)
			throw new IllegalArgumentException("other Matrix cannot be null");

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < resultMatrix.rows; i++) {
			for (int j = 0; j < resultMatrix.columns; j++) {

				resultMatrix.internalArray[i][j] = this.internalArray[i][j] - other.internalArray[i][j];
			}
		}

		return resultMatrix;
	}

	public Matrix multiply(final Matrix other) {
		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(other.columns).build();

		if (this.columns != other.rows)
			throw new RuntimeException("this.columns != other.rows ( " + this.rows + "x" + this.columns
					+ " cannot multiply " + other.rows + "x" + other.columns + "     because " + this.columns + " != "
					+ other.rows + " ).  Cannot multiply these two matrices");

		for (int i = 0; i < resultMatrix.rows; i++) {
			for (int j = 0; j < resultMatrix.columns; j++) {
				for (int k = 0; k < this.columns; k++) {

					resultMatrix.internalArray[i][j] += this.internalArray[i][k] * other.internalArray[k][j];
				}
			}
		}

		return resultMatrix;
	}

	public Matrix getColumn(final int columnNumber) {

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(1).build();

		for (int i = 0; i < this.rows; i++) {

			resultMatrix.internalArray[i][columnNumber] = this.internalArray[i][columnNumber];
		}

		return resultMatrix;

	}

	public Matrix getRow(final int rowNumber) {
		Matrix resultMatrix = new Matrix.Builder().setRows(1).setColumns(this.columns).build();

		for (int j = 0; j < this.columns; j++) {

			resultMatrix.internalArray[rowNumber][j] = this.internalArray[rowNumber][j];
		}

		return resultMatrix;

	}

	public Matrix elementWiseMultiply(final Matrix other) {

		if (other == null)
			throw new IllegalArgumentException("other Matrix cannot be null");

		if (this.getRows() != other.getRows() || this.getColumns() != other.getColumns()) {
			throw new RuntimeException(
					"this.columns != other.rows ( " + this.rows + "x" + this.columns + " cannot multiply " + other.rows
							+ "x" + other.columns + " ).  Cannot multiply these two matrices");
		}

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {

				resultMatrix.internalArray[i][j] = this.internalArray[i][j] * other.internalArray[i][j];
			}
		}

		return resultMatrix;

	}

	public Matrix constantElementMultiply(final double element) {

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.columns; j++) {

				resultMatrix.internalArray[i][j] = this.internalArray[i][j] * element;
			}
		}

		return resultMatrix;

	}

	public Matrix applyActivationFunction(final ActivationFunction activationFunction) {

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < this.rows; i++) {

			for (int j = 0; j < this.columns; j++) {

				resultMatrix.internalArray[i][j] = activationFunction.getValue(this.internalArray[i][j]);
			}
		}

		return resultMatrix;

	}

	public Matrix applyActivationFunctionDerivative(final ActivationFunction activationFunction) {

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < this.rows; i++) {

			for (int j = 0; j < this.columns; j++) {

				resultMatrix.internalArray[i][j] = activationFunction.getDerivativeValue(this.internalArray[i][j]);
			}
		}

		return resultMatrix;

	}

	public Matrix clip(final double lower, final double upper) {
		if (lower > upper)
			throw new IllegalArgumentException(" lower cannot be greater than upper ");

		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows).setColumns(this.columns).build();

		for (int i = 0; i < this.rows; i++) {

			for (int j = 0; j < this.columns; j++) {

				double value = this.internalArray[i][j];
				if (value < lower)
					value = lower;

				if (value > upper)
					value = upper;
				resultMatrix.internalArray[i][j] = value;

			}

		}

		return resultMatrix;

	}

	public Matrix rowStack(Matrix other) {
		Matrix resultMatrix = new Matrix.Builder().setRows(this.rows + other.rows)
				.setColumns(Math.max(this.columns, other.columns)).build();

		int index = 0;

		for (int i = 0; i < this.rows; i++) {

			resultMatrix.internalArray[index++] = this.internalArray[i];

		}

		for (int i = 0; i < other.rows; i++) {

			resultMatrix.internalArray[index++] = other.internalArray[i];

		}

		return resultMatrix;

	}

	public void printMatrix() {
		for (int i = 0; i < this.internalArray.length; i++) {
			for (int j = 0; j < this.internalArray[0].length; j++) {
				System.out.print(" " + this.internalArray[i][j]);
			}

			System.out.println();
		}

	}

	public String toString() {
		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < this.internalArray.length; i++) {
			for (int j = 0; j < this.internalArray[0].length; j++) {
				sb.append(" " + this.internalArray[i][j]);
			}

			sb.append("\n");
		}

		return sb.toString();

	}

}
