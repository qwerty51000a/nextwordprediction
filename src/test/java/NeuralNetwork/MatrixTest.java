package NeuralNetwork;

import org.junit.Test;

import neuralnetwork.Matrix;

import static org.junit.Assert.*;

import org.junit.Before;

public class MatrixTest {

	Matrix mat1 = new Matrix.Builder().setRows(2).setColumns(2).build();

	Matrix mat2 = new Matrix.Builder().setRows(2).setColumns(2).build();

	// maximum error difference
	double delta = 0.0001d;

	@Before
	public void before() {

		mat1.setValue(0, 0, 1);
		mat1.setValue(0, 1, 10);
		mat1.setValue(1, 0, 50);
		mat1.setValue(1, 1, 100);

		mat2.setValue(0, 0, 2);
		mat2.setValue(0, 1, 20);
		mat2.setValue(1, 0, 70);
		mat2.setValue(1, 1, 200);
	}

	@Test
	public void testAdd() {

		Matrix resultMatrix = mat1.add(mat2);

		assertEquals(resultMatrix.getValue(0, 0), 3, delta);
		assertEquals(resultMatrix.getValue(0, 1), 30, delta);
		assertEquals(resultMatrix.getValue(1, 0), 120, delta);
		assertEquals(resultMatrix.getValue(1, 1), 300, delta);

	}

	@Test
	public void testSubtract() {

		Matrix resultMatrix = mat1.subtract(mat2);

		assertEquals(resultMatrix.getValue(0, 0), -1, delta);
		assertEquals(resultMatrix.getValue(0, 1), -10, delta);
		assertEquals(resultMatrix.getValue(1, 0), -20, delta);
		assertEquals(resultMatrix.getValue(1, 1), -100, delta);

	}

	@Test
	public void testMultiply() {

		Matrix resultMatrix = mat1.multiply(mat2);

		assertEquals(resultMatrix.getValue(0, 0), 702, delta);
		assertEquals(resultMatrix.getValue(0, 1), 2020, delta);
		assertEquals(resultMatrix.getValue(1, 0), 7100, delta);
		assertEquals(resultMatrix.getValue(1, 1), 21000, delta);

	}

}
