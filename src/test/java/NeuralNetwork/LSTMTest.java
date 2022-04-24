package NeuralNetwork;

import java.util.List;
import org.junit.Test;

import neuralnetwork.LSTM;
import neuralnetwork.Matrix;

import org.junit.Before;

public class LSTMTest {

	
	
	LSTM lstm = new LSTM(2, 3, 0.05);

	
	Matrix x11 = new Matrix.Builder().setRows(2).setColumns(1).build();



	Matrix x12 = new Matrix.Builder().setRows(2).setColumns(1).build();
	
	Matrix y1 = new Matrix.Builder().setRows(3).setColumns(1).build();


	Matrix x21 = new Matrix.Builder().setRows(2).setColumns(1).build();

	Matrix x22 = new Matrix.Builder().setRows(2).setColumns(1).build();

	
	Matrix y2 = new Matrix.Builder().setRows(3).setColumns(1).build();

	
	List<Matrix> input1 = List.of(x11, x12);
	Matrix output1 = y1;

	List<Matrix> input2 = List.of(x21, x22);
	Matrix output2 = y2;
	
	Matrix xTest = new Matrix.Builder().setRows(2).setColumns(1).build();


	@Before
	public void before()
	{
		
		x11.setValue(0, 0, 1);
		x11.setValue(1, 0, 0);
		
		x12.setValue(0, 0, 1);
		x12.setValue(1, 0, 0);
		
		y1.setValue(0, 0, 0);
		y1.setValue(1, 0, 1);
		y1.setValue(2, 0, 0);
		
		x21.setValue(0, 0, 0);
		x21.setValue(1, 0, 1);
		
		x22.setValue(0, 0, 0);
		x22.setValue(1, 0, 1);
		
		
		y2.setValue(0, 0, 0);
		y2.setValue(1, 0, 0);
		y2.setValue(2, 0, 1);




		
	}
	
	
	
	
	
	

	
	@Test
	public void testTrainedLSTM()
	{
		
		xTest.setValue(0, 0, 0);
		xTest.setValue(1, 0, 1);
		
		lstm.train(List.of(input1, input2), List.of(y1, y2), 5000, -5, 5);

		
		Matrix output1 = lstm.getOutput(input1);
		
		Matrix output2 = lstm.getOutput(input2);
		
		//output1 must be very close to y1
		for(int i = 0; i < output1.getRows(); i++)
		{
			Matrix diff = output1.subtract(y1);
			
			assert(diff.getValue(i,0) < 0.01);
		}
		
		//output2 must be very close to y2
		for(int i = 0; i < output2.getRows(); i++)
		{
			Matrix diff = output2.subtract(y2);
			
			assert(diff.getValue(i,0) < 0.01);
		}
	}
	
	
	


}
