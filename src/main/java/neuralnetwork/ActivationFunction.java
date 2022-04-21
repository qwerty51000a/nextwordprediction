package neuralnetwork;

public enum ActivationFunction implements ActivationFunctionInterface {

	RELU {
		public double getValue(double x)

	{
			return x > 0 ? x : 0;
		}

		public double getDerivativeValue(double x)

	{

			return x > 0 ? 1 : 0;
		}
		
		
		
	}
	
	

	,

	SIGMOID {
		public double getValue(double x)

	{
			return (1 / (1 + Math.exp(-x)));
		}

		public double getDerivativeValue(double x)

	{

			return getValue(x) * (1 - getValue(x));

		}
		
		
	}

	,

	TANH {
		
	
		public double getValue(double x)

	{
			return Math.tanh(x);
		}

		
		public double getDerivativeValue(double x)

	{
			// 1 - [f'(x)] ^ 2
			return 1 - Math.pow(TANH.getValue(x), 2);

		}
		
		
	}

	


}
