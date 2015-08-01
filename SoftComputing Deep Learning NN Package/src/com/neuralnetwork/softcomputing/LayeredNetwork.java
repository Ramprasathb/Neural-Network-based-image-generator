package com.neuralnetwork.softcomputing;

import java.util.List;

public class LayeredNetwork implements Supervised_Interface {
	
	int input_dimension;
	int output_dimension;
	List<Feedforward_Interface> layers;
	boolean dropoutMode = false;
	
	public LayeredNetwork(List<Feedforward_Interface> layers) throws Exception {
		this.layers = layers;
		input_dimension = layers.get(0).GetInputDimension();
		output_dimension = layers.get(layers.size()-1).GetOutputDimension();
		if (layers.size() == 0) {
			throw new Exception("Error");
		}
		for (int i = 1; i < layers.size(); i++) {
			if (layers.get(i).GetInputDimension() != layers.get(i-1).GetOutputDimension()) {
				throw new Exception("Error in Layers");
			}
		}
	}

	public double[] Next(double[] input, double[] target_output) throws Exception {
		
		double[] next = input;
		for (Feedforward_Interface layer : layers) {
			next = layer.forwardPropogation(next);
		}
		double[] output = next;
		
		if (target_output != null) {
			double[] delta = new double[output_dimension];
			for (int k = 0; k < output_dimension; k++) {
				delta[k] = target_output[k] - output[k];
			}
			
			next = delta;
			for (int i = layers.size() - 1; i >= 0; i--) {
				if (next == null) {
					throw new Exception("Error in Backpropogation");
				}
				try {
					next = layers.get(i).backPropogation(next);
				}
				catch (Exception e) {
					throw new Exception(e.getMessage() + " caught at layer " + i);
				}
			}
		}
		return output;
	}

	public double[] Next(double[] input) throws Exception {
		return Next(input, null);
	}
}
