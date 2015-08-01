package com.neuralnetwork.softcomputing;

public interface Feedforward_Interface {
	int GetInputDimension();
	int GetOutputDimension();
	double[] forwardPropogation(double[] input);
	double[] backPropogation(double[] delta) throws Exception;
}
