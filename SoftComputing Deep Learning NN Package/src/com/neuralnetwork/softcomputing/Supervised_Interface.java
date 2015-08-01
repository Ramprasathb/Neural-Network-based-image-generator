package com.neuralnetwork.softcomputing;

public interface Supervised_Interface {
	double[] Next(double[] input, double[] target_output) throws Exception;
	double[] Next(double[] input) throws Exception;
}
