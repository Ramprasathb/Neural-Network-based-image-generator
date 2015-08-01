package com.neuralnetwork.softcomputing;

public class IdentityNeuron extends Neuron_Abstract {
	@Override
	public double Activate(double x) {
		return x;
	}

	@Override
	public double Derivative(double x) {
		return 1;
	}
}

