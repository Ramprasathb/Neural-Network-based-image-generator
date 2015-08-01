package com.neuralnetwork.softcomputing;

import java.util.Random;

public class Layer implements Feedforward_Interface {
	
	int input_dimension;
	int biasedInput_index;
	int output_dimension;
	Neuron_Abstract neuron;
	double learning_rate;
	public double[][] weights;
	double[] input_a;
	double[] output_a;
	double[] output_sums;
	Random random;
	double DELTA_WARNING_SIZE = 1000;
	
	public Layer(Random random, int input_dimension, int output_dimension, Neuron_Abstract neuron, double init_weight_range, double learning_rate) {
		this.random = random;
		this.input_dimension = input_dimension;
		this.output_dimension = output_dimension;
		this.neuron = neuron;
		this.learning_rate = learning_rate;
		double input_dim_inverse = 1.0 / Math.sqrt(input_dimension+1);
		this.learning_rate *= input_dim_inverse;
		biasedInput_index = input_dimension;
		weights = new double[output_dimension][input_dimension+1];
		for (int k = 0; k < output_dimension; k++) {
			for (int i = 0; i < input_dimension+1; i++) {
				weights[k][i] = random.nextGaussian() * init_weight_range * input_dim_inverse;
			}
		}
		input_a = new double[input_dimension];
		output_sums = new double[output_dimension];
		output_a = new double[output_dimension];
	}

	public int GetInputDimension() {
		return input_dimension;
	}

	public int GetOutputDimension() {
		return output_dimension;
	}

	public double[] forwardPropogation(double[] input) {
		input_a = input;
		for (int k = 0; k < output_dimension; k++) {
			output_sums[k] = 0;
			output_a[k] = 0;
			for (int i = 0; i < input_dimension; i++) {
				output_sums[k] += weights[k][i] * input[i];
			}
			output_sums[k] += weights[k][biasedInput_index];
			output_a[k] = neuron.Activate(output_sums[k]);
		}
		return output_a;
	}

	public double[] backPropogation(double[] delta) throws Exception {
		double[] delta_at_input = new double[input_dimension];
		for (int k = 0; k < output_dimension; k++) {
			if (Math.abs(delta[k]) > DELTA_WARNING_SIZE) {
				throw new Exception("WARNING: delta["+k+"] > DELTA_WARNING_SIZE: " + delta[k] + ". Suggest using a smaller initial weight range");
			}
			double _predelta = neuron.Derivative(output_sums[k]) * delta[k];
			for (int i = 0; i < input_dimension; i++) {
				delta_at_input[i] += _predelta * weights[k][i];
				weights[k][i] += input_a[i] * _predelta * learning_rate;
			}
			weights[k][biasedInput_index] += _predelta * learning_rate;
		}
		return delta_at_input;
	}
}
