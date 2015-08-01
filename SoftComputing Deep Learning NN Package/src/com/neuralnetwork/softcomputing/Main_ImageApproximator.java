package com.neuralnetwork.softcomputing;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main_ImageApproximator {
	public static void main(String[] args) throws Exception {
		System.out.println("Image Approximation");
		
		Random random = new Random(333);

		String folder = "data/";
		String file = "ram.jpg";
		ImageGenerator imageGenerator = new ImageGenerator(random, folder, file);

		int epoches = 500;
		
		double initial_weight_range = 0.3;
		double learning_rate = 0.1;
		Neuron_Abstract neuron = new RectifiedLinearNeuron(0.01);
		
		int input_dimension = imageGenerator.GetObservationDimension();
		int hidden_dimension = 300;
		int total_layers = 5;
		int output_dimension = imageGenerator.GetActionDimension();
		
		Neuron_Abstract linear_neuron = new IdentityNeuron();
		List<Feedforward_Interface> layers = new ArrayList<Feedforward_Interface>();
		layers.add(new Layer(random, input_dimension, hidden_dimension, neuron, initial_weight_range, learning_rate));
		for (int i = 0; i < total_layers; i++) {
			layers.add(new Layer(random, hidden_dimension, hidden_dimension, neuron, initial_weight_range, learning_rate));
		}
		layers.add(new Layer(random, hidden_dimension, output_dimension, linear_neuron, initial_weight_range, learning_rate));
		LayeredNetwork layeredNetwork = new LayeredNetwork(layers);
		
		int low_at = 1;
		double low = Double.POSITIVE_INFINITY;
		for (int t = 0; t < epoches; t++)
		{
			imageGenerator.SetValidationMode(false);
			imageGenerator.evaluateSupervisedLearningFitness(layeredNetwork);
			imageGenerator.SetValidationMode(true);
			double validation = imageGenerator.evaluateSupervisedLearningFitness(layeredNetwork);
			
			System.out.println("\n\t Epoch Number "+(t+1)+" :\t" + (1-validation));
			
			if (1 - validation < low) {
				low = 1 - validation;
				low_at = t;
			}
			System.out.println("\tLowest error  = " + low + " at Epoch number : " + (low_at+1));
		}

		System.out.println("\n\ndone.");
	}

}
 