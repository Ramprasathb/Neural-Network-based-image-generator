package com.neuralnetwork.softcomputing;


public interface SupervisedEvaluator_Interface {
	double evaluateSupervisedLearningFitness(Supervised_Interface agent) throws Exception;
	int GetActionDimension();
	int GetObservationDimension();
	void SetValidationMode(boolean validation);
	
}
