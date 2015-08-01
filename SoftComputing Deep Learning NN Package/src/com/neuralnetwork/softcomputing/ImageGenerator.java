package com.neuralnetwork.softcomputing;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ImageGenerator implements SupervisedEvaluator_Interface {
	
	class PixelLocation {
		public PixelLocation(int h, int w) {
			this.h = h;
			this.w = w;
		}
		public int h;
		public int w;
	}

	private double[][][] data;
	private int height;
	private int width;
	private boolean validation_mode = false;
	private int iteration = 0;
	private boolean visualization = true;
	private Random random;
	
	private List<PixelLocation> fixed_pixelsLoc = new ArrayList<PixelLocation>();
	private String folder;

	public ImageGenerator(Random r, String folder, String file) {
		this.folder = folder;
		data = ImageApproximator_Util.imageToMatrix(folder + file);
		height = data.length;
		width = data[0].length;
		System.out.println("Dimension : "+height+" x " + width);
		this.random = r;
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				fixed_pixelsLoc.add(new PixelLocation(h,w));
			}
		}
	}
	
	public double evaluateSupervisedLearningFitness(Supervised_Interface learner) throws Exception {
		double tot = 0;
		double max_tot = 0;
		double[][][] guess = new double[height][width][3];
		List<PixelLocation> pixels_m = new ArrayList<PixelLocation>();
		if (validation_mode == false) {
			List<PixelLocation> pixels = new ArrayList<PixelLocation>();
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					pixels.add(new PixelLocation(h,w));
				}
			}
			while (pixels.size() > 0) {
				int loc = random.nextInt(pixels.size());
				pixels_m.add(pixels.get(loc));
				pixels.remove(loc);
			}
		}
		else {
			pixels_m = fixed_pixelsLoc;
		}

		int count = 0;
		for (PixelLocation p : pixels_m) {
			if (count%2000 == 1999) {
				System.out.print(".");
			}
			count++;
			double[] actual_output;
			double[] input = {(double)p.h/((double)(height-1)), (double)p.w/((double)(width-1))};
			if (validation_mode) {
				actual_output = learner.Next(input);
			}
			else {
				actual_output = learner.Next(input, data[p.h][p.w]);
			}

			for (int c = 0; c < 3; c++) {
				tot += 1 - Math.abs(data[p.h][p.w][c] - actual_output[c]);
				max_tot++;
				guess[p.h][p.w][c] = actual_output[c];
			}
		}
		System.out.print("\n");
		if (validation_mode && visualization) {
			System.out.println("making visualization v"+iteration+".");
			ImageApproximator_Util.matrixToImage(guess, folder + "learned_" + (iteration++) + ".jpg");
		}
		return tot/max_tot;
	}

	public int GetActionDimension() {
		return 3;
	}

	public int GetObservationDimension() {
		return 2;
	}

	public void SetValidationMode(boolean validation) {
		validation_mode = validation;
	}
}
