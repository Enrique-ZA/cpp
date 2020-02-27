// Toy Neural Network rewritten for c++
// Author: Daniel Shifmann
// Website: https://shiffman.net/
// Github: https://github.com/CodingTrain/
// Youtube: https://www.youtube.com/user/shiffman
// additional functions work with openframeworks (example ofRandom())
// see openframeworks documentation

//--------------------------------------------------------------

// Matrix

class Matrix {
	public:

	int rows, cols;
	vector<vector<float>> values;

	Matrix(int rows, int cols) {
		this->rows = rows;
		this->cols = cols;
		for (int i = 0; i < rows; i++) {
			values[i].resize(cols);
		}
	}

	Matrix() {
		rows = 1;
		cols = 1;
		for (int i = 0; i < rows; i++) {
			values[i].resize(cols);
		}
	}

	Matrix copy() {
		Matrix result(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.values[i][j] = values[i][j];
			}
		}
		return result;
	}

	vector<vector<string>> stringify(Matrix a) {
		vector<vector<string>> result;
		for (int i = 0; i < a.rows; i++) {
			result[i].resize(a.cols);
		}

		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				result[i][j] = to_string(i) + " " + to_string(j) + " " + to_string(a.values[i][j]) + " ";
			}
		}
		return result;
	}

	void multiply(float n) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				values[i][j] *= n;
			}
		}
	}

	void add(float n) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				values[i][j] += n;
			}
		}
	}

	Matrix random(int rows, int cols) {
		Matrix result(rows, cols);
		result.randomize();
		return result;
	}

	void randomize() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				values[i][j] = ofRandom(-1, 1);
			}
		}
	}

	Matrix subtract(Matrix a, Matrix b) {
		Matrix result(a.rows, a.cols);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				result.values[i][j] = a.values[i][j] - b.values[i][j];
			}
		}
		return result;
	}

	Matrix FromArray(vector<float> arr) {
		Matrix result(arr.size(), 1);
		for (int i = 0; i < result.rows; i++) {
			result.values[i][0] = arr[i];
		}
		return result;
	}

	vector<float> toArray() {
		int n = rows + cols;
		vector<float> arr(n,0);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				arr[i] = values[i][j];
			}
		}
		return arr;
	}

	void add(Matrix other) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				values[i][j] += other.values[i][j];
			}
		}
	}

	void multiply(Matrix other) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				values[i][j] *= other.values[i][j];
			}
		}
	}

	Matrix transpose(Matrix a) {
		Matrix result(a.cols, a.rows);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				result.values[j][i] = a.values[i][j];
			}
		}
		return result;
	}

	Matrix Product(Matrix first, Matrix second) {
		if (first.cols != second.rows) {
			throw ERROR;
		}
		else {
			Matrix a = first;
			Matrix b = second;
			Matrix result(a.rows, b.cols);
			for (int i = 0; i < result.rows; i++) {
				for (int j = 0; j < result.cols; j++) {
					float sum = 0;
					for (int k = 0; k < a.cols; k++) {
						sum += a.values[i][k] * b.values[k][j];
					}
					result.values[i][j] = sum;
				}
			}
			return result;
		}
	}
};
//--------------------------------------------------------------

// Neural Network

float sigmoid(float x) {
	return 1 / (1 + (float)exp(-x));
}

float dsigmoid(float y) {
	return y * (1 - y);
}

float tanH(float x) {
	float y = (float)tanh(x);
	return y;
}

float dtanh(float x) {
	float y = 1 / (pow((float)cosh(x), 2));
	return y;
}

float ofGaussian() {
	float v1, v2, s;
	do {
		v1 = 2 * ofRandomf() - 1;
		v2 = 2 * ofRandomf() - 1;
		s = v1 * v1 + v2 * v2;
	} while (s >= 1 || s == 0);
	float multiplier = sqrt(-2 * log(s) / s);
	return v1 * multiplier;
}

class NeuralNetwork {

	public:

	int inputNodes, hiddenNodes, outputNodes;

	float LearningRate = 0.1;

	Matrix IHWeights, HOWeights, Hbias, Obias, input, hidden, output;

	NeuralNetwork copy(NeuralNetwork nn) {
		inputNodes = nn.inputNodes;
		hiddenNodes = nn.hiddenNodes;
		outputNodes = nn.outputNodes;

		IHWeights = nn.IHWeights.copy();
		HOWeights = nn.HOWeights.copy();
		Hbias = nn.Hbias.copy();
		Obias = nn.Obias.copy();
		return nn;
	}

	NeuralNetwork(int input, int hidden, int output) {
		inputNodes = input;
		hiddenNodes = hidden;
		outputNodes = output;

		IHWeights = IHWeights.random(hiddenNodes, inputNodes);
		HOWeights = HOWeights.random(outputNodes, hiddenNodes);
		Hbias = Hbias.random(hiddenNodes, 1);
		Obias = Obias.random(outputNodes, 1);
	}

	NeuralNetwork(int input, int hidden, int output, float lr) {
		inputNodes = input;
		hiddenNodes = hidden;
		outputNodes = output;

		IHWeights = IHWeights.random(hiddenNodes, inputNodes);
		HOWeights = HOWeights.random(outputNodes, hiddenNodes);
		Hbias = Hbias.random(hiddenNodes, 1);
		Obias = Obias.random(outputNodes, 1);
		setLearingRate(lr);
	}

	NeuralNetwork copy() {
		return *this;
	}

	float mut(float val, float rate) {
		if (ofRandom(1) < rate) {
			return val + ofGaussian() * .1;
		}
		else {
			return val;
		}
	}

	void mutate(float rate) {
		for (int i = 0; i < IHWeights.rows; i++) {
			for (int j = 0; j < IHWeights.cols; j++) {
				float val = IHWeights.values[i][j];
				IHWeights.values[i][j] = mut(val, rate);
			}
		}

		for (int i = 0; i < HOWeights.rows; i++) {
			for (int j = 0; j < HOWeights.cols; j++) {
				float val = HOWeights.values[i][j];
				HOWeights.values[i][j] = mut(val, rate);
			}
		}

		for (int i = 0; i < Hbias.rows; i++) {
			for (int j = 0; j < Hbias.cols; j++) {
				float val = Hbias.values[i][j];
				Hbias.values[i][j] = mut(val, rate);
			}
		}

		for (int i = 0; i < Obias.rows; i++) {
			for (int j = 0; j < Obias.cols; j++) {
				float val = Obias.values[i][j];
				Obias.values[i][j] = mut(val, rate);
			}
		}
	}

	void setLearingRate(float rate) {
		LearningRate = rate;
	}

	vector<float> feedForward(vector<float> inputArray) {
		input = input.FromArray(inputArray);

		hidden = hidden.Product(IHWeights, input);
		hidden.add(Hbias);

		for (int i = 0; i < hidden.rows; i++) {
			for (int j = 0; j < hidden.cols; j++) {
				float val = hidden.values[i][j];
				hidden.values[i][j] = sigmoid(val);
			}
		}

		output = output.Product(HOWeights, hidden);
		output.add(Obias);

		for (int i = 0; i < output.rows; i++) {
			for (int j = 0; j < output.cols; j++) {
				float val = output.values[i][j];
				output.values[i][j] = sigmoid(val);
			}
		}

		return output.toArray();
	}

	void train(vector<float> inputArray, vector<float> targetArray) {
		feedForward(inputArray);

		Matrix targets = targets.FromArray(targetArray);
		Matrix outputErrors = outputErrors.subtract(targets, output);

		Matrix gradient = output.copy();
		for (int i = 0; i < gradient.rows; i++) {
			for (int j = 0; j < gradient.cols; j++) {
				float val = gradient.values[i][j];
				gradient.values[i][j] = dsigmoid(val);
			}
		}

		gradient.multiply(outputErrors);
		gradient.multiply(LearningRate);

		Matrix hiddenT = hiddenT.transpose(hidden);
		Matrix DHOWeights = DHOWeights.Product(gradient, hiddenT);

		HOWeights.add(DHOWeights);

		Obias.add(gradient);

		Matrix HOWeightsT = HOWeightsT.transpose(HOWeights);
		Matrix hiddenErrors = hiddenErrors.Product(HOWeightsT, outputErrors);

		Matrix hiddenGradient = hidden.copy();
		for (int i = 0; i < hiddenGradient.rows; i++) {
			for (int j = 0; j < hiddenGradient.cols; j++) {
				float val = hiddenGradient.values[i][j];
				hiddenGradient.values[i][j] = dsigmoid(val);
			}
		}

		hiddenGradient.multiply(hiddenErrors);
		hiddenGradient.multiply(LearningRate);

		Matrix inputT = inputT.transpose(input);
		Matrix DIHWeights = DIHWeights.Product(hiddenGradient, inputT);

		IHWeights.add(DIHWeights);

		Hbias.add(hiddenGradient);
	}
};
