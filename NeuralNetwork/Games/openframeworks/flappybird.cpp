// Flappy Bird Neural Network
// in openframeworks
// using Toy Neural Network
// this is the ofApp.cpp file
// for c++ by https://github.com/Enrique-ZA

#include <stdlib.h>
#include "ofApp.h"
#include <math.h> 
#include <string>
#include <iostream>
//--------------------------------------------------------------

// Matrix

class Matrix {
	public:

	int rows, cols;
	vector<vector<float>> values;

	Matrix(int rows, int cols) {
		this->rows = rows;
		this->cols = cols;
		values.resize(rows);
		for (int i = 0; i < rows; i++) {
			values[i].resize(cols);
		}
	}

	Matrix() {
		rows = 1;
		cols = 1;
		values.resize(rows);
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
		result.resize(a.rows);
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
		vector<float> arr;
		arr.resize(n);
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

	NeuralNetwork() {
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
			return val + ofGaussian() * 0.1;
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
//--------------------------------------------------------------

// Global variables

int width = 640;
int height = 480;
int nnInput = 5;
int nnHidden = 8;
int nnOutput = 2;

// Classes

class Agent {

	public:

	float y = height / 2;
	float x = 64;
	float w = 32;
	float gravity = 0.7;
	float lift = -16;
	float velocity = 0;
	long score = 0;
	float fitness = 0;
	NeuralNetwork brain;

	Agent() {
		NeuralNetwork n(nnInput, nnHidden, nnOutput);
		brain = brain.copy(n);
	}

	Agent(NeuralNetwork n) {
		brain = brain.copy(n);
	}

	void update();
	void mutate();
	void think();
	void up();
	void render();
	bool offScreen();
};

class Pipe {

	public:

	float spacing = 75;
	float top = ofRandom(height / 8, (height / 2)*1.5);
	float bottom = height - (top + spacing);
	float x = width;
	float w = 80;
	float speed = 6;
	bool highlight = false;

	void render();
	void update();
	bool offScreen();
	bool hits(Agent &agent);
};

// Class variables

//Agent agent;
int total = 350;
long counter = 0;
long generation = 1;
vector<Agent> agents;
vector<Agent> savedAgents;
vector<Pipe> pipes;

// Class functions

void Agent::think() {
	vector<float> inputs;
	inputs.resize(nnInput);
	if (pipes.size() > 0) {

		Pipe closest;
		float closestDist = INFINITY;
		int n = pipes.size();
		for (int i = 0; i < n; i++) {
			float d = (pipes[i].x + pipes[i].w) - x;
			if (d < closestDist && d > 0) {
				closest = pipes[i];
				closestDist = d;
			}
		}

		inputs[0] = ofMap(y, 0, height, 0, 1);
		inputs[1] = ofMap(closest.top, 0, height, 0, 1);
		inputs[2] = ofMap(closest.bottom, 0, height, 0, 1);
		inputs[3] = ofMap(closest.x, 0, width, 0, 1);
		inputs[4] = velocity / 10;
	} else {
		inputs[0] = ofRandom(0, 1);
		inputs[1] = ofRandom(0, 1);
		inputs[2] = ofRandom(0, 1);
		inputs[3] = ofRandom(0, 1);
		inputs[4] = ofRandom(0, 1);
	}

	vector<float> outputs = brain.feedForward(inputs);

	if (outputs[0] > outputs[1]) {
		if (outputs[0] > 0.8 && velocity >= 0) {
			up();
		}
	}
}

void Agent::update() {
	score++;

	velocity += gravity;
	y += velocity;
}

void Agent::mutate() {
	brain.mutate(0.1);
}

void Agent::up() {
	velocity += lift;
}

bool Agent::offScreen() {
	if (y > height) {
		return true;
	}
	else if (y < 0) {
		return true;
	}
	return false;
}

void Agent::render() {
	ofSetColor(255, 255, 0, 100);
	ofFill();
	ofDrawEllipse(x, y, w, w);
}

bool Pipe::offScreen() {
	return (x < -w);
}

bool Pipe::hits(Agent &agent) {
	if (agent.y < top || agent.y > height - bottom) {
		if (agent.x > x && agent.x < x + w) {
			highlight = true;
			return true;
		}
	}
	highlight = false;
	return false;
}

void Pipe::update() {
	x -= speed;
}

void Pipe::render() {
	if (!highlight) {
		ofSetColor(0, 255, 0, 255);
	}
	else {
		ofSetColor(255, 0, 0, 255);
	}	
	ofFill();
	ofDrawRectangle(x, 0, w, top);
	ofDrawRectangle(x, height-bottom, w, bottom);
}

// Genetic Algorithm

void calcFitness() {
	long sum = 0;
	for (auto &agent : savedAgents) {
		sum += agent.score;
	}
	for (auto &agent : savedAgents) {
		agent.fitness = (float)(agent.score)/sum;
	}
}

Agent selectCandidate() {
	int index = 0;
	float r = ofRandom(1);
	while (r > 0) {
		if (index >= total) {
			index = total - 1;
		}
		if (index < 0) {
			index = 0;
		}
		r -= savedAgents[index].fitness;
		index++;
	}
	index--;
	if (index < 0 || index > (savedAgents.size() - 1)) {
		index = savedAgents.size() - 1;
	}
	Agent agent = savedAgents[index];
	Agent child(agent.brain);
	child.mutate();
	return child;
}

void nextGeneration() {

	system("CLS");

	cout << "generation: " << generation << endl;

	if (agents.size() > 0) {
		cout << "ERROR agents size!" << endl;
	}

	calcFitness();

	cout << "Best score: " << savedAgents[savedAgents.size() - 1].score << endl;

	for (int i = 0; i < total; i++) {
		agents.push_back(selectCandidate());
	}

	if (agents.size() != total) {
		cout << "ERROR saved agents size!" << endl;
	}

	generation++;
}

//--------------------------------------------------------------
void ofApp::setup() {
	for (int i = 0; i < total; i++) {
		agents.push_back(Agent());
	}
	pipes.push_back(Pipe());
}

//--------------------------------------------------------------
void ofApp::update(){
    if (agents.size() > 0) {
        for (int i = agents.size() - 1; i >= 0; i--) {
            agents[i].think();
            agents[i].update();
        }

        for (int j = agents.size() - 1; j >= 0; j--) {
            if (agents[j].offScreen()) {
                savedAgents.push_back(agents[j]);
                agents.erase(agents.begin() + j);
            }
        }

        if (pipes.size() > 0) {
            for (auto &pipe : pipes) {
                for (int j = agents.size() - 1; j >= 0; j--) {
                    if (pipe.hits(agents[j])) {
                        //cout << "HIT" << endl;
                        savedAgents.push_back(agents[j]);
                        agents.erase(agents.begin() + j);
                    }
                }
            }

            for (int i = pipes.size() - 1; i >= 0; i--) {
                pipes[i].update();
                if (pipes[i].offScreen()) {
                    pipes.erase(pipes.begin() + i);
                }
            }
        }

        if (counter % 75 == 0 && counter > 0) {
            pipes.push_back(Pipe());
        }
    }

    if (agents.size() == 0) {
        counter = 0;
        nextGeneration();
        if (pipes.size() > 0) {
            pipes.clear();
        }
        if (pipes.size() == 0) {
            pipes.push_back(Pipe());
        }
        if (savedAgents.size() > 0) {
            savedAgents.clear();
        }
        if (savedAgents.size() > 0) {
            cout << "ERROR saved agents size!" << endl;
        }
    }
    counter++;
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofSetBackgroundColor(0, 0, 0, 255);

	if (agents.size() > 0) {
		for (int i = agents.size() - 1; i >= 0; i--) {
			agents[i].render();
		}
		if (pipes.size() > 0) {
			for (int i = pipes.size() - 1; i >= 0; i--) {
				pipes[i].render();
			}
		}		
	}	
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
