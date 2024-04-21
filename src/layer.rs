use std::f64::consts::E;

use rand::Rng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn d_sigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

pub struct Layer {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    pub output_num: usize,
    pub weights: Vec<f64>,
    pub delta: Vec<f64>,
}

impl Layer {
    pub fn new(inputs: &Vec<f64>, output_num: usize) -> Layer {
        let inputs: Vec<f64> = inputs.clone();
        let outputs: Vec<f64> = Vec::with_capacity(output_num);
        let weights: Vec<f64> = Self::init_weights(inputs.len(), output_num);
        let delta: Vec<f64> = Vec::new();

        Layer {
            inputs,
            outputs,
            output_num,
            weights,
            delta
        }
    }

    fn init_weights(input_num: usize, output_num: usize) -> Vec<f64> {
        let mut weights: Vec<f64> = Vec::new();

        for _ in 0..input_num*output_num {
            let rnd: f64 = rand::thread_rng().gen_range(0.0..1.0);
            weights.push(rnd);
        }

        weights
    }

    fn get_weight(num_in: usize, weights: &Vec<f64>, output: usize, input: usize) -> f64 {
        let index: usize = output * num_in + input;
        weights[index]
    }

    pub fn forward(&mut self) {
        let num_in: usize = self.inputs.len();
        let num_out: usize = self.output_num;
        
        for i in 0..num_out {
            let mut output: f64 = 0.0;
            for j in 0..self.inputs.len() {
                output += self.inputs[j] * Self::get_weight(num_in, &self.weights, i, j);
            }
            self.outputs.push(sigmoid(output));
        }
    }

    pub fn backward(&mut self, desired_out: &Vec<f64>) {
        for i in 0..self.weights.len() {
            let weight = self.weights[i];

            self.weights[i] += d_sigmoid(weight);
        }

        self.outputs.clear();

        for i in 0..self.output_num {
            for j in 0..self.inputs.len() {
                let val = desired_out[i] / Self::get_weight(self.inputs.len(), &self.weights, i, j);
                self.outputs.push(val);
            }
        }
    }
}