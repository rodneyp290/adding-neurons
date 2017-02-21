const numeric = require('numeric');
const utils = require('./utils.js');

class Neuron {

  /***
   * Constructor requires an input_size (number of inputs), but
   * also accepts two optional arguments specifying the
   * minimum (w_min) and maximum (w_max) for the initial weights
   * that will be generated (default w_min = -0.5, w_max = 0.5 )
   **/
  constructor(input_size, w_min, w_max) {
    // initial optional weight min & max
    w_min = w_min === undefined ? -1 : w_min;
    w_max = w_max === undefined ? +1 : w_max;
    this.inputs = numeric.rep(0, input_size)
    this.weights = utils.create_random_array(input_size, w_min, w_max);
    this.w_grads = numeric.rep(0, input_size);
    this.i_grads = numeric.rep(0, input_size);
    this.final_grad = 0;
    this.output = 0;
  }

  /***
   * Forwards the inputs through the neuron storing the intermediate
   * values which may be needed for for back propagation
   * TODO: Change unnecessary class variables into local variables once
   *       backprop/learning is working
   **/
  forward(inputs) {
    this.inputs = inputs;
    this.agg_inputs = this.aggregation(inputs);
    this.output = this.activation(this.agg_inputs);
    return this.output;
  }

  /***
  * Aggregation function to join all the inputs together for
  * activation
  **/
  aggregation(inputs) {
    let weighted = numeric.mul(inputs, this.weights);
    let weightedSum = numeric.sum(weighted);
    return weightedSum;
  }

  /***
   * Activation function for the neuron after weights and
   * aggregation have been applied.
   * Often sigmoid, but just keeping simple 1z for now
   * (Separated for maintainablility)
   **/
  activation(z) {
    return 1 * z;
  }

  /***
   * backpropagates the passed in gradient through the neuron
   * to be used to learn
   **/
  backpropagate(grad) {
    this.final_grad = grad;
    this.pre_act_grad = this.derive_activation(grad);
    let grads = this.derive_aggregation(grad);
    this.i_grads = grads[0];
    this.w_grads = grads[1];
    return this.i_grads;
  }

  /***
   * Calculates the gradient pre-activation given the
   * resulting gradient
   **/
  derive_activation(grad) {
    // a(z) = z, therefore using the power rule
    // a'(z) = 1;
    return 1 * grad;
  }

  /***
   * Calculates the gradients pre-aggregation in respects to
   * each weight, given the resulting gradient
   **/
  derive_aggregation(grad) {
    // a(inputs) = sum(inputs*weights);
    // since derivative of sum is equal to sum of derivatives
    // inner function, we can focus on inputs*weights.
    // this means each gradient is now just input*weights
    // so we are looking at a linear function a(w, i) = iw again.
    // thus a'(w) = i
    // (da/dW)
    let weight_grads = numeric.mul(this.inputs, grad);
    // (da/dI)
    let input_grads = numeric.mul(this.weights, grad); // for further backprop
    return [input_grads, weight_grads];
  }

  /***
   * Adjusts weights by a fraction (step_size) of the gradients
   * (w_grad) calculated with respect to the weights/
   **/
  learn_from_grads(step_size) {
    step_size = (step_size === undefined) ? 0.005 : step_size;
    let regularised_grads = numeric.sub(this.w_grads, this.weights);
    let step = numeric.mul(regularised_grads, step_size);
    this.weights = numeric.add(this.weights, step);
  }
}
//
module.exports = { Neuron }
