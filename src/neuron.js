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
  }

  /***
   * Forwards the inputs through the neuron storing the intermediate
   * values which may be needed for for back propagation
   **/
  forward(inputs) {
    this.inputs = inputs;
    let agg_inputs = this.aggregation(inputs);
    return this.activation(agg_inputs);
  }

  /***
  * Aggregation function to combine all the inputs together for
  * activation
  **/
  aggregation(inputs) {
    let weighted_i = numeric.mul(inputs, this.weights);
    return numeric.sum(weighted_i);
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
    let pre_act_grad = this.backprop_activation(grad)
    let grads = this.backprop_aggregation(pre_act_grad);
    this.i_grads = grads[0];
    this.w_grads = grads[1];
    return this.i_grads;
  }

  /***
   * Calculates the gradient pre-activation given the
   * resulting gradient
   **/
  backprop_activation(grad) {
    // a(z) = z, therefore using the power rule
    // a'(z) = 1;
    return 1 * grad;
  }

  /***
   * Calculates the gradients pre-aggregation in respects to
   * each weight, given the resulting gradient
   **/
  backprop_aggregation(grad) {
    // a(inputs) = sum(inputs*weights);
    // since derivative of sum is equal to sum of derivatives
    // inner function, we can focus on inputs*weights.
    // this means each gradient is now just input*weights
    // so we are looking at a linear function a(w, i) = iw again.
    // thus a'(w) = i*w
    // (da/dW)
    let weight_grads = numeric.mul(this.inputs, grad);
    // (da/dI)
    let input_grads = numeric.mul(this.weights, grad); // for further backprop
    return [input_grads, weight_grads];
  }

  /***
   * Adjusts weights by a fraction (learning_rate) of the gradients
   * (w_grad) calculated with respect to the weights/
   **/
  learn_from_grads(learning_rate) {
    learning_rate = (learning_rate === undefined) ? 0.005 : learning_rate;
    let regularised_grads = numeric.sub(this.w_grads, this.weights);
    let step = numeric.mul(regularised_grads, learning_rate);
    // let step = numeric.mul(this.w_grads, learning_rate);
    this.weights = numeric.sub(this.weights, step);
  }
}
// Export Neuron class
module.exports = { Neuron }
