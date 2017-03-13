const utils = require('../src/utils.js');
const neuron = require('../src/neuron.js');

// rounding utility function
function round_n_places(number, places) {
  return Math.floor((number * (10**places)+0.5))/(10**places)
}

const f = (x,y) => (x + y);

// seeding random
utils.seed_random(2017);

// Sum of Squared Errors, divided by 2
function cost(label, output) {
  return ((1/2)*((label-output)**2));
}

// Cost derivative with respect to neuron output
function cost_derivative(label, output) {
  return (label-output);
}

// creates dataset of summed integers of a given size
function create_dataset(size) {
  let data = [];
  for (let i = 0; i < size; i++) {
    let input = utils.create_random_int_array(2, 0, 100);
    let label = f(input[0], input[1]);
    data.push([input, label]);
  }
  return data;
}

// Percentage of error for correctness tolerance
const ERROR_MARGIN = 0.025;
let training_set = [];
let test_set = [];
let n = {};
let epoch = 0;


function initialise () {
  // initialisation of datasets
  training_set = create_dataset(500000);
  test_set = create_dataset(50000);

  // making a neuron
  n = new neuron.Neuron(2, 0, 2);

  epoch = 0;
}

function forward(inputs) {
  return JSON.stringify({ "output": n.forward(inputs) });
}

function state() {
  return JSON.stringify(n);
}

function train() {
  // calculate learning rate (decreases as time goes on, but will not reach 0)
  let lr = 1/(2*10000*(epoch+1))
  console.log('training with lr of ' + lr);

  // training loop
  for (let i = 0; i < training_set.length; i++) {
    let inputs = training_set[i][0];
    let label = training_set[i][1];
    let output = n.forward(inputs);
    let dc = cost_derivative(label, output);
    if (i > 0 && !(isFinite(dc) && isFinite(n.output) && isFinite(n.w_grads[0]) && isFinite(n.i_grads[0]))) {
      console.error("AHHH! INFINITE NUMBERS!!");
      process.exit(1);
    }
    n.backpropagate(dc);
    n.learn_from_grads(lr);
  }
  epoch++;
}

function test() {
  // testing loop
  let correct = 0;
  for (let i = 0; i < test_set.length; i++) {
    let label = test_set[i][1];
    let output = n.forward(test_set[i][0]);
    if (Math.abs(label - output) <= (label * (ERROR_MARGIN/100))) {
      correct++;
    }
  }

  return JSON.stringify({ "correct": correct, "size": test_set.length });
}

module.exports = { forward, initialise, state, test, train };
