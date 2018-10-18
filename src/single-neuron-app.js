const utils = require('../src/utils.js');
const neuron = require('../src/neuron.js');


const f = (x,y) => (x + y);

// seeding random
utils.seed_random(2017);

// Sum of Squared Errors, divided by 2
function cost(label, output) {
  return ((label-output)**2);
}

// Cost derivative with respect to neuron output
function cost_derivative(label, output) {
  return -2*(label-output);
}

// creates dataset of summed integers of a given size
function create_dataset(size) {
  let data = [];
  for (let i = 0; i < size; i++) {
    let input = utils.create_random_array(2, 0, 100);
    let label = f(input[0], input[1]);
    data.push([input, label]);
  }
  return data;
}

const demo_int_set = [
  [[2.2,1.5],[3.7]],
  [[1024,2048],[3072]]
]
// Demo of expected value versus actual
function run_demo(neuron) {
  console.log('Demo: ')
  for (let i = 0; i < demo_int_set.length; i++) {
    let input = demo_int_set[i][0];
    let label = demo_int_set[i][1];
    let output = neuron.forward(input);
    console.log('  Expected : ' + input[0] + ' + ' + input[1] + ' = ' + label);
    console.log('  Actual   : ' + input[0] + ' + ' + input[1] + ' = ' + output);
  }
}

// Percentage of error for correctness tolerance
const ERROR_MARGIN = 0.01;

// initialisation of datasets
let training_set = create_dataset(500000);
let test_set = create_dataset(50000);

// making a neuron
let n = new neuron.Neuron(2, 0, 2);

// Define number of epochs
let total_epoch = 10;

// demo random weights
run_demo(n);

// Loop through training and testing
for (let epoch = 0; epoch < total_epoch; epoch++) {
  // calculate learning rate (decreases as time goes on, but will not reach 0)
  let lr = 1/(2*100000*(epoch+1))

  // training loop
  for (let i = 0; i < training_set.length; i++) {
    let inputs = training_set[i][0];
    let label = training_set[i][1];
    let output = n.forward(inputs);

    //let c = cost(label,output); // Not actually needed
    let dc = cost_derivative(label,output);

    n.backpropagate(dc);
    n.learn_from_grads(lr);

    // Check for infinite numbers
    if (!(isFinite(dc) && isFinite(output) && isFinite(n.w_grads[0]) && isFinite(n.i_grads[0]))) {
      console.error("AHHH! INFINITE NUMBERS!!");
      process.exit(1);
    }

  }

  console.log('epoch ' + (epoch + 1) + '/' + total_epoch + ' completed');

  // testing loop
  let correct = 0;
  for (let i = 0; i < test_set.length; i++) {
    let label = test_set[i][1];
    let output = n.forward(test_set[i][0]);
    if (Math.abs(label - output) <= (label * (ERROR_MARGIN/100))) {
      correct++;
    }
  }

  let correct_prcnt = utils.round_n_places(100*(correct/test_set.length),2);
  console.log('Accuracy:  ' + correct + '/' + test_set.length + '(' + correct_prcnt + '%) correct within -/+' + ERROR_MARGIN + '%');

}

run_demo(n);
