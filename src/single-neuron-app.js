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

const demo_int_set = create_dataset(2)
// Demo of expected value versus actual
function run_demo(neuron) {
  console.log('Demo: ')
  for (let i = 0; i < demo_int_set.length; i++) {
    let input = demo_int_set[i][0];
    let label = demo_int_set[i][1];
    let output = n.forward(input);
    console.log('  Expected : ' + input[0] + ' + ' + input[1] + ' = ' + label);
    console.log('  Actual   : ' + input[0] + ' + ' + input[1] + ' = ' + output);
  }
}

// Percentage of error for correctness tolerance
const ERROR_MARGIN = 0.025;

// initialisation of datasets
let training_set = create_dataset(500000);
let test_set = create_dataset(50000);

// making a neuron
let n = new neuron.Neuron(2, 0, 2);

// Define number of epoches
let total_epoch = 10;

// run demo to see output with random weights
run_demo();

// Loop through training and testing
for (let epoch = 0; epoch < total_epoch; epoch++) {
  // calculate learning rate (decreases as time goes on, but will not reach 0)
  let lr = 1/(2*10000*(epoch+1))

  // training loop
  for (let i = 0; i < training_set.length; i++) {
    let inputs = training_set[i][0];
    let label = training_set[i][1];
    let output = n.forward(inputs);
    let c = cost(label,output);
    let dc = cost_derivative(label,output);
    if ( i > 0 && !(isFinite(dc) && isFinite(output) && isFinite(n.w_grads[0]) && isFinite(n.i_grads[0]))) {
      console.error("AHHH! INFINITE NUMBERS!!");
      process.exit(1);
    }
    n.backpropagate(dc);
    n.learn_from_grads(lr);
  }

  // testing loop
  let correct = 0;
  for (let i = 0; i < test_set.length; i++) {
    let label = test_set[i][1];
    let output = n.forward(test_set[i][0]);
    if (Math.abs(label - output) <= (label * (ERROR_MARGIN/100))) {
      correct++;
    }
  }

  // results of epoch
  console.log('epoch ' + (epoch + 1) + '/' + total_epoch + ' completed');
  console.log('learning rate: ', lr);
  console.log('Acurracy:');
  let correct_prcnt = Math.floor(100*(100*correct / test_set.length))/100;
  console.log('  ' + correct + '/' + test_set.length + '(' + correct_prcnt + '%) correct within -/+' + ERROR_MARGIN + '%');

  run_demo(n);
}
