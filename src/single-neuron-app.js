const utils = require('../src/utils.js');
const neuron = require('../src/neuron.js');

function round_n_places(number, places) {
  return Math.floor((number * (10**places)+0.5))/(10**places)
}

const f = (x,y) => (x + y);
// function neuron to learn f(x,y,z) = 2x + y - 3z

utils.seed_random(2017);

function create_dataset(size) {
  let inputs = [];
  let labels = [];
  for (let i = 0; i < size; i++) {
    let input = utils.create_random_array(2, -1, 1);
    let output = f(input[0], input[1], input[2])
    inputs.push(input);
    labels.push(output);
  }
  return [inputs, labels];
}

function cost(y1, y2) {
  return ((1/2)*((y1-y2)**2));
}

function cost_derivative(y1, y2) {
  return ((y1-y2)/1);
}

function create_int_dataset(size) {
  let inputs = [];
  let labels = [];
  for (let i = 0; i < size; i++) {
    let input = utils.create_random_int_array(2, 0, 100);
    let output = f(input[0], input[1], input[2])
    inputs.push(input);
    labels.push(output);
  }
  return [inputs, labels];
}

demo_int_set = create_int_dataset(10)
function run_demo(neuron) {
  for (let i = 0; i < demo_int_set[0].length; i++) {
    let xi = demo_int_set[0][i];
    let x1 = demo_int_set[0][i][0];
    let x2 = demo_int_set[0][i][1];
    let y1 = demo_int_set[1][i];
    let y2 = n.forward(xi);
    console.log('Expected : ' + x1 + ' + ' + x2 + ' = ' + y1 );
    console.log('Received : ' + x1 + ' + ' + x2 + ' = ' + y2 + '(cost: ' + cost(y1,y2) + ')' );
  }
}

let training_set = create_int_dataset(500000);
let test_set = create_int_dataset(50000);
let n = new neuron.Neuron(2, 0, 2);
n.activation = (z) => round_n_places(z, 3);

let total_epoch = 100;
let prev_Xi = 0;
let prev_y1 = 0;
let prev_y2 = 0;
let prev_dc = 0;
let prev_n = JSON.stringify(n);

for (let epoch = 0; epoch < total_epoch; epoch++) {
  let lr = 1/(2*10000*(epoch+1))
  for (let i = 0; i < training_set[0].length; i++) {
    let Xi = training_set[0][i];
    let y1 = training_set[1][i];
    let y2 = n.forward(Xi);
    let c = cost(y1,y2);
    let dc = cost_derivative(y1,y2);
    // console.log('dc - ' + dc + ' n.output - ' + n.output);
    // console.log(!(isFinite(dc) && isFinite(n.output)));
    if (i>0&&!(isFinite(dc) && isFinite(n.output) && isFinite(n.w_grads[0]) && isFinite(n.i_grads[0]))) {
      console.log('lr: ' + lr);
      console.log('prev_y1: ' + prev_y1);
      console.log('y1: ' + y1);
      console.log('prev_Xi: ' + prev_Xi);
      console.log('Xi: ' + Xi);
      console.log('prev_y2: ' + prev_y2);
      console.log('y2: ' + y2);
      console.log('prev_dc: ' + prev_dc);
      console.log('dc: ' + dc);
      console.log('c: ' + c);
      console.log(prev_n);
      console.log(n);
      exit();
    } else {
      // console.log('W: ' + n.weights);
      // console.log('Wg: ' + (n.w_grads));
      // console.log('Wgrs: ' + numeric.mul(numeric.sub(n.w_grads, n.weights), lr));
    }
    prev_Xi = Xi;
    prev_y1 = y1;
    prev_y2 = y2;
    prev_dc = dc;
    prev_n = JSON.parse(JSON.stringify(n));
    prev_n.__proto__ = neuron.Neuron.prototype;
    n.backpropagate(dc);
    n.learn_from_grads(lr);
  }
  let sum_cost = 0;
  let correct = 0;
  for (let i = 0; i < test_set.length; i++) {
    let y1 = test_set[1][i]
    let y2 = n.forward(test_set[0][i]);
    let c = cost(y1,y2);
    // console.log('y1: ' + y1);
    // console.log('y2: ' + y2);
    // console.log('c: ' + c);
    if (c === 0) {
      correct++;
    } else {
      sum_cost += c;
    }
  }
  console.log('epoch ' + epoch + '/' + total_epoch + ' completed');
  console.log('learning rate: ', lr);
  console.log('Acurracy:');
  console.log('  ' + correct + '/' + test_set[1].length + ' correct to ' + PRECISION + ' decimal places');
  console.log('  Sum Cost: ' + sum_cost/test_set[1].length);
  console.log('  Average Cost (Whole Pop): ' + sum_cost/test_set[1].length);
  console.log('  Average Cost (Incorrect Pop): ' + sum_cost/(test_set[1].length-correct));

  run_demo(n);
}
