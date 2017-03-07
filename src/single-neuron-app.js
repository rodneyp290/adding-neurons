const utils = require('../src/utils.js');
const neuron = require('../src/neuron.js');

function round_n_places(number, places) {
  return Math.floor((number * (10**places)+0.5))/(10**places)
}

const f = (x,y) => (x + y);
// function neuron to learn f(x,y,z) = 2x + y - 3z

utils.seed_random(2017);

function cost(y1, y2) {
  return ((1/2)*((y1-y2)**2));
}

function cost_derivative(y1, y2) {
  return ((y1-y2)/1);
}

function create_dataset(size) {
  let data = [];
  for (let i = 0; i < size; i++) {
    let input = utils.create_random_int_array(2, 0, 100);
    let label = f(input[0], input[1], input[2])
    data.push([input, label]);
  }
  return data;
}

const demo_int_set = create_dataset(2)
function run_demo(neuron) {
  for (let i = 0; i < demo_int_set.length; i++) {
    let xi = demo_int_set[i][0];
    let x1 = demo_int_set[i][0][0];
    let x2 = demo_int_set[i][0][1];
    let y1 = demo_int_set[i][1];
    let y2 = n.forward(xi);
    console.log('Expected : ' + x1 + ' + ' + x2 + ' = ' + y1);
    console.log('Received : ' + x1 + ' + ' + x2 + ' = ' + y2);
  }
}


const ERROR_MARGIN = 0.025;
let training_set = create_dataset(500000);
let test_set = create_dataset(50000);
let n = new neuron.Neuron(2, 0, 2);


let total_epoch = 10;
run_demo()
for (let epoch = 0; epoch < total_epoch; epoch++) {
  let lr = 1/(2*10000*(epoch+1))
  for (let i = 0; i < training_set.length; i++) {
    let Xi = training_set[i][0];
    let y1 = training_set[i][1];
    let y2 = n.forward(Xi);
    let c = cost(y1,y2);
    let dc = cost_derivative(y1,y2);
    if (i>0&&!(isFinite(dc) && isFinite(n.output) && isFinite(n.w_grads[0]) && isFinite(n.i_grads[0]))) {
      console.err("AHHH! INFINITE NUMBERS!!");
      process.exit(1);
    } else {
    }
    n.backpropagate(dc);
    n.learn_from_grads(lr);
  }
  let sum_cost = 0;
  let correct = 0;
  for (let i = 0; i < test_set.length; i++) {
    let y1 = test_set[i][1];
    let y2 = n.forward(test_set[i][0]);
    let c = cost(y1,y2);
    if (Math.abs(y1-y2) <= (y1 * (ERROR_MARGIN/100))) {
      correct++;
    }
    sum_cost += c;
  }
  console.log('epoch ' + epoch + '/' + total_epoch + ' completed');
  console.log('learning rate: ', lr);
  console.log('Acurracy:');
  let correct_prcnt = Math.floor(100*(100*correct / test_set.length))/100;
  console.log('  ' + correct + '/' + test_set.length + '(' + correct_prcnt + '%) correct within -/+' + ERROR_MARGIN + '%');
  //console.log('  Sum Cost: ' + sum_cost/test_set.length);
  //console.log('  Average Cost (Whole Pop): ' + sum_cost/test_set.length);

  run_demo(n);
}
