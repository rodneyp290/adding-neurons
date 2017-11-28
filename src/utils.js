const numeric = require('numeric');

function seed_random (seed) {
  numeric.seedrandom.seedrandom(seed)
}

function random_float (high, low) {
  return (numeric.seedrandom.random() * (high - low) + low);
}

function create_random_array (n, high, low) {
  let array = [];
  for (let i = 0; i < n; i++) {
    array.push(random_float(high, low));
  }
  return array;
}

function create_random_int_array (n, high, low) {
  let array = [];
  for (let i = 0; i < n; i++) {
    array.push(Math.floor(random_float(high, low)));
  }
  return array;
}

module.exports = { create_random_array, create_random_int_array, seed_random, random_float }
