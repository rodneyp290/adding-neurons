const express = require('express');

const app = express();

const port = 3000;

const api = require('./single-neuron-functions.js');

let initialised = false;

const init_check = () => {
  if (!initialised) {
    console.log('initialising');
    api.initialise();
    initialised = true;
  }
}

app.get('/reset', (req, resp) => {
  console.log('/reset');
  initialised = false;
  api.initialise();
  resp.end(api.state());
})

app.get('/state', (req, resp) => {
  console.log('/state');
  init_check();
  resp.end(api.state());
})

app.get('/train', (req, resp) => {
  console.log('/train');
  init_check();
  api.train();
  resp.end(api.state());
})

app.get('/test', (req, resp) => {
  console.log('/test');
  init_check();
  resp.end(api.test());
})

app.get('/forward', (req, resp) => {
  console.log('/forward');
  init_check();
  let inputs = [req.query.a, req.query.b];
  resp.end(api.forward(inputs));
})

app.listen(port, err => {
  if (err) {
    return console.error('ERROR: ', err)
  }

  console.log(`server is listening on ${port}`)
})
