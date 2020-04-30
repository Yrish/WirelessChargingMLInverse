# Overview

We have an ANSYS simulation that takes 7 simulation input parameters and
produces 5 simulation output values.  We have a collection of thousands of
records of 7 various input values and the actual output values for these inputs
that the simulation produced.

The simulation can be thought of as a function `f`.  Given 5 output values, we
would like to find 7 input values that can produce this output.  We are seeking
to approximate any right inverse of `f`, call it `g`, such that `f(g(x)) = id`.

We implement two ANN (artificial neural network) approaches and compare the
accuracy of them.

## First approach: linear network

We construct, straightforwardly, a mostly linear neural that includes dense
layers.  It is input the simulation output parameters and outputs predicted
simulation input parameters.  This is the first model that we consider.

## Second approach: GAN

We compare the accuracy of the first model with that of the second: a GAN.

A GAN consists of two subnetworks: a generator and a discriminator.  The
discriminator is input either real or generated (sometimes called "fake") data.
Real data can come from training data, and generated data comes from the second
network: the generator.

The generator is input parameters that determine the generated content, and it
is tasked with fooling the discriminator: its goal is to maximize the
discriminator's error, to make generated content as similar to real / training
data as possible, as far as the discriminator network can tell.

In our application, ultimately, we would like a generator model that takes as
model input 5 desired simulation output values plus an additional `n >= 0`
input models for generation, and that produces as model output 7 simulation
input values.  The discriminator will have access to both the 5 simulation
output values that were provided as input to the generator (but no other
values) and the 7 simulation input values that the generator output.  The
discriminator thus has an input dimensionality of 12.  If it's fed real data,
the real data comes from real training data (arbitrarily chosen 7 simulation
inputs and the 5 actual outputs that the ANSYS simulation produced).

## Third approach: GAN with reversed model.

A third approach is similar to the second but, rather than assuming that output
from the generator is fake when training the discriminator, uses the reverse
GAN model to estimate the accuracy of generated input.
