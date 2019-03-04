"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN0.py
"""

import os
import csv
import time
import sys
sys.path.append("./../ABAGAIL/ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 13
HIDDEN_LAYER1 = 20
HIDDEN_LAYER2 = 20
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = './../NN_OUTPUT/BP_LOG.txt'
# OUTFILE = './../NN_OUTPUT/BP_LOG2.txt'


def initialize_instances(infile):
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) <= 0 else 1))
            instances.append(instance)

    return instances
	

def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc
	
	
def train(oa, network, oaName, training_ints, validation_ints, testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print ("\nError results for %s\n---------------------------" % (oaName,))
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
    	times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
    	    MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_val, acc_val = errorOnDataSet(network,validation_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,times[-1]);print(txt)
            with open(OUTFILE.replace('XXX',oaName),'a+') as f:
                f.write(txt)

def main():
    """Run this experiment"""
    training_ints = initialize_instances('./../data/heart_train.csv')
    testing_ints = initialize_instances('./../data/heart_test.csv')
    validation_ints = initialize_instances('./../data/heart_validation.csv')
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    rule = RPROPUpdateRule()
    oa_names = ["Backprop"]
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER],relu)
    train(BatchBackPropagationTrainer(data_set,classification_network,measure,rule), classification_network, 'Backprop', training_ints, validation_ints, testing_ints, measure)
        


if __name__ == "__main__":
    with open(OUTFILE,'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_val','MSE_tst','acc_trg','acc_val','acc_tst','elapsed'))
    main()