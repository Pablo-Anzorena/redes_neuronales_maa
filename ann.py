from __future__ import division
import numpy as np
import sys
import random
import math

EXECUTION_NUMBER=100

def sigmoid_function(lst_weights, lst_inputs):
    result = lst_weights[0] + reduce(lambda a, b: a+b, [x*y for x,y in zip(lst_weights[1:],lst_inputs)])
    return 1 / (1 + math.pow(math.e, -result))

#combinacion lineal de los pesos y las entradas
def linear_output_function(lst_weights, lst_inputs):
    return reduce(lambda a, b: a+b, [x*y for x,y in zip(lst_weights,lst_inputs)])

#calcula la salida de las neuronas de toda la red utilizando el input, los pesos y la funcion de activacion correspondiente
def output_units_network(input_network, input_hidden_weights, hidden_output_weights):
    output_map = {'output':[], 'hidden':[]}
    for hidden_weights in input_hidden_weights:
        output_map['hidden'].append(sigmoid_function(hidden_weights, input_network))
    for output_weights in hidden_output_weights:
        output_map['output'].append(linear_output_function(output_weights, output_map['hidden']))
    return output_map

def calculate_output_error(target_output, output_values):
    error_result = []
    for x in range(len(output_values)):
        error_result.append(target_output[x]-output_values[x])
    return error_result

def calculate_hidden_error(hidden_output_weights, output_error, hidden_values):
    error_result = []
    for x in range(len(hidden_values)):
        downstream = 0
        for y in range(len(output_error)):
            downstream += output_error[y] * hidden_output_weights[y][x]
        error_result.append(hidden_values[x] * (1-hidden_values[x]) * downstream)
    return  error_result

def update_weights(learning_rate, input_hidden_weights, hidden_error, input_network, hidden_output_weights, output_error, input_hidden):
    for j in range(len(input_hidden_weights)):
        input_hidden_weights[j][0] = input_hidden_weights[j][0] + (learning_rate*hidden_error[j])
        for x in range(len(input_hidden_weights[j][1:])):
            input_hidden_weights[j][x+1] = input_hidden_weights[j][x+1] + (learning_rate*hidden_error[j]*input_network[x])
    for j in range(len(hidden_output_weights)):
        for x in range(len(hidden_output_weights[j])):
            hidden_output_weights[j][x] = hidden_output_weights[j][x] + (learning_rate*output_error[j]*input_hidden[x])

def ann(training_examples, learning_rate, nbr_inputs, nbr_outputs, nbr_hiddens):
    #inicializo los pesos a valores pequenos cercanos a 0
    input_hidden_weights = np.random.uniform(low=-0.05, high=0.05, size=(nbr_hiddens,nbr_inputs+1))
    hidden_output_weights = np.random.uniform(low=-0.05, high=0.05, size=(nbr_outputs,nbr_hiddens))
    for cont in range(EXECUTION_NUMBER):
        for training_example in training_examples:
            input_training = training_example[0]
            target_output = training_example[1]
            output_map = output_units_network(input_training, input_hidden_weights, hidden_output_weights)
            output_error = calculate_output_error(target_output, output_map['output'])
            hidden_error = calculate_hidden_error(hidden_output_weights, output_error, output_map['hidden'])
            #stochastic approx
            update_weights(learning_rate, input_hidden_weights, hidden_error, input_training, hidden_output_weights, output_error, output_map['hidden'])
    return (input_hidden_weights, hidden_output_weights)

#############################################################################
#################################### MAIN ###################################
#############################################################################

if len(sys.argv) != 4 or sys.argv[1] not in ['f','g','h']:
    print 'ERROR de parametros'
    print 'PARAM1:f,g o h'
    print 'PARAM2:hidden_units'
    print 'PARAM3:learning_rate'
    exit()

function = sys.argv[1]
nbr_hiddens = int(sys.argv[2])
learning_rate = float(sys.argv[3])

training_examples = []
nbr_inputs = 1
nbr_outputs = 1


points = np.arange(-1.00,1.05, 0.05)
if function == 'f':#f=x^3-x^2+1
    for point in points:
        training_examples.append(([point],[math.pow(point,3)-math.pow(point,2)+1]))
elif function == 'g':#g=sin(pi*1.5*x)
    for point in points:
        training_examples.append(([point],[math.sin(1.5*math.pi*point)]))
elif function == 'h':#h=1-x^2-y^2
    nbr_inputs = 2
    for index in range(len(points)):
        training_examples.append(([points[index],points[index]],[1-math.pow(points[index],2)-math.pow(points[index],2)])) 

duple = ann(training_examples, learning_rate, nbr_inputs, nbr_outputs, nbr_hiddens)
input_hidden_weights = duple[0]
hidden_output_weights = duple[1]

res = ''
pointsx = np.arange(-1.00,1.05, 0.05)
for pointx in pointsx:
    if function == 'h':
        pointsy = np.arange(-1.00,1.05, 0.05)
        for pointy in pointsy:
            output_map = output_units_network([pointx,pointy], input_hidden_weights, hidden_output_weights)
            res += '(' + str(pointx) + ',' + str(pointy) + ') ' + str(output_map['output'][0]) + ',' + '\n'
    else:
        output_map = output_units_network([pointx], input_hidden_weights, hidden_output_weights)
        res += str(output_map['output'][0]) + ',' + '\n'
print res

