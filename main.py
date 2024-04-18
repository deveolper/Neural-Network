# 3.	Maak gebruik van arrays
# Als het gebruik van externe libraries alleen is toegestaan in het geval van numpy, dan forceert dit om numpy te gebruiken. Python heeft namelijk geen arrays, maar regel 6 maakt dat optioneel
# 5.	We raden het niet aan maar voor de Python die-hards, mag het ook in Python. Echter, zorg dat je of Numpy gebruikt als library of, een andere structuur die arrays vervangt. Zie de opmerking hieronder voor meer info hierover
# Dus numpy arrays zijn toegestaan. Maar array-vervangers zijn dus verplicht. Een list vervangt ook een array.
# 6.	Gebruik geen externe bibliotheken voor het bouwen van het neurale netwerk, met uitzondering van #5 als je daarvoor kiest.
# Ik wil graag nadruk leggen op "externe" en op "als je daarvoor kiest" wat dus betekent dat het optioneel is om geen externe libs te gebruiken
# Ook wil ik graag aangeven dat er een verschil is tussen built-in, extern, en intern.

# 7.	Instrueer de LLM dat het geen backpropagation mag gebruiken en ook niet Gradient Descent algoritme. Vraag of het een simpele manier kan gebruiken op basis van de ‘error’ om de gewichten te trainen
# Dus we moeten Hill climbing of een genetic algorithm implementeren

# 6.	Train het netwerk met behulp van de gegeven training samples en evalueer de prestaties ervan.
# Probleem: er zijn geen training samples gegeven

import random
import math
import typing
import copy


TEST_SIZE = 0.15
VALIDATION_SIZE = 0.30
TRAIN_SIZE = 1 - TEST_SIZE - VALIDATION_SIZE

FOURTH_ROOT_OF_SIZE = 15 # 15 => 50625 samples

# Omdat er geen training samples gegeven zijn, ga ik hier wat definieren.
# Probleemomschrijving: checken of de som van de 4 getallen minstens 2.0 is.
X = [
    [a / FOURTH_ROOT_OF_SIZE, b / FOURTH_ROOT_OF_SIZE, c / FOURTH_ROOT_OF_SIZE, d / FOURTH_ROOT_OF_SIZE]
    for a in range(FOURTH_ROOT_OF_SIZE)
    for b in range(FOURTH_ROOT_OF_SIZE)
    for c in range(FOURTH_ROOT_OF_SIZE)
    for d in range(FOURTH_ROOT_OF_SIZE)
]

Y = [
    [1.0] if sum(testcase) >= 2.0 else [0.0]
    for testcase in X
]

data_size = len(X)

order = random.sample(range(data_size), data_size)

X = [
    X[order_element]
    for order_element in order
]

Y = [
    Y[order_element]
    for order_element in order
]

X_TRAIN, X_VALIDATION, X_TEST = X[:int(data_size * TRAIN_SIZE)], X[int(data_size * TRAIN_SIZE):int(data_size * TRAIN_SIZE) + int(data_size * VALIDATION_SIZE)], X[int(data_size * TRAIN_SIZE) + int(data_size * VALIDATION_SIZE):]
Y_TRAIN, Y_VALIDATION, Y_TEST = Y[:int(data_size * TRAIN_SIZE)], Y[int(data_size * TRAIN_SIZE):int(data_size * TRAIN_SIZE) + int(data_size * VALIDATION_SIZE)], Y[int(data_size * TRAIN_SIZE) + int(data_size * VALIDATION_SIZE):]


class Node:
    def __init__(self, bias: float, weights: list[float], activation: typing.Callable[[float], float]) -> None:
        self.bias = bias
        self.weights = weights
        self.activation = activation

    def forward(self, inputs: list[float]) -> float:
        return self.activation(
            sum(
                weight * activation
                for weight, activation in zip(self.weights, inputs)
            )
        )


class Layer:
    node: list[Node]

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes

    def forward(self, inputs: list[float]) -> list[float]:
        return [
            node.forward(inputs)
            for node in self.nodes
        ]


class NeuralNetwork:
    layers: list[Layer]

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def sum_of_squared_errors(self, expected_outputs: list[list[float]], actual_outputs: list[list[float]]) -> float:
        total_error = 0.0

        for expected, actual in zip(expected_outputs, actual_outputs):
            for expected_item, actual_item in zip(expected, actual):
                total_error += (expected_item - actual_item) ** 2

        return total_error
    
    def sum_of_absolute_errors(self, actual_outputs: list[list[float]], calculated_outputs: list[list[float]]) -> float:
        summation: float = 0.0

        for actual, calculated in zip(actual_outputs, calculated_outputs):
            for actual_item, calculated_item in zip(actual, calculated):
                summation += abs(actual_item - calculated_item)

        return summation / len(actual_outputs)

    def forward(self, inputs: list[float]) -> list[float]:
        last_layer_outputs = inputs

        for layer in self.layers:
            last_layer_outputs = layer.forward(last_layer_outputs)

        return last_layer_outputs

    def cost_from_inputs(self, expected_outputs: list[list[float]], inputs: list[list[float]]) -> float:
        actual_outputs = [
            self.forward(testcase)
            for testcase in inputs
        ]
        return self.cost(expected_outputs, actual_outputs)

    def cost(self, expected_outputs: list[list[float]], actual_outputs: list[list[float]]) -> float:
        return sum(
            self.cost_for_testcase(expected_output, actual_output)
            for expected_output, actual_output in zip(expected_outputs, actual_outputs)
        )

    def forward_and_cost_for_testcase(self, expected_output: list[float], inputs: list[float]) -> float:
        return self.cost_for_testcase(expected_output, self.forward(inputs))

    def cost_for_testcase(self, expected_output: list[float], actual_output: list[float]) -> float:
        return sum((expected - actual) ** 2 for expected, actual in zip(expected_output, actual_output))

    def train(self, data: list[list[float]], labels: list[list[float]], validation_data: list[list[float]], validation_labels: list[list[float]], epochs: int, delta: float):
        best_validation_cost = self.cost_from_inputs(validation_labels, validation_data)

        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}, validation cost: {best_validation_cost}")

            old_network = copy.deepcopy(self)

            self._train(data, labels, delta)

            new_cost = self.cost_from_inputs(validation_labels, validation_data)

            if new_cost >= best_validation_cost:
                self.layers = copy.deepcopy(old_network.layers)
                break
            else:
                best_validation_cost = new_cost

    def _train(self, data: list[list[float]], labels: list[list[float]], delta: float):
        best_cost = self.cost_from_inputs(labels, data)

        for layer in self.layers:
            for node in layer.nodes:
                for index, weight in enumerate(node.weights):
                    print(f"training cost: {best_cost:.10}          ", end="\r")

                    best_weight = weight

                    for new_weight in [max(best_weight - delta, -1.0), best_weight, min(best_weight + delta, 1.0)]:
                        node.weights[index] = new_weight

                        new_cost = self.cost_from_inputs(labels, data)

                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_weight = new_weight

                    node.weights[index] = best_weight

                best_bias = node.bias

                for bias in [max(node.bias - delta, -1.0), node.bias, min(node.bias + delta, 1.0)]:

                    node.bias = bias

                    new_cost = self.cost_from_inputs(labels, data)

                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_bias = bias

                node.bias = best_bias


def sigmoid(x: float) -> float:
    return 1.0 / (1 + math.e ** (-x))


def main():
    # Structure
    nodes_per_layer = [
        4, 2, 1
    ]
    # Weights zijn willekeurig geïnitieerd. Alleen de biases zijn 0.0.
    nn = NeuralNetwork([
        Layer([
            Node(0.0, [random.random() for _ in range(0 if index == 0 else nodes_per_layer[index-1])], sigmoid)
            for _ in range(node_count)
        ])
        for index, node_count in enumerate(nodes_per_layer)
    ])

    epochs = 5
    delta = 1.0

    last_cost = nn.cost_from_inputs(Y, X)

    improved_cost = False

    while True:
        print()
        print(f"Delta: {delta}")
        nn.train(X_TRAIN, Y_TRAIN, X_VALIDATION, Y_VALIDATION, epochs, delta)

        cost = nn.cost_from_inputs(Y, X)

        if cost >= last_cost:
            delta /= 10
        else:
            improved_cost = True

        last_cost = cost

        if delta < 1e-10:
            if not improved_cost:
                break
            improved_cost = False
            delta = 1.0

    print(f"test cost: {nn.cost(Y_TEST, X_TEST)}")

    print(nn.forward([0.1, 0.3, 0.2, 0.1]))

    total = 0
    correct = 0

    for x, y in zip(X_TRAIN, Y_TRAIN):
        total += 1

        prediction = nn.forward(x)

        if prediction.index(max(prediction)) == y.index(max(y)):
            correct += 1

    print(f"Training-data accuracy: {correct / total}")

    total = 0
    correct = 0

    for x, y in zip(X_VALIDATION, Y_VALIDATION):
        total += 1

        prediction = nn.forward(x)

        if prediction.index(max(prediction)) == y.index(max(y)):
            correct += 1

    print(f"Validation-data accuracy: {correct / total}")

    total = 0
    correct = 0

    for x, y in zip(X_TEST, Y_TEST):
        total += 1

        prediction = nn.forward(x)

        if prediction.index(max(prediction)) == y.index(max(y)):
            correct += 1

    print(f"Test-data accuracy: {correct / total}")
    print(f"Sum of squared errors: {nn.sum_of_squared_errors(Y, [nn.forward(item) for item in X])}")
    print(f"Sum of absolute errors: {nn.sum_of_absolute_errors(Y, [nn.forward(item) for item in X])}")


if __name__ == "__main__":
    main()
