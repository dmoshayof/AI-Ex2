from collections import defaultdict
import math
import random

# Consts
dataset_path = 'dataset.txt'
output_tree_path = 'tree.txt'
test_path = 'test.txt'
train_path = 'train.txt'
output_path = 'output.txt'
k = 5
cross_k = 5

############################
#######    Utils   #########
############################

# load the train set and attribute list and target attribute
def load_dataset(data):
    training_set = []
    with open(data) as f:
        attributes = f.readline().split()
        for line in f.readlines():
            training_set.append(tuple(line.split()))
    target = attributes[-1]
    return training_set, attributes, target


# Loading the test set and seperate it to features and labels
def load_test(test_file_path):
    """ generating testing set of examples and gold labels """
    test_x = []
    test_y = []
    with open(test_file_path) as f:
        _ = f.readline().split()
        for line in f.readlines():
            values = line.split()
            test_x.append(tuple(values[:-1]))
            test_y.append(values[-1])

    return test_x, test_y


# Generate the attribute space
def create_attribute_space(data, attributes):
    attribute_space = defaultdict(set)
    for example in data:
        for i, value in enumerate(example[:-1]):
            attribute_space[attributes[i]].add(value)
    return attribute_space


# Find the positive target for that data
def get_label_attribute(data):
    if data[0][-1] == 'true' or data[0][-1] == 'false':
        positive_target = 'true'
    elif data[0][-1] == '1' or data[0][-1] == '0':
        positive_target = '1'
    else:
        positive_target = 'yes'
    return positive_target


# split lists into chunks of n
def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# split test into features and labels lists
def split_test_set(test):
    features = []
    labels = []
    for i in test:
        features.append(tuple(i[:-1]))
        labels.append(i[-1])

    return features, labels


# Calculate the accuracy according to predictions and labels
def accuracy(predictions, labels):
    good = 0.
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            good += 1

    accuracy = math.ceil(good / len(predictions) * 100) / 100
    return accuracy


# split the features to unique set
def get_values(data, attributes, attr):
    index = attributes.index(attr)
    return set([entry[index] for entry in data])


# Create a new data set with the best attribute value.
def get_data(data, attributes, best_attr, val):
    exmp_data = []
    index = attributes.index(best_attr)

    for entry in data:
        if entry[index] == val:
            new_entry = []
            for i, value in enumerate(entry):
                if i != index:
                    new_entry.append(value)
            exmp_data.append(tuple(new_entry))
    return exmp_data


# saving all the models predictions to file
def save_to_file_outputs(data, predictions_len, output_file_path):
    with open(output_file_path, 'w') as f:
        for index in range(predictions_len):
            f.write(
                '{}\t{}\t\n'.format(index + 1, data[0][index]))

        f.write('{}\t'.format('', str(data[1])))


############################
####### Decision Tree ######
############################


# A object of decision tree to be able to create a sub-trees
class DecisionTree(object):
    """ Decision tree object """

    def __init__(self, decision=None, value=None, is_leaf=False):
        self.is_leaf = is_leaf
        self.decisions = {}
        self.decision = decision
        self.value = value


# Calculate the entropy of all the data
def entropy(data, attributes, target):
    targets = defaultdict(int)
    index = attributes.index(target)

    for example in data:
        targets[example[index]] += 1

    entropyI = 0.

    for targets in targets.values():
        entropyI += (-targets / len(data)) * math.log(targets / len(data), 2)

    return entropyI


# Calculate the gain of the data
def gain(data, attributes, attr, target):
    f = defaultdict(int)
    index = attributes.index(attr)

    for example in data:
        f[example[index]] += 1

    entropy_s = 0.

    for key in f.keys():
        prob = f[key] / sum(f.values())
        data_subset = [example for example in data if example[index] == key]
        entropy_s += prob * entropy(data_subset, attributes, target)

    return entropy(data, attributes, target) - entropy_s


# pick the best attribute according to high gain
def choose_attribute(data, attributes, target):
    max_gain = 0
    chosen = attributes[0]

    for attr in attributes:
        if attr != target:
            current_gain = gain(data, attributes, attr, target)
            if current_gain > max_gain:
                max_gain = current_gain
                chosen = attr

    return chosen


# Print the evaluated tree in recursive way
def print_tree(tree, depth=0):
    if tree.is_leaf:
        return tree.value
    reprs = []
    items = tree.decisions.items()
    # sort the decisions in alphabetic order by their decision value
    sorted_items = sorted(items, key=lambda item: item[0])
    for value, sub_tree in sorted_items:
        if sub_tree.is_leaf:
            sub_tree_repr = '{}{}{}={}:{}'.format('\t' * depth, '|' * (depth > 0), tree.decision, value,
                                                  print_tree(sub_tree, depth + 1))
        else:
            sub_tree_repr = '{}{}{}={}\n{}'.format('\t' * depth, '|' * (depth > 0), tree.decision, value,
                                                   print_tree(sub_tree, depth + 1))
        reprs.append(sub_tree_repr)
    return '\n'.join(reprs)


# Calculate which is the major class from the data
def mode_major(data, attributes, target, positive_target):
    freq = defaultdict(int)
    index = attributes.index(target)

    for sample in data:
        freq[sample[index]] += 1

    # calculating max frequent class for breaking the tie will be taken positive target
    major = max(freq.items(), key=lambda item: (item[1], item[0] == positive_target))[0]
    return major


# Built tree with the DTL algorithm (Id3)
def DTL(data, attributes, target, attribute_space, positive_target):
    values = [example[attributes.index(target)] for example in data]
    default = mode_major(data, attributes, target, positive_target)

    # if the data is empty of only one attribute left
    if not data or len(attributes) == 1:
        return DecisionTree(value=default, is_leaf=True)
    # all values are the same no need to calculate- return this
    elif values.count(values[0]) == len(values):
        return DecisionTree(value=values[0], is_leaf=True)
    else:
        best = choose_attribute(data, attributes, target)
        # create a new decision root
        tree = DecisionTree(decision=best)

        best_values = get_values(data, attributes, best)
        best_possible_values = attribute_space.get(best)
        # find the best attribute
        for value in best_values:
            examples = get_data(data, attributes, best, value)
            new_attributes = list(attributes)
            new_attributes.remove(best)
            sub_tree = DTL(examples, new_attributes, target, attribute_space, positive_target)
            tree.decisions[value] = sub_tree

        # the left overs features
        for val in best_possible_values - best_values:
            tree.decisions[val] = DecisionTree(value=default, is_leaf=True)
    return tree


# Recursively on the tree to predict the output for a specific sample.
def tree_predication(example, attributes, tree):
    if tree.is_leaf:
        return tree.value
    decision = tree.decision
    value = example[attributes.index(decision)]
    return tree_predication(example, attributes, tree.decisions[value])


############################
#########   KNN   ##########
############################

# Calculate the Hamming Distance between two samples
def hamming_distance_calc(sample, training_sample):
    return sum(train_attr != test_attr for test_attr, train_attr in zip(sample, training_sample))


# Predict the output label with knn algorithm
def predict_by_knn(query_example, training_set, k):
    distances = []
    for i, sample in enumerate(training_set):
        distances.append((i, sample[-1], hamming_distance_calc(query_example, sample[:-1])))
    increasing_distance = sorted(distances, key=lambda x: (x[2], x[0]))
    neighbors = [neighbor for i, neighbor in enumerate(increasing_distance) if i < k]
    f = defaultdict(int)
    for n in neighbors:
        f[n[1]] += 1
    prediction = max(f.items(), key=lambda x: x[1])[0]
    return prediction


############################
#######  Naive-Bayes #######
############################

# compare all values to a specific attribute and count it all.
def calc_count(data, attributes, label_i, cu_value):
    count = 0.
    i = attributes.index(label_i)
    for sample in data:
        if sample[i] == cu_value:
            count += 1
    return count


# classify the data according to naive Bayes algoritm
def predict_by_naiveBayes(sample, training_set, attributes, label_i):
    attribute_space = create_attribute_space(training_set, attributes)
    values = get_values(training_set, attributes, label_i)
    last_count = {value: calc_count(training_set, attributes, label_i, value)
                               for value in values}
    probs = []
    for value in values:
        # prior probability
        probability = last_count[value] / len(training_set)
        for index, attribute in enumerate(sample):
            n_data = [entry for entry in training_set if entry[index] == attribute]
            probability *= (calc_count(n_data, attributes, label_i, value) + 1) / \
                    (last_count[value] + len(attribute_space[attributes[index]]))
        probs.append((value, probability))

    prediction = max(probs, key=lambda item: item[1])[0]
    return prediction


##### Cross validation #######

# Train the tree model and run test on the current fold. print output to file
def tree_train_and_test(training_set, testing_set, attributes, target, gold_labels, fold=0):
    tree_predictions = []
    attribute_space = create_attribute_space(training_set, attributes)
    positive_target = get_label_attribute(training_set)
    tree = DTL(training_set, attributes, target, attribute_space, positive_target)
    for example in testing_set:
        tree_predictions.append(tree_predication(example, attributes, tree))
    tree_accuracy = accuracy(tree_predictions, gold_labels)
    with open(output_tree_path, 'w') as f:
        f.write(print_tree(tree))
    save_to_file_outputs((tree_predictions, tree_accuracy), len(gold_labels), "output" + str(fold) + ".txt")

# Run the knn model with test set and save output to file
def knn_train_and_test(training_set, testing_set, attributes, target, gold_labels, fold=0):
    knn_predictions = []
    attribute_space = create_attribute_space(training_set, attributes)
    positive_target = get_label_attribute(training_set)
    for example in testing_set:
        knn_predictions.append(predict_by_knn(example, training_set, k))
    knn_accuracy = accuracy(knn_predictions, gold_labels)
    save_to_file_outputs((knn_predictions, knn_accuracy), len(gold_labels), "knn_output" + str(fold) + ".txt")

# Run the naive_base model with test set and save output to file
def nb_train_and_test(training_set, testing_set, attributes, target, gold_labels, fold=0):
    nb_predictions = []
    attribute_space = create_attribute_space(training_set, attributes)
    positive_target = get_label_attribute(training_set)
    for example in testing_set:
        nb_predictions.append(predict_by_naiveBayes(example, training_set, attributes, target))
    nb_accuracy = accuracy(nb_predictions, gold_labels)
    save_to_file_outputs((nb_predictions, nb_accuracy), len(gold_labels), "nb_output" + str(fold) + ".txt")

# Run cross validation on the dataset- split the data to
def run_cross_validation(training_set, attributes, target):
    random.shuffle(training_set)
    folds = list(split_list(training_set, round(len(training_set) / 5)))
    test_x = []
    labels = []
    for i, fold in enumerate(folds):
        test = fold
        foldI = folds.pop(i)
        train = list(set().union(*folds))
        test_x, labels = split_test_set(test)
        tree_train_and_test(train, test_x, attributes, target, labels, i)
        folds.insert(i, foldI)


def save_accuracy(t_accu, k_accu, nb_accu):
    with open('accuracy.txt', 'w') as f:
        accuracy = '{}\t{}\t{}\t'.format(t_accu, k_accu, nb_accu)
        f.write(accuracy)
    f.close()
    return accuracy


def save_output(tree, t_accu, k_accu, nb_accu):
    with open(output_tree_path, 'w') as f:
        f.write(print_tree(tree))
        f.write('\n')
        accuracy = '{}\t{}\t{}\t'.format(t_accu, k_accu, nb_accu)
        f.write(accuracy)
    f.close()


def main():
    training_set, attributes, target = load_dataset(train_path)
    test_x, test_y = load_test(test_path)
    attribute_space = create_attribute_space(training_set, attributes)
    label_attr = get_label_attribute(training_set)

    tree = DTL(training_set, attributes, target, attribute_space, label_attr)

    tree_predictions = []
    knn_predictions = []
    nb_predictions = []

    # predicting
    for example in test_x:
        tree_predictions.append(tree_predication(example, attributes, tree))
        knn_predictions.append(predict_by_knn(example, training_set, k))
        nb_predictions.append(predict_by_naiveBayes(example, training_set, attributes, target))

    # calculate accuracies
    tree_accuracy = accuracy(tree_predictions, test_y)
    knn_accuracy = accuracy(knn_predictions, test_y)
    nb_accuracy = accuracy(nb_predictions, test_y)

    # output predictions to file
    save_output(tree, tree_accuracy, knn_accuracy, nb_accuracy)

    # run_cross_validation(training_set, attributes, target)
    print()


main()
