##############
# Name: Sehej Oberoi
# email: oberois@purdue.edu
# Date: 10/21/2020

import numpy as np
import sys
import os
import math
import random

def entropy(freqs):
    """
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    if all_freq == 0:
      return 0
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy

def infor_gain(before_split_freqs, after_split_freqs):
    """
    gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
    >>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    0.02922
    """
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


class Node(object):
    def __init__(self, l, r, attr, thresh):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh


class Tree(object):
	def __init__(self, ____):
		pass

def ID3(train_data, label, remainingDepth, minSplit):
    if remainingDepth == 1 or len(train_data) < minSplit:
      count = 0
      for i in label[label.columns[-1]]:
        count += i

      if count > len(label)/2.0:
        return 1
      return 0

    if len(train_data) == 0:
      return 0

    if len(train_data) == 1:
      return label[label.columns[0]][0]

    maxGain = 0
    attribute = ""
    maxThreshold = 1;
    # 1. use a for loop to calculate the infor-gain of every attribute
    for att in train_data:
        # 1.1 pick a threshold
        """
        threshold = 1
        small = train_data[att][0]
        large = train_data[att][0]
        total = 0

        for x in train_data[att]:
          if x > large:
            large = x
          elif x < small:
            small = x
          total += x

        if (large != 1 or small != 0):
          threshold=total/(1.0 * train_data.shape[0])
        """
        ma = train_data[att].max()
        mi = train_data[att].min()
        threshold = 0
        if ma <= 1 and mi >= 0:    # treat it as a boolean
          threshold = 1
        else:
          # its continous so use quartiles as possible thresholds and choose max gain
          quants = train_data[att].quantile([0.25, 0.5, 0.75])
          best = -1
          for q in quants:
            lessZero = 0
            lessOne = 0
            moreZero = 0
            moreOne = 0

            count = 0
            for x in train_data[att]:
              if x < threshold:
                if label[label.columns[0]][count] == 0:
                  lessZero += 1
                else:
                  lessOne += 1
              else:
                if label[label.columns[0]][count] == 0:
                  moreZero += 1
                else:
                  moreOne += 1
              count += 1

            less=(lessZero, lessOne)
            more=(moreZero, moreOne)
            before=(lessZero + moreZero, lessOne + moreOne)


            if before[0] == 0:    # becasue then entropy is 0
              return 1
            if before[1] == 0:
              return 0

            info = infor_gain(before, (less, more))

            if info > best:
              threshold = q
              best = info

        # 1.2 split the data using the threshold
        lessZero = 0
        lessOne = 0
        moreZero = 0
        moreOne = 0

        count = 0
        for x in train_data[att]:
          if x < threshold:
            if label[label.columns[0]][count] == 0:
              lessZero += 1
            else:
              lessOne += 1
          else:
            if label[label.columns[0]][count] == 0:
              moreZero += 1
            else:
              moreOne += 1
          count += 1

        less=(lessZero, lessOne)
        more=(moreZero, moreOne)
        before=(lessZero + moreZero, lessOne + moreOne)
        if before[0] == 0:    # becasue then entropy is 0
          return 1
        if before[1] == 0:
          return 0

        # 1.3 calculate the infor_gain
        info = infor_gain(before, (less, more))
        if info > maxGain:
          maxGain = info
          attribute = att
          maxThreshold = threshold
    # 2. pick the attribute that achieve the maximum infor-gain
      # max value is already stored
    # 3. build a node to hold the data;
    if attribute == '':
      ans = 1
      if before[0] > before[1]:
        ans = 0
      return ans

    current_node = Node(None, None, attribute, maxThreshold)
    # 4. split the data into two parts.

    left_part_train_data = train_data.loc[train_data[attribute] < maxThreshold]
    left_part_train_data.reset_index(inplace=True, drop=True)
    left_part_train_label = label.loc[train_data[attribute] < maxThreshold]
    left_part_train_label.reset_index(inplace=True, drop=True)
    right_part_train_data = train_data.loc[train_data[attribute] >= maxThreshold]
    right_part_train_data.reset_index(inplace=True, drop=True)
    right_part_train_label = label.loc[train_data[attribute] >= maxThreshold]
    right_part_train_label.reset_index(inplace=True, drop=True)


    # 5. call ID3() for the left parts of the data
    left_subtree = ID3(left_part_train_data, left_part_train_label, remainingDepth - 1, minSplit)

    # 6. call ID3() for the right parts of the data.
    right_subtree = ID3(right_part_train_data, right_part_train_label, remainingDepth - 1, minSplit)

    current_node.left_subtree = left_subtree
    current_node.right_subtree = right_subtree

    return current_node

def postPrune(root, cur, data):
  if isinstance(cur, Node) == False:
    return cur

  left_part_data = data.loc[data[root.attribute] < root.threshold]
  left_part_data.reset_index(inplace=True, drop=True)
  right_part_data = data.loc[data[root.attribute] >= root.threshold]
  right_part_data.reset_index(inplace=True, drop=True)

  root.left_subtree = postPrune(root, cur.left_subtree, left_part_data)
  root.right_subtree = postPrune(root, cur.right_subtree, right_part_data)

  postClass = 0
  if isLeaf(root.left_subtree) and isLeaf(root.right_subtree):
    # try pruning
    count = 0
    for i in range(0, len(data)):
      count += data[data.columns[-1]][i]

    if count > len(data)/2.0:
      postClass = 1

  correctPredict = 0
  for i in range(0, len(validationSet)):
    curTrav = root
    while isinstance(curTrav, Node):
      if curTrav == cur:
        #print("curTrav = cur")
        curTrav = postClass
      else:
        if validationSet[curTrav.attribute][i] < curTrav.threshold:
          curTrav = curTrav.left_subtree
        else:
          curTrav = curTrav.right_subtree

    if curTrav == validationSet[validationSet.columns[-1]][i]:
      correctPredict += 1

  global validCorrect
  if correctPredict > validCorrect:
    validCorrect = correctPredict
    return postClass

  return cur




def isLeaf(node):
  if node == None:
    return False

  if isinstance(node, Node) == False:
    return True


def getClass(root, row):
  cur = root
  while isinstance(cur, Node):
    if row[cur.attribute] < cur.threshold:
      cur = cur.left_subtree
    else:
      cur = cur.right_subtree

  return cur


def size(root):
  if root == None:
    return 0

  if isinstance(root, Node) == False:
    return 1

  return 1 + size(root.left_subtree) + size(root.right_subtree)



if __name__ == "__main__":
    # parse arguments
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model', type=str, default='vanilla')
    parser.add_argument('--crossValidK', type=int, default=5)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--minSplit', type=int, default=0)
    parser.add_argument('--fillMethod', type=str, default='median')
    args = parser.parse_args()

    if args.trainFolder == None:
      print("No training foler provided")
      exit(1)

    dataFile = str(args.trainFolder) + '/train-file.data'
    labelFile = str(args.trainFolder) + '/train-file.label'
    data = pd.read_csv(dataFile, delimiter=',', index_col=None, engine='python')
    label = pd.read_csv(labelFile, delimiter=',', index_col=None, engine='python')

    # join labels and features for k fold selection
    data = data.join(label)
    data[data.columns] = data[data.columns].apply(pd.to_numeric, errors='coerce')

    # deciding what to do with na values
    if args.fillMethod == 'iterpolate':
      data = data.interpolate(method ='linear', limit_direction ='forward')
    elif args.fillMethod == 'median':
      data = data.fillna(data.median())
    elif args.fillMethod == 'mode':
      for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    elif args.fillMethod == 'drop':
      data = data.dropna()

    trees = list()
    totalAcc = 0
    totalTAcc = 0
    totalSize = 0
    for i in range(0, args.crossValidK):
      validationSet = data.sample(frac = 1/(1.0 * args.crossValidK))
      validationSet.reset_index(inplace=True, drop=True)
      trainSet = pd.concat([data, validationSet, validationSet]).drop_duplicates(keep=False)
      trainSet.reset_index(inplace=True, drop=True)

    # build decision tree
      if args.model == 'vanilla' or args.model == 'postPrune':
        root = ID3(trainSet.iloc[:, :-1], trainSet.iloc[:, -1:], len(trainSet) + 1, 0)
      elif args.model == 'depth':
        root = ID3(trainSet.iloc[:, :-1], trainSet.iloc[:, -1:], args.depth, 0)
      elif args.model == 'minSplit':
        root = ID3(trainSet.iloc[:, :-1], trainSet.iloc[:, -1:], len(trainSet) + 1, args.minSplit)
      else:
        print("Invalid Model")
        exit(1)

      trees.append(root)

      trainCorrect = 0
      for j in range(0, len(trainSet)):
        if getClass(root, trainSet.loc[j])== trainSet[trainSet.columns[-1]][j]:
          trainCorrect += 1

      validCorrect = 0
      for j in range(0, len(validationSet)):
        if getClass(root, validationSet.loc[j]) == validationSet[validationSet.columns[-1]][j]:
          validCorrect += 1

      if args.model == 'postPrune':
        trees[i] = postPrune(root, root, trainSet)

      totalAcc += validCorrect/(1.0*len(validationSet))
      totalTAcc += trainCorrect/(1.0*len(trainSet))

      print("fold=" + str(i + 1) + ", train set accuracy=" + str(trainCorrect/(1.0 * len(trainSet))) + ", validation set accuracy=" + str(validCorrect/(1.0 * len(validationSet))))
      totalSize += size(trees[i])



    # predict on testing set & evaluate the testing accuracy
    if args.testFolder:
      testDataFile = str(args.testFolder) + '/test-file.data'
      testLabelFile = str(args.testFolder) + '/test-file.label'
      testData = pd.read_csv(testDataFile, delimiter=',', index_col=None, engine='python')
      labelData = pd.read_csv(testLabelFile, delimiter=',', index_col=None, engine='python')
      testData = testData.join(labelData)
      correct = 0

      for i in range(0, len(testData)):
        classification = 0
        for j in range(0, len(trees)):
          if getClass(trees[j], testData.loc[i]) == 1:
            classification += 1

        if classification > len(trees)/2.0:
          classification = 1
        else:
          classification = 0

        if classification == labelData[labelData.columns[-1]][i]:
          correct += 1

      print("Test set accuracy=" + str(correct/(1.0 * len(testData))))
      print(totalSize/(1.0*args.crossValidK), totalTAcc/(1.0*args.crossValidK), totalAcc/(1.0*args.crossValidK))
