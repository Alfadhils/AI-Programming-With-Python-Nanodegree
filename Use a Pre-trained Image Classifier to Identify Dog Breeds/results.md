Result Overview

Project Objectives:
1. Identifying which pet images are of dogs and which pet images aren't of dogs
2. Classifying the breeds of dogs, for the images that are of dogs

Pet Classifier:
- Number of total images = 40
- Number of dog images = 30
- Number of not-a-dog images = 10

Results table

| Architecture | % not-a-dog correct | % dog correct | % breeds correct | % match labels |
|--------------|---------------------|--------------|-----------------|----------------|
| ResNet       | 90.0%               | 100.0%       | 90.0%           | 82.5%          |
| AlexNet      | 100.0%              | 100.0%       | 80.0%           | 75.0%          |
| VGG          | 100.0%              | 100.0%       | 93.3%           | 87.5%          |

For the pet classifier, we can conclude VGG is the best architecture due to its overall accuracy.

Upload Image Classifier:
- Number of total images = 4
- Number of dog images = 2
- Number of not-a-dog images = 2

Results table

| Architecture | % not-a-dog correct | % dog correct | % breeds correct | % match labels | Time Taken |
|--------------|---------------------|--------------|-----------------|----------------|------------|
| ResNet       | 100.0%              | 100.0%       | 00.0%           | 50.0%          | 4 sec      |
| AlexNet      | 100.0%              | 100.0%       | 00.0%           | 50.0%          | 1 sec      |
| VGG          | 100.0%              | 100.0%       | 00.0%           | 50.0%          | 8 sec      |

Notable results:
1. For the uploaded image classifier, the dog images are called dog_01 and dog_02, hence the breed is denoted as 'dog' while their actual breed is foxhound. If the files are called foxhound_01 and foxhound_02, then the breeds correct will be 100%.
2. Because the performance of every model in the uploaded image classifier is the same, we can consider the time taken and its number of parameters. Because we do not have access to the parameter count, we can use the time taken as our measure. We conclude that AlexNet is the best because of a significantly lower time taken of 1 sec.

References :
1. Workspace hints section
2. Chat GPT
3. Numpy Documentation
