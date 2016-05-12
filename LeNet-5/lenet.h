#pragma once

#define LENGTH_KERNEL0	5
#define LENGTH_KERNEL1	4
#define LENGTH_SAMPLE	2

#define LENGTH_FEATURE0	28
#define LENGTH_FEATURE1	24//(LENGTH_INPUT - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	12//(LENGTH_INPUT_1 >> 1)
#define LENGTH_FEATURE3	8//(LENGTH_INPUT_2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	4//(LENGTH_INPUT_3 >> 1)
#define LENGTH_FEATURE5	1//(LENGTH_INPUT_4 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE6	1

#define LAYER0			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define LAYER6          84
#define LAYER7          10

#define ALPHA 0.05

typedef unsigned char uint8;
typedef uint8 image[LENGTH_FEATURE0][LENGTH_FEATURE0];


typedef struct LeNet5
{
	double weight0_1[LAYER0][LAYER1][LENGTH_KERNEL0][LENGTH_KERNEL0];
	double weight1_2[LAYER1][LENGTH_SAMPLE][LENGTH_SAMPLE];
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL0][LENGTH_KERNEL0];
	double weight3_4[LAYER3][LENGTH_SAMPLE][LENGTH_SAMPLE];
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL1][LENGTH_KERNEL1];
	double weight5_6[LAYER5][LAYER6][LENGTH_FEATURE5][LENGTH_FEATURE5];

	double bias0_1[LAYER1];
	double bias1_2[LAYER2];
	double bias2_3[LAYER3];
	double bias3_4[LAYER4];
	double bias4_5[LAYER5];
	double bias5_6[LAYER6];

}LeNet5;

typedef struct Feature
{
	double value0[LAYER0][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double value1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double value2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double value3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double value4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double value5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double value6[LAYER6][LENGTH_FEATURE6][LENGTH_FEATURE6];
}Feature;

void TrainBatch(LeNet5 *lenet, image *input, uint8 *result, int batchSize);

void Train(LeNet5 *lenet, image input, uint8 result);

uint8 Predict(LeNet5 *lenet, image input);

void Initial(LeNet5 *lenet, double(*rand)());
