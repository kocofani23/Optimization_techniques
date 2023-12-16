#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAX_WORDS 10000
#define MAX_WORD_LENGTH 100
#define LEARNING_RATE 0.001
#define EPOCHS 100
#define BETA1 0.9 // Exponential decay rates for moment estimates
#define BETA2 0.999
#define EPSILON 1e-8 // Small value to avoid division by zero

// Structure to hold the dictionary and other necessary data

typedef struct
{
    char** words;           // Array to hold unique words in the dictionary
    int wordCount;          // Number of unique words in the dictionary
} Dictionary;

int wordExists(const char* word, Dictionary* dict)
{            //function to check if a word exists already in the dictionary or not

    int i;
    for(i=0; i<dict->wordCount; i++){
        if(strcmp(word, dict->words[i]) == 0){
            return 1;                   // Word already exists in the dictionary
        }
    }
    return 0; // Word does not exist in the dictionary
}

// Function to add a word to the dictionary
void addWordToDictionary(const char* word, Dictionary* dict)
{
    if (!wordExists(word, dict)) {
        dict->words = realloc(dict->words, (dict->wordCount + 1) * sizeof(char*));
        dict->words[dict->wordCount] = malloc((strlen(word) + 1) * sizeof(char));
        strcpy(dict->words[dict->wordCount], word);
        dict->wordCount++;
    }
}

// Function to read files and create a dictionary
Dictionary createDictionary(const char* fileA, const char* fileB)
{
    FILE *fpA, *fpB;
    char word[MAX_WORD_LENGTH];
    Dictionary dict = { .words = NULL, .wordCount = 0 };

    // Open files for reading
    fpA = fopen(fileA, "r");
    fpB = fopen(fileB, "r");
    if (fpA == NULL || fpB == NULL) {
        printf("Error opening files.\n");
        return dict;
    }

    // Read file A
    while (fscanf(fpA, "%s", word) == 1) {
        addWordToDictionary(word, &dict);
    }

    // Read file B
    while (fscanf(fpB, "%s", word) == 1) {
        addWordToDictionary(word, &dict);
    }

    // Close files
    fclose(fpA);
    fclose(fpB);

    return dict;
}


// Function to split data into training and test sets
void splitData(const Dictionary* dict, const char* fileA, const char* fileB, double** trainSetA, double** trainSetB, double** testSetA, double** testSetB)
{
    FILE *fpA, *fpB;
    char word[MAX_WORD_LENGTH];         //maximum word length 100 characters
    int wordCount = dict->wordCount;
    int i;
    double TRAIN_SET_SIZE = wordCount * (0.8);

    // Allocate memory for train and test sets

    *trainSetA = (double*)calloc(wordCount, sizeof(double));
    *trainSetB = (double*)calloc(wordCount, sizeof(double));
    *testSetA = (double*)calloc(wordCount, sizeof(double));
    *testSetB = (double*)calloc(wordCount, sizeof(double));

    // Open files for reading
    fpA = fopen(fileA, "r");
    fpB = fopen(fileB, "r");
    if(fpA == NULL || fpB == NULL){
        printf("Error opening files.\n");
        return;
    }

    // Read file A and populate trainSetA and testSetA

    while(fscanf(fpA, "%s", word) == 1){
        for(i=0; i<wordCount; i++){
            {
                if(i < TRAIN_SET_SIZE){
                    *trainSetA[i] = wordExists(word, dict);
                }
                else{
                    *testSetA[i] = wordExists(word, dict);
                }
            }
        }
    }

    while(fscanf(fpB, "%s", word) == 1){
        for(i=0; i<wordCount; i++){
                if(i < TRAIN_SET_SIZE){
                    *trainSetB[i] = wordExists(word, dict);
                }
                else{
                    *testSetB[i] = wordExists(word, dict);
                }
            }
        }

    // Close files
    fclose(fpA);
    fclose(fpB);
}

// Function to initialize 'w' array with random values
void initializeW(double* w, int size) {

    int i;

    srand(time(NULL));
    // Initialize 'w' with random values between -1 and 1

    for(i=0; i<size; i++){
        w[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    printf("\n\n");

}

double dotProduct(const double w, const double* x, int size)
{

    double result = 0.0;
    int i;
    for(i=0; i<size; i++){
        result += w * (*x);
    }
    return result;


}

// Function for gradient descent optimization

void gradient_descent(double** trainSetA, double** trainSetB, double* w, int size)
{
    //gradient descent optimization

    int epoch;
    int i, j;
    double errorA, errorB;
    double predictionA = 0.0, predictionB = 0.0;
    double gradientA = 0.0, gradientB = 0.0;
    double TRAIN_SET_SIZE = size * 0.8;
    clock_t start, end;
    double duration;
    double loss = 0.0;



    printf("****************************\t [GRADIENT DESCENT]\t****************************\n");

    start = clock();
    for(epoch=0; epoch<EPOCHS; epoch++){
        errorA = 0.0, errorB = 0.0;

        // Calculate error for set A
        for(i=0; i<TRAIN_SET_SIZE; i++) {
            predictionA = tanh(dotProduct(w[i], trainSetA[i], TRAIN_SET_SIZE));
            errorA += (1 - predictionA * predictionA); // Gradient of tanh function         1 for file A
        }
        errorA /= TRAIN_SET_SIZE; // Mean squared error

        // Calculate error for set B
        for(i=0; i<TRAIN_SET_SIZE; i++){
            predictionB = tanh(dotProduct(w[i], trainSetB[i], TRAIN_SET_SIZE));
            errorB += (-1 - predictionB * predictionB); // Gradient of tanh function        -1 for file B
        }
        errorB /= TRAIN_SET_SIZE; // Mean squared error
        // Update 'w' using gradient descent

        for(i=0; i<TRAIN_SET_SIZE; i++){
            // Calculate gradient for set A
            for(j=0; j<size; j++){
                predictionA = tanh(dotProduct(w[i], trainSetA[i], TRAIN_SET_SIZE));
                gradientA += (1 - predictionA * predictionA) * trainSetA[j][i]; // Gradient of tanh function
            }
            gradientA /= TRAIN_SET_SIZE; // Mean gradient

            // Calculate gradient for set B
            for(j=0; j<TRAIN_SET_SIZE; j++){
                predictionB = tanh(dotProduct(w[i], trainSetB[i], TRAIN_SET_SIZE));
                gradientB += (1 - predictionB * predictionB) * trainSetB[j][i]; // Gradient of tanh function
            }
            gradientB /= TRAIN_SET_SIZE; // Mean gradient

            // Update weights using gradient descent
            w[i] -= LEARNING_RATE * (gradientA - gradientB); // Update rule
            loss += (errorA + errorB);
        }

        loss /= TRAIN_SET_SIZE;

        // Print error at each epoch (for observation)
        printf("Epoch %d \tLoss: %lf\n", epoch + 1,  loss);
    }
    end = clock();
    duration = (double)(end - start)/CLOCKS_PER_SEC;
    printf("\nDuration of gradient descent in seconds: %lf\n", duration);

    //create file with array w


    FILE *filePointer;
    filePointer = fopen("W_GD5.txt", "w"); // Open a file in write mode ("w")


    if (filePointer == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    // Write the array elements to the file
    for(i=0; i<size; i++) {
        fprintf(filePointer, "%lf ", w[i]);
    }

    fclose(filePointer); // Close the file

    printf("Array has been saved to file successfully.\n");

}

// Function for stochastic gradient descent optimization
void stochastic_gradient_descent(double** trainSetA, double** trainSetB, double* w, int size)
 {

    int epoch;
    int i, j;
    double errorA = 0.0, errorB = 0.0;
    double predictionA = 0.0, predictionB = 0.0;
    double gradientA = 0.0, gradientB = 0.0;
    double TRAIN_SET_SIZE = size * 0.8;
    clock_t start, end;
    double duration;
    double loss = 0.0;



    printf("****************************\t [STOCHASTIC GRADIENT DESCENT] \t****************************\n");

    start = clock();
    for(epoch=0; epoch<EPOCHS; epoch++) {

        // Update 'w' using stochastic gradient descent
        for(i=0; i<TRAIN_SET_SIZE; i++){
            int randomIndexA = rand() % size;
            int randomIndexB = rand() % size;

            predictionA = tanh(dotProduct(w[i], trainSetA[i], TRAIN_SET_SIZE));
            predictionB = tanh(dotProduct(w[i], trainSetB[i], TRAIN_SET_SIZE));

            gradientA = (1 - predictionA * predictionA);
            gradientB = (1 - predictionB * predictionB);

            // Update weights using stochastic gradient descent
            for(j=0; j<size; j++){
                w[j] -= LEARNING_RATE * (gradientA * (*trainSetA[randomIndexA]) - gradientB * (*trainSetB[randomIndexB]));
            }

            errorA = 1 - gradientA; // Accumulate error for set A       1 for file A
            errorB = -1 + gradientB; // Accumulate error for set B      -1 for file B
            loss += (errorA + errorB);
        }
         loss /= TRAIN_SET_SIZE;

        // Print error at each epoch (for observation)
        printf("Epoch %d \tLoss: %lf\n", epoch + 1,  loss);
    }

    end = clock();
    duration = (double)(end - start)/CLOCKS_PER_SEC;
    printf("\nDuration of stochastic gradient descent in seconds: %lf\n", duration);

    //create file with array w


    FILE *filePointer;
    filePointer = fopen("W_SGD5.txt", "w"); // Open a file in write mode ("w")

    if (filePointer == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    // Write the array elements to the file
    for(i=0; i<size; i++) {
        fprintf(filePointer, "%lf ", w[i]);
    }

    fclose(filePointer); // Close the file

    printf("Array has been saved to file successfully.\n");
}

// Function for ADAM optimization
void ADAM_optimization(double** trainSetA, double** trainSetB, double* w, int size)
 {

    int epoch;
    int i, j;
    double errorA = 0.0, errorB = 0.0;
    double predictionA, predictionB;
    double gradientA, gradientB;
    double correctedM[size], correctedV[size];
    double TRAIN_SET_SIZE = size * 0.8;
    clock_t start, end;
    double duration;
    double loss = 0.0;


    double* m = (double*)calloc(size, sizeof(double)); // Initialize 1st moment vector
    double* v = (double*)calloc(size, sizeof(double)); // Initialize 2nd moment vector


    printf("****************************\t [ADAM OPTIMIZATION] \t****************************\n");


    start = clock();
    for(epoch=0; epoch<EPOCHS; epoch++){
        // Update 'w' using ADAM optimization
        for(i=0; i<TRAIN_SET_SIZE; i++){
            int randomIndexA = rand() % size;
            int randomIndexB = rand() % size;

            predictionA = tanh(dotProduct(w[i], trainSetA[i], size));
            predictionB = tanh(dotProduct(w[i], trainSetB[i], size));

            gradientA = (1 - predictionA * predictionA);
            gradientB = (1 - predictionB * predictionB);

            // Update moment estimates
            for(j=0; j<size; j++){
                m[j] = BETA1 * m[j] + (1 - BETA1) * (gradientA * (*trainSetA[randomIndexA]) - gradientB * (*trainSetB[randomIndexB]));
                v[j] = BETA2 * v[j] + (1 - BETA2) * (gradientA * (*trainSetA[randomIndexA]) - gradientB * (*trainSetB[randomIndexB])) * (gradientA * (*trainSetA[randomIndexA]) - gradientB * (*trainSetB[randomIndexB]));
            }

            // Bias-corrected moment estimates
            for(j=0; j<size; j++) {
                correctedM[j] = m[j] / (1 - pow(BETA1, epoch + 1));
                correctedV[j] = v[j] / (1 - pow(BETA2, epoch + 1));
            }

            // Update weights using ADAM optimization
            for(j=0; j<size; j++) {
                w[j] -= LEARNING_RATE * correctedM[j] / (sqrt(correctedV[j]) + EPSILON);
            }

            errorA = 1 - gradientA; // Accumulate error for set A           1 for file A
            errorB = -1 + gradientB; // Accumulate error for set B          -1 for file B
            loss += (errorA + errorB);
        }
        loss /= TRAIN_SET_SIZE;

        // Print error at each epoch (for observation)
        printf("Epoch %d \tLoss: %lf\n", epoch + 1,  loss);
    }

    end = clock();

    duration = (double)(end - start)/CLOCKS_PER_SEC;
    printf("\nDuration of ADAM optimization in seconds: %lf\n", duration);

    free(m);
    free(v);

    //create file with array w

    FILE *filePointer;
    filePointer = fopen("W_AO5.txt", "w"); // Open a file in write mode ("w")

    if (filePointer == NULL) {
        printf("Error opening the file.\n");
        return;
    }

    // Write the array elements to the file
    for(i=0; i<size; i++) {
        fprintf(filePointer, "%lf ", w[i]);
    }

    fclose(filePointer); // Close the file

    printf("Array has been saved to file successfully.\n");
}


int main()
{
    const char* fileA = "fileA.txt"; // Replace with file names
    const char* fileB = "fileB.txt";

    int i;

    Dictionary dict = createDictionary(fileA, fileB);

    double w_GD[dict.wordCount];            //gradient descent array
    double w_SGD[dict.wordCount];           //stochastic gradient descent array
    double w_AO[dict.wordCount];            //ADAM optimization array

    //allocate memory for train and test sets as 2D arrays

    double** trainSetA = (double**)malloc(dict.wordCount * sizeof(double*));
    for(i=0; i<dict.wordCount; i++){
        trainSetA[i] = (double*)calloc(dict.wordCount, sizeof(double));
    }

    double** trainSetB = (double**)malloc(dict.wordCount * sizeof(double*));
    for(i=0; i<dict.wordCount; i++){
        trainSetB[i] = (double*)calloc(dict.wordCount, sizeof(double));
    }

    double** testSetA = (double**)malloc(dict.wordCount * sizeof(double*));
    for(i=0; i<dict.wordCount; i++){
        testSetA[i] = (double*)calloc(dict.wordCount, sizeof(double));
    }

    double** testSetB = (double**)malloc(dict.wordCount * sizeof(double*));
    for(i=0; i<dict.wordCount; i++){
        testSetB[i] = (double*)calloc(dict.wordCount, sizeof(double));
    }

    splitData(&dict, fileA, fileB, trainSetA, trainSetB, testSetA, testSetB);

    //optimization functions here
    //for every optimization function, create a new array w

    initializeW(&w_GD, dict.wordCount);
    gradient_descent(trainSetA, trainSetB, &w_GD, dict.wordCount);

    initializeW(&w_SGD, dict.wordCount);
    stochastic_gradient_descent(trainSetA, trainSetB, &w_SGD, dict.wordCount);

    initializeW(&w_AO, dict.wordCount);
    ADAM_optimization(trainSetA, trainSetB, &w_AO, dict.wordCount);


    // Free allocated memory for train and test sets
    for(i=0; i<dict.wordCount; i++){
        free(trainSetA[i]);
    }
    free(trainSetA);

    for(i=0; i<dict.wordCount; i++){
        free(trainSetB[i]);
    }
    free(trainSetB);

    for(i=0; i<dict.wordCount; i++){
        free(testSetA[i]);
    }
    free(testSetA);

    for(i=0; i<dict.wordCount; i++){
        free(testSetB[i]);
    }
    free(testSetB);

    // Free dictionary memory
    for(i=0; i<dict.wordCount; i++){
        free(dict.words[i]);
    }
    free(dict.words);

    return 0;

}
