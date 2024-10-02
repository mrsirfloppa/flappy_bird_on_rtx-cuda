#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <windows.h>  // For Sleep(), GetAsyncKeyState(), and clearing the console

#define SCREEN_WIDTH 40   // Width of the console display
#define SCREEN_HEIGHT 20  // Height of the console display
#define BIRD_WIDTH 1
#define BIRD_HEIGHT 1
#define PIPE_WIDTH 1
#define PIPE_GAP 5
#define GRAVITY 0.1f
#define JUMP_VELOCITY -1.0f
#define HIGH_SCORE_FILE "highscore.txt"
#define CLOUD_SPEED 1
#define NUM_CLOUDS 3  // Number of cloud groups in the background
#define CLOUD_WIDTH 4  // Width of each cloud group
#define CLOUD_HEIGHT 2  // Height of each cloud group

// Function to clear the screen in Windows
void clearScreen() {
    system("cls");  // Clear console screen on Windows
}

// CUDA Kernel to initialize cuRAND states
__global__ void initCurand(unsigned int seed, curandState* states) {
    int idx = threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);  // Initialize cuRAND state for each thread
}

// CUDA Kernel to update the bird's position (gravity, jump physics)
__global__ void updateBird(float* birdY, float* birdVelocity, bool* isJumping) {
    if (*isJumping) {
        *birdVelocity = JUMP_VELOCITY;
        *isJumping = false;
    }
    *birdVelocity += GRAVITY;
    *birdY += *birdVelocity;
    if (*birdY + BIRD_HEIGHT >= SCREEN_HEIGHT) {
        *birdY = SCREEN_HEIGHT - BIRD_HEIGHT;  // Ground collision
        *birdVelocity = 0;
    }
    if (*birdY < 0) {
        *birdY = 0;  // Ceiling collision
        *birdVelocity = 0;
    }
}

// CUDA Kernel to update pipe positions using cuRAND for random number generation
__global__ void updatePipes(int* pipeX, int* pipeY, int pipeSpeed, curandState* states) {
    int idx = threadIdx.x;

    // Move pipes to the left
    pipeX[idx] -= pipeSpeed;
    if (pipeX[idx] + PIPE_WIDTH < 0) {
        pipeX[idx] = SCREEN_WIDTH;

        // Randomly generate new pipe position on GPU using cuRAND
        int gap = curand(&states[idx]) % (SCREEN_HEIGHT - PIPE_GAP - 5);
        pipeY[idx] = gap + 2;  // Ensure the pipe doesn't go too low or too high
    }
}

// CUDA Kernel to update cloud group positions in the background
__global__ void updateClouds(int* cloudX, int cloudSpeed, int* cloudY, curandState* states) {
    int idx = threadIdx.x;
    cloudX[idx] -= cloudSpeed;
    if (cloudX[idx] < 0) {
        cloudX[idx] = SCREEN_WIDTH;  // Reset cloud group position when it moves off screen
        // Randomize cloud Y position (so clouds don't always appear on the same row)
        cloudY[idx] = 1 + (curand(&states[idx]) % 5);  // Random Y row between 1 and 5
    }
}

// CUDA Kernel to check for collisions and update the score
__global__ void checkCollisionAndScore(float birdY, int birdX, int* pipeX, int* pipeY, int* score, bool* collision) {
    int idx = threadIdx.x;

    // Check if bird is in line with a pipe
    if (pipeX[idx] == birdX) {
        if (birdY < pipeY[idx] || birdY > (pipeY[idx] + PIPE_GAP)) {
            *collision = true;  // Collision with pipe
        }
    }

    // Increase score when bird passes a pipe
    if (pipeX[idx] == birdX - 1 && !(*collision)) {
        atomicAdd(score, 1);  // Increment score using atomic addition
    }
}

// Function to load the high score from a file
int loadHighScore() {
    std::ifstream file(HIGH_SCORE_FILE);
    int highScore = 0;
    if (file.is_open()) {
        file >> highScore;
        file.close();
    }
    return highScore;
}

// Function to save the high score to a file
void saveHighScore(int highScore) {
    std::ofstream file(HIGH_SCORE_FILE);
    if (file.is_open()) {
        file << highScore;
        file.close();
    }
}

// Function to render the game in the console, including the score and clouds
void renderGame(float birdY, int* pipeX, int* pipeY, int score, int highScore, int* cloudX, int* cloudY) {
    char screen[SCREEN_HEIGHT][SCREEN_WIDTH + 1];  // Screen buffer, +1 for null terminator
    
    // Initialize screen with spaces
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            screen[y][x] = ' ';
        }
        screen[y][SCREEN_WIDTH] = '\0';  // Null terminator for each row
    }

    // Place cloud groups on screen
    char cloudShapes[CLOUD_HEIGHT][CLOUD_WIDTH] = {  // Cloud group pattern
        {' ', 'O', 'o', ' '},
        {'o', 'O', ' ', 'o'}
    };

    for (int i = 0; i < NUM_CLOUDS; i++) {
        if (cloudX[i] >= 0 && cloudX[i] < SCREEN_WIDTH) {
            // Render each cloud group
            for (int dy = 0; dy < CLOUD_HEIGHT; dy++) {
                for (int dx = 0; dx < CLOUD_WIDTH; dx++) {
                    if (cloudX[i] + dx < SCREEN_WIDTH && cloudY[i] + dy < SCREEN_HEIGHT) {
                        screen[cloudY[i] + dy][cloudX[i] + dx] = cloudShapes[dy][dx];
                    }
                }
            }
        }
    }

    // Place bird on screen (represented by 'B')
    int birdPosY = (int)birdY;
    if (birdPosY >= 0 && birdPosY < SCREEN_HEIGHT) {
        screen[birdPosY][5] = 'B';  // Bird is always at x = 5 for simplicity
    }

    // Place pipes on screen (represented by '|')
    for (int i = 0; i < 2; i++) {
        // Upper pipe
        if (pipeX[i] >= 0 && pipeX[i] < SCREEN_WIDTH) {
            for (int y = 0; y < pipeY[i]; y++) {
                if (y >= 0 && y < SCREEN_HEIGHT) {
                    screen[y][pipeX[i]] = '|';
                }
            }
            // Lower pipe
            for (int y = pipeY[i] + PIPE_GAP; y < SCREEN_HEIGHT; y++) {
                if (y >= 0 && y < SCREEN_HEIGHT) {
                    screen[y][pipeX[i]] = '|';
                }
            }
        }
    }

    // Print the screen to the console
    clearScreen();
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        printf("%s\n", screen[y]);
    }

    // Print the score and high score
    printf("\nScore: %d  |  High Score: %d\n", score, highScore);
}

// Function to initialize all game variables
void initializeGame(float* birdY, float* birdVelocity, bool* isJumping, int* pipeX, int* pipeY, int* cloudX, int* cloudY, int* score, bool* collision, curandState* d_states) {
    *birdY = SCREEN_HEIGHT / 2;
    *birdVelocity = 0;
    *isJumping = false;

    pipeX[0] = SCREEN_WIDTH;
    pipeX[1] = SCREEN_WIDTH + (SCREEN_WIDTH / 2);
    pipeY[0] = 5;
    pipeY[1] = 10;

    cloudX[0] = SCREEN_WIDTH - 10;
    cloudX[1] = SCREEN_WIDTH - 20;
    cloudX[2] = SCREEN_WIDTH - 30;

    cloudY[0] = 1;
    cloudY[1] = 3;
    cloudY[2] = 5;

    *score = 0;
    *collision = false;

    // Reinitialize curand states for pipes and clouds
    initCurand<<<1, 2 + NUM_CLOUDS>>>(time(NULL), d_states);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    // Clear high score if -c or --clear flag is passed
    if (argc > 1 && (strcmp(argv[1], "-c") == 0 || strcmp(argv[1], "--clear") == 0)) {
        saveHighScore(0);
        printf("High score cleared!\n");
        return 0;
    }

    // Load the high score from file
    int highScore = loadHighScore();

    // Bird variables
    float birdY = SCREEN_HEIGHT / 2;
    float birdVelocity = 0;
    bool isJumping = false;

    // Allocate memory for bird properties on device
    float* d_birdY;
    float* d_birdVelocity;
    bool* d_isJumping;
    cudaMalloc((void**)&d_birdY, sizeof(float));
    cudaMalloc((void**)&d_birdVelocity, sizeof(float));
    cudaMalloc((void**)&d_isJumping, sizeof(bool));

    cudaMemcpy(d_birdY, &birdY, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_birdVelocity, &birdVelocity, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_isJumping, &isJumping, sizeof(bool), cudaMemcpyHostToDevice);

    // Pipe variables
    int pipeX[2] = {SCREEN_WIDTH, SCREEN_WIDTH + (SCREEN_WIDTH / 2)};
    int pipeY[2] = {5, 10};
    int pipeSpeed = 1;

    // Allocate memory for pipes on device
    int* d_pipeX;
    int* d_pipeY;
    cudaMalloc((void**)&d_pipeX, 2 * sizeof(int));
    cudaMalloc((void**)&d_pipeY, 2 * sizeof(int));

    cudaMemcpy(d_pipeX, pipeX, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pipeY, pipeY, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Cloud variables (for background)
    int cloudX[NUM_CLOUDS] = {SCREEN_WIDTH - 10, SCREEN_WIDTH - 20, SCREEN_WIDTH - 30};
    int cloudY[NUM_CLOUDS] = {1, 3, 5};  // Random starting Y positions

    // Allocate memory for clouds on device
    int* d_cloudX;
    int* d_cloudY;
    cudaMalloc((void**)&d_cloudX, NUM_CLOUDS * sizeof(int));
    cudaMalloc((void**)&d_cloudY, NUM_CLOUDS * sizeof(int));
    cudaMemcpy(d_cloudX, cloudX, NUM_CLOUDS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cloudY, cloudY, NUM_CLOUDS * sizeof(int), cudaMemcpyHostToDevice);

    // Score and collision variables
    int score = 0;
    bool collision = false;
    int* d_score;
    bool* d_collision;
    cudaMalloc((void**)&d_score, sizeof(int));
    cudaMalloc((void**)&d_collision, sizeof(bool));

    cudaMemcpy(d_score, &score, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_collision, &collision, sizeof(bool), cudaMemcpyHostToDevice);

    // cuRAND setup
    curandState* d_states;
    cudaMalloc((void**)&d_states, (2 + NUM_CLOUDS) * sizeof(curandState));  // For pipes and clouds
    initCurand<<<1, 2 + NUM_CLOUDS>>>(time(NULL), d_states);
    cudaDeviceSynchronize();

    bool isPaused = false;

    // Game loop (with pause and restart functionality)
    while (true) {
        initializeGame(&birdY, &birdVelocity, &isJumping, pipeX, pipeY, cloudX, cloudY, &score, &collision, d_states);

        cudaMemcpy(d_birdY, &birdY, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_birdVelocity, &birdVelocity, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_isJumping, &isJumping, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pipeX, pipeX, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pipeY, pipeY, 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cloudX, cloudX, NUM_CLOUDS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cloudY, cloudY, NUM_CLOUDS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_score, &score, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision, &collision, sizeof(bool), cudaMemcpyHostToDevice);

        for (int frame = 0; frame < 500 && !collision; ++frame) {
            // Pause and resume functionality
            if (GetAsyncKeyState('P') & 0x8000) {
                isPaused = !isPaused;
                while (isPaused) {
                    printf("Game Paused. Press 'P' to resume.\n");
                    if (GetAsyncKeyState('P') & 0x8000) {
                        isPaused = false;
                    }
                    Sleep(100);
                }
            }

            // Check for user input (spacebar jump)
            if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
                isJumping = true;  // Set jump flag when spacebar is pressed
                cudaMemcpy(d_isJumping, &isJumping, sizeof(bool), cudaMemcpyHostToDevice);
            }

            // Update bird
            updateBird<<<1, 1>>>(d_birdY, d_birdVelocity, d_isJumping);
            cudaDeviceSynchronize();

            // Update pipes
            updatePipes<<<1, 2>>>(d_pipeX, d_pipeY, pipeSpeed, d_states);
            cudaDeviceSynchronize();

            // Update clouds (background)
            updateClouds<<<1, NUM_CLOUDS>>>(d_cloudX, CLOUD_SPEED, d_cloudY, d_states + 2);  // Cloud states start at index 2
            cudaDeviceSynchronize();

            // Check for collision and update score
            checkCollisionAndScore<<<1, 2>>>(birdY, 5, d_pipeX, d_pipeY, d_score, d_collision);
            cudaDeviceSynchronize();

            // Copy updated bird, pipes, clouds, and score back to host
            cudaMemcpy(&birdY, d_birdY, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(pipeX, d_pipeX, 2 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(pipeY, d_pipeY, 2 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cloudX, d_cloudX, NUM_CLOUDS * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(cloudY, d_cloudY, NUM_CLOUDS * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&score, d_score, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&collision, d_collision, sizeof(bool), cudaMemcpyDeviceToHost);

            // Render game to the console
            renderGame(birdY, pipeX, pipeY, score, highScore, cloudX, cloudY);

            // Sleep to slow down the game loop (in milliseconds)
            Sleep(100);  // 100 ms delay for each frame to make the game playable

            // Game over logic
            if (collision) {
                printf("Game Over! Final Score: %d\n", score);
                if (score > highScore) {
                    highScore = score;
                    saveHighScore(highScore);
                    printf("New High Score: %d\n", highScore);
                }
                printf("Press 'R' to restart or 'Q' to quit.\n");

                while (true) {
                    if (GetAsyncKeyState('R') & 0x8000) {
                        break;  // Restart the game
                    }
                    if (GetAsyncKeyState('Q') & 0x8000) {
                        // Quit the game
                        return 0;
                    }
                    Sleep(100);
                }
            }
        }
    }

    // Free memory
    cudaFree(d_birdY);
    cudaFree(d_birdVelocity);
    cudaFree(d_isJumping);
    cudaFree(d_pipeX);
    cudaFree(d_pipeY);
    cudaFree(d_cloudX);
    cudaFree(d_cloudY);
    cudaFree(d_score);
    cudaFree(d_collision);
    cudaFree(d_states);

    return 0;
}
