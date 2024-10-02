#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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

// Function to check for collision with pipes
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

// Function to render the game in the console, including the score
void renderGame(float birdY, int* pipeX, int* pipeY, int score) {
    char screen[SCREEN_HEIGHT][SCREEN_WIDTH + 1];  // Screen buffer, +1 for null terminator
    
    // Initialize screen with spaces
    for (int y = 0; y < SCREEN_HEIGHT; y++) {
        for (int x = 0; x < SCREEN_WIDTH; x++) {
            screen[y][x] = ' ';
        }
        screen[y][SCREEN_WIDTH] = '\0';  // Null terminator for each row
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

    // Print the score
    printf("\nScore: %d\n", score);
}

int main() {
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
    cudaMalloc((void**)&d_states, 2 * sizeof(curandState));
    initCurand<<<1, 2>>>(time(NULL), d_states);
    cudaDeviceSynchronize();

    // Game loop (simplified, with user input for jumping)
    for (int frame = 0; frame < 500 && !collision; ++frame) {
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

        // Check for collision and update score
        checkCollisionAndScore<<<1, 2>>>(birdY, 5, d_pipeX, d_pipeY, d_score, d_collision);
        cudaDeviceSynchronize();

        // Copy updated bird and pipes back to host
        cudaMemcpy(&birdY, d_birdY, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pipeX, d_pipeX, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pipeY, d_pipeY, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&score, d_score, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&collision, d_collision, sizeof(bool), cudaMemcpyDeviceToHost);

        // Render game to the console
        renderGame(birdY, pipeX, pipeY, score);

        // Sleep to slow down the game loop (in milliseconds)
        Sleep(100);  // 100 ms delay for each frame to make the game playable
    }

    if (collision) {
        printf("Game Over! Final Score: %d\n", score);
    }

    // Free memory
    cudaFree(d_birdY);
    cudaFree(d_birdVelocity);
    cudaFree(d_isJumping);
    cudaFree(d_pipeX);
    cudaFree(d_pipeY);
    cudaFree(d_score);
    cudaFree(d_collision);
    cudaFree(d_states);

    return 0;
}
