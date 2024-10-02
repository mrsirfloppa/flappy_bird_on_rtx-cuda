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
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
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

        // Use cuRAND to generate random pipe Y position
        int gap = curand(&states[idx]) % (SCREEN_HEIGHT - PIPE_GAP - 5);
        pipeY[idx] = gap + 2;  // Ensure the pipe doesn't go too low or too high
    }
}

// Function to check for collision with pipes
bool checkCollision(float birdY, int birdX, int* pipeX, int* pipeY) {
    for (int i = 0; i < 2; i++) {
        if (pipeX[i] == birdX) {  // Check if bird is in line with a pipe
            if (birdY < pipeY[i] || birdY > (pipeY[i] + PIPE_GAP)) {
                return true;  // Collision with upper or lower pipe
            }
        }
    }
    return false;
}

// Function to render the game in the console
void renderGame(float birdY, int* pipeX, int* pipeY) {
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

    // cuRAND setup
    curandState* d_states;
    cudaMalloc((void**)&d_states, 2 * sizeof(curandState));
    initCurand<<<1, 2>>>(time(NULL), d_states);
    cudaDeviceSynchronize();

    bool collision = false;

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

        // Copy updated bird and pipes back to host
        cudaMemcpy(&birdY, d_birdY, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pipeX, d_pipeX, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pipeY, d_pipeY, 2 * sizeof(int), cudaMemcpyDeviceToHost);

        // Check for collision
        collision = checkCollision(birdY, 5, pipeX, pipeY);
        if (collision) {
            printf("Game Over! You collided with a pipe.\n");
            break;
        }

        // Render game to the console
        renderGame(birdY, pipeX, pipeY);

        // Sleep to slow down the game loop (in milliseconds)
        Sleep(100);  // 100 ms delay for each frame to make the game playable
    }

    // Free memory
    cudaFree(d_birdY);
    cudaFree(d_birdVelocity);
    cudaFree(d_isJumping);
    cudaFree(d_pipeX);
    cudaFree(d_pipeY);
    cudaFree(d_states);

    return 0;
}
