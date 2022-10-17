/**
 * Laboratorio 6
 * 
 * Programacion de microprocesadores
 * 
 * Producto punto entre dos vectores de 768 elementos
 * 
 * 
 * Angel Castellanos 21700
 * Jose Pablo Santisteban 21153
 */

#include <stdio.h>

  
#include <cuda_runtime.h>


__global__ void
dotProduct(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        //multiplicacion de elementos de vectores
        C[i] = A[i] * B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    //Variable que contendra el resultado de la operacion
    float resultado = 0;

    // codigo de error a manejar
    cudaError_t err = cudaSuccess;

    // imprime el tamaño de los vectores
    int numElements = 768;
    size_t size = numElements * sizeof(float);
    printf("[Suma de vectores de %d elementos]\n", numElements);

    // Reserva de memoria para los vectores en el host 
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // verifica que la reserva de memoria haya sido exitosa
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Error al asignar la memoria!\n");
        exit(EXIT_FAILURE);
    }

    // Inicializa los vectores en el host 
    for (int i = 0; i < numElements; ++i)
    {
        float numRand1 = (500 + rand() % (5000 - 500)) / (float)RAND_MAX;
        float numRand2 = (500 + rand() % (5000 - 500)) / (float)RAND_MAX;
        h_A[i] = numRand1;
        h_B[i] = numRand2;
    }

    // Reserva de memoria para los vectores en el device
    float* d_A = NULL;
    err = cudaMalloc((void**)&d_A, size);

    // verifica que la reserva de memoria haya sido exitosa en el device 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo en reserval la memoria, ver código %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reserva de memoria para los vectores en el device y verifica que la reserva de memoria haya sido exitosa
    float* d_B = NULL;
    err = cudaMalloc((void**)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo en reservar la memoria en el device del vector B, ver código %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* d_C = NULL;
    err = cudaMalloc((void**)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo en reservar la memoria en el device del vector C, ver código %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copia los vectores A y B del host al device
    printf("Copiando los vectores del host al device...\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // verifica que la copia de memoria de los vectores A y B haya sido exitosa
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al copiar el vector A del host al device (código de error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al copiar el vector B del host al device (código de error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Lanzamiento del kernel CUda con  %d bloques de %d hilos\n", blocksPerGrid, threadsPerBlock);
    dotProduct <<<blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, numElements);
    err = cudaGetLastError();


    // verifica que el lanzamiento del kernel haya sido exitoso
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al multiplicar los vectores (código de error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copia el resultado del device (d_C) al host (h_C)
    printf("Copiando el resultado del device al host...\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al copiar el vector c del host al devide (código de error  %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //prueba que la suma de los vectores sea correcta
    for (int i = 0; i < numElements; ++i)
    {
        float multiplicacion = h_A[i] * h_B[i];
        if (fabs(multiplicacion / h_C[i]) != 1)
        {
            fprintf(stderr, "Resultado de verificación falló en el elemento %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Prueba pasada\n");


    //Suma el resultado de la suma de los vectores
    for (int i = 0; i < numElements; i++)
    {
        resultado = resultado + (float)h_C[i];
    }

    printf("El resultado total es: %f \n", resultado);

    

    // Libera la memoria reservada en el device
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al liberar el vector A del device (código de error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al liberar el vector B del device (código de error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo al liberar el vector C del device (código de error %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Libera la memoria reservada en el host
    free(h_A);
    free(h_B);
    free(h_C);


    // Reinicia el device y verifica que la reinicialización haya sido exitosa
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Fallo a desinicializar el device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Finalizado\n");
    return 0;
}

