#include <math.h>
#include <stdio.h>
#include <time.h>
#include <malloc.h>
#include <mpi.h>

void gpu(int,int,int,int,float*,float*);

int main(int argc, char *argv[])
{
        int my_rank;
        int PR;
        MPI_Status status;
        MPI_Request request;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &PR);

        int i, j, k;
        int n = 20;
        int HEIGHT = n/PR;

        size_t size = (HEIGHT+1) * (n+1) * sizeof(float);
        float *T = (float *)malloc(size);
        for (i = 0; i < HEIGHT; i++)
                for (j = 0; j < n; j++)
                        T[i * n + j] = 0;
        float *T_old = (float *)malloc(size);
        for (i = 0; i < HEIGHT; i++)
                for (j = 0; j < n; j++)
                        T_old[i * n + j] = 0;

        if(my_rank == 0)
        {
                for (i = 0; i < n+1; i++)
                {
                        T_old[0*n+i] = 500;
                        T[0*n+i] = 500;
                }
        }

        if(my_rank == PR-1)
        {
                for (i = 0; i < n+1; i++)
                {
                        T_old[HEIGHT*n+i] = 500;
                        T[HEIGHT*n+i] = 500;
                }
        }

        for (i = 0; i < HEIGHT+1; i++)
        {
                T_old[i*n+0] = 500;
                T[i*n+0] = 500;
                T_old[i*n+n] = 500;
                T[i*n+n] = 500;
        }

        printf("PR %d\n", PR);

        printf("HEIGHT %d\n", HEIGHT);


        int firstRow = (my_rank)*HEIGHT + 1;
        int lastRow = firstRow + HEIGHT - 1;
        if(my_rank < PR-1 && PR>1)
                lastRow += 1;
        printf("firstRow %d\n", firstRow);
        printf("lastRow %d\n", lastRow);
        double time;

        printf("%d\n", my_rank);

        time = MPI_Wtime();

        for(k = 0;k < 10000;k++)
        {
                gpu(k, my_rank, n, HEIGHT, T, T_old);

                if(my_rank>0)
                        MPI_Send(&(T[0*n+0]),n,MPI_FLOAT,my_rank-1,6,MPI_COMM_WORLD);
                if(my_rank<PR-1)
                        MPI_Send(&(T[(HEIGHT-1)*n+0]),n,MPI_FLOAT,my_rank+1,6,MPI_COMM_WORLD);
                if(my_rank<PR-1)
                        MPI_Recv(&(T[HEIGHT*n+0]),n,MPI_FLOAT,my_rank+1,6,MPI_COMM_WORLD,&status);
                if(my_rank>0)
                        MPI_Recv(&(T[0*n+0]),n,MPI_FLOAT,my_rank-1,6,MPI_COMM_WORLD,&status);

                k++;

                gpu(k, my_rank, n, HEIGHT, T_old, T);

                if(my_rank>0)
                        MPI_Send(&(T_old[0*n+0]),n,MPI_FLOAT,my_rank-1,6,MPI_COMM_WORLD);
                if(my_rank<PR-1)
                        MPI_Send(&(T_old[(HEIGHT-1)*n+0]),n,MPI_FLOAT,my_rank+1,6,MPI_COMM_WORLD);
                if(my_rank<PR-1)
                        MPI_Recv(&(T_old[HEIGHT*n+0]),n,MPI_FLOAT,my_rank+1,6,MPI_COMM_WORLD,&status);
                if(my_rank>0)
                        MPI_Recv(&(T_old[0*n+0]),n,MPI_FLOAT,my_rank-1,6,MPI_COMM_WORLD,&status);
        }

        if(my_rank>0)
                MPI_Send(&(T_old[0*n+0]),(HEIGHT+1) * (n+1),MPI_FLOAT,0,6,MPI_COMM_WORLD);

        if(my_rank == 0)
        {
                printf("%f\n", MPI_Wtime()-time);

                float *T_node = (float *)malloc(size);
                /*for(j = 0; j < HEIGHT+1; j++)
                {
                        MPI_Recv(&(T_node[j*n+0]),n+1,MPI_FLOAT,my_rank+1,6,MPI_COMM_WORLD,&status);
                }*/
                //MPI_Recv(&(T_node[0*n+0]),(HEIGHT+1) * (n+1),MPI_FLOAT,1,6,MPI_COMM_WORLD,&status);
                /*(for(j = 0; j < HEIGHT+1; j++)
                {
                        for(i = 0; i<n+1; i++)
                                printf("%.0f ", T_old[j*n+i]);
                        printf("\n");
                }*/
                FILE *S;
                S = fopen("results.txt", "w");
                for (i = 0; i < HEIGHT+1; i++)
                {
                        for (j = 0; j < n+1; j++)
                        {
                                fprintf(S,"%.0f ", T_old[i*n+i]);
                        }
                        fprintf(S, "\n");
                }
                fclose(S);
                int thread;
                if(PR>1)
                {
                        for(thread = 1; thread<PR; thread++)
                        {
                                MPI_Recv(&(T_node[0*n+0]),(HEIGHT+1) * (n+1),MPI_FLOAT,thread,6,MPI_COMM_WORLD,&status);
                                /*for(j = 1; j < HEIGHT+1; j++)
                                {
                                        for(i = 0; i<n+1; i++)
                                                printf("%.0f ", T_node[j*n+i]);
                                        printf("\n");
                                }*/
                        }
                }
                free(T_node);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        free(T);
        free(T_old);

        MPI_Finalize();
        return 0;
}

