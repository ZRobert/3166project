/*============================================================================
 *  Name: Robert Payne
 *  Student ID: 800703655
 *  Date: 4/3/2014
 *  Class: ITCS 3166-091
 *  Professor: Dr. Tzacheva
 *============================================================================
 *  Discription: This program is an implementation of the sliding 
 *  window algorithm using the MPI library to simulate the server
 *  and client. To compile this source, it is best to use a 
 *  terminal compiler (gcc, clang) with the MPI library installed. Ubuntu and
 *  Mac OSX come with compilers installed that can compile using
 *  the following command in the terminal:
 *
 *  $   mpicc slidingwindow.c -o slidingwindow
 *
 *  And can be run and executed using the following command:
 *
 *  $   mpirun -n 2 ./slidingwindow
 *
 *  This program uses the 2 argument from the command line to 
 *  run the program using 2 processes. In this code, process rank 0
 *  represents the server, while every other process represents
 *  a client. The server waits for a connection request from the
 *  client, then sends frames for the array 'a' to the client
 *  which stores the data in array 'b'. The throughput is measured
 *  by the (N * sizeof(int) * 8)/ the time stamp. The results of 
 *  different size buffers are displayed at the bottom of this
 *  file. Due to the nature of MPI, it is common for the output
 *  to appear out of order since this is determined by how the
 *  operating system schedules the individual processes.
 *
 *  NOTE: FRAMEBUFFERSIZE is adjusted in the source to change the
 *  amount of data that the server sends to the client with each
 *  packet. It is also possible to simulate multiple client request
 *  by changing the "while(controlMessage != DISCONNECT)" to an
 *  infinite loop and using a different number after the "-n" in
 *  the argument. But since it's an infinite loop, it will never
 *  reach the execution time output in this mode.
 ===========================================================================*/

#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DISCONNECT 1000
#define CONNECT 1001
#define N 512
#define FRAMEBUFFERSIZE 8
#define PRINT(x) printf(" %i", x)
#define MPI_START(p,rank,argc, argv)                                \
    MPI_Init(argc, argv);                                           \
    MPI_Comm_size(MPI_COMM_WORLD, p);                               \
    MPI_Comm_rank(MPI_COMM_WORLD, rank)


struct timeval tv1, tv2;        //timevalues for measuring the execution time
MPI_Status status;              //status needed for MPI function calls


//_____________________________________________________________________________
//  Function the server executes; waits for a connection to be made to
//  a client and sends the data in chunks determined by the FRAMEBUFFERSIZE
//-----------------------------------------------------------------------------
void server_func(int rank, int a[N]){

    int controlMessage = -1;    //used to connect/disconnect with client(s)
    int currentFrame;           //a counter for sending out the proper frame
    if(rank == 0){              //rank check to ensure only the server is executing
        
        while(controlMessage != DISCONNECT){    //Check for connection
            printf("Server: LISTENING FOR CONNECTION REQUEST\n");
            //recv the connection request
            MPI_Recv(&controlMessage, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            printf("Server: CONNECTION REQUEST RECEIVED FROM: %i\n", status.MPI_SOURCE);
            //initialize the current frame to 0
            currentFrame = 0;
        
            //send the frames until all the data has been sent
            while(currentFrame * FRAMEBUFFERSIZE <= N - FRAMEBUFFERSIZE){
                //send out the next frame if needed
                if(currentFrame * FRAMEBUFFERSIZE <= N - FRAMEBUFFERSIZE){
                    //send a frame worth of data to the client
                    MPI_Send(&a[currentFrame * FRAMEBUFFERSIZE], FRAMEBUFFERSIZE, MPI_INT, status.MPI_SOURCE, rank, MPI_COMM_WORLD);
                    currentFrame++;     //advance the frame
                }
            }
        }
    }
}

//_____________________________________________________________________________
//  Client executes this function to start a connection and receive data
//  from the server
//-----------------------------------------------------------------------------
void client_func(int rank, int b[N]){
    
    int i;                      //iterator for loop
    int controlMessage = -1;    //control message that's sent to the server
    int currentFrame = 0;       //tracks the frames being received
    
    if(rank != 0){              //rank check, all ranks not 0 are considered
                                //clients
        controlMessage = CONNECT;   //send a connect request to the server
        printf("Client: CONNECTION REQUESTED\n");
        MPI_Send(&controlMessage, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
        
        //while the server still has data to send
        while(currentFrame * FRAMEBUFFERSIZE <= N - FRAMEBUFFERSIZE){
            //receive the proper frame and store it into array b
            MPI_Recv(&b[currentFrame * FRAMEBUFFERSIZE], FRAMEBUFFERSIZE, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            //prints out what was just received on the console
            printf("Client %i RCVD: ", rank);
            for(i = 0; i < FRAMEBUFFERSIZE; i++){
                printf(" %i", b[currentFrame * FRAMEBUFFERSIZE + i]);
            }
            printf("\n");
            //advance to the next frame
            currentFrame ++;

        }
        //send a disconnect message to the server
        controlMessage = DISCONNECT;
        printf("Client: DISCONNECTING\n");
        MPI_Send(&controlMessage, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }
}

//_____________________________________________________________________________
//   Take a time stamp before the sliding window starts
//-----------------------------------------------------------------------------
void start_timer(int rank){
    if(rank == 0){
        gettimeofday(&tv1, NULL);
    }
}

//_____________________________________________________________________________
//  Outputs the time the sliding window took to execute
//-----------------------------------------------------------------------------
void stop_timer(int rank){
    double elapsed_time = (tv2.tv_sec -tv1.tv_sec)+((tv2.tv_usec - tv1.tv_usec)/1000000.0);
    double intSize = sizeof(int) * N * 8;
    if(rank == 0){
        gettimeofday(&tv2, NULL);
        elapsed_time = (tv2.tv_sec -tv1.tv_sec)+((tv2.tv_usec - tv1.tv_usec)/1000000.0);
        printf("elapsed_time=\t%lf (seconds)\n", elapsed_time);
        printf("%lf\n",intSize);
        printf("throughput = %lf bits/sec\n", intSize/elapsed_time);
    }
    //terminate the processes so the program can exit
    MPI_Finalize();
}

//_____________________________________________________________________________
//  main
//-----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    
    int i;              //iterator for loop
    int p, rank = 0;    //p = number of processes; rank is a process identifier
    int a[512];         //data that is being sent from server to client
    int b[512] = {};    //where the data from the server is being stored
  
    //populate a[] with values of 0 to N-1
    for(i = 0; i < N; i++){
        a[i] = i;
    }
    
    //initialize MPI
    MPI_START(&p, &rank, &argc, &argv);
    //take a time stamp for starting the sliding frame
    start_timer(rank);
    //function that the server will execute
    server_func(rank, a);
    //function that the client will execute
    client_func(rank, b);
    //stop taking time and measure the throughput
    stop_timer(rank);
   
    return 0;
}

/*===============================================
 *      RESULTS OF THROUGHPUT TESTING            |
 *===============================================
 *  FRAMEBUFFERSIZE |   THROUGHPUT               |
 *-----------------------------------------------
 *  512:            |   35008547.008547 bites/sec|
 *-----------------------------------------------
 *  256:            |   18143964.562569 bits/sec |
 *-----------------------------------------------
 *  128:            |   10835978.835979 bits/sec |
 *-----------------------------------------------
 *  64:             |   10708496.732026 bits/sec |
 *-----------------------------------------------
 *  32:             |   11268225.584594 bits/sec |
 *-----------------------------------------------
 *  16:             |   8885032.537961 bits/sec  |
 *-----------------------------------------------
 *  8:              |   8449716.348633 bits/sec  |
 *-----------------------------------------------
 *  4:              |   6823823.406914 bits/sec  |
 *-----------------------------------------------
 *  2:              |   2835583.246798 bits/sec  |
 *-----------------------------------------------
 *  1:              |   2398828.696925 bits/sec  |
 *===============================================
 *  Conclusion: The increase in the number of 
 *  packets slows down the bit rate
 *  considerably, since each time the server
 *  has to prepare each packet.
 */