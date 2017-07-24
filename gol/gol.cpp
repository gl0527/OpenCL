#include <GL/freeglut.h>
#include <CL/cl2.hpp>
#include "../Common.h"

// global constants
const char* kernelString = STRINGIFY (
    __kernel void GOL (__global char* in, const int width, const int height, __global char* out)
    {
        int2 threadID_2D = (int2) (get_global_id (0), get_global_id (1));

        if (threadID_2D.x < width && threadID_2D.y < height)
        {
            char aliveNeighbors = 0;
            for (int y = threadID_2D.y-1; y <= threadID_2D.y+1; ++y)
                for (int x = threadID_2D.x-1; x <= threadID_2D.x+1; ++x)
                {
                    int row = (y + height) % height;
                    int col = (x + width) % width; 
                    aliveNeighbors += in [row * width + col];
                } 
            int threadID_1D = threadID_2D.y * width + threadID_2D.x;
            aliveNeighbors -= in [threadID_1D];
            out [threadID_1D] = (aliveNeighbors == 3) || (aliveNeighbors == 2 && in [threadID_1D]);
        }
    }
);

const char* kernelName      = "GOL";
const int WIDTH             = 800;
const int HEIGHT            = 600;
const int NUM_OF_CELLS      = WIDTH * HEIGHT;
const cl_uint workDim       = 2;
const cl_float3 aliveColor  = {1.0f, 1.0f, 0.0f};
const cl_float3 deadColor   = {0.1f, 0.1f, 0.1f};

// global variables
bool keysPressed [256]      = { false };
bool running                = true;

size_t globalWorkSize [workDim];

cl_context context          = nullptr;
cl_command_queue commands   = nullptr;
cl_program program          = nullptr;
cl_kernel kernel            = nullptr;
                              
char* hostBuffer            = nullptr;
cl_float3* image            = nullptr;
cl_mem deviceBuffer         = nullptr;
cl_mem deviceBuffer_out     = nullptr;

cl_int err                  = CL_SUCCESS;


// OpenCL
bool InitOpenCL (void)
{
    // get available platforms - we want to get maximum 1 platform
    cl_platform_id platform = nullptr;
    err = clGetPlatformIDs (1, &platform, nullptr);
    if (CheckCLError (err) == false)
        return false;
    
    const char MAXLENGTH = 40;
    char vendor [MAXLENGTH];
    clGetPlatformInfo (platform, CL_PLATFORM_VENDOR, MAXLENGTH, vendor, nullptr);
    printf ("GOU vendor: %s\n", vendor);

    // get available GPU devices - we want to get maximum 1 device
    cl_device_id device = nullptr;
    err = clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (CheckCLError (err) == false)
        return false;

    char deviceName [MAXLENGTH];
    clGetDeviceInfo (device, CL_DEVICE_NAME, MAXLENGTH, &deviceName, nullptr);
    printf ("GPU device: %s\n", deviceName);

    // creation of OpenCL context with given properties
    cl_context_properties props [] = { 
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 
    };
    context = clCreateContext (props, 1, &device, nullptr, nullptr, &err);
    if (context == nullptr || CheckCLError (err) == false)
        return false;

    // creation of OpenCL command queue with the CL_QUEUE_PROFILING_ENABLE property
    commands = clCreateCommandQueue (context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (commands == nullptr || CheckCLError (err) == false)
        return false;

    // creation of the program
    program = clCreateProgramWithSource (context, 1, &kernelString, nullptr, &err);
    if (CheckCLError (err) == false)
        return false;

    // compilation of the program
    err = clBuildProgram (program, 1, &device, nullptr, nullptr, nullptr);
    if (CheckCLError (err) == false)
    {
        size_t logLength;
        char* log = nullptr;
        clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);
        try {
            log = new char [logLength];
        } catch (const std::bad_alloc& ba) {
            std::cerr << "Bad alloc exception was caught:" << ba.what () << '\n';

            return false;
        }
        clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, logLength, log, 0);
        std::cout << log << std::endl;
        if (log != nullptr)
            delete [] log;

        return false;
    }

    // creation of the kernel
    kernel = clCreateKernel (program, kernelName, &err);
    if (CheckCLError (err) == false)
        return false;

    // initialization of host and device data
    globalWorkSize [0] = WIDTH;
    globalWorkSize [1] = HEIGHT;

    try {
        image = new cl_float3 [NUM_OF_CELLS];
        hostBuffer = new char [NUM_OF_CELLS];
    } catch (const std::bad_alloc& ba) {
        std::cerr << "Bad alloc exception was caught: " << ba.what () << '\n';

        return false;
    }

    for (int i = 0; i < NUM_OF_CELLS; ++i)
        hostBuffer [i] = ( (float) rand () / RAND_MAX < 0.3) ? 1 : 0;

    deviceBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, NUM_OF_CELLS, nullptr, &err);
    if (deviceBuffer == nullptr || CheckCLError (err) == false)
        return false;

    deviceBuffer_out = clCreateBuffer (context, CL_MEM_READ_WRITE, NUM_OF_CELLS, nullptr, &err);
    if (deviceBuffer_out == nullptr || CheckCLError (err) == false)
        return false;

    err = clEnqueueWriteBuffer (commands, deviceBuffer, CL_TRUE, 0, NUM_OF_CELLS, hostBuffer, 0, nullptr, nullptr);
    if (CheckCLError (err) == false)
        return false;
    
    return true;
}


void RunOpenCL (void)
{
    // setting the kernel arguments
    err = clSetKernelArg (kernel, 0, sizeof (cl_mem), &deviceBuffer); 
    err |= clSetKernelArg (kernel, 1, sizeof (int), &WIDTH); 
    err |= clSetKernelArg (kernel, 2, sizeof (int), &HEIGHT); 
    err |= clSetKernelArg (kernel, 3, sizeof (cl_mem), &deviceBuffer_out); 
    if (CheckCLError (err) == false)
        exit (-1);

    // kernel execution
    err = clEnqueueNDRangeKernel (commands, kernel, workDim, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (CheckCLError (err) == false)
        exit (-1);

    // getting back the results
    clFinish (commands);
    err = clEnqueueReadBuffer (commands, deviceBuffer_out, CL_TRUE, 0, NUM_OF_CELLS, hostBuffer, 0, nullptr, nullptr);
    if (CheckCLError (err) == false)
        exit (-1);

    // swap the device buffers for the next step of the computation
    std::swap (deviceBuffer, deviceBuffer_out);
    
    // updating the image
    for (int i = 0; i < NUM_OF_CELLS; ++i)
        image [i] = (hostBuffer [i] == 1) ? aliveColor : deadColor;
}


void DestroyOpenCL (void)
{
    // free data
    clReleaseKernel (kernel);
    clReleaseProgram (program);
    clReleaseMemObject (deviceBuffer_out);
    clReleaseMemObject (deviceBuffer);
    clReleaseCommandQueue (commands);
    clReleaseContext (context);
    if (hostBuffer != nullptr)
        delete [] hostBuffer;
    if (image != nullptr)
        delete [] image;
}


// OpenGL
void InitOpenGL (void)
{
    glClearColor (0.17f, 0.4f, 0.6f, 1.0f);
    glDisable (GL_DEPTH_TEST);
}


void Display (void)
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (running)
        RunOpenCL ();
    glDrawPixels (WIDTH, HEIGHT, GL_RGBA, GL_FLOAT, image);

    glutSwapBuffers ();
}


void Idle (void)
{
    glutPostRedisplay ();
}


void KeyDown (unsigned char key, int /*x*/, int /*y*/)
{
    keysPressed [key] = true;
}

void KeyUp (unsigned char key, int /*x*/, int /*y*/)
{
    keysPressed [key] = false;
    switch (key) {
        case 27:
            DestroyOpenCL ();
            exit (0);
            break;
        case 32:
            running = !running;
            break;
    }
}


void MouseClick (int /*button*/, int /*state*/, int /*x*/, int /*y*/)
{
}


void MouseMove (int /*x*/, int /*y*/)
{
}


void Reshape (int /*newWidth*/, int /*newHeight*/)
{
}


int main (int argc, char* argv [])
{
    srand (time (0));

    if (InitOpenCL () == false)
        return 1;

    glutInit 				(&argc, argv);
    glutInitContextVersion 	(3, 0);
    glutInitContextFlags 	(GLUT_CORE_PROFILE | GLUT_DEBUG);
    glutInitDisplayMode 	(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize 		(WIDTH, HEIGHT);
    glutCreateWindow 		("Game of life");

    InitOpenGL ();

    glutDisplayFunc 	(Display);
    glutIdleFunc 		(Idle);
    glutReshapeFunc 	(Reshape);
    glutKeyboardFunc 	(KeyDown);
    glutKeyboardUpFunc 	(KeyUp);
    glutMouseFunc 		(MouseClick);
    glutMotionFunc 		(MouseMove);

    glutMainLoop ();

    return 0;
}

