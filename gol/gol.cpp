#ifdef WINDOWS
#include <GL\freeglut.h>
#include <CL\cl2.hpp>
#elif defined (__linux__)
#include <GL/freeglut.h>
#include <CL/cl2.hpp>
#endif
#include "Common.h"


const char* programSource = STRINGIFY (
    __kernel
    void GOL (__global char* in, const int width, const int height, __global char* out)
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

size_t screenWidth              = 800;
size_t screenHeight             = 600;
size_t globalWorkSize [2]       = { 0 };
bool keysPressed [256]          = { false };
bool isRunning                  = true;

cl_context context              = nullptr;
cl_command_queue commands       = nullptr;
cl_program program              = nullptr;
cl_kernel kernel                = nullptr;
                              
char* hostBuffer                = nullptr;
cl_float3* image                = nullptr;
cl_mem deviceBufferIn           = nullptr;
cl_mem deviceBufferOut          = nullptr;

cl_int errorCode                = CL_SUCCESS;


bool AllocateData (void)
{
    globalWorkSize[0] = screenWidth;
    globalWorkSize[1] = screenHeight;

    // (re)allocating host data
    if (hostBuffer != nullptr)
        delete [] hostBuffer;

    if (image != nullptr)
        delete [] image;

    try {
        image = new cl_float3 [screenWidth * screenHeight];
        hostBuffer = new char [screenWidth * screenHeight];
    } catch (const std::bad_alloc& ba) {
        std::cerr << "Bad alloc exception was caught: " << ba.what () << '\n';

        return false;
    }

    // (re)allocating device data
    clReleaseMemObject (deviceBufferIn);
    clReleaseMemObject (deviceBufferOut);

    deviceBufferIn = clCreateBuffer (context, CL_MEM_READ_WRITE, screenWidth * screenHeight, nullptr, &errorCode);
    if (deviceBufferIn == nullptr || !CheckCLError (errorCode))
        return false;

    deviceBufferOut = clCreateBuffer (context, CL_MEM_READ_WRITE, screenWidth * screenHeight, nullptr, &errorCode);
    if (deviceBufferOut == nullptr || !CheckCLError (errorCode))
        return false;

    return true;
}


bool InitData (void)
{
    // initializing host and device data
    for (size_t i = 0; i < screenWidth * screenHeight; ++i)
        hostBuffer [i] = (static_cast<float> (rand ()) / RAND_MAX < 0.3) ? 1 : 0;

    errorCode = clEnqueueWriteBuffer (commands, deviceBufferIn, CL_TRUE, 0, screenWidth * screenHeight, hostBuffer, 0, nullptr, nullptr);
    if (!CheckCLError (errorCode))
        return false;

    return true;
}


// OpenCL
bool InitOpenCL (void)
{
    // get available platforms - we want to get maximum 1 platform
    cl_platform_id platform = nullptr;
    errorCode = clGetPlatformIDs (1, &platform, nullptr);
    if (!CheckCLError (errorCode))
        return false;
    
    const char MAXLENGTH = 40;
    char vendorName [MAXLENGTH];
    clGetPlatformInfo (platform, CL_PLATFORM_VENDOR, MAXLENGTH, vendorName, nullptr);
    printf ("GPU vendor: %s\n", vendorName);

    // get available GPU devices - we want to get maximum 1 device
    cl_device_id device = nullptr;
    errorCode = clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (!CheckCLError (errorCode))
        return false;

    char deviceName [MAXLENGTH];
    clGetDeviceInfo (device, CL_DEVICE_NAME, MAXLENGTH, deviceName, nullptr);
    printf ("GPU device: %s\n", deviceName);

    // creation of OpenCL context with given properties
    cl_context_properties props [] = { 
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 
    };
    context = clCreateContext (props, 1, &device, nullptr, nullptr, &errorCode);
    if (context == nullptr || !CheckCLError (errorCode))
        return false;

    // creation of OpenCL command queue with the CL_QUEUE_PROFILING_ENABLE property
    commands = clCreateCommandQueue (context, device, CL_QUEUE_PROFILING_ENABLE, &errorCode);
    if (commands == nullptr || !CheckCLError (errorCode))
        return false;

    // creation of the program
    program = clCreateProgramWithSource (context, 1, &programSource, nullptr, &errorCode);
    if (!CheckCLError (errorCode))
        return false;

    // compilation of the program
    errorCode = clBuildProgram (program, 1, &device, nullptr, nullptr, nullptr);
    if (!CheckCLError (errorCode))
    {
        size_t logLength;
        char* log = nullptr;
        clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLength);
        try {
            log = new char [logLength];
        } catch (const std::bad_alloc& ba) {
            std::cerr << "Bad alloc exception was caught: " << ba.what () << '\n';

            return false;
        }
        clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, logLength, log, 0);
        std::cout << log << std::endl;
        if (log != nullptr)
            delete [] log;

        return false;
    }

    // creation of the kernel
    kernel = clCreateKernel (program, "GOL", &errorCode);
    if (!CheckCLError (errorCode))
        return false;

    // allocation and initialization of host and device data
    if (!AllocateData () || !InitData ())
        return false;
    
    return true;
}


void RunOpenCL (void)
{
    // setting the kernel arguments
    errorCode = clSetKernelArg (kernel, 0, sizeof (cl_mem), &deviceBufferIn); 
    errorCode |= clSetKernelArg (kernel, 1, sizeof (int), &screenWidth); 
    errorCode |= clSetKernelArg (kernel, 2, sizeof (int), &screenHeight); 
    errorCode |= clSetKernelArg (kernel, 3, sizeof (cl_mem), &deviceBufferOut); 
    if (!CheckCLError (errorCode))
        exit (-1);

    // kernel execution
    errorCode = clEnqueueNDRangeKernel (commands, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (!CheckCLError (errorCode))
        exit (-1);

    // getting back the results
    clFinish (commands);
    errorCode = clEnqueueReadBuffer (commands, deviceBufferOut, CL_TRUE, 0, screenWidth * screenHeight, hostBuffer, 0, nullptr, nullptr);
    if (!CheckCLError (errorCode))
        exit (-1);

    // swap the device buffers for the next step of the computation
    std::swap (deviceBufferIn, deviceBufferOut);
    
    // updating the image
    for (size_t i = 0; i < screenWidth * screenHeight; ++i)
        image [i] = (hostBuffer [i] == 1) ? cl_float3 {0.22f, 1.0f, 0.08f} : cl_float3 {0.0f, 0.0f, 0.0f};
}


void DestroyOpenCL (void)
{
    // free data
    clReleaseKernel (kernel);
    clReleaseProgram (program);
    clReleaseMemObject (deviceBufferOut);
    clReleaseMemObject (deviceBufferIn);
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
    // TODO texturaba kellene rajzolni, mert ez kurva lassu!!!
    if (isRunning)
    {
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawPixels (screenWidth, screenHeight, GL_RGBA, GL_FLOAT, image);
        glutSwapBuffers ();
    }
}


void Idle (void)
{
    if (isRunning)
        RunOpenCL ();
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
            isRunning = !isRunning;
            break;

        case 'R': case 'r':
            if (!InitData ())
                exit (-1);
            break;
    }
}


void MouseClick (int /*button*/, int /*state*/, int /*x*/, int /*y*/)
{
}


void MouseMove (int /*x*/, int /*y*/)
{
}


void Reshape (int newWidth, int newHeight)
{
    screenWidth = newWidth;
    screenHeight = newHeight;

    if (AllocateData () && InitData ())
        glViewport (0, 0, screenWidth, screenHeight);
    else
        exit (-1);
}


int main (int argc, char* argv [])
{
    srand (time (0));

    if (!InitOpenCL ())
        return 1;

    glutInit (&argc, argv);
    glutInitContextVersion (3, 0);
    glutInitContextFlags (GLUT_CORE_PROFILE | GLUT_DEBUG);
    glutInitDisplayMode (GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (screenWidth, screenHeight);
    glutCreateWindow ("Game of life");

    InitOpenGL ();

    glutDisplayFunc (Display);
    glutIdleFunc (Idle);
    glutReshapeFunc (Reshape);
    glutKeyboardFunc (KeyDown);
    glutKeyboardUpFunc (KeyUp);
    glutMouseFunc (MouseClick);
    glutMotionFunc (MouseMove);

    glutMainLoop ();

    return 0;
}

