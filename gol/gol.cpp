#include <GL/freeglut.h>
#include <CL/cl2.hpp>
#include "../Common.h"

const char* kernelString = STRINGIFY(
    __kernel void gol(__global char* in, const int width, const int height, __global char* out)
    {
        int2 id = (int2)(get_global_id(0), get_global_id(1));
        if (id.x < width && id.y < height)
        {
            char aliveNeighbors = 0;
            for (int y = id.y-1; y <= id.y+1; ++y)
                for (int x = id.x-1; x <= id.x+1; ++x)
                {
                    int row = (y + height) % height;
                    int col = (x + width) % width; 
                    aliveNeighbors += in[row * width + col];
                } 
            int idx = id.y * width + id.x;
            aliveNeighbors -= in[idx];
            out[idx] = (aliveNeighbors == 3) || (aliveNeighbors == 2 && in[idx] == 1) ? 1 : 0;
        }
    }
);

const char* kernelName = "gol";

const int WIDTH = 512;
const int HEIGHT = 512;
bool keysPressed[256];

const cl_uint workDim = 2;
size_t globalWorkSize[workDim];

cl_platform_id platform = NULL;
cl_device_id device = NULL;
cl_context context = NULL;
cl_command_queue commands = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;

char* hostBuffer = NULL;
cl_float3* image = NULL;
cl_mem deviceBuffer = NULL;
cl_mem deviceBuffer_out = NULL;

const cl_float3 aliveColor = {1.0f, 1.0f, 1.0f};
const cl_float3 deadColor = {0.0f, 0.0f, 0.0f};

cl_int err = CL_SUCCESS;

// OpenCL
void initOpenCL(void)
{
    // get available platforms - we want to get maximum 1 platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (!CheckCLError(err)) exit(-1);
    
    const char MAXLENGTH = 40;
    char vendor[MAXLENGTH];
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, MAXLENGTH, vendor, NULL);
    printf("Vendor: %s\n", vendor);

    // get available GPU devices - we want to get maximum 1 device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (!CheckCLError(err)) exit(-1);

    char deviceName[MAXLENGTH];
    clGetDeviceInfo(device, CL_DEVICE_NAME, MAXLENGTH, &deviceName, NULL);
    printf("Device name: %s\n", deviceName);

    // creation of OpenCl context with given properties
    cl_context_properties props[] = { 
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 
    };
    context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if(!context || !CheckCLError(err)) exit(-1);

    // creation of OpenCL command queue with the CL_QUEUE_PROFILING_ENABLE property
    commands = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(!commands || !CheckCLError(err)) exit(-1);

    // creation of the program
    program = clCreateProgramWithSource(context, 1, &kernelString, NULL, &err);
    if (!CheckCLError(err)) exit(-1);

    // compilation of the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (!CheckCLError(err))
    {
        size_t logLength;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLength);
        char* log = new char[logLength];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLength, log, 0);
        std::cout << log << std::endl;
        delete[] log;
        exit(-1);
    }

    // creation of the kernel
    kernel = clCreateKernel(program, kernelName, &err);
    if(!CheckCLError(err)) exit(-1);

    // initialization of host and device data
    globalWorkSize[0] = WIDTH;
    globalWorkSize[1] = HEIGHT;

    image = new cl_float3[WIDTH * HEIGHT];
    hostBuffer = new char[WIDTH * HEIGHT];

    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        hostBuffer[i] = ((float) rand() / RAND_MAX < 0.3) ? 1 : 0;

    deviceBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * WIDTH * HEIGHT, NULL, &err);
    if (!deviceBuffer || !CheckCLError(err)) exit(-1);

    deviceBuffer_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * WIDTH * HEIGHT, NULL, &err);
    if (!deviceBuffer_out || !CheckCLError(err)) exit(-1);

    err = clEnqueueWriteBuffer(commands, deviceBuffer, CL_TRUE, 0, sizeof(char) * WIDTH * HEIGHT, hostBuffer, 0, NULL, NULL);
    if (!CheckCLError(err)) exit(-1);
}

void runOpenCL(void)
{
    // setting the kernel arguments
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceBuffer); 
    err |= clSetKernelArg(kernel, 1, sizeof(int), &WIDTH); 
    err |= clSetKernelArg(kernel, 2, sizeof(int), &HEIGHT); 
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &deviceBuffer_out); 
    if (!CheckCLError(err)) exit(-1);

    // kernel execution
    err = clEnqueueNDRangeKernel(commands, kernel, workDim, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (!CheckCLError(err)) exit(-1);

    // getting back the results
    clFinish(commands);
    err = clEnqueueReadBuffer(commands, deviceBuffer_out, CL_TRUE, 0, sizeof(char) * WIDTH * HEIGHT, hostBuffer, 0, NULL, NULL);
    if (!CheckCLError(err)) exit(-1);

    // swap the device buffers for the next step of the computation
    std::swap(deviceBuffer, deviceBuffer_out);
    
    // output of results
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        image[i] = (hostBuffer[i] == 1) ? aliveColor : deadColor;

    glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_FLOAT, image);
}

void destroyOpenCL(void)
{
    // free data
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(deviceBuffer_out);
    clReleaseMemObject(deviceBuffer);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    delete[] hostBuffer;
    delete[] image;
}

// OpenGL
void initOpenGL(void)
{
    glClearColor(0.17f, 0.4f, 0.6f, 1.0f);
    glDisable(GL_DEPTH_TEST);
}

void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    runOpenCL();

    glutSwapBuffers();
}

void idle(void) {
    glutPostRedisplay();
}

void keyDown(unsigned char key, int x, int y) {
    keysPressed[key] = true;
}

void keyUp(unsigned char key, int x, int y) {
    keysPressed[key] = false;
    switch (key) {
        case 27:
            destroyOpenCL();
            exit(0);
            break;
    }
}

void mouseClick(int button, int state, int x, int y) {

}

void mouseMove(int x, int y) {

}

void reshape(int newWidth, int newHeight) {

}

int main(int argc, char* argv[]) {
    srand(time(0));

    glutInit(&argc, argv);
    glutInitContextVersion(3, 0);
    glutInitContextFlags(GLUT_CORE_PROFILE | GLUT_DEBUG);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Game of life");

    initOpenGL();

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyDown);
    glutKeyboardUpFunc(keyUp);
    glutMouseFunc(mouseClick);
    glutMotionFunc(mouseMove);

    initOpenCL();
    glutMainLoop();
    return 0;
}

