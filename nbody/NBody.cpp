#include <iostream>
#include <vector>
#include <GL/freeglut.h>

#include "../Common.h"

// global constants
const std::string PROGRAM_SOURCE = STRINGIFY (
	/* -*- mode: c++ -*- */

	// *************
	// Simulation
	// *************
	__constant float dt = 1.0e-3;
	__constant float G  = 5.0e-2;
	__constant float eps  = 1.0e-1;

	__kernel
    void SimulationKernel (__global float4* particles, const int BODY_NUM)
	{
		int id = get_global_id (0);
		float2 F = (float2) (0.0f, 0.0f);

		for (int i = 0; i < BODY_NUM; ++i)
		{
			if (i != id)
			{
				float2 r = particles [i].xy - particles [id].xy;
				float l = length (r);
				F += r / pow (l * l + eps * eps, 1.5f);
			}
		}
		F *= G;

		float2 vel = particles [id].zw + F * dt;
		float2 pos = particles [id].xy + vel * dt;

		particles [id] = (float4) (pos, vel);
	}

	// *************
	// Visualization
	// *************
	__kernel
	void VisualizationClear (const int width, const int height, __global float4* visualizationBuffer)
    {
	    int2 id = (int2) (get_global_id (0), get_global_id (1));

	    if (id.x < width && id.y < height)
		    visualizationBuffer [id.x + id.y * width] = (float4) (0.0f);
	}


	int2 Sampl (int2 coord, int width, int height)
	{
		int x = max (min (width - 1, coord.x), 0);
		int y = max (min (height - 1, coord.y), 0);
		return (int2) (x,y);
	}

	__constant float r = 2.0e-3;

	__kernel
	void Visualization (const int width, const int height, __global float4* visualizationBuffer, __global float4* particleBuffer)
	{
		int id = get_global_id (0);
		float4 posdir = particleBuffer [id];
		int w = width * r;
		for (int i = -w; i <= w; ++i)
		for (int j = -w; j <= w; ++j)
		{
			int2 coord = Sampl ( (int2) (posdir.x * (width -1) + i, posdir.y * (height -1) + j), width , height);
			visualizationBuffer [coord.x + coord.y * width] = (float4) (1, 1, 1, 1);
		}
	}
);

const size_t BODY_NUM = 5000;

// global variables
bool keysPressed [256] = { false };
int visualizationWidth = 512;
int visualizationHeight = 512;

// visualization buffers
size_t visualizationBufferSize [2];
cl_float4* visualizationBufferCPU = nullptr;
cl::Buffer visualizationBufferGPU;

cl_int errorCode = CL_SUCCESS;

// simulation buffer
// position + velocity
cl::Buffer particlesBufferGPU;
cl_float4* particlesBufferCPU = nullptr;

// kernels
cl::Context context;
cl::CommandQueue queue;
cl::Program program;

cl::Kernel visualizationClearKernel;
cl::Kernel visualizationKernel;
cl::Kernel simulationKernel;


bool ResetSimulation (void)
{
    for (size_t i = 0; i < BODY_NUM; ++i)
    {
        float p1 = static_cast<float> (rand ()) / RAND_MAX;
        float p2 = static_cast<float> (rand ()) / RAND_MAX;
        float v1 = 2.0 * static_cast<float> (rand ()) / RAND_MAX - 1.0;
        float v2 = 2.0 * static_cast<float> (rand ()) / RAND_MAX - 1.0;
        particlesBufferCPU [i] = { p1, p2, v1, v2 };
    }
    errorCode = queue.enqueueWriteBuffer (particlesBufferGPU, true, 0, sizeof (cl_float4) * BODY_NUM, particlesBufferCPU);
    if (errorCode != CL_SUCCESS)
        return false;

    return true;
}


bool AllocateVisualizationBuffers (void)
{
	visualizationBufferSize [0] = visualizationWidth;
	visualizationBufferSize [1] = visualizationHeight;
	if (visualizationBufferCPU != nullptr)
		delete [] visualizationBufferCPU;

    try {
        visualizationBufferCPU = new cl_float4 [visualizationWidth * visualizationHeight];
    } catch (const std::bad_alloc& ba) {
        std::cout << ba.what () << std::endl;

        return false;
    }
	visualizationBufferGPU = cl::Buffer (context, CL_MEM_READ_WRITE, sizeof (cl_float4) * visualizationWidth * visualizationHeight, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

    return true;
}


bool InitSimulation (void)
{
    std::vector<cl::Platform> platforms;

	cl::Platform::get (&platforms);
	if (platforms.size () == 0)
	{
		std::cout << "Unable to find suitable platform." << std::endl;

		return false;
	}

	cl_context_properties properties [] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms [0]) (), 0 };

	context = cl::Context (CL_DEVICE_TYPE_GPU, properties, nullptr, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES> (&errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

	queue = cl::CommandQueue (context, devices [0], 0, &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

	program = cl::Program (context, PROGRAM_SOURCE);
    errorCode = program.build (devices);
    if (errorCode != CL_SUCCESS) {
        std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG> (devices [0]);
        std::cerr << "Build log:\n" << buildlog << std::endl;

        return false;
    }

	visualizationClearKernel = cl::Kernel (program, "VisualizationClear", &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

	visualizationKernel = cl::Kernel (program, "Visualization", &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

	simulationKernel = cl::Kernel (program, "SimulationKernel", &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

    try {
        particlesBufferCPU = new cl_float4 [BODY_NUM];
    } catch (const std::bad_alloc& ba) {
        std::cout << ba.what () << std::endl;

        return false;
    }

	particlesBufferGPU = cl::Buffer (context, CL_MEM_READ_WRITE, sizeof (cl_float4) * BODY_NUM, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
        return false;

	if (!AllocateVisualizationBuffers ())
        return false;

	if (!ResetSimulation ())
	    return false;

    return true;
}


void RunSimulationKernel (void)
{
    errorCode = simulationKernel.setArg (0, particlesBufferGPU);
    errorCode |= simulationKernel.setArg (1, (int)BODY_NUM);
    if (errorCode != CL_SUCCESS)
        exit (-1);

    errorCode = queue.enqueueNDRangeKernel (simulationKernel, cl::NullRange, cl::NDRange (BODY_NUM), cl::NullRange, nullptr, nullptr);
    if (errorCode != CL_SUCCESS)
        exit (-1);
}


void RunVisualizationKernels (void)
{
	errorCode = visualizationClearKernel.setArg (0, visualizationWidth);
	errorCode |= visualizationClearKernel.setArg (1, visualizationHeight);
	errorCode |= visualizationClearKernel.setArg (2, visualizationBufferGPU);
    if (errorCode != CL_SUCCESS)
        exit (-1);

	cl::Event event;
	errorCode = queue.enqueueNDRangeKernel (visualizationClearKernel, cl::NullRange, cl::NDRange (visualizationWidth, visualizationHeight), cl::NullRange, nullptr, &event);
    if (errorCode != CL_SUCCESS)
        exit (-1);

	errorCode = visualizationKernel.setArg (0, visualizationWidth);
	errorCode |= visualizationKernel.setArg (1, visualizationHeight);
	errorCode |= visualizationKernel.setArg (2, visualizationBufferGPU);
	errorCode |= visualizationKernel.setArg (3, particlesBufferGPU);
    if (errorCode != CL_SUCCESS)
        exit (-1);
	
    errorCode = queue.enqueueNDRangeKernel (visualizationKernel, cl::NullRange, cl::NDRange (BODY_NUM), cl::NullRange, nullptr, &event);
    if (errorCode != CL_SUCCESS)
        exit (-1);

	errorCode = 
        queue.enqueueReadBuffer (visualizationBufferGPU, CL_TRUE, 0, sizeof (cl_float4) * visualizationWidth * visualizationHeight, visualizationBufferCPU, nullptr, &event);
    if (errorCode != CL_SUCCESS)
        exit (-1);

	glDrawPixels (visualizationWidth, visualizationHeight, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}


void DestroySimulation (void)
{
    if (visualizationBufferCPU != nullptr)
        delete [] visualizationBufferCPU;
    if (particlesBufferCPU != nullptr)
        delete [] particlesBufferCPU;
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

	RunSimulationKernel ();
	RunVisualizationKernels ();

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

	case 'R': case 'r':
		ResetSimulation ();
		break;

	case 27:
        DestroySimulation ();
		exit (0);
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
	visualizationWidth = newWidth;
	visualizationHeight = newHeight;
	AllocateVisualizationBuffers ();
	glViewport (0, 0, visualizationWidth, visualizationHeight);
}


int main (int argc, char* argv [])
{
    srand (time (0));

	// OpenCL processing
	if (!InitSimulation ())
        return -1;

	glutInit (&argc, argv);
	glutInitContextVersion (3, 0);
	glutInitContextFlags (GLUT_CORE_PROFILE | GLUT_DEBUG);
	glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize (visualizationWidth, visualizationHeight);
	glutCreateWindow ("NBODY");

	InitOpenGL ();

	glutDisplayFunc (Display);
	glutIdleFunc (Idle);
	glutReshapeFunc (Reshape);
	glutKeyboardFunc (KeyDown);
	glutKeyboardUpFunc (KeyUp);
	glutMouseFunc (MouseClick);
	glutMotionFunc (MouseMove);

	glutMainLoop ();
	return (0);
}

