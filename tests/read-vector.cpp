// read-vector.cpp
#include <iostream>
#include <ctime>
#include <stdlib.h>
using namespace std;

int main(int argc, char** argv)
{
    int size =  atoi(argv[1]);
	int *data = (int *) malloc (sizeof(int) * size);
	int next = 0;
	int times = 1000;
    int maxstride = 8388608; // 32MiB

	for(int stride = 1; stride <= maxstride; stride = stride * 2)
	{
		clock_t begin = clock();
		for(int t = 0; t <= times; t++)
		{
			for(int i = 0; i <= size; i = i + stride)
			{
				next = data[i];
			}
		}
		clock_t end = clock();
		double elapsed_secs = double(end-begin) / CLOCKS_PER_SEC / times / (size / stride);
		cout << stride << " " << elapsed_secs << endl;
	}
    free(data);
}

