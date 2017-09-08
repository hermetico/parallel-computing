// read-vector.cpp
#include <iostream>
#include <ctime>
#include <stdlib.h>
using namespace std;

int main(int argc, char** argv)
{
    int size =  atoi(argv[1]);
    char *data = (char *) malloc (sizeof(char) * size); // 1 integer --> 1 byte
	char next = 0;
	int times = 1000;
    int maxstride = size / 2;// half of the size

	for(int stride = 4; stride <= maxstride; stride = stride * 2)
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
    return (int) next;
}

