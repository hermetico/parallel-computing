// read-vector-2.cpp
#include <iostream>
#include <ctime>

using namespace std;
int main(int argc, char** argv)
{
	int data[1048576];
	int next = 0;
	int times = 100;
	for(int j = 1; j <= 512; j = j * 2)
	{
		clock_t begin = clock();
	
		for(int t = 0; t <= times; t++)
		{
			for(int x = 0; x < j; x++){
				for(int i = x; i <= 1048576; i = i + j)
				{
					next = data[i];
				}	
			}
		}
		clock_t end = clock();
		double elapsed_secs = double(end-begin) / CLOCKS_PER_SEC / times;
		cout << j << " " << elapsed_secs << endl;
	}
}

