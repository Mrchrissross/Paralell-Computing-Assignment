#include "WeatherTool.h"

int main(int argc, char **argv) 
{
	// Start a timer that will be used to time exactly how long it takes to complete
	// all tasks from start to finish.
	Timer timerTotalStart = Clock::now();

	// Selecting the required data file.
		const char* path = "temp_lincolnshire.txt";
		//const char* path = "temp_lincolnshire_short.txt";
		//const char* path = "temp_lincolnshire_even_shorter.txt";

	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0; int device_id = 0;

	// Platform and device selection.
	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}


	// Used to demonstrate other platforms.
	/*cout << "Platform: ";
	cin >> platform_id;

	cout << endl;*/

	// Display the selected device.
	std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; 

	// Provide feedback to the user.
	std::printf("Reading File...");

	// Start Timer.
	Timer timerStart = Clock::now();

	// Load all the records from the text files.
	std::vector<float> temperatures = ReadFile(path);

	// Stop Timer.
	Timer timerStop = Clock::now();

	std::printf("\rSorting Data...");

	// Sort the Records
	Data data = SortData(platform_id, device_id, temperatures);

	// Display all the information and statics calculated.
	#pragma region Display Information

	Timer timerTotalStop = Clock::now();

	std::printf("\r_____________________________________________");
	
	std::cout << endl;
	std::cout << endl;
	std::cout << "Information					Kernel Run Times " << endl;
	std::cout << "____________________________________________________________________________________" << endl;
	std::cout << endl;

	std::cout << "Filename: " << path << endl;
	std::cout << "Total of " << temperatures.size() << " temperatures processed." << "	| Read time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count() << "(ms)" << std::endl;
	std::cout << "AVG: " << data.avg << "					| Addition time: " << data.sumTime << "(ms)" << std::endl;
	std::cout << "MIN: " << data.min << "					| Minimum time: " << data.minTime << "(ms)" << std::endl;
	std::cout << "MAX: " << data.max << "						| Maximum time: " << data.maxTime << "(ms)" << std::endl;
	std::cout << "STD: " << data.stdv << "					| Standard Deviation time: " << data.stdvTime << "(ms)" << std::endl;
	std::cout << "MED: " << data.median << "					| Sort time: " << data.sortTime << "(ms)" << std::endl;
	std::cout << endl;
	std::cout << "						| Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(timerTotalStop - timerTotalStart).count() << "(ms)" << std::endl;

	std::cout << endl;

	std::cout << "EXTRAS" << std::endl;
	std::cout << "SUM: " << data.sum << std::endl;
	std::cout << "VAR: " << data.variance << std::endl;
	std::cout << "1QT: " << data.LQT << std::endl;
	std::cout << "3QT: " << data.HQT << std::endl;

	#pragma endregion

	int x;
	cin >> x;

	return 0;
}

std::vector<float> ReadFile(const char* path)
{
	std::vector<float> temperatures;

	// Read the file.
	std::ifstream file(path);
	std::string data;

	// Read each line in file.
	while (std::getline(file, data))
	{
		string element;
		stringstream line(data);
		vector<string> elements;

		// Go through each element on the line, and place them into the elements vector. 
		while (line >> element)
			elements.push_back(element);

		// Add the last element to the temperatures vector.
		temperatures.push_back(std::stof(elements[5]));
	}

	return temperatures;
}

// This is the main part of the program, where the sort is called and statistics are gathered from it.
Data SortData(int platform_id, int device_id, std::vector<float> temperatures)
{
	//Load Kernels
	Kernel sortKernel(platform_id, device_id, "my_kernels.cl", "float_selection_sort_local", true);
	Kernel sortKernel2(platform_id, device_id, "my_kernels.cl", "float_selection_sort", false);
	Kernel sortKernel3(platform_id, device_id, "my_kernels.cl", "sort_oddeven", false);
	Kernel sumKernel(platform_id, device_id, "my_kernels.cl", "float_reduce_add_4", true);
	Kernel minKernel(platform_id, device_id, "my_kernels.cl", "float_reduce_min", true);
	Kernel maxKernel(platform_id, device_id, "my_kernels.cl", "float_reduce_max", true);
	Kernel varKernel(platform_id, device_id, "my_kernels.cl", "float_reduce_variance", false);
	//	^^Initialise	^^Platform	  ^^Device		^^FileName			^^Command		 ^^Scratch

	// Store the output from the kernel into a structure.
	Data data = Data();

	#pragma region SortData

	// Start timer.
	Timer timerStart = Clock::now();
	
	// Single sort function.
	std::vector<float> E = Sort(sortKernel, temperatures);

	// Multiple sort function - used for testing sort algorithms and finding out which is the most efficient.
	//std::vector<float> E = SortTest(5, sortKernel, temperatures);

	// End timer.
	Timer timerStop = Clock::now();

	// Display how long it took in milliseconds to complete the above function.
	data.sortTime = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count();

	#pragma endregion

	// This section records the data found ready to be displayed.
	#pragma region SetData

	// Provide user feedback.
	std::printf("\rCalculating Statistics...");

	// Create a variable for easy access to the size of the A vector.
	size_t size = E.size();

		#pragma region Min-Max-Median

		timerStart = Clock::now();

		data.min = minKernel.RunKernel(temperatures)[0];						// • The smallest temperature, first element of the output (sorted) vector.

		timerStop = Clock::now();
		data.minTime = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count();

		data.LQT = E[size / 4];													// • The 1st quartile, the element at the quarter point.

		if (size % 2 == 0) data.median = (E[size / 2 - 1] + E[size / 2]) / 2;	// • The median, If the size of vector A is even, the average between the two middle elements are calculated. (size = 14, element 7 & 8 are taken added together and divided by 2)
		else data.median = E[size / 2];											//	 If it is odd, the middle element is used. (size = 17. element 9 is chosen)	
		
		data.HQT = E[(3 * size) / 4];											// • The 3rd quartile, three quarters.

		timerStart = Clock::now();

		data.max = maxKernel.RunKernel(temperatures)[0];						// • The highest temperature, last element of the output (sorted) vector.

		timerStop = Clock::now();
		data.maxTime = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count();

		#pragma endregion

		#pragma region Average

		timerStart = Clock::now();

		data.sum = (float)sumKernel.RunKernel(temperatures)[0];					// • The sum of all elements in the vector reduced via kernel.

		data.avg = (data.sum / size);											// • The mean is acquired by dividing the sum by the total size of the input vector. 

		timerStop = Clock::now();
		data.sumTime = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count();

		#pragma endregion

		#pragma region Standard Deviation
		
		timerStart = Clock::now();

		vector<float> variance = varKernel.RunKernel(temperatures, data.avg);	// • The variance is obtained by subtracting each element by the mean, and then squaring and added them together.
		for (int i = 0; i < size; i++)											//   Once all elements have been processed, this sum is then divided by the size of the vector.
			data.variance += variance[i];										
		data.variance /= size;

		data.stdv = sqrt(data.variance);										// • The standard deviation is simply obtained by through the squareroot of variance.
	
		timerStop = Clock::now();
		data.stdvTime = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count();

		#pragma endregion

	#pragma endregion

	return data;
}

// Tests the sorting algoithm and displays an average time. Temperatures are still returned.
std::vector<float> SortTest(int count, Kernel sortKernel, std::vector<float> temperatures)
{
	// Create a vector that will hold all of the times.
	std::vector <float> times;
	// Create an E vector that will be used to hold the sorted data.
	std::vector<float> E;

	// Loop through the sorting process for x number of times.
	for (int i = 0; i < count; i++)
	{
		std::printf("\rSorting...             ");

		Timer timerStart = Clock::now();

		E = Sort(sortKernel, temperatures);

		Timer timerStop = Clock::now();

		// Add all the times to a vector.
		times.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart).count());
	}

	// Create a mean variable to be used for calculating the average between sort times.
	float mean = 0;

	std::printf("\rTimes:             ");
	cout << endl;

	// Loop through each time found in the times vector.
	for (float time : times)
	{
		// Display the time.
		cout << time << endl;
		// Add it to the mean variable.
		mean += time;
	}

	cout << "Total Sort Time: " << mean << "(ms)" << endl;

	// Calculate mean.
	mean /= times.size();

	// Display the mean.
	cout << "Mean Sort Time: " << mean << "(ms)" << endl;

	return E;
}

std::vector<float> Sort(Kernel sortKernel, std::vector<float> temperatures)
{
	// Create vectors to be used for sorting.
	std::vector<float> A;
	std::vector<float> B;
	std::vector<float> C;
	std::vector<float> D;
	std::vector<float> output;

	// Divide the temperatures equally between four vectors.
	for (int i = 0; i < temperatures.size(); i+=2)
	{
		if (i < temperatures.size() / 2)
		{
			A.push_back(temperatures[i]);
			if(i + 1 < temperatures.size() / 2)
				B.push_back(temperatures[i + 1]);
		}
		else
		{
			C.push_back(temperatures[i]);
			if (i + 1 < temperatures.size())
				D.push_back(temperatures[i + 1]);
		}
	}

	// Sort all of the partitions.

	A = sortKernel.RunKernel(A);

	B = sortKernel.RunKernel(B);

	std::printf("\rSorting Partitions...");

	C = sortKernel.RunKernel(C);

	D = sortKernel.RunKernel(D);

	// Two iterators are created to be used in the next for loop.
	int iteratorA = 0;
	int iteratorB = 0;
	int iteratorC = 0;
	int iteratorD = 0;

	// As the two vectors have now been sorted, a final sort will take place merging the two vectors.
	for (float temperature : temperatures)
	{
		float AB;
		float CD;

		bool Abool = false;
		bool Cbool = false;

		// If the element in the A vector is lower than the one in B,C,D...
		if (A[iteratorA] < B[iteratorB])
		{
			// Add it to the E vector.
			AB = A[iteratorA];
			// This bool is used to notify which vector was the lowest.
			Abool = true;
		}
		else
			// Do the same but for B.
			AB = B[iteratorB];

		if (C[iteratorC] < D[iteratorD])
		{
			CD = C[iteratorC];
			Cbool = true;
		}
		else
			CD = D[iteratorD];

		// Increase the iterator on the lowest vector so that the element is not repeated.
		if (AB < CD)
		{
			output.push_back(AB);

			if(Abool)
				iteratorA++;
			else
				iteratorB++;
		}
		else
		{
			output.push_back(CD);

			if (Cbool)
				iteratorC++;
			else
				iteratorD++;
		}

		// If the size of the E vector is the same as temperature, the loop is broken.
		if (output.size() == temperatures.size())
			break;
	}

	return output;
}

