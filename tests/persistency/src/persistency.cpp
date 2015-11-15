//////////////////////////////////////////////////////////////////////////////////////////
//																						//
//	The MIT License (MIT)																//
//																						//
//	Copyright (c) 2015 Pablo Ramon Soria												//
//																						//
//	Permission is hereby granted, free of charge, to any person obtaining a copy		//
//	of this software and associated documentation files (the "Software"), to deal		//
//	in the Software without restriction, including without limitation the rights		//
//	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell			//
//	copies of the Software, and to permit persons to whom the Software is				//
//	furnished to do so, subject to the following conditions:							//
//																						//
//	The above copyright notice and this permission notice shall be included in all		//
//	copies or substantial portions of the Software.										//
//																						//
//	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR			//
//	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
//	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE			//
//	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER				//
//	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,		//
//	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE		//
//	SOFTWARE.																			//
//																						//
//////////////////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <Svmpp.h>

using namespace std;
using namespace svmpp;

int main(int _argc, char **_argv) {
	TrainSet set;
	set.addEntry({0,0}, 0);
	set.addEntry({0,1}, 0);
	set.addEntry({1,0}, 0);
	set.addEntry({1,1}, 1);

	Svm svm;

	// Setting parameters
	Svm::Params params;
	params.svm_type = C_SVC;
	params.kernel_type = LINEAR;
	params.cache_size = 100;
	params.C = 1;
	params.eps = 1e-5;
	params.p = 0.1;
	params.shrinking = 0;
	params.probability = 0;
	params.nr_weight = 0;
	params.weight_label = nullptr;
	params.weight = nullptr;

	// Train
	svm.train(params, set);

	// save and load model
	svm.save("testModel");

	Svm svmNew;
	svmNew.load("testModel");

	// Test
	Query query1({2,2});
	double res = svmNew.predict(query1);
	assert(res == 1);

	Query query2({0,0});
	res = svmNew.predict(query2);
	assert(res == 0);
}