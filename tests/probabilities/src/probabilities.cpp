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
#include <cmath>
#include <Svmpp.h>

using namespace std;
using namespace svmpp;

int main(int _argc, char **_argv) {
	
	TrainSet set;

	for (unsigned i = 0; i < 90; i++) {
		set.addEntry({0.5*cos(4*(i*3.1416/180)), 0.5*sin(4*(i*3.1416/180))}, 0);
	}
	
	for (unsigned i = 0; i < 90; i++) {
		set.addEntry({2.0*cos(4*(i*3.1416/180)), 2.0*sin(4*(i*3.1416/180))}, 1);
	}

	Svm svm;
	
	// Setting parameters
	Svm::Params params;
	params.svm_type = C_SVC;
	params.kernel_type = RBF;
	params.cache_size = 100;
	params.gamma = 0.01;
	params.C = 10;
	params.eps = 1e-5;
	params.p = 0.1;
	params.shrinking = 0;
	params.probability = 1;
	params.nr_weight = 0;
	params.weight_label = nullptr;
	params.weight = nullptr;


	svm.train(params, set);
	assert(svm.hasProbabilities());
	// Evaluate data
	Query query1({0,0});
	std::vector<double> probs1;
	double res1 = svm.predict(query1, probs1);
	assert(res1 == 0);
	assert(probs1[0] > probs1[1]);

	Query query2({5,5});
	std::vector<double> probs2;
	double res2 = svm.predict(query2, probs2);
	assert(res2 == 1);
	assert(probs2[1] > probs2[0]);
}