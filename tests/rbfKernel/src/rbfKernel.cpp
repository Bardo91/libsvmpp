//
//
//
//

#include <cassert>
#include <cmath>
#include <Svmpp.h>

using namespace std;
using namespace svmpp;

int main(int _argc, char **_argv) {
	
	TrainSet set;

	for (unsigned i = 0; i < 90; i++) {
		set.addEntry({2*cos(4*(i*3.1416/180)), 2*sin(4*(i*3.1416/180))}, 0);
	}
	
	for (unsigned i = 0; i < 90; i++) {
		set.addEntry({0.5*cos(4*(i*3.1416/180)), 0.5*sin(4*(i*3.1416/180))}, 1);
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
	params.probability = 0;
	params.nr_weight = 0;
	params.weight_label = nullptr;
	params.weight = nullptr;


	svm.train(params, set);

	// Evaluate data
	Query query1({0,0});
	double res1 = svm.predict(query1);
	assert(res1 == 1);

	Query query2({3,3});
	double res2 = svm.predict(query2);
	assert(res2 == 0);

}