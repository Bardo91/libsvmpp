//
//
//
//

#include <cassert>
#include <Svmpp.h>

using namespace std;
using namespace svmpp;

int main(int _argc, char **_argv) {
	vector<double> x1 = {0,0};
	vector<double> x2 = {0,1};
	vector<double> x3 = {1,0};
	vector<double> x4 = {1,1};

	TrainSet set;
	set.addEntry(x1, 0);
	set.addEntry(x2, 0);
	set.addEntry(x3, 0);
	set.addEntry(x4, 1);

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

	svm.train(params, set);
	
	
	Query query({2,2}, 1);
	
	double res = svm.predict(query);

	assert(res == 1);

	return 1;
}