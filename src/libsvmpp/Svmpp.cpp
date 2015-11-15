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


#include "Svmpp.h"

#include <cassert>
#include <algorithm>

namespace svmpp {
	//-----------------------------------------------------------------------------------------------------------------
	// Train struct
	//-----------------------------------------------------------------------------------------------------------------
	void TrainSet::addEntry(double * _x, unsigned _dimension, double _y) {
		addEntry(std::vector<double>(_x, _x+_dimension), _y);
	}

	//-----------------------------------------------------------------------------------------------------------------
	void TrainSet::addEntry(const std::vector<double>& _x, double _y) {
		mX.push_back(_x);
		mY.push_back(_y);
	}

	//-----------------------------------------------------------------------------------------------------------------
	void TrainSet::addEntries(const std::vector<std::vector<double>>& _X, const std::vector<double>& _Y){
		mX.insert(mX.end(), _X.begin(), _X.end());
		mY.insert(mY.end(), _Y.begin(), _Y.end());
	}

	//-----------------------------------------------------------------------------------------------------------------
	TrainSet::Problem TrainSet::problem() const {
		Problem problem;
		
		assert(mX.size() == mY.size());

		// Set size of train set.
		problem.l = mX.size();

		// Set labels of data
		problem.y = new double[problem.l];
		for (unsigned i = 0; i < mY.size(); i++) {
			problem.y[i] = mY[i];
		}

		// Set data
		int dims = mX[0].size();
		problem.x = new svm_node*[problem.l];
		for (unsigned i = 0; i < mX.size(); i++) {
			problem.x[i] = new svm_node[dims+1];
			for (unsigned j = 0; j < dims; j++) {
				problem.x[i][j].index = j;
				problem.x[i][j].value = mX[i][j];
			}
			problem.x[i][dims].index = -1;
		}

		return problem;
	}


	//-----------------------------------------------------------------------------------------------------------------
	// Query struct
	//-----------------------------------------------------------------------------------------------------------------
	Query::Query(double * _x, unsigned _dimension): Query(std::vector<double>(_x, _x + _dimension)) {
	}

	//-----------------------------------------------------------------------------------------------------------------
	Query::Query(const std::vector<double>& _x) {
		int dims = _x.size();
		mNode = new svm_node[dims+1];
		for (unsigned i = 0; i < dims; i++) {
			mNode[i].index = i;
			mNode[i].value = _x[i];
		}
		mNode[dims].index = -1;
	}

	//-----------------------------------------------------------------------------------------------------------------
	Query::Node Query::node() const {
		return mNode;
	}

	//-----------------------------------------------------------------------------------------------------------------
	// Svm class
	//-----------------------------------------------------------------------------------------------------------------
	void Svm::save(std::string _file) const {
		// 666 Not implemented yet.
	}

	//-----------------------------------------------------------------------------------------------------------------
	void Svm::load(std::string _file) {
		// 666 Not implemented yet.
	}

	//-----------------------------------------------------------------------------------------------------------------
	void Svm::train(const Params & _params, const TrainSet & _trainSet) {
		mModel = *svm_train(&(_trainSet.problem()), &_params);
	}

	//-----------------------------------------------------------------------------------------------------------------
	double Svm::predict(const Query & _query) const {
		return svm_predict(&mModel, _query.node());
	}

	//-----------------------------------------------------------------------------------------------------------------
	double Svm::predict(const Query & _query, std::vector<double> &_probs) const {
		double *probs = new double[mModel.nr_class];
		svm_predict_probability(&mModel, _query.node(), probs);
		
		int maxIndex;
		double maxProb=0;
		for (unsigned i = 0; i < mModel.nr_class;i++) {
			_probs.push_back(probs[i]);
			if (maxProb < probs[i]) {
				maxProb = probs[i];
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	//-----------------------------------------------------------------------------------------------------------------
	bool Svm::hasProbabilities() const {
		return svm_check_probability_model(&mModel);
	}

	//-----------------------------------------------------------------------------------------------------------------
	Svm::Params Svm::params() const {
		return mParams;
	}

}	// namespace svmpp

