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


#ifndef LIBSVMPP_SVMPP_H_
#define LIBSVMPP_SVMPP_H_

#include <svm.h>

#include <string>
#include <vector>

namespace svmpp {
	//-----------------------------------------------------------------------------------------------------------------
	/// Struct that holds information for training the svm
	struct TrainSet {
	public:
		typedef svm_problem Problem;
		
		void addEntry(double *_x, double _y);
		void addEntry(const std::vector<double> &_x, double _y);

		void addEntries(const std::vector<std::vector<double>> &_X, const std::vector<double> &_Y);

		Problem problem() const;
	private:
		std::vector<std::vector<double>>	mX;
		std::vector<double>					mY;
	};	// Struct TrainSet


	//-----------------------------------------------------------------------------------------------------------------
	struct Query {
	public:
		typedef svm_node *Node;

		Query(double *_x);
		Query(const std::vector<double> &_x);

		Node node() const;
	private:
		Node mNode;
	};	// Struct Query


	//-----------------------------------------------------------------------------------------------------------------
	/// SVM wrapper of libsvm svm implementation.
	class Svm {
	public:
		/// Structure that holds SVM params. Renamed svm_params from libsvm
		typedef svm_parameter Params;

		/// Save SVM into a file.
		void save(std::string _file) const;

		/// Load SVM from a file.
		void load(std::string _file);

		/// Train SVM with the given dataset.
		void train(const Params &_params, const TrainSet &_trainSet);

		///
		double predict(const Query &_query) const;
		
		///
		double predict(const Query &_query, double *_probs) const;

		///
		bool hasProbabilities() const;

		Params params() const;
	private:
		typedef svm_model	Model;

		Model	mModel;
		Params	mParams;
	};

}	//	namespace svmpp

#endif	//	LIBSVMPP_SVMPP_H_