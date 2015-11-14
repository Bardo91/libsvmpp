//
//
//
//
//


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
		
		bool addEntry(double *_x, double _y);
		bool addEntry(const std::vector<double> &_x, double _y);

		bool addEntries(const std::vector<std::vector<double>> &_X, const std::vector<double> &_Y);

		Problem problem() const;
	private:
		Problem mProblem;
	};	// Struct TrainSet


	//-----------------------------------------------------------------------------------------------------------------
	struct Query {
	public:
		typedef svm_node Node;

		Query(double *_x, double _y);
		Query(const std::vector<double> &_x, double _y);

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