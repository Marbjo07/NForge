#ifndef ERROR_HANDLER_H 
#define ERROR_HANDLER_H

#define LOG(x) std::cerr

namespace ErrorHandler {
	namespace Flags {
		static bool errorNotImplementedEnabled = true;
		static bool cudaSyncLogEnabled = true;
	}

	enum class ErrorTypes {
		NOT_IMPLEMENTED
	};
	
	/*
	ErrorHandler::log
	Why:
		Common error are in a standarized format
	Takes: 
		- ErrorTypes errorType: error type
		- std::string where: A string of where the error occured, usually the function name
	*/
	static void log(ErrorTypes errorType, std::string where) {
		switch (errorType) {
		case (ErrorTypes::NOT_IMPLEMENTED):
			if (Flags::errorNotImplementedEnabled) {
				LOG(ERROR) << where << "(..) not implemented";
			}
			break;
		default:
			LOG(ERROR) << "Error type not found. ironic ;)";
			break;
		}
	}

	/*
	ErrorHandler::cudaSyncLog
	What:
		- Does not garantee that the device is synced, as the function can be disabled
		- Syncs device and logs message if an error is detected
	Takes:
		- std::string errorDescription: a short description of the error
	*/
	static void cudaSyncLog(std::string errorDescription) {
		if (!ErrorHandler::Flags::cudaSyncLogEnabled) return;

		#if 0

		cudaError_t status = cudaDeviceSynchronize();
		if (status != cudaError_t::cudaSuccess) {
			LOG(ERROR) << "cuda error, " << errorDescription << " status: " << status;
		}
		#endif
	}
};

#endif // ERROR_HANDLER_H