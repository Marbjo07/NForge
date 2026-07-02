#ifndef UTILS_H
#define UTILS_H

#include "nforge/nforge.h"

static constexpr Backend backends[] = {Backend::CPU,

#ifdef NFORGE_WITH_CUDA
                                       Backend::CUDA
#endif

};

template <typename A, typename B>
bool tensor_equal(const A& a, const B& b) {
	return a.isEqual(b);
}

template <typename A, typename B>
bool tensor_not_equal(const A& a, const B& b) {
	return a.isNotEqual(b);
}

static std::string getBackendString(Backend backend) {
	switch (backend) {
		case Backend::CPU:
			return "CPU";
		case Backend::CUDA:
			return "CUDA";
		default:
			return "UNKNOWN";
	}
}

#endif  // UTILS_H