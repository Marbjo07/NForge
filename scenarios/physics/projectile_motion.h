#include <cmath>
#include <iostream>

#include "nforge/nforge.h"

constexpr double PI = 3.14159265358979323846;

struct ProjectileMotionParams {
	float dt = 0.001;
	float grav = 9.81;
	float initialSpeed = 10;
	float angle = 10;  // degrees
};

struct ProjectileMotionResults {
	Tensor position = Tensor({2}, 0.0f);
	Tensor speed = Tensor({2}, 0.0f);
	float t;

	ProjectileMotionResults(Tensor _position, Tensor _speed, float _t)
	    : position(_position), speed(_speed), t(_t) {}
};

ProjectileMotionResults simulateProjectileMotion(ProjectileMotionParams params) {
	Tensor s({2}, 0.0f), v({2}, 0.0f), a({2}, 0.0f);

	float angleRad = params.angle * PI / 180.0;
	v[0] = params.initialSpeed * std::cos(angleRad);
	v[1] = params.initialSpeed * std::sin(angleRad);

	a[1] = -params.grav;

	float t = 0;
	while (s.toVector()[1] >= 0) {
		v += a * params.dt;
		s += v * params.dt;
		t += params.dt;
	}

	ProjectileMotionResults res(s, v, t);
	return res;
}
