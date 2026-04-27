#include <iostream>
#include <cmath>

#include "nforge/nforge.h"

struct SphereSlideParams {
    float dt = 0.001;
    float grav = 9.81;
    float mass = 1;
    float radius = 1;
    float initalXSpeed = 0.001;
};

struct SphereSlideResults {
    Tensor position = Tensor({2}, 0.0f);
    Tensor speed = Tensor({2}, 0.0f);
    float t;

    SphereSlideResults(Tensor _position, Tensor _speed, float _t)
        : position(_position), speed(_speed), t(_t) {}

    bool approxEqual(const SphereSlideResults& other, float tol = 1e-3f) const {
        auto p1 = position.toVector(), p2 = other.position.toVector();
        auto s1 = speed.toVector(),    s2 = other.speed.toVector();

        for (int i = 0; i < 2; i++) {
            if (std::abs(p1[i] - p2[i]) > tol) return false;
            if (std::abs(s1[i] - s2[i]) > tol) return false;
        }
        return std::abs(t - other.t) < tol;
    }
};

float length(std::vector<float> x) {
    float sum = 0;
    for (float e : x) {
        sum += e * e;
    }

    sum = powf(sum, 1 / (float)x.size());
    return sum;
}

SphereSlideResults simulateSphereSlide(SphereSlideParams params) {
    Tensor s({2}, 0), v({2}, 0), a({2}, 0);
    Tensor G({2}, 0);
    G[1] = -params.mass * params.grav;

    v[0] = params.initalXSpeed;
    s[1] = params.radius;

    float t = 0;
    Tensor N = Tensor({2}, 0) - G;

    while (length(N.toVector()) > 0) {
        Tensor aGrav = G / Tensor(params.mass);
        // p = s + v * dt + a/2 * dt^2
        Tensor positionFree = s + v * Tensor(params.dt) + aGrav * Tensor(0.5f * params.dt * params.dt);

        float distCenter = length(positionFree.toVector());

        if (distCenter >= params.radius) {
            N = Tensor({2}, 0);
            break;
        }

        // Position mapped to sphere
        Tensor positionNext = positionFree * Tensor(params.radius / distCenter);

        
        Tensor correction = positionNext - positionFree;
        // Normal force is correction 
        // F = m * dx / dt^2
        N = correction * Tensor(params.mass / (params.dt * params.dt)); 

        v = (positionNext - s) * Tensor(1.0f / params.dt);
        s = positionNext;

        t += params.dt;
    }

    SphereSlideResults res{s, v, t};
    return res;
}
