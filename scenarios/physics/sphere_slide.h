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
};

float length(std::vector<float> x) {
    float sum = 0;
    for (float e : x) {
        sum += e * e;
    }


    return sqrt(sum);
}

SphereSlideResults simulateSphereSlide(SphereSlideParams params) {
    Tensor s({2}, 0);
    s[1] = params.radius;
    
    Tensor v({2}, 0);
    v[0] = params.initalXSpeed;
    
    Tensor G({2}, 0);
    G[1] = -params.mass * params.grav;
    
    Tensor a = G / params.mass;

    float t = 0;

    while (true) {
        // p = s + v * dt + a/2 * dt^2
        Tensor position = s + v * params.dt + a * 0.5  * params.dt * params.dt;

        float distCenter = length(position.toVector());
        if (distCenter >= params.radius) { // does not fall into the sphere
            break;
        }

        // Position mapped to sphere
        position *= Tensor(params.radius / distCenter);

        v = (position - s) * (1 / params.dt);
        s = position;

        t += params.dt;
    }

    SphereSlideResults res{s, v, t};
    return res;
}
