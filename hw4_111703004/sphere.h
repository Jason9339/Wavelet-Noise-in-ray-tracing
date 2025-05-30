#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"
#include "ray.h"
#include <algorithm> // for fmax

class sphere : public hittable {
public:
    vec3 center;
    float radius;
    float w_r;
    float w_t;
    vec3 Kd;  // diffuse coefficient

    // 保證半徑非負以避免非法幾何
    sphere() = default;
    sphere(
        vec3 c,
        float r,
        float w_ri = 0.0f,
        float w_ti = 0.0f,
        vec3 kd = vec3(1.0f, 1.0f, 1.0f)  // 預設白色
    ) : center(c), radius(r), w_r(w_ri), w_t(w_ti), Kd(kd) {}

    bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        // 假設球心為 C，射線為 P(t) = O + tD，球體方程為：|P(t) - C|^2 = r^2
        // 展開：|O + tD - C|^2 = r^2
        //      = (O - C + tD) · (O - C + tD)
        //      = (O - C)·(O - C) + 2t(D·(O - C)) + t^2(D·D)
        //      = t^2(D·D) + 2t(D·(O - C)) + (O - C)·(O - C) - r^2 = 0
        // 一般形式：a·t^2 + b·t + c = 0
        // 可得：
        //   a = D·D
        //   b = 2·D·(O - C)
        //   c = (O - C)·(O - C) - r^2

        // 若設 h = D·(C - O) 則 b = -2h
        // 所以可以改寫為：t^2·a - 2t·h + c = 0
        
        vec3 oc = center - r.origin(); // 向量從 ray 原點指向球心
        float a = r.direction().squared_length(); 
        float h = dot(r.direction(), oc);         
        float c_term = oc.squared_length() - radius * radius; 

        float discriminant = h * h - a * c_term; // 判別式

        if (discriminant < 0.0f) return false;

        float sqrt_discriminant = sqrt(discriminant);

        // 優先取靠近的交點（從球外部打進來）
        float root = (h - sqrt_discriminant) / a;
        if (root < t_min || root > t_max) {
            // 如果不合法，再嘗試另一個解（可能是從球內部打出）
            root = (h + sqrt_discriminant) / a;
            if (root < t_min || root > t_max)
                return false;
        }

        // 設定交點資訊
        rec.t = root;
        rec.p = r.point_at_parameter(rec.t);
        rec.Kd = Kd;

        // 利用 outward normal 決定法線方向是否朝向射線外部
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal); // 自動判斷 front_face 並設定 normal

        return true;
    }
};

#endif
