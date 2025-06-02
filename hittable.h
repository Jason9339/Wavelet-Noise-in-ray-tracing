#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include <memory>
#include "aabb.h"

using std::shared_ptr;

class material;

struct hit_record {
    vec3 p;
    vec3 normal;
    float t;
    // float w_r = 0.0f;
    // float w_t = 0.0f;
    // vec3 Kd;
    double u;
    double v;
    shared_ptr<material> mat;

    // 正面追蹤，讓法線始終指向射線
    bool front_face;
    
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
  public:
    virtual ~hittable() = default;
    // tmin：避免自相交 (Self-Intersection) / 處理浮點數精度問題 (Shadow Acne / Surface Acne)
    // ray_tmax：定義最遠的可接受交點距離 => 尋找最近的交點、限定搜索範圍的終點（例如陰影射線）、性能優化
    virtual bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const = 0;

    virtual aabb bounding_box() const = 0;
};

#endif