#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

using namespace std;

class hittable_list : public hittable {
  public:
    // shared_ptr 內有一個計數器，use_count() 可以回傳有幾個 shared_ptr 共享同一物件
    // shared_ptr 有自動記憶體管理(無需手動 delete)
    vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (const auto& object : objects) {
            if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    aabb bounding_box() const override {
        aabb output_box;
        bool first_box = true;

        for (const auto& object : objects) {
            aabb temp_box = object->bounding_box();
            output_box = first_box ? temp_box : aabb(output_box, temp_box);
            first_box = false;
        }

        return output_box;
    }
};

#endif
