
#include "SK_Serializer.hpp"

namespace {

    using namespace nlohmann;

    constexpr float NAN_f = NAN;

    inline
        void add(json& it, std::string k, sl::float3& in) {
        it[k] = { in.x, in.y, in.z };
    }

    inline
        void add(json& it, std::string k, sl::float4& in) {
        it[k] = { in.x, in.y, in.z, in.w };
    }

    template<typename T>
    inline
        void add(json& it, std::string k, std::vector<sl::Vector2<T>>& in) {
        for (auto& i_ : in)
            it[k].push_back({ i_.x, i_.y });
    }

    inline
        void add(json& it, std::string k, std::vector<sl::float3>& in) {
        for (auto& i_ : in)
            it[k].push_back({ i_.x, i_.y, i_.z });
    }

    inline
        void add(json& it, std::string k, std::vector<sl::float4>& in) {
        for (auto& i_ : in)
            it[k].push_back({ i_.x, i_.y, i_.z, i_.w });
    }

    inline
        void add(json& it, std::string k, std::vector<float>& in) {
        for (auto& i_ : in)
            it[k].push_back(i_);
    }
    inline
        void add(json& it, std::string k, std::vector<std::array<float,6>>& in) {
        for (auto& i_ : in)
            it[k].push_back({i_[0], i_[1], i_[2], i_[3], i_[4], i_[5]});
    }

    json serialize_(sl::BodyData& data) {
        json out;
        out["id"] = data.id;
        out["unique_object_id"] = std::string(data.unique_object_id.c_str());
        out["tracking_state"] = data.tracking_state;
        out["action_state"] = data.action_state;
        add(out, "position", data.position);
        add(out, "velocity", data.velocity);
        out["position_covariance"] = { data.position_covariance[0], data.position_covariance[1], data.position_covariance[2],
                                        data.position_covariance[3], data.position_covariance[4], data.position_covariance[5] };
        add(out, "bounding_box_2d", data.bounding_box_2d);
        out["confidence"] = data.confidence;
        add(out, "bounding_box", data.bounding_box);
        add(out, "dimensions", data.dimensions);
        add(out, "keypoint_2d", data.keypoint_2d);
        add(out, "keypoint", data.keypoint);
        add(out, "keypoint_cov", data.keypoint_covariances);
        add(out, "head_bounding_box_2d", data.head_bounding_box_2d);
        add(out, "head_bounding_box", data.head_bounding_box);
        add(out, "head_position", data.head_position);
        add(out, "keypoint_confidence", data.keypoint_confidence);
        add(out, "local_position_per_joint", data.local_position_per_joint);
        add(out, "local_orientation_per_joint", data.local_orientation_per_joint);
        add(out, "global_root_orientation", data.global_root_orientation);
        return out;
    }

    template<typename T>
    inline
        void get(T& data, json in, std::string k) {
        data = in[k].get<T>();
    }

    inline 
    void read(sl::float3& data, json in, std::string k) {
        auto i_ = in[k];
        if (i_[0].is_null()) 
            data = sl::float3(NAN_f, NAN_f, NAN_f);
        else
            data = sl::float3(i_[0].get<float>(), i_[1].get<float>(), i_[2].get<float>());
    }

    inline
        void read(sl::float4& data, json in, std::string k) {
        auto i_ = in[k];
        data = sl::float4(i_[0].get<float>(), i_[1].get<float>(), i_[2].get<float>(), i_[3].get<float>());
    }

    inline
        void read(float* data, json in, std::string k) {
        auto i_ = in[k];
        for (int i = 0; i < 6; i++)
            data[i] = i_[i].get<float>();
    }

    template<typename T>
    inline
        void read(std::vector<sl::Vector2<T>>& data, json in, std::string k) {

        for(auto &i_ : in[k])
            if (i_[0].is_null())
                data.push_back(sl::Vector2<T>(0, 0));
            else
                data.push_back(sl::Vector2<T>(i_[0].get<T>(), i_[1].get<T>()));
    }

    inline
        void read(std::vector<float>& data, json in, std::string k) {
        for (auto& i_ : in[k])
            if (i_.is_null())
                data.push_back(NAN_f);
            else
                data.push_back(i_.get<float>());
    }

    template<typename T>
    inline
        void read(std::vector<sl::Vector3<T>>& data, json in, std::string k) {
        for (auto& i_ : in[k])
            if (i_[0].is_null())
                data.push_back(sl::Vector3<T>(NAN_f, NAN_f, NAN_f));
            else
            data.push_back(sl::Vector3<T>(i_[0].get<T>(), i_[1].get<T>(), i_[2].get<T>()));
    }

    template<typename T>
    inline
        void read(std::vector<sl::Vector4<T>>& data, json in, std::string k) {
        for (auto& i_ : in[k])
            if (i_[0].is_null())
                data.push_back(sl::Vector4<T>(NAN_f, NAN_f, NAN_f, NAN_f));
            else
            data.push_back(sl::Vector4<T>(i_[0].get<T>(), i_[1].get<T>(), i_[2].get<T>(), i_[3].get<T>()));
    }

    inline
        void read(std::vector<std::array<float, 6>>& data, json in, std::string k) {
        for (auto& i_ : in[k])
        {
            std::array<float,6> t;
            if (i_[0].is_null())
            {
                for (int i = 0; i < 6; i++)
                    t[i] = NAN_f;
            }
            else{
                for (int i = 0; i < 6; i++)
                    t[i] = i_[i].get<float>();
            }
            data.push_back(t);
        }
    }


    sl::BodyData deserialize_(json& data) {
        sl::BodyData out;
        get(out.id, data, "id");
        std::string str;
        get(str, data, "unique_object_id");
        out.unique_object_id.set(str.c_str());
        get(out.tracking_state, data, "tracking_state");
        get(out.action_state, data, "action_state");
        read(out.position, data, "position");
        read(out.velocity, data, "velocity");
        read(out.position_covariance, data, "position_covariance");
        read(out.bounding_box_2d, data, "bounding_box_2d");
        get(out.confidence, data, "confidence");
        read(out.bounding_box, data, "bounding_box");
        read(out.dimensions, data, "dimensions");
        read(out.keypoint_2d, data, "keypoint_2d");
        read(out.keypoint, data, "keypoint");
        read(out.head_bounding_box_2d, data, "head_bounding_box_2d");
        read(out.head_bounding_box, data, "head_bounding_box");
        read(out.head_position, data, "head_position");
        read(out.keypoint_confidence, data, "keypoint_confidence");
        read(out.local_position_per_joint, data, "local_position_per_joint");
        read(out.local_orientation_per_joint, data, "local_orientation_per_joint");
        read(out.global_root_orientation, data, "global_root_orientation");
        read(out.keypoint_covariances, data, "keypoint_cov");
        return out;
    }
}

std::string string_to_hex(const std::string& in) {
    std::stringstream ss;

    ss << std::hex << std::setfill('0');
    for (size_t i = 0; in.length() > i; ++i) {
        ss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(in[i]));
    }

    return ss.str(); 
}

std::string hex_to_string(const std::string& in) {
    std::string output;

    if ((in.length() % 2) != 0) {
        throw std::runtime_error("String is not valid length ...");
    }

    size_t cnt = in.length() / 2;

    for (size_t i = 0; cnt > i; ++i) {
        uint32_t s = 0;
        std::stringstream ss;
        ss << std::hex << in.substr(i * 2, 2);
        ss >> s;

        output.push_back(static_cast<unsigned char>(s));
    }

    return output;
}

nlohmann::json sk::serialize(sl::Bodies& data) {
    nlohmann::json out;
    out["is_new"] = data.is_new;
    out["is_tracked"] = data.is_tracked;
    out["timestamp"] = data.timestamp.data_ns;
    for (auto& it : data.body_list)
        out["body_list"].push_back(serialize_(it));
    return out;
}

sl::Bodies sk::deserialize(nlohmann::json & data) {
    sl::Bodies out;
    get(out.is_new, data, "is_new");
    get(out.is_tracked, data, "is_tracked");
    get(out.timestamp.data_ns, data, "timestamp");
    for (auto& it : data["body_list"])
        out.body_list.push_back(deserialize_(it));
    return out;
}
